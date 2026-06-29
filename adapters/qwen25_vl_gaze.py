from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from adapters.qwen25_vl import Qwen25VLAdapter, _resolve_device_map
from vlm_eval.gaze_regions import assign_panels_to_tokens, region_positions_from_ids


ATTENTION_IMPL_NAME = "vlm_gaze_panel_bias"
IMAGE_PAD_TOKEN = "<|image_pad|>"


@dataclass
class Qwen25VLGazeAdapter(Qwen25VLAdapter):
    attention_impl: str = ATTENTION_IMPL_NAME
    last_generation_metadata: dict[str, Any] | None = None
    _attention_records: list[dict[str, float | int]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Install optional Qwen dependencies with `uv sync --extra qwen`.") from exc

        _register_attention_impl()
        self._torch = torch
        self._process_vision_info = process_vision_info
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        device_map = _resolve_device_map(self.device_map, torch)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            attn_implementation=self.attention_impl,
        )
        self._model.eval()
        self.name = f"{self.model_id}-gaze"
        self._configure_attention_modules()

    @property
    def eval_config(self) -> dict[str, Any]:
        config = super().eval_config
        config.update({"attention_impl": self.attention_impl})
        return config

    def prepare_inputs(self, image: Any, prompt: str) -> Any:
        messages = [
            {
                "role": "user",
                "content": [
                    _image_content(image, self.max_pixels, self.min_pixels),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        return self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

    def model_dims(self) -> tuple[int, int, int]:
        cfg = self._model.config
        n_layers = int(cfg.text_config.num_hidden_layers)
        n_heads = int(cfg.text_config.num_attention_heads)
        spatial_merge = int(getattr(cfg.vision_config, "spatial_merge_size", 2))
        return n_layers, n_heads, spatial_merge

    def find_image_token_range(self, inputs: Any) -> tuple[int, int]:
        image_token_id = _get_image_token_id(self._processor, self._model)
        token_ids = inputs["input_ids"][0].tolist()
        positions = [idx for idx, token_id in enumerate(token_ids) if token_id == image_token_id]
        if not positions:
            raise ValueError(f"No image tokens with id={image_token_id} found in prompt inputs.")
        return int(positions[0]), int(positions[-1] + 1)

    def panel_token_masks(self, inputs: Any, panel_widths: list[int], n_panels: int) -> dict[int, Any]:
        _, _, spatial_merge = self.model_dims()
        region_ids, _, _ = assign_panels_to_tokens(inputs["image_grid_thw"], panel_widths, spatial_merge)
        img_start, img_end = self.find_image_token_range(inputs)
        usable_region_ids = region_ids[: max(0, img_end - img_start)]
        positions = region_positions_from_ids(img_start, usable_region_ids, n_panels)
        masks = {}
        for panel_idx, panel_positions in positions.items():
            mask = self._torch.zeros(inputs["input_ids"].shape[1], dtype=self._torch.bool, device=self._model.device)
            if panel_positions:
                mask[self._torch.tensor(panel_positions, dtype=self._torch.long, device=self._model.device)] = True
            masks[panel_idx] = mask
        return masks

    def collect_panel_attention(self, inputs: Any, panel_masks: dict[int, Any]) -> np.ndarray:
        self._clear_attention_state()
        self._set_record_masks({f"panel_{idx + 1}": mask for idx, mask in panel_masks.items()})
        with self._torch.no_grad():
            output = self._model(**inputs, output_attentions=True, return_dict=True)
        self._clear_attention_state()

        layers = []
        for attention in output.attentions:
            layers.append(attention[0, :, -1, :].detach().float().cpu().numpy())
        attn_at_query = np.stack(layers, axis=0)

        n_layers, n_heads = attn_at_query.shape[:2]
        panel_attention = np.zeros((n_layers, n_heads, len(panel_masks)), dtype=np.float64)
        for panel_idx, mask in panel_masks.items():
            mask_np = mask.detach().cpu().numpy().astype(bool)
            usable = mask_np[: attn_at_query.shape[-1]]
            panel_attention[:, :, panel_idx] = attn_at_query[:, :, : usable.shape[0]][:, :, usable].sum(axis=-1)
        return panel_attention

    def generate_steered(
        self,
        inputs: Any,
        panel_masks: dict[int, Any],
        head_specs: list[tuple[int, int]],
        target_panel: int,
        *,
        max_new_tokens: int,
        swap_bias: float = 10000.0,
        decode_only: bool = False,
    ) -> str:
        self._clear_attention_state()
        self._set_panel_bias(
            panel_masks=panel_masks,
            head_specs=head_specs,
            target_panel=target_panel,
            swap_bias=swap_bias,
            decode_only=decode_only,
        )
        try:
            generated = self._model.generate(**inputs, **self._generation_kwargs(max_new_tokens=max_new_tokens))
        finally:
            self.last_generation_metadata = {"attention": _summarize_attention_records(self._attention_records)}
            self._clear_attention_state()

        input_len = int(inputs["input_ids"].shape[1])
        generated_trimmed = generated[:, input_len:]
        return self._processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def generate_steered_dynamic(
        self,
        inputs: Any,
        panel_masks: dict[int, Any],
        head_specs: list[tuple[int, int]],
        schedule: list[tuple[int, int]],
        *,
        max_new_tokens: int,
        swap_bias: float = 10000.0,
    ) -> str:
        self._clear_attention_state()
        self._set_dynamic_panel_bias(
            panel_masks=panel_masks,
            head_specs=head_specs,
            schedule=schedule,
            swap_bias=swap_bias,
            prompt_length=int(inputs["input_ids"].shape[1]),
        )
        try:
            generated = self._model.generate(**inputs, **self._generation_kwargs(max_new_tokens=max_new_tokens))
        finally:
            self.last_generation_metadata = {"attention": _summarize_attention_records(self._attention_records)}
            self._clear_attention_state()

        input_len = int(inputs["input_ids"].shape[1])
        return self._processor.batch_decode(
            generated[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def generate_unsteered(self, inputs: Any, *, max_new_tokens: int) -> str:
        self._clear_attention_state()
        generated = self._model.generate(**inputs, **self._generation_kwargs(max_new_tokens=max_new_tokens))
        input_len = int(inputs["input_ids"].shape[1])
        return self._processor.batch_decode(
            generated[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def generate_with_panel_tracking(
        self,
        inputs: Any,
        panel_masks: dict[int, Any],
        head_specs: list[tuple[int, int]],
        *,
        max_new_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        self._clear_attention_state()
        self._set_tracking(panel_masks=panel_masks, head_specs=head_specs, prompt_length=int(inputs["input_ids"].shape[1]))
        try:
            generated = self._model.generate(**inputs, **self._generation_kwargs(max_new_tokens=max_new_tokens))
            input_len = int(inputs["input_ids"].shape[1])
            text = self._processor.batch_decode(
                generated[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            trajectory = _trajectory_from_records(self._attention_records)
        finally:
            self._clear_attention_state()
        return text, trajectory

    def _configure_attention_modules(self) -> None:
        for layer_idx, layer in enumerate(_language_layers(self._model)):
            attn = layer.self_attn
            attn._vlm_gaze_layer_idx = layer_idx
            attn._vlm_gaze_bias_by_head = {}
            attn._vlm_gaze_dynamic_bias_by_head = {}
            attn._vlm_gaze_attention_masks = {}
            attn._vlm_gaze_attention_records = self._attention_records
            attn._vlm_gaze_track_heads = []

    def _clear_attention_state(self) -> None:
        self._attention_records.clear()
        for layer in _language_layers(self._model):
            attn = layer.self_attn
            attn._vlm_gaze_bias_by_head = {}
            attn._vlm_gaze_dynamic_bias_by_head = {}
            attn._vlm_gaze_attention_masks = {}
            attn._vlm_gaze_decode_only = False
            attn._vlm_gaze_prompt_length = 0
            attn._vlm_gaze_track_heads = []

    def _set_record_masks(self, masks: dict[str, Any]) -> None:
        for layer in _language_layers(self._model):
            layer.self_attn._vlm_gaze_attention_masks = masks

    def _set_panel_bias(
        self,
        panel_masks: dict[int, Any],
        head_specs: list[tuple[int, int]],
        target_panel: int,
        swap_bias: float,
        decode_only: bool,
    ) -> None:
        target_mask = panel_masks[target_panel]
        other_mask = None
        for panel_idx, mask in panel_masks.items():
            if panel_idx == target_panel:
                continue
            other_mask = mask if other_mask is None else (other_mask | mask)

        by_layer: dict[int, list[int]] = {}
        for layer_idx, head_idx in head_specs:
            by_layer.setdefault(int(layer_idx), []).append(int(head_idx))

        for layer_idx, layer in enumerate(_language_layers(self._model)):
            layer.self_attn._vlm_gaze_decode_only = decode_only
            layer.self_attn._vlm_gaze_attention_masks = {f"panel_{target_panel + 1}": target_mask}
            layer.self_attn._vlm_gaze_bias_by_head = {
                head_idx: {
                    "boost_mask": target_mask,
                    "suppress_mask": other_mask,
                    "swap_bias": float(swap_bias),
                }
                for head_idx in by_layer.get(layer_idx, [])
            }

    def _set_dynamic_panel_bias(
        self,
        panel_masks: dict[int, Any],
        head_specs: list[tuple[int, int]],
        schedule: list[tuple[int, int]],
        swap_bias: float,
        prompt_length: int,
    ) -> None:
        if not schedule:
            raise ValueError("Dynamic gaze schedule must contain at least one target.")

        normalized_schedule = sorted((int(step), int(panel_idx)) for step, panel_idx in schedule)
        by_layer: dict[int, list[int]] = {}
        for layer_idx, head_idx in head_specs:
            by_layer.setdefault(int(layer_idx), []).append(int(head_idx))

        for layer_idx, layer in enumerate(_language_layers(self._model)):
            layer.self_attn._vlm_gaze_prompt_length = int(prompt_length)
            layer.self_attn._vlm_gaze_attention_masks = {
                f"panel_{panel_idx + 1}": panel_masks[panel_idx] for _, panel_idx in normalized_schedule
            }
            layer.self_attn._vlm_gaze_dynamic_bias_by_head = {
                head_idx: {
                    "panel_masks": panel_masks,
                    "schedule": normalized_schedule,
                    "swap_bias": float(swap_bias),
                }
                for head_idx in by_layer.get(layer_idx, [])
            }

    def _set_tracking(
        self,
        panel_masks: dict[int, Any],
        head_specs: list[tuple[int, int]],
        prompt_length: int,
    ) -> None:
        by_layer: dict[int, list[int]] = {}
        for layer_idx, head_idx in head_specs:
            by_layer.setdefault(int(layer_idx), []).append(int(head_idx))

        masks = {f"panel_{idx + 1}": mask for idx, mask in panel_masks.items()}
        for layer_idx, layer in enumerate(_language_layers(self._model)):
            layer.self_attn._vlm_gaze_prompt_length = int(prompt_length)
            layer.self_attn._vlm_gaze_attention_masks = masks
            layer.self_attn._vlm_gaze_track_heads = by_layer.get(layer_idx, [])


def make_adapter(**kwargs: Any) -> Qwen25VLGazeAdapter:
    return Qwen25VLGazeAdapter(**kwargs)


def _register_attention_impl() -> None:
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen_modeling

    if ATTENTION_IMPL_NAME not in qwen_modeling.ALL_ATTENTION_FUNCTIONS:
        qwen_modeling.ALL_ATTENTION_FUNCTIONS.register(ATTENTION_IMPL_NAME, _gaze_attention_forward)


def _gaze_attention_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[Any, Any]:
    import torch
    from torch import nn
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    decode_only = bool(getattr(module, "_vlm_gaze_decode_only", False))
    should_bias = not (decode_only and query.shape[-2] > 1)
    if should_bias:
        for head_idx, spec in getattr(module, "_vlm_gaze_bias_by_head", {}).items():
            boost_mask = _token_mask_for_key_length(spec.get("boost_mask"), key_states.shape[-2])
            suppress_mask = _token_mask_for_key_length(spec.get("suppress_mask"), key_states.shape[-2])
            swap_bias = float(spec.get("swap_bias", 0.0))
            if suppress_mask is not None:
                attn_weights[:, int(head_idx), :, suppress_mask] -= swap_bias
            if boost_mask is not None:
                attn_weights[:, int(head_idx), :, boost_mask] += swap_bias
        decode_step = max(0, int(key_states.shape[-2]) - int(getattr(module, "_vlm_gaze_prompt_length", 0)))
        for head_idx, spec in getattr(module, "_vlm_gaze_dynamic_bias_by_head", {}).items():
            panel_masks = spec.get("panel_masks", {})
            target_panel = _current_schedule_target(spec.get("schedule", []), decode_step)
            if target_panel is None or target_panel not in panel_masks:
                continue
            swap_bias = float(spec.get("swap_bias", 0.0))
            boost_mask = _token_mask_for_key_length(panel_masks[target_panel], key_states.shape[-2])
            suppress_mask = None
            for panel_idx, panel_mask in panel_masks.items():
                if int(panel_idx) == int(target_panel):
                    continue
                suppress_mask = panel_mask if suppress_mask is None else (suppress_mask | panel_mask)
            suppress_mask = _token_mask_for_key_length(suppress_mask, key_states.shape[-2])
            if suppress_mask is not None:
                attn_weights[:, int(head_idx), :, suppress_mask] -= swap_bias
            if boost_mask is not None:
                attn_weights[:, int(head_idx), :, boost_mask] += swap_bias

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    records = getattr(module, "_vlm_gaze_attention_records", None)
    named_masks = getattr(module, "_vlm_gaze_attention_masks", {})
    if records is not None and named_masks:
        record = {
            "layer_idx": int(getattr(module, "_vlm_gaze_layer_idx", -1)),
            "query_len": int(query.shape[-2]),
            "key_len": int(key_states.shape[-2]),
        }
        query_slice = slice(-1, None) if query.shape[-2] > 1 else slice(None)
        track_heads = [int(head_idx) for head_idx in getattr(module, "_vlm_gaze_track_heads", [])]
        head_slice = track_heads if track_heads else slice(None)
        if track_heads:
            record["tracked_heads"] = len(track_heads)
        for name, token_mask in named_masks.items():
            mask = _token_mask_for_key_length(token_mask, key_states.shape[-2])
            if mask is not None:
                selected = attn_weights[:, head_slice, query_slice, :]
                mass = selected[..., mask].sum(dim=-1).float().mean()
                record[f"{name}_attention_mass"] = float(mass.detach().cpu())
        if any(key.endswith("_attention_mass") for key in record):
            records.append(record)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _token_mask_for_key_length(token_mask: Any | None, key_len: int) -> Any | None:
    if token_mask is None or not bool(token_mask.any()):
        return None
    if token_mask.numel() == key_len:
        return token_mask
    if token_mask.numel() > key_len:
        return token_mask[:key_len]

    import torch

    padding = torch.zeros(key_len - token_mask.numel(), dtype=torch.bool, device=token_mask.device)
    return torch.cat([token_mask, padding], dim=0)


def _current_schedule_target(schedule: list[tuple[int, int]], decode_step: int) -> int | None:
    if not schedule:
        return None
    target_panel = int(schedule[0][1])
    for start_step, panel_idx in schedule:
        if decode_step >= int(start_step):
            target_panel = int(panel_idx)
        else:
            break
    return target_panel


def _get_image_token_id(processor: Any, model: Any) -> int:
    tokenizer = processor.tokenizer
    token_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    if token_id is not None and token_id >= 0:
        return int(token_id)
    config_token_id = getattr(model.config, "image_token_id", None)
    if config_token_id is not None:
        return int(config_token_id)
    raise ValueError("Could not resolve Qwen image token id.")


def _image_content(image: Any, max_pixels: int, min_pixels: int) -> dict[str, Any]:
    content = {"type": "image", "image": image}
    if max_pixels > 0:
        content["max_pixels"] = max_pixels
    if min_pixels > 0:
        content["min_pixels"] = min_pixels
    return content


def _language_layers(model: Any) -> Any:
    if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    if hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Could not find Qwen language layers.")


def _summarize_attention_records(records: list[dict[str, float | int]]) -> dict[str, Any]:
    mass_keys = sorted({key for record in records for key in record if key.endswith("_attention_mass")})
    summary = {"n_records": len(records)}
    for key in mass_keys:
        values = [float(record[key]) for record in records if key in record]
        summary[f"mean_{key}"] = sum(values) / len(values) if values else None
    return summary


def _trajectory_from_records(records: list[dict[str, float | int]]) -> dict[str, Any]:
    decode_records = [record for record in records if int(record.get("query_len", 0)) == 1]
    mass_keys = sorted(
        key
        for key in {key for record in decode_records for key in record}
        if key.endswith("_attention_mass")
    )
    by_key_len: dict[int, list[dict[str, float | int]]] = {}
    for record in decode_records:
        by_key_len.setdefault(int(record["key_len"]), []).append(record)

    steps = []
    for step_idx, key_len in enumerate(sorted(by_key_len)):
        layer_records = by_key_len[key_len]
        panel_masses = {
            key.removesuffix("_attention_mass"): _mean(
                [float(record[key]) for record in layer_records if key in record]
            )
            for key in mass_keys
        }
        steps.append(
            {
                "step": step_idx,
                "key_len": key_len,
                "n_layer_records": len(layer_records),
                "panel_masses": panel_masses,
            }
        )
    return {
        "n_decode_steps": len(steps),
        "panel_names": [key.removesuffix("_attention_mass") for key in mass_keys],
        "steps": steps,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
