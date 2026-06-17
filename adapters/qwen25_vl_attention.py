from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adapters.qwen25_vl import Qwen25VLAdapter


ATTENTION_IMPL_NAME = "vlm_image_token_bias"


@dataclass
class Qwen25VLAttentionAdapter(Qwen25VLAdapter):
    attention_alpha: float = 0.0
    layer_selection: str = "baseline"
    last_generation_metadata: dict[str, Any] | None = None
    _attention_records: list[dict[str, float | int]] = field(default_factory=list, init=False)
    _boosted_layers: set[int] = field(default_factory=set, init=False)

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
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device_map,
            attn_implementation=ATTENTION_IMPL_NAME,
        )
        self.name = f"{self.model_id}-attention-{self.layer_selection}-alpha{self.attention_alpha}"
        self._configure_attention_modules()

    @property
    def eval_config(self) -> dict[str, Any]:
        config = super().eval_config
        config.update(
            {
                "attention_alpha": self.attention_alpha,
                "layer_selection": self.layer_selection,
                "boosted_layers": sorted(self._boosted_layers),
                "attention_impl": ATTENTION_IMPL_NAME,
            }
        )
        return config

    def _configure_attention_modules(self) -> None:
        layers = _language_layers(self._model)
        self._boosted_layers = _select_layers(len(layers), self.layer_selection)
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            attn._vlm_layer_idx = layer_idx
            attn._vlm_attention_alpha = self.attention_alpha if layer_idx in self._boosted_layers else 0.0
            attn._vlm_attention_records = self._attention_records
            attn._vlm_image_token_mask = None
            attn._vlm_attention_masks = {}

    def _before_generate(self, inputs: Any, *, include_image: bool) -> None:
        self._attention_records.clear()
        self.last_generation_metadata = None
        image_token_id = getattr(self._model.config, "image_token_id", None)
        if include_image and image_token_id is not None:
            image_mask = inputs.input_ids[0].eq(image_token_id).detach()
        else:
            image_mask = None

        for layer in _language_layers(self._model):
            layer.self_attn._vlm_image_token_mask = image_mask
            layer.self_attn._vlm_attention_masks = {"image": image_mask} if image_mask is not None else {}

    def _after_generate(self) -> None:
        self.last_generation_metadata = {
            "attention": _summarize_attention_records(self._attention_records, self._boosted_layers),
            "attention_alpha": self.attention_alpha,
            "layer_selection": self.layer_selection,
        }


def make_adapter(
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    max_new_tokens: int = 16,
    max_pixels: int = 1048576,
    min_pixels: int = 0,
    do_sample: bool = True,
    temperature: float | None = 0.7,
    top_p: float | None = None,
    top_k: int | None = None,
    include_image: bool = True,
    seed: int | None = 0,
    device_map: str = "auto",
    prompt_mode: str = "baseline",
    attention_alpha: float = 0.0,
    layer_selection: str = "baseline",
) -> Qwen25VLAttentionAdapter:
    return Qwen25VLAttentionAdapter(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        include_image=include_image,
        seed=seed,
        device_map=device_map,
        prompt_mode=prompt_mode,
        attention_alpha=attention_alpha,
        layer_selection=layer_selection,
    )


def _register_attention_impl() -> None:
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen_modeling

    if ATTENTION_IMPL_NAME not in qwen_modeling.ALL_ATTENTION_FUNCTIONS:
        qwen_modeling.ALL_ATTENTION_FUNCTIONS.register(ATTENTION_IMPL_NAME, _image_bias_attention_forward)


def _image_bias_attention_forward(
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

    image_mask = _token_mask_for_key_length(getattr(module, "_vlm_image_token_mask", None), key_states.shape[-2])
    query_slice = slice(-1, None) if query.shape[-2] > 1 else slice(None)
    alpha = float(getattr(module, "_vlm_attention_alpha", 0.0))
    if image_mask is not None and alpha != 0.0:
        attn_weights[..., query_slice, image_mask] += alpha

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    records = getattr(module, "_vlm_attention_records", None)
    named_masks = getattr(module, "_vlm_attention_masks", {})
    if records is not None and (image_mask is not None or named_masks):
        record = {
            "layer_idx": int(getattr(module, "_vlm_layer_idx", -1)),
            "query_len": int(query.shape[-2]),
            "key_len": int(key_states.shape[-2]),
        }
        if image_mask is not None:
            image_mass = attn_weights[..., query_slice, image_mask].sum(dim=-1).float().mean()
            record["image_attention_mass"] = float(image_mass.detach().cpu())
        for name, token_mask in named_masks.items():
            mask = _token_mask_for_key_length(token_mask, key_states.shape[-2])
            if mask is not None:
                mass = attn_weights[..., query_slice, mask].sum(dim=-1).float().mean()
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


def _select_layers(num_layers: int, layer_selection: str) -> set[int]:
    if layer_selection == "baseline":
        return set()
    if layer_selection == "all":
        return set(range(num_layers))
    if layer_selection == "last50":
        return set(range(num_layers // 2, num_layers))
    if layer_selection == "last25":
        return set(range((3 * num_layers) // 4, num_layers))
    raise ValueError(f"Unknown layer_selection: {layer_selection}")


def _language_layers(model: Any) -> Any:
    if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    if hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Could not find Qwen language layers on model.model.language_model.layers or model.model.layers.")


def _summarize_attention_records(records: list[dict[str, float | int]], boosted_layers: set[int]) -> dict[str, Any]:
    mass_keys = sorted(
        {
            key
            for record in records
            for key in record
            if key.endswith("_attention_mass")
        }
    )
    all_masses = [float(record["image_attention_mass"]) for record in records if "image_attention_mass" in record]
    boosted_masses = [
        float(record["image_attention_mass"])
        for record in records
        if int(record["layer_idx"]) in boosted_layers and "image_attention_mass" in record
    ]
    by_layer: dict[int, list[float]] = {}
    for record in records:
        if "image_attention_mass" in record:
            by_layer.setdefault(int(record["layer_idx"]), []).append(float(record["image_attention_mass"]))

    summary = {
        "n_records": len(records),
        "mean_image_attention_mass": _mean(all_masses),
        "mean_boosted_layer_image_attention_mass": _mean(boosted_masses),
        "by_layer": {
            str(layer_idx): {
                "n_records": len(values),
                "mean_image_attention_mass": _mean(values),
            }
            for layer_idx, values in sorted(by_layer.items())
        },
    }
    for key in mass_keys:
        summary[f"mean_{key}"] = _mean([float(record[key]) for record in records if key in record])
    return summary


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
