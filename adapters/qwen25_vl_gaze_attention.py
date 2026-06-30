from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapters.qwen25_vl import Qwen25VLAdapter, _resolve_device_map


ATTENTION_IMPL_NAME = "vlm_gaze_head_image_bias"


@dataclass
class Qwen25VLGazeAttentionAdapter(Qwen25VLAdapter):
    attention_alpha: float = 0.0
    gaze_ranking_path: str = "segments/gaze_heads_qwen25/runs/gaze_discovery_merged_0_500/gaze_head_ranking.json"
    top_k_gaze: int = 0
    decode_only: bool = False
    last_generation_metadata: dict[str, Any] | None = None
    _attention_records: list[dict[str, float | int]] = field(default_factory=list, init=False)
    _boosted_heads: list[tuple[int, int]] = field(default_factory=list, init=False)
    _boosted_heads_by_layer: dict[int, list[int]] = field(default_factory=dict, init=False)

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
            attn_implementation=ATTENTION_IMPL_NAME,
        )
        self._model.eval()
        self.name = f"{self.model_id}-gaze-attention-top{self.top_k_gaze}-alpha{self.attention_alpha}"
        self._configure_attention_modules()

    @property
    def eval_config(self) -> dict[str, Any]:
        config = super().eval_config
        config.update(
            {
                "attention_alpha": self.attention_alpha,
                "gaze_ranking_path": self.gaze_ranking_path,
                "top_k_gaze": self.top_k_gaze,
                "decode_only": self.decode_only,
                "boosted_heads": [{"layer": layer, "head": head} for layer, head in self._boosted_heads],
                "attention_impl": ATTENTION_IMPL_NAME,
            }
        )
        return config

    def _configure_attention_modules(self) -> None:
        self._boosted_heads = _load_gaze_heads(Path(self.gaze_ranking_path), self.top_k_gaze)
        self._boosted_heads_by_layer = _heads_by_layer(self._boosted_heads)
        for layer_idx, layer in enumerate(_language_layers(self._model)):
            attn = layer.self_attn
            attn._vlm_gaze_image_layer_idx = layer_idx
            attn._vlm_gaze_image_attention_alpha = float(self.attention_alpha)
            attn._vlm_gaze_image_head_indices = self._boosted_heads_by_layer.get(layer_idx, [])
            attn._vlm_gaze_image_token_mask = None
            attn._vlm_gaze_image_decode_only = bool(self.decode_only)
            attn._vlm_gaze_image_attention_records = self._attention_records

    def _before_generate(self, inputs: Any, *, include_image: bool) -> None:
        self._attention_records.clear()
        self.last_generation_metadata = None
        image_token_id = getattr(self._model.config, "image_token_id", None)
        if include_image and image_token_id is not None:
            image_mask = inputs.input_ids[0].eq(image_token_id).detach()
        else:
            image_mask = None

        for layer in _language_layers(self._model):
            layer.self_attn._vlm_gaze_image_token_mask = image_mask

    def _after_generate(self) -> None:
        self.last_generation_metadata = {
            "attention": _summarize_attention_records(self._attention_records, self._boosted_heads_by_layer),
            "attention_alpha": self.attention_alpha,
            "top_k_gaze": self.top_k_gaze,
            "decode_only": self.decode_only,
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
    gaze_ranking_path: str = "segments/gaze_heads_qwen25/runs/gaze_discovery_merged_0_500/gaze_head_ranking.json",
    top_k_gaze: int = 0,
    decode_only: bool = False,
) -> Qwen25VLGazeAttentionAdapter:
    return Qwen25VLGazeAttentionAdapter(
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
        gaze_ranking_path=gaze_ranking_path,
        top_k_gaze=top_k_gaze,
        decode_only=decode_only,
    )


def _register_attention_impl() -> None:
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen_modeling

    if ATTENTION_IMPL_NAME not in qwen_modeling.ALL_ATTENTION_FUNCTIONS:
        qwen_modeling.ALL_ATTENTION_FUNCTIONS.register(ATTENTION_IMPL_NAME, _gaze_image_bias_attention_forward)


def _gaze_image_bias_attention_forward(
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

    image_mask = _token_mask_for_key_length(
        getattr(module, "_vlm_gaze_image_token_mask", None),
        key_states.shape[-2],
    )
    query_slice = slice(-1, None) if query.shape[-2] > 1 else slice(None)
    alpha = float(getattr(module, "_vlm_gaze_image_attention_alpha", 0.0))
    head_indices = [int(head_idx) for head_idx in getattr(module, "_vlm_gaze_image_head_indices", [])]
    decode_only = bool(getattr(module, "_vlm_gaze_image_decode_only", False))
    should_bias = image_mask is not None and alpha != 0.0 and head_indices and not (decode_only and query.shape[-2] > 1)
    if should_bias:
        for head_idx in head_indices:
            attn_weights[:, head_idx, query_slice, image_mask] += alpha

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    records = getattr(module, "_vlm_gaze_image_attention_records", None)
    if records is not None and image_mask is not None:
        record = {
            "layer_idx": int(getattr(module, "_vlm_gaze_image_layer_idx", -1)),
            "query_len": int(query.shape[-2]),
            "key_len": int(key_states.shape[-2]),
        }
        image_mass = attn_weights[..., query_slice, image_mask].sum(dim=-1).float().mean()
        record["image_attention_mass"] = float(image_mass.detach().cpu())
        if head_indices:
            selected = attn_weights[:, head_indices, query_slice, :]
            selected_mass = selected[..., image_mask].sum(dim=-1).float().mean()
            record["boosted_head_image_attention_mass"] = float(selected_mass.detach().cpu())
            record["boosted_heads"] = len(head_indices)
        records.append(record)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _load_gaze_heads(ranking_path: Path, top_k: int) -> list[tuple[int, int]]:
    if top_k <= 0:
        return []
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    if not isinstance(ranking, list):
        raise ValueError(f"Expected gaze ranking list in {ranking_path}")
    heads = []
    for row in ranking[:top_k]:
        try:
            heads.append((int(row["layer"]), int(row["head"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed gaze head row in {ranking_path}: {row!r}") from exc
    return heads


def _heads_by_layer(heads: list[tuple[int, int]]) -> dict[int, list[int]]:
    by_layer: dict[int, list[int]] = {}
    for layer_idx, head_idx in heads:
        by_layer.setdefault(int(layer_idx), []).append(int(head_idx))
    return {layer_idx: sorted(set(heads)) for layer_idx, heads in by_layer.items()}


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


def _language_layers(model: Any) -> Any:
    if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    if hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Could not find Qwen language layers on model.model.language_model.layers or model.model.layers.")


def _summarize_attention_records(records: list[dict[str, float | int]], boosted_heads_by_layer: dict[int, list[int]]) -> dict[str, Any]:
    all_masses = [float(record["image_attention_mass"]) for record in records if "image_attention_mass" in record]
    boosted_masses = [
        float(record["boosted_head_image_attention_mass"])
        for record in records
        if "boosted_head_image_attention_mass" in record
    ]
    boosted_layer_masses = [
        float(record["image_attention_mass"])
        for record in records
        if int(record["layer_idx"]) in boosted_heads_by_layer and "image_attention_mass" in record
    ]
    return {
        "n_records": len(records),
        "mean_image_attention_mass": _mean(all_masses),
        "mean_boosted_layer_image_attention_mass": _mean(boosted_layer_masses),
        "mean_boosted_head_image_attention_mass": _mean(boosted_masses),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
