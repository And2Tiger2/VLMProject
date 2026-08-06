from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from adapters.qwen25_vl_gaze_attention import _repeat_kv
from vlm_eval.mechanistic_heads.patching import LazyProjectedHeads


MECHANISTIC_ATTENTION_IMPL = "vlm_mechanistic_capture"


@dataclass
class CaptureStore:
    layer_inputs: dict[int, Any] = field(default_factory=dict)
    layer_outputs: dict[int, Any] = field(default_factory=dict)
    attention_outputs: dict[int, Any] = field(default_factory=dict)
    mlp_outputs: dict[int, Any] = field(default_factory=dict)
    raw_heads: dict[int, Any] = field(default_factory=dict)
    attention_probabilities: dict[int, Any] = field(default_factory=dict)
    projected_heads: dict[int, Any] = field(default_factory=dict)

    def clear(self) -> None:
        for value in vars(self).values():
            if isinstance(value, dict):
                value.clear()


def _detach(value: Any, *, to_cpu: bool) -> Any:
    if isinstance(value, tuple):
        return tuple(_detach(item, to_cpu=to_cpu) for item in value)
    if hasattr(value, "detach"):
        value = value.detach()
        return value.cpu() if to_cpu else value
    return value


class Qwen3CaptureHooks:
    """Layer/module hooks plus raw-head capture from the custom backend."""

    def __init__(
        self,
        model: Any,
        *,
        layers: list[int] | None = None,
        to_cpu: bool = True,
    ) -> None:
        self.model = model
        self.layers = layers
        self.to_cpu = to_cpu
        self.store = CaptureStore()
        self._handles: list[Any] = []

    def __enter__(self) -> CaptureStore:
        language_layers = get_language_layers(self.model)
        selected = self.layers if self.layers is not None else list(range(len(language_layers)))
        for layer_idx in selected:
            layer = language_layers[layer_idx]
            layer.self_attn._vlm_mechanistic_store = self.store
            layer.self_attn._vlm_mechanistic_capture_raw = True
            layer.self_attn._vlm_mechanistic_to_cpu = self.to_cpu
            self._handles.extend(
                [
                    layer.register_forward_pre_hook(self._pre_hook(layer_idx)),
                    layer.register_forward_hook(self._post_hook("layer_outputs", layer_idx)),
                    layer.self_attn.register_forward_hook(
                        self._post_hook("attention_outputs", layer_idx)
                    ),
                    layer.mlp.register_forward_hook(self._post_hook("mlp_outputs", layer_idx)),
                ]
            )
        return self.store

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        for layer in get_language_layers(self.model):
            for name in (
                "_vlm_mechanistic_store",
                "_vlm_mechanistic_capture_raw",
                "_vlm_mechanistic_to_cpu",
            ):
                if hasattr(layer.self_attn, name):
                    delattr(layer.self_attn, name)
        self._handles.clear()

    def _pre_hook(self, layer_idx: int) -> Callable[..., None]:
        def hook(module: Any, args: tuple[Any, ...]) -> None:
            self.store.layer_inputs[layer_idx] = _detach(args[0], to_cpu=self.to_cpu)

        return hook

    def _post_hook(self, field_name: str, layer_idx: int) -> Callable[..., None]:
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            first = output[0] if isinstance(output, tuple) else output
            getattr(self.store, field_name)[layer_idx] = _detach(first, to_cpu=self.to_cpu)

        return hook


def mechanistic_attention_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Eager Qwen attention with exact raw-head/map capture.

    The returned raw output retains Transformers' expected
    `[batch, query, heads, head_dim]` layout. No intervention is applied here;
    post-W_O patches are injected by attention-module hooks.
    """

    import torch
    from torch import nn

    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        logits = logits + attention_mask
    probabilities = nn.functional.softmax(logits, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    probabilities = nn.functional.dropout(
        probabilities, p=dropout, training=module.training
    )
    replacement = getattr(module, "_vlm_mechanistic_attention_replacement", None)
    if replacement is not None:
        if callable(replacement):
            probabilities = replacement(probabilities)
        else:
            if replacement.shape != probabilities.shape:
                raise ValueError(
                    f"replacement map {replacement.shape} != attention {probabilities.shape}"
                )
            probabilities = replacement.to(probabilities.device, probabilities.dtype)
    knockout = getattr(module, "_vlm_mechanistic_image_knockout_mask", None)
    if knockout is not None:
        from vlm_eval.mechanistic_heads.ablations import image_attention_knockout

        probabilities = image_attention_knockout(probabilities, knockout)
    raw = torch.matmul(probabilities, value_states).transpose(1, 2).contiguous()
    head_scales = getattr(module, "_vlm_mechanistic_head_scales", None)
    if head_scales:
        raw = raw.clone()
        for head_idx, scale in head_scales.items():
            raw[:, :, int(head_idx), :] *= float(scale)
    batch_scales = getattr(module, "_vlm_mechanistic_batch_head_scales", None)
    if batch_scales:
        if len(batch_scales) != raw.shape[0]:
            raise RuntimeError("per-batch head scaling does not match attention batch")
        raw = raw.clone()
        for batch_idx, (head_idx, scale) in enumerate(batch_scales):
            raw[batch_idx, :, int(head_idx), :] *= float(scale)

    store = getattr(module, "_vlm_mechanistic_store", None)
    if store is not None and getattr(module, "_vlm_mechanistic_capture_raw", False):
        layer_idx = int(module.layer_idx)
        to_cpu = bool(getattr(module, "_vlm_mechanistic_to_cpu", True))
        store.raw_heads[layer_idx] = _detach(raw, to_cpu=to_cpu)
        store.attention_probabilities[layer_idx] = _detach(
            probabilities, to_cpu=to_cpu
        )
        # Do not materialize [batch, sequence, heads, model_width] here: long
        # multimodal prompts can turn that derived tensor into a 10–40 GiB
        # allocation. Exact post-W_O contributions are projected lazily after
        # callers select the required positions and heads.
        stored_raw = store.raw_heads[layer_idx]
        store.projected_heads[layer_idx] = LazyProjectedHeads(
            stored_raw, module.o_proj.weight.detach()
        )
    return raw, probabilities


def register_qwen3_mechanistic_attention() -> None:
    from transformers.masking_utils import (
        ALL_MASK_ATTENTION_FUNCTIONS,
        eager_mask,
    )
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    if MECHANISTIC_ATTENTION_IMPL not in modeling_qwen3_vl.ALL_ATTENTION_FUNCTIONS:
        modeling_qwen3_vl.ALL_ATTENTION_FUNCTIONS.register(
            MECHANISTIC_ATTENTION_IMPL, mechanistic_attention_forward
        )
    # Transformers selects the causal-mask builder independently from the
    # attention kernel. Unregistered custom implementations are assumed to be
    # external backends that manage causality themselves, and consequently
    # receive ``attention_mask=None``. Our implementation is eager attention
    # and must receive the same explicit causal/padding mask as HF eager.
    if MECHANISTIC_ATTENTION_IMPL not in ALL_MASK_ATTENTION_FUNCTIONS:
        ALL_MASK_ATTENTION_FUNCTIONS.register(
            MECHANISTIC_ATTENTION_IMPL, eager_mask
        )


def get_language_layers(model: Any) -> Any:
    # Base Qwen3-VL exposes ``model.language_model.layers``. PEFT wraps that
    # object in one or more ``base_model``/``model`` containers, so walk only
    # those known wrapper edges and stop at the first language stack. This
    # deliberately avoids a generic module traversal, which could mistake the
    # vision transformer's layers for the language layers.
    queue = [model]
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        language_model = getattr(candidate, "language_model", None)
        if language_model is not None and hasattr(language_model, "layers"):
            return language_model.layers
        nested_model = getattr(candidate, "model", None)
        if nested_model is not None and nested_model is not candidate:
            queue.append(nested_model)
        base_model = getattr(candidate, "base_model", None)
        if base_model is not None and base_model is not candidate:
            queue.append(base_model)
    raise AttributeError("could not find Qwen language layers")
