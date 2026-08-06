from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from vlm_eval.mechanistic_heads.capture import Qwen3CaptureHooks, get_language_layers
from vlm_eval.mechanistic_heads.likelihood import (
    append_answer_tokens,
    candidate_sequence_log_likelihood,
)


def _assert_attention_normalized(probabilities: Any, *, label: str) -> None:
    """Validate probability rows at a tolerance appropriate for their dtype.

    Qwen's eager attention casts the float32 softmax back to the model dtype.
    A bfloat16 row can therefore differ from one by a few 1e-3 even before an
    intervention.  Keep this as a real normalization guard without applying a
    float32-only tolerance to bfloat16 inference.
    """

    import torch

    sums = probabilities.float().sum(dim=-1)
    tolerance = 1e-4
    if probabilities.dtype.is_floating_point:
        tolerance = max(tolerance, float(torch.finfo(probabilities.dtype).eps))
    if not torch.allclose(
        sums, torch.ones_like(sums), atol=tolerance, rtol=0
    ):
        maximum = float((sums - 1).abs().max().detach().cpu())
        raise RuntimeError(
            f"{label} broke normalization: max_abs={maximum:.6g}, "
            f"tolerance={tolerance:.6g}"
        )


@dataclass(frozen=True)
class CapturedPrefill:
    inputs: Any
    store: Any
    image_positions: list[int]
    prompt_length: int
    spans: Any
    answer_length: int = 0


def bounded_head_microbatch(
    requested: int,
    sequence_length: int,
    *,
    token_budget: int = 8192,
) -> int:
    """Bound intervention batches for eager multimodal attention memory.

    Candidate-head rows are independent, so reducing this batch changes only
    execution time. The cap retains batches of 32 for short synthetic prompts,
    while using four rows around 2k tokens and one row around 8k tokens.
    """

    if requested <= 0 or sequence_length <= 0 or token_budget <= 0:
        raise ValueError("microbatch inputs and token budget must be positive")
    return max(1, min(int(requested), int(token_budget) // int(sequence_length)))


def capture_prefill(
    runtime: Any,
    *,
    image_path: Any,
    prompt: str,
    layers: list[int],
    to_cpu: bool = False,
) -> CapturedPrefill:
    image = (
        image_path.convert("RGB")
        if hasattr(image_path, "convert")
        else Image.open(image_path).convert("RGB")
    )
    inputs = runtime.prepare(image, prompt, prompt_mode="raw")
    image_token_id = int(runtime.model.config.image_token_id)
    image_positions = inputs.input_ids[0].eq(image_token_id).nonzero().flatten().tolist()
    if not image_positions:
        raise RuntimeError("processed input has no visual tokens")
    with runtime.torch.no_grad(), Qwen3CaptureHooks(
        runtime.model, layers=layers, to_cpu=to_cpu
    ) as store:
        runtime.model(**inputs, use_cache=False, return_dict=True)
    return CapturedPrefill(
        inputs=inputs,
        store=store,
        image_positions=[int(value) for value in image_positions],
        prompt_length=int(inputs.input_ids.shape[1]),
        spans=runtime.trace_spans(inputs, prompt),
    )


def capture_teacher_forced(
    runtime: Any,
    *,
    image_path: Any,
    prompt: str,
    answer: str,
    layers: list[int],
    to_cpu: bool = False,
) -> CapturedPrefill:
    image = image_path.convert("RGB") if hasattr(image_path, "convert") else Image.open(image_path).convert("RGB")
    inputs = runtime.prepare(image, prompt, prompt_mode="raw")
    answer_ids = runtime.answer_token_ids(answer)
    kwargs, answer_ids, prompt_length = append_answer_tokens(inputs, answer_ids)
    image_token_id = int(runtime.model.config.image_token_id)
    image_positions = inputs.input_ids[0].eq(image_token_id).nonzero().flatten().tolist()
    with runtime.torch.no_grad(), Qwen3CaptureHooks(
        runtime.model, layers=layers, to_cpu=to_cpu
    ) as store:
        runtime.model(**kwargs, use_cache=False, return_dict=True)
    return CapturedPrefill(inputs=inputs, store=store, image_positions=[int(value) for value in image_positions], prompt_length=prompt_length, spans=runtime.trace_spans(inputs, prompt, answer_length=int(answer_ids.shape[1])), answer_length=int(answer_ids.shape[1]))


def candidate_margin(
    runtime: Any,
    inputs: Any,
    *,
    positive_answer: str,
    negative_answer: str,
) -> tuple[float, dict[str, float]]:
    scores = {}
    with runtime.torch.no_grad():
        for name, answer in (("positive", positive_answer), ("negative", negative_answer)):
            token_ids = runtime.answer_token_ids(answer)
            score = candidate_sequence_log_likelihood(runtime.model, inputs, token_ids)
            scores[name] = float(score.total_log_probability[0].detach().cpu())
    return scores["positive"] - scores["negative"], scores


def repeat_model_inputs(inputs: Any, repeats: int) -> dict[str, Any]:
    """Repeat one processed multimodal example into an identical model batch."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    output = {}
    for key, value in dict(inputs).items():
        if not hasattr(value, "shape"):
            output[key] = value
            continue
        if value.ndim == 0:
            output[key] = value
        elif key == "position_ids" and value.ndim >= 3 and value.shape[1] == 1:
            output[key] = value.repeat(1, repeats, *([1] * (value.ndim - 2)))
        else:
            output[key] = value.repeat(repeats, *([1] * (value.ndim - 1)))
    return output


def batched_candidate_margin(
    runtime: Any,
    inputs: dict[str, Any],
    *,
    positive_answer: str,
    negative_answer: str,
) -> tuple[Any, dict[str, Any]]:
    scores = {}
    with runtime.torch.no_grad():
        for name, answer in (("positive", positive_answer), ("negative", negative_answer)):
            token_ids = runtime.answer_token_ids(answer)
            score = candidate_sequence_log_likelihood(runtime.model, inputs, token_ids)
            scores[name] = score.total_log_probability.detach()
    return scores["positive"] - scores["negative"], scores


def scope_positions(
    capture: CapturedPrefill,
    scope: str,
) -> list[int]:
    if scope == "all_image_tokens":
        return capture.image_positions
    if scope == "last_image_token":
        return [capture.image_positions[-1]]
    if scope == "final_prompt_token":
        return [capture.prompt_length - 1]
    if scope == "all_prefill_positions":
        return list(range(capture.prompt_length))
    if scope == "system_tokens":
        return capture.spans.system.indices()
    if scope == "user_prompt":
        return capture.spans.user_prompt.indices()
    raise ValueError(f"unsupported patch scope: {scope}")


@contextmanager
def projected_head_patch(
    model: Any,
    *,
    layer_idx: int,
    head_idx: int,
    donor_projected: Any,
    recipient_projected: Any,
    positions: list[int],
    donor_positions: list[int] | None = None,
    scale: float = 1.0,
) -> Iterator[None]:
    layer = get_language_layers(model)[layer_idx]
    import torch

    recipient_index = torch.as_tensor(
        positions, dtype=torch.long, device=recipient_projected.device
    )
    donor_index = torch.as_tensor(
        donor_positions if donor_positions is not None else positions,
        dtype=torch.long,
        device=donor_projected.device,
    )
    if donor_index.numel() != recipient_index.numel():
        raise ValueError("donor and recipient patch scopes must have equal cardinality")

    def hook(module: Any, args: tuple[Any, ...], output: Any) -> Any:
        current, *rest = output if isinstance(output, tuple) else (output,)
        if max(positions) >= current.shape[1]:
            raise RuntimeError("patch position exceeds current sequence length")
        patched = current.clone()
        donor = donor_projected[0].index_select(0, donor_index)[:, head_idx, :]
        recipient = recipient_projected[0].index_select(0, recipient_index)[:, head_idx, :]
        delta = float(scale) * (donor - recipient)
        # Captures used by the locked validation paths are intentionally
        # offloaded to CPU.  Move only the selected positions/head back to the
        # active model device instead of retaining every captured layer on the
        # GPU or relying on implicit cross-device indexing.
        current_index = recipient_index.to(device=current.device)
        delta = delta.to(device=current.device, dtype=current.dtype)
        # A serial head intervention may still score a microbatch. Apply the
        # same candidate-head patch independently to every row; this lets the
        # instrumentation compare serial and all-head-batched hooks under the
        # exact same model batch and vision-encoder numerical path.
        for batch_idx in range(current.shape[0]):
            patched[batch_idx, current_index, :] = (
                current[batch_idx, current_index, :] + delta
            )
        return (patched, *rest) if isinstance(output, tuple) else patched

    handle = layer.self_attn.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def batched_projected_head_patch(
    model: Any,
    *,
    layer_idx: int,
    head_indices: list[int],
    donor_projected: Any,
    recipient_projected: Any,
    positions: list[int],
    donor_positions: list[int] | None = None,
) -> Iterator[None]:
    """Patch a different head in each batch row for one layer scan."""

    import torch

    layer = get_language_layers(model)[layer_idx]
    recipient_index = torch.as_tensor(
        positions, dtype=torch.long, device=recipient_projected.device
    )
    donor_index = torch.as_tensor(
        donor_positions if donor_positions is not None else positions,
        dtype=torch.long,
        device=donor_projected.device,
    )
    if donor_index.numel() != recipient_index.numel():
        raise ValueError("donor and recipient patch scopes must have equal cardinality")

    def hook(module: Any, args: tuple[Any, ...], output: Any) -> Any:
        current, *rest = output if isinstance(output, tuple) else (output,)
        if current.shape[0] != len(head_indices):
            raise RuntimeError(
                f"patch batch {len(head_indices)} != model batch {current.shape[0]}"
            )
        patched = current.clone()
        current_index = recipient_index.to(device=current.device)
        for batch_idx, head_idx in enumerate(head_indices):
            donor = donor_projected[0].index_select(0, donor_index)[:, head_idx, :]
            recipient = recipient_projected[0].index_select(0, recipient_index)[:, head_idx, :]
            delta = (donor - recipient).to(device=current.device, dtype=current.dtype)
            patched[batch_idx, current_index, :] = (
                current[batch_idx, current_index, :] + delta
            )
        return (patched, *rest) if isinstance(output, tuple) else patched

    handle = layer.self_attn.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def module_activation_patch(
    model: Any,
    *,
    layer_idx: int,
    module_kind: str,
    donor_activation: Any,
    positions: list[int],
) -> Iterator[None]:
    layer = get_language_layers(model)[layer_idx]
    module = {
        "full_residual": layer,
        "attention_output": layer.self_attn,
        "mlp_output": layer.mlp,
    }.get(module_kind)
    if module is None:
        raise ValueError(f"unsupported module patch: {module_kind}")
    import torch

    index = torch.as_tensor(positions, dtype=torch.long, device=donor_activation.device)

    def hook(target: Any, args: tuple[Any, ...], output: Any) -> Any:
        current, *rest = output if isinstance(output, tuple) else (output,)
        patched = current.clone()
        current_index = index.to(device=current.device)
        replacement = donor_activation[0].index_select(0, index).to(
            device=current.device, dtype=current.dtype
        )
        patched[0, current_index, :] = replacement
        return (patched, *rest) if isinstance(output, tuple) else patched

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def projected_head_scaling(
    model: Any, head_scales: dict[tuple[int, int], float]
) -> Iterator[None]:
    """Scale selected raw heads before the shared output projection."""

    layers = get_language_layers(model)
    by_layer: dict[int, dict[int, float]] = {}
    for (layer_idx, head_idx), scale in head_scales.items():
        by_layer.setdefault(int(layer_idx), {})[int(head_idx)] = float(scale)
    for layer_idx, scales in by_layer.items():
        layers[layer_idx].self_attn._vlm_mechanistic_head_scales = scales
    try:
        yield
    finally:
        for layer_idx in by_layer:
            if hasattr(layers[layer_idx].self_attn, "_vlm_mechanistic_head_scales"):
                delattr(layers[layer_idx].self_attn, "_vlm_mechanistic_head_scales")


@contextmanager
def selected_image_attention_knockout(
    model: Any,
    *,
    layer_idx: int,
    head_indices: list[int],
    image_key_positions: list[int],
) -> Iterator[None]:
    """Zero selected heads' image-key probabilities and renormalize.

    This intervention is applied inside the custom attention backend before
    ``A_h V_h`` is formed. It leaves every unselected head unchanged and
    refuses a knockout that removes all valid mass from any selected query.
    """

    import torch

    module = get_language_layers(model)[int(layer_idx)].self_attn
    if hasattr(module, "_vlm_mechanistic_attention_replacement"):
        raise RuntimeError("an attention-map intervention is already installed")

    def replacement(probabilities: Any) -> Any:
        keys = torch.as_tensor(
            image_key_positions, dtype=torch.long, device=probabilities.device
        )
        result = probabilities.clone()
        for head_idx in head_indices:
            selected = result[:, int(head_idx), :, :]
            selected[..., keys] = 0
            denominator = selected.sum(dim=-1, keepdim=True)
            if bool((denominator <= 0).any()):
                raise RuntimeError("image-attention knockout removed all valid mass")
            result[:, int(head_idx), :, :] = selected / denominator
        return result

    module._vlm_mechanistic_attention_replacement = replacement
    try:
        yield
    finally:
        delattr(module, "_vlm_mechanistic_attention_replacement")


@contextmanager
def projected_head_set_replacement(
    model: Any,
    *,
    replacements: dict[tuple[int, int], Any],
    recipient_projected: dict[int, Any],
    positions: list[int],
) -> Iterator[None]:
    """Replace several exact post-W_O head contributions at aligned positions.

    ``replacements[(layer, head)]`` may be ``None`` for zero ablation, a
    ``[positions, hidden]`` tensor, or a ``[1, positions, hidden]`` tensor.
    The shared ``o_proj`` bias remains untouched and is therefore counted once.
    """

    import torch

    layers = get_language_layers(model)
    by_layer: dict[int, list[tuple[int, Any]]] = {}
    for (layer_idx, head_idx), value in replacements.items():
        by_layer.setdefault(int(layer_idx), []).append((int(head_idx), value))
    handles = []
    for layer_idx, values in by_layer.items():
        if layer_idx not in recipient_projected:
            raise KeyError(f"missing recipient projected contributions for layer {layer_idx}")
        recipient = recipient_projected[layer_idx]
        index = torch.as_tensor(positions, dtype=torch.long, device=recipient.device)

        def hook(module: Any, args: tuple[Any, ...], output: Any, *, layer_values=values, recipient_tensor=recipient, position_index=index) -> Any:
            current, *rest = output if isinstance(output, tuple) else (output,)
            if current.shape[0] != 1:
                raise RuntimeError("projected head-set replacement expects batch size one")
            patched = current.clone()
            current_index = position_index.to(device=current.device)
            delta = torch.zeros(
                (position_index.numel(), current.shape[-1]),
                dtype=current.dtype,
                device=current.device,
            )
            for head_idx, replacement in layer_values:
                old = recipient_tensor[0].index_select(0, position_index)[:, head_idx, :]
                if replacement is None:
                    new = torch.zeros_like(old)
                else:
                    new = replacement
                    if new.ndim == 3:
                        new = new[0]
                    new = new.to(device=current.device, dtype=current.dtype)
                    if new.shape != old.shape:
                        raise ValueError(
                            f"replacement shape {tuple(new.shape)} != {tuple(old.shape)}"
                        )
                delta = delta + new - old.to(device=current.device, dtype=current.dtype)
            patched[0, current_index, :] = current[0, current_index, :] + delta
            return (patched, *rest) if isinstance(output, tuple) else patched

        handles.append(layers[layer_idx].self_attn.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def batched_head_scaling(model: Any, *, layer_idx: int, head_indices: list[int], scale: float) -> Iterator[None]:
    layer = get_language_layers(model)[layer_idx]
    layer.self_attn._vlm_mechanistic_batch_head_scales = [(int(head), float(scale)) for head in head_indices]
    try: yield
    finally: delattr(layer.self_attn, "_vlm_mechanistic_batch_head_scales")


@contextmanager
def visual_attention_map_patch(
    model: Any,
    *,
    layer_idx: int,
    head_idx: int,
    donor_probabilities: Any,
    donor_query_position: int,
    recipient_query_position: int,
    donor_visual_positions: list[int],
    recipient_visual_positions: list[int],
) -> Iterator[None]:
    """Replace an aligned visual-key pattern while preserving recipient mass."""

    if len(donor_visual_positions) != len(recipient_visual_positions):
        raise ValueError("visual-key transplantation requires equal aligned token counts")
    import torch

    layer = get_language_layers(model)[layer_idx]
    donor_keys = torch.as_tensor(
        donor_visual_positions, dtype=torch.long, device=donor_probabilities.device
    )

    def replacement(probabilities: Any) -> Any:
        if recipient_query_position >= probabilities.shape[-2]:
            raise RuntimeError("recipient query position is outside current attention map")
        recipient_keys = torch.as_tensor(
            recipient_visual_positions, dtype=torch.long, device=probabilities.device
        )
        donor_slice = donor_probabilities[
            0, head_idx, donor_query_position, :
        ].index_select(-1, donor_keys).to(probabilities.device, probabilities.dtype)
        donor_mass = donor_slice.sum()
        if float(donor_mass) <= 0:
            raise RuntimeError("donor visual attention slice has zero mass")
        result = probabilities.clone()
        current = result[0, head_idx, recipient_query_position, :]
        recipient_mass = current.index_select(-1, recipient_keys).sum()
        current[recipient_keys] = donor_slice / donor_mass * recipient_mass
        result[0, head_idx, recipient_query_position, :] = current
        _assert_attention_normalized(result, label="attention-map patch")
        return result

    layer.self_attn._vlm_mechanistic_attention_replacement = replacement
    try:
        yield
    finally:
        delattr(layer.self_attn, "_vlm_mechanistic_attention_replacement")


@contextmanager
def visual_attention_map_patch_many(
    model: Any,
    *,
    layer_idx: int,
    head_idx: int,
    donor_probabilities: Any,
    donor_query_positions: list[int],
    recipient_query_positions: list[int],
    donor_visual_positions: list[int],
    recipient_visual_positions: list[int],
) -> Iterator[None]:
    """Transplant aligned visual-key patterns for multiple answer queries."""
    if len(donor_query_positions) != len(recipient_query_positions):
        raise ValueError("query alignments must have equal cardinality")
    if len(donor_visual_positions) != len(recipient_visual_positions):
        raise ValueError("visual-key alignments must have equal cardinality")
    import torch
    layer = get_language_layers(model)[layer_idx]
    donor_keys = torch.as_tensor(donor_visual_positions, dtype=torch.long, device=donor_probabilities.device)
    def replacement(probabilities: Any) -> Any:
        result = probabilities.clone(); recipient_keys = torch.as_tensor(recipient_visual_positions, dtype=torch.long, device=probabilities.device)
        for donor_query, recipient_query in zip(donor_query_positions, recipient_query_positions):
            if donor_query >= donor_probabilities.shape[-2] or recipient_query >= probabilities.shape[-2]:
                raise RuntimeError("aligned answer query is outside an attention map")
            source = donor_probabilities[0, head_idx, donor_query, :].index_select(-1, donor_keys).to(probabilities.device, probabilities.dtype)
            if float(source.sum()) <= 0: raise RuntimeError("donor visual slice has zero mass")
            current = result[0, head_idx, recipient_query, :]; mass = current.index_select(-1, recipient_keys).sum(); current[recipient_keys] = source / source.sum() * mass; result[0, head_idx, recipient_query, :] = current
        _assert_attention_normalized(result, label="attention-map patch")
        return result
    layer.self_attn._vlm_mechanistic_attention_replacement = replacement
    try: yield
    finally: delattr(layer.self_attn, "_vlm_mechanistic_attention_replacement")


@contextmanager
def batched_visual_attention_map_patch_many(
    model: Any,
    *,
    layer_idx: int,
    head_indices: list[int],
    donor_probabilities: Any,
    donor_query_positions: list[int],
    recipient_query_positions: list[int],
    donor_visual_positions: list[int],
    recipient_visual_positions: list[int],
) -> Iterator[None]:
    """Apply a different head-map transplant to every model batch row."""

    if len(donor_query_positions) != len(recipient_query_positions):
        raise ValueError("query alignments must have equal cardinality")
    if len(donor_visual_positions) != len(recipient_visual_positions):
        raise ValueError("visual-key alignments must have equal cardinality")
    import torch

    layer = get_language_layers(model)[layer_idx]
    donor_keys = torch.as_tensor(
        donor_visual_positions,
        dtype=torch.long,
        device=donor_probabilities.device,
    )

    def replacement(probabilities: Any) -> Any:
        if probabilities.shape[0] != len(head_indices):
            raise RuntimeError(
                f"attention transplant batch {len(head_indices)} != model batch {probabilities.shape[0]}"
            )
        recipient_keys = torch.as_tensor(
            recipient_visual_positions,
            dtype=torch.long,
            device=probabilities.device,
        )
        result = probabilities.clone()
        for batch_idx, head_idx in enumerate(head_indices):
            for donor_query, recipient_query in zip(
                donor_query_positions, recipient_query_positions
            ):
                if (
                    donor_query >= donor_probabilities.shape[-2]
                    or recipient_query >= probabilities.shape[-2]
                ):
                    raise RuntimeError("aligned answer query is outside an attention map")
                source = donor_probabilities[
                    0, head_idx, donor_query, :
                ].index_select(-1, donor_keys).to(
                    probabilities.device, probabilities.dtype
                )
                source_mass = source.sum()
                if float(source_mass) <= 0:
                    raise RuntimeError("donor visual slice has zero mass")
                current = result[batch_idx, head_idx, recipient_query, :]
                recipient_mass = current.index_select(-1, recipient_keys).sum()
                current[recipient_keys] = source / source_mass * recipient_mass
                result[batch_idx, head_idx, recipient_query, :] = current
        _assert_attention_normalized(result, label="batched attention-map patch")
        return result

    layer.self_attn._vlm_mechanistic_attention_replacement = replacement
    try:
        yield
    finally:
        delattr(layer.self_attn, "_vlm_mechanistic_attention_replacement")
