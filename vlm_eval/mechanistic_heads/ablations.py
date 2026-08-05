from __future__ import annotations

from typing import Any


def zero_ablation(value: Any) -> Any:
    return value.new_zeros(value.shape)


def mean_ablation(value: Any, reference: Any, *, dim: int = 0) -> Any:
    mean = reference.mean(dim=dim, keepdim=True)
    return mean.expand_as(value)


def resample_ablation(value: Any, reference: Any, indices: Any) -> Any:
    sampled = reference.index_select(0, indices.to(reference.device))
    if sampled.shape != value.shape:
        raise ValueError(f"resampled shape {sampled.shape} != target {value.shape}")
    return sampled


def scale_projected_head(value: Any, scale: float) -> Any:
    return value * float(scale)


def image_attention_knockout(probabilities: Any, image_mask: Any) -> Any:
    """Set image-key probability to zero and renormalize remaining keys."""

    result = probabilities.clone()
    result[..., image_mask] = 0
    denominator = result.sum(dim=-1, keepdim=True)
    if bool((denominator <= 0).any()):
        raise ValueError("image knockout removed all valid attention keys")
    return result / denominator
