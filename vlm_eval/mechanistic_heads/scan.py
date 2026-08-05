from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from vlm_eval.mechanistic_heads.patching import batched_single_head_patches


@dataclass(frozen=True)
class HeadPatchBatch:
    head_indices: list[int]
    patched_outputs: Any


def iter_head_patch_batches(
    recipient_output: Any,
    donor_projected_heads: Any,
    recipient_projected_heads: Any,
    *,
    head_microbatch: int = 32,
) -> Iterator[HeadPatchBatch]:
    if head_microbatch <= 0:
        raise ValueError("head_microbatch must be positive")
    n_heads = donor_projected_heads.shape[-2]
    for start in range(0, n_heads, head_microbatch):
        indices = list(range(start, min(start + head_microbatch, n_heads)))
        yield HeadPatchBatch(
            head_indices=indices,
            patched_outputs=batched_single_head_patches(
                recipient_output,
                donor_projected_heads,
                recipient_projected_heads,
                indices,
            ),
        )


def symmetric_bidirectional_score(
    forward_margin_shift: float, reverse_margin_shift: float
) -> float:
    """Positive when each donor moves the recipient margin toward donor answer."""

    return 0.5 * (float(forward_margin_shift) + float(reverse_margin_shift))
