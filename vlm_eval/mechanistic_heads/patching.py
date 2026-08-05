from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def projected_head_contributions(
    raw_heads: Any, o_proj_weight: Any
) -> Any:
    """Return each head's exact bias-free post-W_O contribution.

    Args:
        raw_heads: tensor shaped ``[..., n_heads, head_dim]`` containing A_h V_h.
        o_proj_weight: tensor shaped ``[model_width, n_heads * head_dim]``.
    Returns:
        tensor shaped ``[..., n_heads, model_width]``.
    """

    if raw_heads.ndim < 3:
        raise ValueError("raw_heads must have [..., heads, head_dim] dimensions")
    n_heads, head_dim = raw_heads.shape[-2:]
    if o_proj_weight.ndim != 2:
        raise ValueError("o_proj_weight must be a matrix")
    if o_proj_weight.shape[1] != n_heads * head_dim:
        raise ValueError(
            f"o_proj input width {o_proj_weight.shape[1]} != {n_heads}*{head_dim}"
        )
    # [model_width, heads, head_dim] -> [heads, head_dim, model_width]
    per_head_weight = o_proj_weight.reshape(
        o_proj_weight.shape[0], n_heads, head_dim
    ).permute(1, 2, 0)
    return raw_heads.unsqueeze(-2).matmul(per_head_weight).squeeze(-2)


def reconstruct_attention_output(
    projected_heads: Any, o_proj_bias: Any | None = None
) -> Any:
    result = projected_heads.sum(dim=-2)
    return result if o_proj_bias is None else result + o_proj_bias


def patch_projected_head(
    recipient_output: Any,
    donor_projected: Any,
    recipient_projected: Any,
) -> Any:
    if donor_projected.shape != recipient_projected.shape:
        raise ValueError("donor and recipient projected head shapes differ")
    if recipient_output.shape != donor_projected.shape:
        raise ValueError("attention output and projected head shapes differ")
    # A self/identity patch is semantically a no-op. Returning the original
    # tensor also avoids a final-bit roundoff from `(x + y) - y`.
    if donor_projected is recipient_projected or bool(
        donor_projected.equal(recipient_projected)
    ):
        return recipient_output
    return recipient_output + donor_projected - recipient_projected


def batched_single_head_patches(
    recipient_output: Any,
    donor_projected_heads: Any,
    recipient_projected_heads: Any,
    head_indices: list[int] | None = None,
) -> Any:
    """Stack one independently patched recipient per requested head."""

    if donor_projected_heads.shape != recipient_projected_heads.shape:
        raise ValueError("donor and recipient projected-head tensors differ")
    indices = head_indices or list(range(donor_projected_heads.shape[-2]))
    return recipient_output.unsqueeze(0) + donor_projected_heads[..., indices, :].movedim(
        -2, 0
    ) - recipient_projected_heads[..., indices, :].movedim(-2, 0)


@dataclass(frozen=True)
class AttentionTransplant:
    probabilities: Any
    normalization_rule: str


def transplant_attention_map(
    recipient: Any,
    donor: Any,
    *,
    recipient_key_indices: Any | None = None,
    donor_key_indices: Any | None = None,
    atol: float = 1e-5,
) -> AttentionTransplant:
    """Transplant aligned attention, refusing silent sequence truncation.

    With equal key lengths, the full distribution is replaced. With unequal
    lengths, explicit equal-cardinality visual-key indices are mandatory. The
    donor visual pattern is normalized within that slice and receives the
    recipient's original total visual mass, preserving nonvisual probabilities.
    """

    import torch

    if recipient.shape[:-1] != donor.shape[:-1]:
        raise ValueError("attention query/batch/head dimensions are not aligned")
    if recipient_key_indices is None and donor_key_indices is None:
        if recipient.shape[-1] != donor.shape[-1]:
            raise ValueError(
                "unequal key lengths require explicit aligned visual-key indices"
            )
        result = donor.clone()
        rule = "full aligned distribution replacement"
    else:
        if recipient_key_indices is None or donor_key_indices is None:
            raise ValueError("both recipient and donor key indices are required")
        recipient_key_indices = torch.as_tensor(
            recipient_key_indices, dtype=torch.long, device=recipient.device
        )
        donor_key_indices = torch.as_tensor(
            donor_key_indices, dtype=torch.long, device=donor.device
        )
        if recipient_key_indices.numel() != donor_key_indices.numel():
            raise ValueError("aligned visual-key slices must have equal cardinality")
        result = recipient.clone()
        old_mass = recipient.index_select(-1, recipient_key_indices).sum(-1, keepdim=True)
        donor_slice = donor.index_select(-1, donor_key_indices)
        denom = donor_slice.sum(-1, keepdim=True)
        if bool((denom <= 0).any()):
            raise ValueError("donor visual slice has zero attention mass")
        result[..., recipient_key_indices] = donor_slice / denom * old_mass
        rule = "visual slice shape replacement preserving recipient visual mass"
    sums = result.float().sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=atol, rtol=0):
        raise RuntimeError("transplanted attention does not sum to one")
    return AttentionTransplant(result, rule)
