from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class LazyProjectedHeads:
    """Lazy exact post-W_O head contributions.

    A dense ``[batch, sequence, heads, model_width]`` tensor can require tens
    of GiB for high-resolution multimodal prompts.  Keep the much smaller raw
    ``A_h V_h`` tensor and materialize only the positions and heads selected by
    downstream patching code.  The object intentionally implements just the
    tensor indexing operations used by this repository.
    """

    _DIMENSIONS = ("batch", "sequence", "head", "width")

    def __init__(
        self,
        raw_heads: Any,
        o_proj_weight: Any,
        *,
        indices: dict[str, Any] | None = None,
        present: dict[str, bool] | None = None,
    ) -> None:
        import torch

        if raw_heads.ndim != 4:
            raise ValueError("lazy projected heads require [batch, sequence, heads, head_dim]")
        n_heads, head_dim = raw_heads.shape[-2:]
        if o_proj_weight.ndim != 2 or o_proj_weight.shape[1] != n_heads * head_dim:
            raise ValueError("o_proj weight does not match captured raw heads")
        self.raw_heads = raw_heads
        self.o_proj_weight = o_proj_weight
        sizes = {
            "batch": raw_heads.shape[0],
            "sequence": raw_heads.shape[1],
            "head": raw_heads.shape[2],
            "width": o_proj_weight.shape[0],
        }
        self._indices = indices or {
            name: torch.arange(size, device=raw_heads.device)
            for name, size in sizes.items()
        }
        self._present = present or {name: True for name in self._DIMENSIONS}

    @property
    def device(self) -> Any:
        return self.raw_heads.device

    @property
    def dtype(self) -> Any:
        return self.raw_heads.dtype

    @property
    def shape(self) -> Any:
        import torch

        return torch.Size(
            [self._indices[name].numel() for name in self._DIMENSIONS if self._present[name]]
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def _view(self, key: Any) -> "LazyProjectedHeads":
        import torch

        keys = list(key if isinstance(key, tuple) else (key,))
        if keys.count(Ellipsis) > 1:
            raise IndexError("only one ellipsis is supported")
        current_names = [name for name in self._DIMENSIONS if self._present[name]]
        if Ellipsis in keys:
            position = keys.index(Ellipsis)
            fill = len(current_names) - (len(keys) - 1)
            keys = keys[:position] + [slice(None)] * fill + keys[position + 1 :]
        if len(keys) > len(current_names):
            raise IndexError("too many indices for lazy projected heads")
        keys += [slice(None)] * (len(current_names) - len(keys))
        indices = dict(self._indices)
        present = dict(self._present)
        for name, selector in zip(current_names, keys):
            base = indices[name]
            if isinstance(selector, int):
                position = selector if selector >= 0 else base.numel() + selector
                if position < 0 or position >= base.numel():
                    raise IndexError(f"{name} index is out of range")
                indices[name] = base[position : position + 1]
                present[name] = False
            else:
                selected = base[selector]
                if selected.ndim == 0:
                    selected = selected.reshape(1)
                    present[name] = False
                indices[name] = selected.to(device=self.raw_heads.device, dtype=torch.long)
        return LazyProjectedHeads(
            self.raw_heads,
            self.o_proj_weight,
            indices=indices,
            present=present,
        )

    def __getitem__(self, key: Any) -> Any:
        view = self._view(key)
        # All repository hot paths select one head before asking for the
        # resulting tensor. Keep broader intermediate views lazy.
        return view if view._present["head"] else view.materialize()

    def index_select(self, dim: int, index: Any) -> "LazyProjectedHeads":
        names = [name for name in self._DIMENSIONS if self._present[name]]
        dim = dim if dim >= 0 else len(names) + dim
        if dim < 0 or dim >= len(names):
            raise IndexError("lazy projected-head dimension is out of range")
        name = names[dim]
        indices = dict(self._indices)
        selector = index.to(device=self.raw_heads.device, dtype=self._indices[name].dtype)
        indices[name] = self._indices[name].index_select(0, selector)
        return LazyProjectedHeads(
            self.raw_heads,
            self.o_proj_weight,
            indices=indices,
            present=dict(self._present),
        )

    def materialize(self) -> Any:
        import torch

        batch = self._indices["batch"]
        sequence = self._indices["sequence"]
        heads = self._indices["head"]
        width = self._indices["width"]
        raw = self.raw_heads.index_select(0, batch).index_select(1, sequence).index_select(2, heads)
        n_heads, head_dim = self.raw_heads.shape[-2:]
        weight = self.o_proj_weight.reshape(
            self.o_proj_weight.shape[0], n_heads, head_dim
        ).permute(1, 2, 0)
        weight = weight.index_select(0, heads.to(weight.device)).index_select(
            2, width.to(weight.device)
        )
        weight = weight.to(raw.device)
        result = torch.einsum("bshd,hdw->bshw", raw, weight)
        for original_dim, name in reversed(list(enumerate(self._DIMENSIONS))):
            if not self._present[name]:
                result = result.squeeze(original_dim)
        return result

    def sum(self, *args: Any, **kwargs: Any) -> Any:
        return self.materialize().sum(*args, **kwargs)

    def equal(self, other: Any) -> bool:
        other_value = other.materialize() if isinstance(other, LazyProjectedHeads) else other
        return bool(self.materialize().equal(other_value))


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
