from __future__ import annotations

import json
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from adapters.qwen25_vl import Qwen25VLAdapter, _resolve_device_map, _resolve_torch_dtype


ATTENTION_IMPL_NAME = "vlm_gaze_head_image_bias"


@dataclass
class Qwen25VLGazeAttentionAdapter(Qwen25VLAdapter):
    attention_alpha: float = 0.0
    attention_controller: str = "fixed"
    target_attention_mass: float = 0.0
    max_attention_alpha: float = 5.0
    record_token_confidence: bool = True
    gaze_ranking_path: str = "segments/gaze_heads_qwen25/runs/gaze_discovery_merged_0_500/gaze_head_ranking.json"
    top_k_gaze: int = 0
    head_selection: str = "gaze_global"
    head_selection_seed: int = 0
    decode_only: bool = False
    last_generation_metadata: dict[str, Any] | None = None
    _attention_records: list[dict[str, float | int]] = field(default_factory=list, init=False)
    _boosted_heads: list[tuple[int, int]] = field(default_factory=list, init=False)
    _boosted_heads_by_layer: dict[int, list[int]] = field(default_factory=dict, init=False)
    _token_confidence_metadata: dict[str, float | int | None] = field(
        default_factory=dict, init=False
    )

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
            torch_dtype=_resolve_torch_dtype(torch),
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
                "attention_controller": self.attention_controller,
                "target_attention_mass": self.target_attention_mass,
                "max_attention_alpha": self.max_attention_alpha,
                "gaze_ranking_path": self.gaze_ranking_path,
                "top_k_gaze": self.top_k_gaze,
                "head_selection": self.head_selection,
                "head_selection_seed": self.head_selection_seed,
                "decode_only": self.decode_only,
                "boosted_heads": [{"layer": layer, "head": head} for layer, head in self._boosted_heads],
                "attention_impl": ATTENTION_IMPL_NAME,
            }
        )
        return config

    def _configure_attention_modules(self) -> None:
        layers = _language_layers(self._model)
        first_attn = layers[0].self_attn
        n_heads = int(
            getattr(first_attn, "num_heads", 0)
            or getattr(self._model.config, "num_attention_heads", 0)
            or getattr(
                getattr(self._model.config, "text_config", None),
                "num_attention_heads",
                0,
            )
        )
        if not n_heads:
            raise RuntimeError("Could not determine the number of Qwen attention heads")
        self._boosted_heads = select_image_boost_heads(
            ranking_path=Path(self.gaze_ranking_path),
            n_select=self.top_k_gaze,
            selection=self.head_selection,
            seed=self.head_selection_seed,
            n_layers=len(layers),
            n_heads=n_heads,
        )
        self._boosted_heads_by_layer = _heads_by_layer(self._boosted_heads)
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            attn._vlm_gaze_image_layer_idx = layer_idx
            attn._vlm_gaze_image_attention_alpha = float(self.attention_alpha)
            attn._vlm_gaze_image_attention_controller = self.attention_controller
            attn._vlm_gaze_image_target_attention_mass = float(
                self.target_attention_mass
            )
            attn._vlm_gaze_image_max_attention_alpha = float(
                self.max_attention_alpha
            )
            attn._vlm_gaze_image_head_indices = self._boosted_heads_by_layer.get(layer_idx, [])
            attn._vlm_gaze_image_token_mask = None
            attn._vlm_gaze_image_decode_only = bool(self.decode_only)
            attn._vlm_gaze_image_attention_records = self._attention_records

    def _before_generate(self, inputs: Any, *, include_image: bool) -> None:
        self._attention_records.clear()
        self._token_confidence_metadata.clear()
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
            "attention_controller": self.attention_controller,
            "target_attention_mass": self.target_attention_mass,
            "max_attention_alpha": self.max_attention_alpha,
            "top_k_gaze": self.top_k_gaze,
            "head_selection": self.head_selection,
            "head_selection_seed": self.head_selection_seed,
            "decode_only": self.decode_only,
            "token_confidence": dict(self._token_confidence_metadata),
        }

    def _generation_kwargs(self, max_new_tokens: int | None = None) -> dict[str, Any]:
        kwargs = super()._generation_kwargs(max_new_tokens=max_new_tokens)
        if self.record_token_confidence:
            kwargs.update(return_dict_in_generate=True, output_scores=True)
        return kwargs

    def _after_generate_output(self, generated: Any, inputs: Any) -> None:
        if not self.record_token_confidence or not hasattr(generated, "scores"):
            return
        scores = list(generated.scores)
        sequences = generated.sequences
        input_length = int(inputs.input_ids.shape[1])
        generated_ids = sequences[0, input_length : input_length + len(scores)]
        chosen_probabilities: list[float] = []
        for score, token_id in zip(scores, generated_ids):
            log_probability = self._torch.log_softmax(score[0].float(), dim=-1)[
                int(token_id)
            ]
            chosen_probabilities.append(float(log_probability.exp().detach().cpu()))
        if not chosen_probabilities:
            self._token_confidence_metadata.update(
                {
                    "n_tokens": 0,
                    "geometric_mean_probability": None,
                    "first_token_probability": None,
                    "minimum_token_probability": None,
                }
            )
            return
        mean_log = sum(math.log(max(value, 1e-12)) for value in chosen_probabilities) / len(
            chosen_probabilities
        )
        self._token_confidence_metadata.update(
            {
                "n_tokens": len(chosen_probabilities),
                "geometric_mean_probability": math.exp(mean_log),
                "first_token_probability": chosen_probabilities[0],
                "minimum_token_probability": min(chosen_probabilities),
            }
        )


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
    attention_controller: str = "fixed",
    target_attention_mass: float = 0.0,
    max_attention_alpha: float = 5.0,
    gaze_ranking_path: str = "segments/gaze_heads_qwen25/runs/gaze_discovery_merged_0_500/gaze_head_ranking.json",
    top_k_gaze: int = 0,
    head_selection: str = "gaze_global",
    head_selection_seed: int = 0,
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
        attention_controller=attention_controller,
        target_attention_mass=target_attention_mass,
        max_attention_alpha=max_attention_alpha,
        gaze_ranking_path=gaze_ranking_path,
        top_k_gaze=top_k_gaze,
        head_selection=head_selection,
        head_selection_seed=head_selection_seed,
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

    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    image_mask = _token_mask_for_key_length(
        getattr(module, "_vlm_gaze_image_token_mask", None),
        key_states.shape[-2],
    )
    query_slice = slice(-1, None) if query.shape[-2] > 1 else slice(None)
    alpha = float(getattr(module, "_vlm_gaze_image_attention_alpha", 0.0))
    controller = str(
        getattr(module, "_vlm_gaze_image_attention_controller", "fixed")
    )
    head_indices = [int(head_idx) for head_idx in getattr(module, "_vlm_gaze_image_head_indices", [])]
    decode_only = bool(getattr(module, "_vlm_gaze_image_decode_only", False))
    target_mass = float(
        getattr(module, "_vlm_gaze_image_target_attention_mass", 0.0)
    )
    max_alpha = float(
        getattr(module, "_vlm_gaze_image_max_attention_alpha", 5.0)
    )
    should_bias = (
        image_mask is not None
        and head_indices
        and (alpha != 0.0 or controller == "target_mass")
        and not (decode_only and query.shape[-2] > 1)
    )
    target_diagnostics: dict[str, float] = {}
    if should_bias:
        # Full-sequence mode must affect every prefill query, just like the
        # reference steering mask which broadcasts over the query dimension.
        # ``query_slice`` remains last-query-only below for compact telemetry.
        bias_query_slice = slice(None)
        if controller == "fixed":
            for head_idx in head_indices:
                attn_weights[:, head_idx, bias_query_slice, image_mask] += alpha
        elif controller == "target_mass":
            effective_alpha, baseline_mass = _target_mass_alpha(
                attn_weights[:, head_indices, bias_query_slice, :],
                image_mask=image_mask,
                target_mass=target_mass,
                max_alpha=max_alpha,
            )
            for position, head_idx in enumerate(head_indices):
                attn_weights[:, head_idx, bias_query_slice, image_mask] += (
                    effective_alpha[:, position, :, None].to(attn_weights.dtype)
                )
            target_diagnostics = {
                "preboost_head_image_attention_mass": float(
                    baseline_mass.float().mean().detach().cpu()
                ),
                "mean_effective_alpha": float(
                    effective_alpha.float().mean().detach().cpu()
                ),
                "alpha_cap_fraction": float(
                    effective_alpha.eq(max_alpha).float().mean().detach().cpu()
                ),
            }
        else:
            raise ValueError(f"Unknown attention controller: {controller}")

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
        record.update(target_diagnostics)
        records.append(record)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _target_mass_alpha(
    selected_logits: Any,
    *,
    image_mask: Any,
    target_mass: float,
    max_alpha: float,
) -> tuple[Any, Any]:
    """Return exact positive logit bias needed to reach a target image mass."""
    if not 0.0 < target_mass < 1.0:
        raise ValueError("target_attention_mass must be strictly between 0 and 1")
    if max_alpha <= 0.0:
        raise ValueError("max_attention_alpha must be positive")
    import torch

    probabilities = torch.softmax(selected_logits.float(), dim=-1)
    baseline_mass = probabilities[..., image_mask].sum(dim=-1)
    epsilon = torch.finfo(probabilities.dtype).eps
    clipped = baseline_mass.clamp(epsilon, 1.0 - epsilon)
    target_log_odds = math.log(target_mass / (1.0 - target_mass))
    baseline_log_odds = torch.log(clipped) - torch.log1p(-clipped)
    effective_alpha = (target_log_odds - baseline_log_odds).clamp(
        min=0.0, max=max_alpha
    )
    effective_alpha = torch.where(
        torch.isfinite(effective_alpha),
        effective_alpha,
        torch.full_like(effective_alpha, max_alpha),
    )
    return effective_alpha, baseline_mass


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


def select_image_boost_heads(
    *,
    ranking_path: Path,
    n_select: int,
    selection: str,
    seed: int,
    n_layers: int,
    n_heads: int,
) -> list[tuple[int, int]]:
    """Select an exact, reproducible head set for benchmark interventions."""
    if n_select <= 0:
        return []
    ranking_rows = json.loads(ranking_path.read_text(encoding="utf-8"))
    ranking = [
        (int(row["layer"]), int(row["head"])) for row in ranking_rows
    ]
    valid = [
        head
        for head in ranking
        if 0 <= head[0] < n_layers and 0 <= head[1] < n_heads
    ]
    if selection == "gaze_global":
        selected = valid[:n_select]
    elif selection in {"gaze_early", "gaze_middle", "gaze_late"}:
        bands = {
            "gaze_early": (0, 11),
            "gaze_middle": (12, 23),
            "gaze_late": (24, 35),
        }
        low, high = bands[selection]
        selected = [
            head for head in valid if low <= head[0] <= min(high, n_layers - 1)
        ][:n_select]
    elif selection in {"layer_matched_random", "layer_matched_low"}:
        reference = valid[:n_select]
        layer_counts: dict[int, int] = {}
        for layer, _ in reference:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        scores_path = ranking_path.with_name("gaze_scores.npy")
        scores = None
        if selection == "layer_matched_low":
            if not scores_path.exists():
                raise FileNotFoundError(
                    f"Layer-matched low selection requires {scores_path}"
                )
            import numpy as np

            scores = np.load(scores_path)
        import torch

        selected = []
        excluded = set(reference)
        for layer, count in sorted(layer_counts.items()):
            candidates = [
                (layer, head)
                for head in range(n_heads)
                if (layer, head) not in excluded
            ]
            if selection == "layer_matched_low":
                candidates.sort(
                    key=lambda item: float(scores[item[0], item[1]])
                )
            else:
                generator = torch.Generator().manual_seed(seed * 10_000 + layer)
                order = torch.randperm(
                    len(candidates), generator=generator
                ).tolist()
                candidates = [candidates[index] for index in order]
            selected.extend(candidates[:count])
    elif selection == "paper_random":
        import torch

        reference = set(valid[:n_select])
        candidates = [
            (layer, head)
            for layer in range(20, min(35, n_layers - 1) + 1)
            for head in range(n_heads)
            if (layer, head) not in reference
        ]
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(candidates), generator=generator).tolist()
        selected = [candidates[index] for index in order[:n_select]]
    else:
        raise ValueError(f"Unknown head selection: {selection}")
    if len(selected) != n_select or len(set(selected)) != n_select:
        raise RuntimeError(
            f"{selection} produced {len(set(selected))} unique heads; "
            f"expected exactly {n_select}"
        )
    return selected


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


def _repeat_kv(hidden_states: Any, n_rep: int) -> Any:
    """Expand KV heads without depending on one Transformers model module."""
    batch, n_kv_heads, sequence_length, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, n_kv_heads, n_rep, sequence_length, head_dim
    )
    return expanded.reshape(batch, n_kv_heads * n_rep, sequence_length, head_dim)


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
    effective_alphas = [
        float(record["mean_effective_alpha"])
        for record in records
        if "mean_effective_alpha" in record
    ]
    preboost_masses = [
        float(record["preboost_head_image_attention_mass"])
        for record in records
        if "preboost_head_image_attention_mass" in record
    ]
    cap_fractions = [
        float(record["alpha_cap_fraction"])
        for record in records
        if "alpha_cap_fraction" in record
    ]
    return {
        "n_records": len(records),
        "mean_image_attention_mass": _mean(all_masses),
        "mean_boosted_layer_image_attention_mass": _mean(boosted_layer_masses),
        "mean_boosted_head_image_attention_mass": _mean(boosted_masses),
        "mean_preboost_head_image_attention_mass": _mean(preboost_masses),
        "mean_effective_alpha": _mean(effective_alphas),
        "mean_alpha_cap_fraction": _mean(cap_fractions),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
