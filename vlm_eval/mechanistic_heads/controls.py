from __future__ import annotations

from collections import Counter
import random
from typing import Iterable


Head = tuple[int, int]


def layer_matched_control_draws(
    selected: Iterable[Head],
    *,
    n_layers: int,
    n_heads: int,
    n_draws: int = 20,
    seed: int = 0,
    excluded: Iterable[Head] = (),
) -> list[list[Head]]:
    selected = list(selected)
    if n_draws < 20:
        raise ValueError("scientific control distributions require at least 20 draws")
    layer_counts = Counter(layer for layer, _ in selected)
    forbidden = set(selected) | set(excluded)
    draws: list[list[Head]] = []
    for draw_idx in range(n_draws):
        rng = random.Random(seed * 100_003 + draw_idx)
        draw: list[Head] = []
        for layer, count in sorted(layer_counts.items()):
            if not 0 <= layer < n_layers:
                raise ValueError(f"selected layer {layer} outside architecture")
            candidates = [
                (layer, head)
                for head in range(n_heads)
                if (layer, head) not in forbidden
            ]
            if len(candidates) < count:
                raise ValueError(f"not enough layer-{layer} control heads")
            draw.extend(rng.sample(candidates, count))
        draws.append(sorted(draw))
    return draws


def multivariate_matched_control_draws(
    selected: Iterable[Head],
    head_features: dict[Head, dict[str, float]],
    *,
    feature_names: tuple[str, ...] = (
        "image_attention",
        "projected_output_norm",
        "attention_entropy",
        "gaze_score",
        "general_causal_importance",
    ),
    n_draws: int = 20,
    seed: int = 0,
    nearest_pool: int = 8,
) -> list[list[Head]]:
    """Repeated controls matched on layer and standardized head diagnostics."""

    import math

    selected = list(selected)
    if n_draws < 20:
        raise ValueError("scientific control distributions require at least 20 draws")
    universe = set(head_features)
    if not set(selected) <= universe:
        raise ValueError("selected heads are missing feature rows")
    scales = {}
    for name in feature_names:
        values = [float(features[name]) for features in head_features.values() if name in features]
        if len(values) != len(head_features):
            raise ValueError(f"feature {name} is missing for one or more heads")
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        scales[name] = (mean, math.sqrt(variance) or 1.0)

    def distance(left: Head, right: Head) -> float:
        if left[0] != right[0]:
            return math.inf
        return sum(
            (
                (head_features[left][name] - head_features[right][name])
                / scales[name][1]
            )
            ** 2
            for name in feature_names
        ) ** 0.5

    draws = []
    forbidden = set(selected)
    for draw_idx in range(n_draws):
        rng = random.Random(seed * 1_000_003 + draw_idx)
        used: set[Head] = set()
        draw = []
        targets = list(selected)
        rng.shuffle(targets)
        for target in targets:
            candidates = sorted(
                universe - forbidden - used,
                key=lambda candidate: (distance(target, candidate), candidate),
            )
            candidates = [candidate for candidate in candidates if math.isfinite(distance(target, candidate))]
            if not candidates:
                raise ValueError(f"no matched control candidate for {target}")
            choice = rng.choice(candidates[: min(nearest_pool, len(candidates))])
            used.add(choice)
            draw.append(choice)
        draws.append(sorted(draw))
    return draws
