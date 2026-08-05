from __future__ import annotations

from collections import defaultdict
import random
from typing import Callable, Iterable, TypeVar


T = TypeVar("T")


def group_split(
    rows: Iterable[T],
    *,
    group_key: Callable[[T], str],
    fractions: dict[str, float],
    seed: int,
) -> dict[str, list[T]]:
    if not fractions or any(value <= 0 for value in fractions.values()):
        raise ValueError("split fractions must all be positive")
    total = sum(fractions.values())
    normalized = {key: value / total for key, value in fractions.items()}
    groups: dict[str, list[T]] = defaultdict(list)
    for row in rows:
        groups[str(group_key(row))].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    targets = {name: normalized[name] * len(keys) for name in normalized}
    counts = {name: 0 for name in normalized}
    assignment: dict[str, str] = {}
    for group in keys:
        split = max(normalized, key=lambda name: (targets[name] - counts[name], name))
        assignment[group] = split
        counts[split] += 1
    result = {name: [] for name in normalized}
    for group, group_rows in groups.items():
        result[assignment[group]].extend(group_rows)
    return result
