from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_completed_keys(path: Path, fields: Iterable[str]) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    field_list = list(fields)
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed.add(tuple(row.get(field) for field in field_list))
    return completed


def row_key(row: dict[str, Any], fields: Iterable[str]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in fields)
