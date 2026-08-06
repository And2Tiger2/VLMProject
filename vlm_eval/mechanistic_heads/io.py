from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


def union_fieldnames(rows: Iterable[dict[str, Any]], *, fallback: str = "empty") -> list[str]:
    """Return a deterministic TSV schema covering every row.

    Scientific runners sometimes emit a compact exclusion row alongside normal
    result rows. Deriving the schema from only the first row makes the run fail
    at write time or silently lose later diagnostic fields.
    """

    fields = {str(key) for row in rows for key in row}
    return sorted(fields) if fields else [fallback]


def write_tsv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fallback: str = "empty",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = union_fieldnames(rows, fallback=fallback)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
