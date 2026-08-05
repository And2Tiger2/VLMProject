from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


ATLAS_COLUMNS = (
    "layer",
    "head",
    "comic_gaze_score",
    "image_attention",
    "count_causal_score",
    "search_causal_score",
    "verification_causal_score",
    "distractor_suppression_score",
    "mmmc_signed_score",
    "vlmbias_semantic_prior_score",
    "vlmbias_context_score",
    "vlmbias_detail_score",
    "projected_output_norm",
    "correct_vs_bias_logit_attribution",
)


def initialize_head_atlas(n_layers: int, n_heads: int) -> list[dict[str, Any]]:
    return [
        {
            column: (layer if column == "layer" else head if column == "head" else None)
            for column in ATLAS_COLUMNS
        }
        for layer in range(n_layers)
        for head in range(n_heads)
    ]


def write_head_atlas(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ATLAS_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in ATLAS_COLUMNS})
