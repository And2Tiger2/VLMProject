from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vlm_eval.qwen3_high_bias_roi_attention import (
    DEFAULT_ROI_ROOT,
    DEFAULT_VLMBIAS,
    prepare_splits,
)
from vlm_eval.qwen3_roi_attention import base_condition, slug


EXPERIMENT_VERSION = "qwen3_vlmbias_roi_context_factorial_v1"
DEFAULT_EXPERIMENT_ROOT = Path(
    "segments/vlm_bias_attention/experiments/qwen3_roi_context_factorial_v1"
)
DEFAULT_RUN_ROOT = Path(
    "segments/vlm_bias_attention/runs/qwen3_roi_context_factorial_v1"
)
DEFAULT_REPORT_ROOT = Path(
    "segments/vlm_bias_attention/reports/qwen3_roi_context_factorial_v1"
)
DEFAULT_STRENGTH = 5.0


def factorial_conditions(strength: float = DEFAULT_STRENGTH) -> list[dict[str, Any]]:
    if strength <= 0:
        raise ValueError("strength must be positive")
    strength_slug = slug(strength)
    definitions = (
        ("baseline", False, False),
        (f"roi_suppress_alpha{strength_slug}", True, False),
        (f"context_boost_alpha{strength_slug}", False, True),
        (f"roi_suppress_context_boost_alpha{strength_slug}", True, True),
    )
    conditions = []
    for name, suppress_roi, boost_context in definitions:
        row = base_condition(
            name,
            region="roi",
            alpha=0.0,
            heads=50,
            selection="gaze_global",
        )
        row.update(
            {
                "mask_variant": "tight",
                "suppress_roi": suppress_roi,
                "boost_context": boost_context,
                "roi_attention_bias": -float(strength) if suppress_roi else 0.0,
                "context_attention_bias": (float(strength) if boost_context else 0.0),
            }
        )
        conditions.append(row)
    return conditions


def prepare_experiment(
    *,
    vlmbias_path: Path = DEFAULT_VLMBIAS,
    roi_root: Path = DEFAULT_ROI_ROOT,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    strength: float = DEFAULT_STRENGTH,
) -> dict[str, Any]:
    splits = prepare_splits(
        vlmbias_path=vlmbias_path,
        roi_root=roi_root,
        out_dir=experiment_root,
    )
    dataset = splits["paths"]["all"]
    conditions = [
        {**row, "split": "all_exploratory", "dataset": dataset}
        for row in factorial_conditions(strength)
    ]
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": "factorial",
        "split": "all_exploratory",
        "n_rows": splits["counts"]["all"],
        "n_visual_groups": splits["group_counts"]["all"],
        "conditions": conditions,
    }
    experiment_root.mkdir(parents=True, exist_ok=True)
    (experiment_root / "factorial_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (experiment_root / "methodology.json").write_text(
        json.dumps(methodology(strength), indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def methodology(strength: float = DEFAULT_STRENGTH) -> dict[str, Any]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "status": "exploratory follow-up; all 114 rows were used after inspecting the prior development and confirmation results",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "dataset": "114 Game Boards and Optical Illusion rows in 79 canonical visual groups",
        "design": "2x2 factorial: tight-ROI suppression off/on by complementary-image-context boosting off/on",
        "conditions": [row["name"] for row in factorial_conditions(strength)],
        "heads": "fixed seed-42 gaze-ranked global top 50 language-attention heads",
        "roi": "reviewed tight binary mask mapped to Qwen visual tokens at >=5% token coverage",
        "context": "all image tokens outside the tight ROI; text tokens are never boosted",
        "intervention": (
            f"Before softmax, add {-float(strength):g} to ROI key logits when suppression is on "
            f"and +{float(strength):g} to complementary image-key logits when context boosting is on."
        ),
        "generation": "deterministic; intervention applies to all prefill and decoding queries",
        "metrics": "accuracy, bias-aligned fraction/count, conditional bias-aligned error rate, invalid rate, per-topic metrics, ROI/context token fractions, and ROI/context attention mass",
        "interpretation": "Because this hypothesis was chosen after inspecting earlier results, use it to choose a direction; a later fresh split or dataset is required for confirmatory inference.",
    }
