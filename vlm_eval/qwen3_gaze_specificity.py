from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vlm_eval.qwen3_attention_methods import (
    confirmation_conditions,
    prepare_stratified_splits,
    write_condition_manifest,
)


EXPERIMENT_VERSION = "qwen3_gaze_specificity_v2"
DEFAULT_EXPERIMENT_ROOT = Path(
    "segments/gaze_heads_qwen3_8b/experiments/gaze_specificity_v2"
)
DEFAULT_RUN_ROOT = Path("segments/gaze_heads_qwen3_8b/runs/gaze_specificity_v2")
DEFAULT_REPORT_ROOT = Path("segments/gaze_heads_qwen3_8b/reports/gaze_specificity_v2")
SOURCE_REPORT_ROOT = Path("segments/gaze_heads_qwen3_8b/reports/attention_methods_v1")
STAGE_SPLITS = {
    "repair": "confirm",
    "controls": "confirm",
    "tune": "dev",
    "final": "confirm",
    "robustness": "all",
}
STAGE_COUNTS = {
    "repair": 4,
    "controls": 33,
    "tune": 21,
    "final": 5,
}


def methodology() -> dict[str, Any]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "pre_registered_stage_order": [
            "repair",
            "controls",
            "tune",
            "final",
            "robustness",
        ],
        "data_policy": {
            "development": "100 VLMBias rows and 25 NaturalBench groups",
            "held_out": "300 disjoint VLMBias rows and 75 disjoint NaturalBench groups",
            "robustness": "all 400 VLMBias rows and all 100 NaturalBench groups",
            "split_seed": 2026,
            "routing_invariant": (
                "The stage owns dataset routing. Condition specs inherited from "
                "development reports may never override the stage split."
            ),
        },
        "repair": (
            "Rerun the previously locked baseline, fixed, target-mass, and "
            "confidence-gated conditions on the same held-out split."
        ),
        "controls": {
            "primary_question": (
                "Does fixed alpha=0.5 on the globally top-50 gaze heads outperform "
                "the distribution of equally sized layer-matched random head sets?"
            ),
            "conditions": {
                "baseline": 1,
                "gaze_top50": 1,
                "layer_matched_random_top50": 20,
                "paper_random_layers_20_35_top50": 10,
                "layer_matched_low_gaze_score_top50": 1,
            },
            "primary_control_family": "layer_matched_random",
            "empirical_test": (
                "One-sided add-one empirical p-values across 20 independent head "
                "draws; minimum attainable p is 1/21."
            ),
            "paired_uncertainty": (
                "10,000 paired cluster-bootstrap replicates, using VLMBias example "
                "and NaturalBench group as the resampling units."
            ),
        },
        "tune": {
            "head_set": "globally top-50 gaze heads, locked before this sweep",
            "fixed_alpha": [0.25, 0.5, 1.0, 2.0, 5.0],
            "target_mass": [0.3, 0.4, 0.5],
            "confidence_threshold": [0.7, 0.8, 0.85, 0.9],
            "confidence_gate_alpha": [0.5, 1.0, 2.0],
            "selection": (
                "Within each family, require VLMBias accuracy, NaturalBench G_Acc, "
                "and invalid-rate guardrails; then minimize bias-aligned fraction, "
                "maximize accuracy, minimize invalid rate, and maximize NaturalBench."
            ),
        },
        "final": (
            "Held-out baseline, the v1 alpha=0.5 anchor, and one development-selected "
            "candidate from each controller family. No held-out metric enters selection."
        ),
        "robustness": (
            "The same five final conditions on all data with temperature 0.7 at "
            "seeds 0, 1, and 2. This is secondary because it reuses development data."
        ),
        "primary_endpoints": [
            "VLMBias bias_aligned_fraction (lower is better)",
            "VLMBias accuracy (higher is better)",
            "NaturalBench G_Acc and Acc (higher is better)",
            "invalid generation rate (lower is better)",
        ],
    }


def repair_conditions(
    controller_selection: dict[str, Any],
    head_selection: dict[str, Any],
) -> list[dict[str, Any]]:
    return confirmation_conditions(controller_selection, head_selection)


def control_conditions() -> list[dict[str, Any]]:
    conditions = [
        _condition("baseline", controller="fixed", alpha=0.0, head_count=0),
        _condition("gaze_top50_alpha0p5", controller="fixed", alpha=0.5),
    ]
    conditions.extend(
        _condition(
            f"layer_matched_random50_seed{seed}",
            controller="fixed",
            alpha=0.5,
            head_selection="layer_matched_random",
            head_seed=seed,
        )
        for seed in range(100, 120)
    )
    conditions.extend(
        _condition(
            f"paper_random50_seed{seed}",
            controller="fixed",
            alpha=0.5,
            head_selection="paper_random",
            head_seed=seed,
        )
        for seed in range(100, 110)
    )
    conditions.append(
        _condition(
            "layer_matched_low50",
            controller="fixed",
            alpha=0.5,
            head_selection="layer_matched_low",
        )
    )
    return conditions


def tune_conditions() -> list[dict[str, Any]]:
    conditions = [_condition("baseline", controller="fixed", alpha=0.0, head_count=0)]
    conditions.extend(
        _condition(
            f"fixed_alpha{_slug(alpha)}",
            controller="fixed",
            alpha=alpha,
        )
        for alpha in (0.25, 0.5, 1.0, 2.0, 5.0)
    )
    conditions.extend(
        _condition(
            f"target_mass{_slug(target)}",
            controller="target_mass",
            target_mass=target,
        )
        for target in (0.3, 0.4, 0.5)
    )
    conditions.extend(
        _condition(
            f"confidence_gate{_slug(threshold)}_alpha{_slug(alpha)}",
            controller="confidence_gate",
            alpha=alpha,
            confidence_threshold=threshold,
        )
        for threshold in (0.7, 0.8, 0.85, 0.9)
        for alpha in (0.5, 1.0, 2.0)
    )
    return conditions


def final_conditions(tune_selection: dict[str, Any]) -> list[dict[str, Any]]:
    conditions = [
        _condition("baseline", controller="fixed", alpha=0.0, head_count=0),
        _condition(
            "anchor_gaze_top50_fixed_alpha0p5",
            controller="fixed",
            alpha=0.5,
        ),
    ]
    for family in ("fixed", "target_mass", "confidence_gate"):
        source = tune_selection["selected_by_family"][family]["spec"]
        conditions.append(
            {
                **_core_spec(source),
                "name": f"final_{family}_{source['name']}",
                "source_tune_condition": source["name"],
            }
        )
    return conditions


def robustness_conditions(
    tune_selection: dict[str, Any], seeds: list[int]
) -> list[dict[str, Any]]:
    return [
        {
            **condition,
            "name": f"{condition['name']}_seed{seed}",
            "seed": seed,
            "do_sample": True,
            "temperature": 0.7,
        }
        for seed in seeds
        for condition in final_conditions(tune_selection)
    ]


def prepare_stage(
    *,
    stage: str,
    vlmbias_path: Path,
    naturalbench_path: Path,
    experiment_root: Path,
    source_report_root: Path,
    report_root: Path,
    seeds: list[int],
) -> dict[str, Any]:
    if stage not in STAGE_SPLITS:
        raise ValueError(f"unknown stage {stage!r}")
    split_manifest = prepare_stratified_splits(
        vlmbias_path=vlmbias_path,
        naturalbench_path=naturalbench_path,
        out_dir=experiment_root,
    )
    controller_selection = None
    head_selection = None
    tune_selection = None
    if stage == "repair":
        controller_selection = _read_json(
            source_report_root / "controller" / "selection.json"
        )
        head_selection = _read_json(source_report_root / "heads" / "selection.json")
        conditions = repair_conditions(controller_selection, head_selection)
    elif stage == "controls":
        conditions = control_conditions()
    elif stage == "tune":
        conditions = tune_conditions()
    else:
        tune_selection = _read_json(report_root / "tune" / "selection.json")
        conditions = (
            final_conditions(tune_selection)
            if stage == "final"
            else robustness_conditions(tune_selection, seeds)
        )
    manifest = write_condition_manifest(
        stage=stage,
        conditions=conditions,
        split_manifest=split_manifest,
        out_path=experiment_root / f"{stage}_manifest.json",
        split=STAGE_SPLITS[stage],
    )
    manifest["experiment_version"] = EXPERIMENT_VERSION
    manifest["methodology"] = str(experiment_root / "methodology.json")
    (experiment_root / f"{stage}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _condition(
    name: str,
    *,
    controller: str,
    alpha: float = 0.0,
    target_mass: float = 0.0,
    confidence_threshold: float | None = None,
    head_selection: str = "gaze_global",
    head_count: int = 50,
    head_seed: int = 0,
) -> dict[str, Any]:
    return {
        "name": name,
        "controller": controller,
        "alpha": float(alpha),
        "target_mass": float(target_mass),
        "max_alpha": 5.0,
        "confidence_threshold": confidence_threshold,
        "head_selection": head_selection,
        "head_count": int(head_count),
        "head_seed": int(head_seed),
    }


def _core_spec(source: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "controller",
        "alpha",
        "target_mass",
        "max_alpha",
        "confidence_threshold",
        "head_selection",
        "head_count",
        "head_seed",
    }
    return {key: source[key] for key in keys}


def _slug(value: float) -> str:
    return str(value).replace(".", "p")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
