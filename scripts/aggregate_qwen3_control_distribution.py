#!/usr/bin/env python3
"""Aggregate the Qwen3 static-steering control-distribution experiment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

from scripts.aggregate_qwen3_static_paper_judgments import (
    _condition_summary,
    _control_head_diagnostics,
    _seed_mean_t_interval,
    paired_cluster_bootstrap,
)


DEFAULT_SEGMENT_ROOT = Path("segments/gaze_heads_qwen3_8b")
DEFAULT_PAPER_SEEDS = list(range(45, 55))
DEFAULT_MATCHED_SEEDS = list(range(45, 55))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-root", type=Path, default=DEFAULT_SEGMENT_ROOT)
    parser.add_argument("--paper-seeds", type=int, nargs="+", default=DEFAULT_PAPER_SEEDS)
    parser.add_argument(
        "--matched-seeds", type=int, nargs="+", default=DEFAULT_MATCHED_SEEDS
    )
    parser.add_argument("--low-seed", type=int, default=42)
    parser.add_argument("--reference-seed", type=int, default=42)
    parser.add_argument("--reference-comics", type=int, default=500)
    parser.add_argument("--n-comics", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=4200)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_SEGMENT_ROOT / "reports" / "control_distribution_100",
    )
    args = parser.parse_args()
    result = aggregate_control_distribution(
        segment_root=args.segment_root,
        paper_seeds=args.paper_seeds,
        matched_seeds=args.matched_seeds,
        low_seed=args.low_seed,
        reference_seed=args.reference_seed,
        reference_comics=args.reference_comics,
        n_comics=args.n_comics,
        top_k=args.top_k,
        n_bootstrap=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


def aggregate_control_distribution(
    *,
    segment_root: Path,
    paper_seeds: list[int],
    matched_seeds: list[int],
    low_seed: int,
    reference_seed: int,
    reference_comics: int,
    n_comics: int,
    top_k: int,
    n_bootstrap: int,
    bootstrap_seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    gaze_condition = f"gaze_top{top_k}"
    expected_rows = n_comics * 6
    reference_path = (
        segment_root
        / "runs"
        / f"static_paper_replication_seed{reference_seed}_top{top_k}"
        f"_merged_0_{reference_comics}"
        / "kimi_judge"
        / "judgments.jsonl"
    )
    try:
        reference_rows = _read_jsonl(reference_path)
        reference_config = _read_json(reference_path.parent / "judgment_config.json")
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read reference gaze judgments: {exc}")
        return _write_result(
            out_dir,
            _failure_result(
                reference_path=reference_path,
                paper_seeds=paper_seeds,
                matched_seeds=matched_seeds,
                low_seed=low_seed,
                errors=errors,
            ),
        )
    _validate_judge_config(reference_config, "reference", errors)
    all_reference_gaze = _condition_map(reference_rows, gaze_condition)
    specs = [
        *(("paper", seed) for seed in paper_seeds),
        *(("layer_matched_random", seed) for seed in matched_seeds),
        ("layer_matched_low", low_seed),
    ]
    run_data: dict[tuple[str, int], dict[str, Any]] = {}
    reference_keys: set[tuple[str, int]] | None = None
    source_paths: dict[str, str] = {}
    for index, (mode, seed) in enumerate(specs):
        condition = f"non_gaze_{mode}_{top_k}"
        run_dir = (
            segment_root
            / "runs"
            / f"static_control_distribution_{mode}_seed{seed}_top{top_k}"
            f"_merged_0_{n_comics}"
        )
        judgments_path = run_dir / "kimi_judge" / "judgments.jsonl"
        source_paths[f"{mode}@{seed}"] = str(judgments_path)
        try:
            rows = _read_jsonl(judgments_path)
            judge_config = _read_json(judgments_path.parent / "judgment_config.json")
            experiment_config = _read_json(run_dir / "experiment_config.json")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read {mode}@{seed}: {exc}")
            continue
        _validate_judge_config(judge_config, f"{mode}@{seed}", errors)
        if len(rows) != expected_rows:
            errors.append(
                f"{mode}@{seed} has {len(rows)} judgments; expected {expected_rows}"
            )
        if any(str(row.get("condition")) != condition for row in rows):
            errors.append(f"{mode}@{seed} contains a condition other than {condition}")
        keys = [_pair_key(row) for row in rows]
        if len(keys) != len(set(keys)):
            errors.append(f"{mode}@{seed} contains duplicate judgment keys")
        parse_failures = sum(
            bool((row.get("judgment") or {}).get("parse_failed")) for row in rows
        )
        if parse_failures:
            errors.append(f"{mode}@{seed} has {parse_failures} parse failures")
        control = {_pair_key(row): row for row in rows}
        if reference_keys is None:
            reference_keys = set(control)
        elif set(control) != reference_keys:
            errors.append(f"{mode}@{seed} does not use the same comic-panel keys")
        selected_gaze = {
            (int(head[0]), int(head[1]))
            for head in experiment_config.get("selected_gaze_heads", [])
        }
        selected_control = {
            (int(head[0]), int(head[1]))
            for head in experiment_config.get("selected_control_heads", [])
        }
        if len(selected_gaze) != top_k or len(selected_control) != top_k:
            errors.append(f"{mode}@{seed} does not contain two exact top-{top_k} sets")
        if selected_gaze & selected_control:
            errors.append(f"{mode}@{seed} control overlaps the gaze set")
        run_data[(mode, seed)] = {
            "control": control,
            "config": experiment_config,
            "bootstrap_index": index,
        }

    if reference_keys is None:
        errors.append("no control judgments were loaded")
        reference_keys = set()
    gaze = {
        key: all_reference_gaze[key]
        for key in reference_keys
        if key in all_reference_gaze
    }
    if len(gaze) != expected_rows or set(gaze) != reference_keys:
        errors.append(
            f"reference gaze subset has {len(gaze)} rows; expected exactly {expected_rows}"
        )
    if errors:
        return _write_result(
            out_dir,
            _failure_result(
                reference_path=reference_path,
                paper_seeds=paper_seeds,
                matched_seeds=matched_seeds,
                low_seed=low_seed,
                errors=errors,
                source_paths=source_paths,
            ),
        )

    per_control: dict[str, Any] = {}
    for (mode, seed), values in run_data.items():
        paired = paired_cluster_bootstrap(
            gaze,
            values["control"],
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed + int(values["bootstrap_index"]),
        )
        per_control[f"{mode}@{seed}"] = {
            "mode": mode,
            "seed": seed,
            **paired,
            "control": _condition_summary(values["control"].values()),
            "control_head_diagnostics": _control_head_diagnostics(values["config"]),
        }

    paper = [per_control[f"paper@{seed}"] for seed in paper_seeds]
    matched = [
        per_control[f"layer_matched_random@{seed}"] for seed in matched_seeds
    ]
    low = per_control[f"layer_matched_low@{low_seed}"]
    gaze_accuracy = mean(
        float(bool((row.get("judgment") or {}).get("correct")))
        for row in gaze.values()
    )
    result = {
        "valid": True,
        "stage": "qwen3_control_distribution",
        "question": (
            "Are top gaze heads more effective than the distribution of equally sized "
            "random controls, after separating layer placement from gaze score?"
        ),
        "judge": "moonshotai/Kimi-VL-A3B-Instruct",
        "numbered_judge_images": True,
        "top_k": top_k,
        "n_comics": n_comics,
        "panels_per_comic": 6,
        "independent_gaze_rows": expected_rows,
        "control_rows_per_draw": expected_rows,
        "paper_seeds": paper_seeds,
        "matched_seeds": matched_seeds,
        "low_seed": low_seed,
        "reference_gaze": str(reference_path),
        "source_paths": source_paths,
        "bootstrap_unit": "comic strip",
        "n_bootstrap": n_bootstrap,
        "gaze": {
            "accuracy": gaze_accuracy,
            "summary": _condition_summary(gaze.values()),
            "counting_policy": "existing seed-42 gaze judgments are counted once",
        },
        "groups": {
            "paper_uniform_layers_20_35": _group_summary(paper, gaze_accuracy),
            "layer_matched_random": _group_summary(matched, gaze_accuracy),
            "layer_matched_low_score": {
                "n_draws": 1,
                "control_accuracy": low["control_accuracy"],
                "control_ci": low["control_ci"],
                "gaze_minus_control": low["delta"],
                "delta_ci": low["delta_ci"],
                "control_head_diagnostics": low["control_head_diagnostics"],
            },
        },
        "per_control": per_control,
        "errors": [],
        "warnings": warnings,
    }
    return _write_result(out_dir, result)


def _group_summary(rows: list[dict[str, Any]], gaze_accuracy: float) -> dict[str, Any]:
    accuracies = [float(row["control_accuracy"]) for row in rows]
    deltas = [float(row["delta"]) for row in rows]
    scores = [
        row["control_head_diagnostics"].get("mean_gaze_score") for row in rows
    ]
    less = sum(value < gaze_accuracy for value in accuracies)
    ties = sum(value == gaze_accuracy for value in accuracies)
    return {
        "n_draws": len(rows),
        "control_accuracy_mean": mean(accuracies),
        "control_accuracy_std": stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "control_accuracy_range": [min(accuracies), max(accuracies)],
        "gaze_empirical_percentile": (less + 0.5 * ties) / len(accuracies),
        "n_controls_at_or_above_gaze": sum(
            value >= gaze_accuracy for value in accuracies
        ),
        "delta_mean": mean(deltas),
        "delta_std": stdev(deltas) if len(deltas) > 1 else 0.0,
        "delta_range": [min(deltas), max(deltas)],
        "delta_draw_t_interval": _seed_mean_t_interval(deltas),
        "mean_control_score_accuracy_correlation": _safe_optional_correlation(
            scores, accuracies
        ),
    }


def _safe_optional_correlation(
    left: list[float | None], right: list[float]
) -> float | None:
    if len(left) < 2 or any(value is None for value in left):
        return None
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.std() == 0 or right_array.std() == 0:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _validate_judge_config(
    config: dict[str, Any], label: str, errors: list[str]
) -> None:
    if config.get("judgment_schema_version") != 6:
        errors.append(
            f"{label} uses judgment schema {config.get('judgment_schema_version')!r}; expected 6"
        )
    if config.get("label_panels") is not True:
        errors.append(f"{label} was not judged with numbered panel images")


def _condition_map(
    rows: list[dict[str, Any]], condition: str
) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        _pair_key(row): row
        for row in rows
        if str(row.get("condition")) == condition
    }


def _pair_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["strip_name"]), int(row["target_panel"])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _failure_result(
    *,
    reference_path: Path,
    paper_seeds: list[int],
    matched_seeds: list[int],
    low_seed: int,
    errors: list[str],
    source_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "valid": False,
        "stage": "qwen3_control_distribution",
        "paper_seeds": paper_seeds,
        "matched_seeds": matched_seeds,
        "low_seed": low_seed,
        "reference_gaze": str(reference_path),
        "source_paths": source_paths or {},
        "errors": errors,
        "warnings": [],
    }


def _write_result(out_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aggregate_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if result.get("valid"):
        (out_dir / "report.md").write_text(_markdown_report(result), encoding="utf-8")
    return result


def _markdown_report(result: dict[str, Any]) -> str:
    gaze = result["gaze"]["accuracy"]
    paper = result["groups"]["paper_uniform_layers_20_35"]
    matched = result["groups"]["layer_matched_random"]
    low = result["groups"]["layer_matched_low_score"]
    return (
        "# Qwen3-VL gaze-head control distribution\n\n"
        f"- Evaluation set: {result['n_comics']} comic strips × 6 panels\n"
        f"- Gaze accuracy (counted once): {gaze:.3f}\n"
        f"- Paper-uniform controls ({paper['n_draws']} draws): "
        f"{paper['control_accuracy_mean']:.3f} ± {paper['control_accuracy_std']:.3f}\n"
        f"- Layer-matched random controls ({matched['n_draws']} draws): "
        f"{matched['control_accuracy_mean']:.3f} ± "
        f"{matched['control_accuracy_std']:.3f}\n"
        f"- Layer-matched lowest-score control: {low['control_accuracy']:.3f}\n"
        f"- Paper-control draws at or above gaze: "
        f"{paper['n_controls_at_or_above_gaze']}/{paper['n_draws']}\n"
        f"- Layer-matched draws at or above gaze: "
        f"{matched['n_controls_at_or_above_gaze']}/{matched['n_draws']}\n"
    )


if __name__ == "__main__":
    main()
