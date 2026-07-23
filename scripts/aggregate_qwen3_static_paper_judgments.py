from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate full Qwen3 static paper-control judgments across control seeds."
    )
    parser.add_argument(
        "--segment-root",
        type=Path,
        default=Path("segments/gaze_heads_qwen3_8b"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--n-comics", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "segments/gaze_heads_qwen3_8b/reports/static_paper_kimi_full"
        ),
    )
    args = parser.parse_args()

    result = aggregate_judgment_runs(
        segment_root=args.segment_root,
        seeds=args.seeds,
        top_k=args.top_k,
        n_comics=args.n_comics,
        n_bootstrap=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


def aggregate_judgment_runs(
    *,
    segment_root: Path,
    seeds: list[int],
    top_k: int,
    n_comics: int,
    n_bootstrap: int,
    bootstrap_seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    gaze_condition = f"gaze_top{top_k}"
    control_condition = f"non_gaze_paper_{top_k}"
    expected_per_condition = n_comics * 6
    errors: list[str] = []
    warnings: list[str] = []
    rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    configs_by_seed: dict[int, dict[str, Any]] = {}
    source_paths: dict[int, str] = {}
    for seed in seeds:
        path = (
            segment_root
            / "runs"
            / f"static_paper_replication_seed{seed}_top{top_k}_merged_0_{n_comics}"
            / "kimi_judge"
            / "judgments.jsonl"
        )
        run_dir = path.parents[1]
        source_paths[seed] = str(path)
        try:
            rows = _read_jsonl(path)
            config = json.loads(
                (path.parent / "judgment_config.json").read_text(encoding="utf-8")
            )
            run_config = json.loads(
                (run_dir / "experiment_config.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read seed {seed} judgments: {exc}")
            continue
        if config.get("judgment_schema_version") != 6:
            errors.append(
                f"seed {seed} uses judgment schema "
                f"{config.get('judgment_schema_version')!r}; expected 6"
            )
        if config.get("label_panels") is not True:
            errors.append(f"seed {seed} was not judged with numbered panel images")
        rows_by_seed[seed] = rows
        configs_by_seed[seed] = run_config
        keys = [_full_key(row) for row in rows]
        if len(keys) != len(set(keys)):
            errors.append(f"seed {seed} contains duplicate judgment keys")
        counts = {
            condition: sum(str(row.get("condition")) == condition for row in rows)
            for condition in (gaze_condition, control_condition)
        }
        unexpected = sorted(
            {
                str(row.get("condition"))
                for row in rows
                if str(row.get("condition"))
                not in {gaze_condition, control_condition}
            }
        )
        if counts != {
            gaze_condition: expected_per_condition,
            control_condition: expected_per_condition,
        }:
            errors.append(
                f"seed {seed} condition counts are {counts}; expected "
                f"{expected_per_condition} each"
            )
        if unexpected:
            errors.append(f"seed {seed} has unexpected conditions: {unexpected}")
        parse_failures = sum(
            bool((row.get("judgment") or {}).get("parse_failed")) for row in rows
        )
        if parse_failures:
            errors.append(f"seed {seed} has {parse_failures} parse failures")

    if len(rows_by_seed) != len(seeds):
        errors.append(
            f"loaded {len(rows_by_seed)} judgment runs; expected {len(seeds)}"
        )
    if errors:
        return _write_result(
            out_dir,
            {
                "valid": False,
                "stage": "static_paper_kimi_aggregate",
                "seeds": seeds,
                "source_paths": source_paths,
                "errors": errors,
                "warnings": warnings,
            },
        )

    reference_seed = seeds[0]
    reference_gaze = _condition_map(
        rows_by_seed[reference_seed], gaze_condition
    )
    repeat_diagnostics: dict[str, Any] = {}
    for seed in seeds[1:]:
        candidate = _condition_map(rows_by_seed[seed], gaze_condition)
        if set(candidate) != set(reference_gaze):
            errors.append(f"seed {seed} repeated gaze keys differ from seed {reference_seed}")
            continue
        generation_disagreements = [
            key
            for key in reference_gaze
            if _generation_signature(reference_gaze[key])
            != _generation_signature(candidate[key])
        ]
        if generation_disagreements:
            errors.append(
                f"seed {seed} has {len(generation_disagreements)} repeated gaze generation "
                f"disagreements; examples: {generation_disagreements[:5]}"
            )
        judgment_disagreements = [
            key
            for key in reference_gaze
            if _judge_signature(reference_gaze[key])
            != _judge_signature(candidate[key])
        ]
        correctness_disagreements = [
            key
            for key in reference_gaze
            if bool((reference_gaze[key].get("judgment") or {}).get("correct"))
            != bool((candidate[key].get("judgment") or {}).get("correct"))
        ]
        repeat_diagnostics[str(seed)] = {
            "compared_with_seed": reference_seed,
            "n_rows": len(reference_gaze),
            "generation_disagreements": len(generation_disagreements),
            "judge_label_disagreements": len(judgment_disagreements),
            "judge_label_agreement": 1.0
            - len(judgment_disagreements) / len(reference_gaze),
            "correctness_disagreements": len(correctness_disagreements),
            "correctness_agreement": 1.0
            - len(correctness_disagreements) / len(reference_gaze),
            "examples": judgment_disagreements[:5],
        }
        if judgment_disagreements:
            warnings.append(
                f"seed {seed} Kimi labels disagree with seed {reference_seed} on "
                f"{len(judgment_disagreements)}/{len(reference_gaze)} repeated gaze rows; "
                "generation text is checked separately"
            )

    seed_results: dict[str, Any] = {}
    for index, seed in enumerate(seeds):
        gaze = _condition_map(rows_by_seed[seed], gaze_condition)
        control = _condition_map(rows_by_seed[seed], control_condition)
        if set(gaze) != set(control):
            errors.append(f"seed {seed} gaze/control judgment keys do not match")
            continue
        paired = paired_cluster_bootstrap(
            gaze,
            control,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed + index,
        )
        seed_results[str(seed)] = {
            **paired,
            "gaze": _condition_summary(gaze.values()),
            "control": _condition_summary(control.values()),
            "control_head_diagnostics": _control_head_diagnostics(
                configs_by_seed[seed]
            ),
        }

    gaze_accuracies = [
        float(seed_results[str(seed)]["gaze_accuracy"]) for seed in seeds
    ]
    control_accuracies = [
        float(seed_results[str(seed)]["control_accuracy"]) for seed in seeds
    ]
    deltas = [float(seed_results[str(seed)]["delta"]) for seed in seeds]
    gaze_accuracy = float(seed_results[str(reference_seed)]["gaze_accuracy"])
    aggregate = {
        "gaze_accuracy_reference_seed": gaze_accuracy,
        "gaze_accuracy_mean_across_judge_repeats": mean(gaze_accuracies),
        "gaze_accuracy_judge_repeat_range": [
            min(gaze_accuracies),
            max(gaze_accuracies),
        ],
        "gaze_cluster_ci": seed_results[str(reference_seed)]["gaze_ci"],
        "control_accuracy_mean_across_seeds": mean(control_accuracies),
        "control_accuracy_seed_std": (
            stdev(control_accuracies) if len(control_accuracies) > 1 else 0.0
        ),
        "control_accuracy_seed_range": [
            min(control_accuracies),
            max(control_accuracies),
        ],
        "delta_mean_across_control_seeds": mean(deltas),
        "delta_seed_std": stdev(deltas) if len(deltas) > 1 else 0.0,
        "delta_seed_standard_error": (
            stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
        ),
        "delta_seed_range": [min(deltas), max(deltas)],
        "delta_control_seed_t_interval": _seed_mean_t_interval(deltas),
        "control_score_accuracy_correlation": _safe_optional_correlation(
            [
                seed_results[str(seed)]["control_head_diagnostics"].get(
                    "mean_gaze_score"
                )
                for seed in seeds
            ],
            control_accuracies,
        ),
        "control_head_pairwise_overlap": _control_head_overlap(
            configs_by_seed, seeds
        ),
    }
    result = {
        "valid": not errors,
        "stage": "static_paper_kimi_aggregate",
        "judge": "moonshotai/Kimi-VL-A3B-Instruct",
        "numbered_judge_images": True,
        "seeds": seeds,
        "top_k": top_k,
        "n_comics": n_comics,
        "rows_per_seed": expected_per_condition * 2,
        "independent_gaze_rows": expected_per_condition,
        "control_rows_per_seed": expected_per_condition,
        "repeated_gaze_policy": (
            f"seed {reference_seed} is the reference; generation rows must match exactly, "
            "while repeated Kimi judgments quantify judge stability"
        ),
        "repeated_gaze_judge_diagnostics": repeat_diagnostics,
        "bootstrap_unit": "comic strip",
        "n_bootstrap": n_bootstrap,
        "source_paths": source_paths,
        "aggregate": aggregate,
        "per_seed": seed_results,
        "errors": errors,
        "warnings": warnings,
    }
    return _write_result(out_dir, result)


def paired_cluster_bootstrap(
    gaze: dict[tuple[str, int], dict[str, Any]],
    control: dict[tuple[str, int], dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    strips = sorted({strip for strip, _ in gaze})
    gaze_by_strip = []
    control_by_strip = []
    for strip in strips:
        gaze_values = [
            float(bool((gaze[(strip, panel)].get("judgment") or {}).get("correct")))
            for panel in range(1, 7)
        ]
        control_values = [
            float(
                bool((control[(strip, panel)].get("judgment") or {}).get("correct"))
            )
            for panel in range(1, 7)
        ]
        gaze_by_strip.append(mean(gaze_values))
        control_by_strip.append(mean(control_values))
    gaze_array = np.asarray(gaze_by_strip, dtype=np.float64)
    control_array = np.asarray(control_by_strip, dtype=np.float64)
    delta_array = gaze_array - control_array
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(strips), size=(n_bootstrap, len(strips)))

    def interval(values: np.ndarray) -> list[float]:
        means = values[indices].mean(axis=1)
        return [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ]

    return {
        "n_comics": len(strips),
        "gaze_accuracy": float(gaze_array.mean()),
        "gaze_ci": interval(gaze_array),
        "control_accuracy": float(control_array.mean()),
        "control_ci": interval(control_array),
        "delta": float(delta_array.mean()),
        "delta_ci": interval(delta_array),
    }


def _condition_summary(rows: Any) -> dict[str, Any]:
    rows = list(rows)
    outcomes = [
        bool((row.get("judgment") or {}).get("correct")) for row in rows
    ]
    per_panel = {
        str(panel): mean(
            [
                float(bool((row.get("judgment") or {}).get("correct")))
                for row in rows
                if int(row["target_panel"]) == panel
            ]
        )
        for panel in range(1, 7)
    }
    return {
        "accuracy": mean(float(value) for value in outcomes),
        "n": len(rows),
        "per_panel_accuracy": per_panel,
        "junk_count": sum(
            bool((row.get("judgment") or {}).get("is_junk")) for row in rows
        ),
        "baseline_match_count": sum(
            bool((row.get("judgment") or {}).get("matches_baseline"))
            for row in rows
        ),
        "empty_count": sum(
            not str(row.get("generated_text", "") or "").strip() for row in rows
        ),
    }


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


def _full_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["strip_name"]),
        str(row["condition"]),
        int(row["target_panel"]),
    )


def _generation_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("generated_text"),
        row.get("baseline_text"),
        row.get("prompt"),
    )


def _judge_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    judgment = row.get("judgment") or {}
    return (
        judgment.get("matched_panel"),
        bool(judgment.get("is_junk")),
        bool(judgment.get("correct")),
        bool(judgment.get("matches_baseline")),
        bool(judgment.get("parse_failed")),
        judgment.get("raw_judge_text"),
    )


def _control_head_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    heads = [
        (int(head[0]), int(head[1]))
        for head in config.get("selected_control_heads", [])
    ]
    ranking_path = Path(str(config.get("gaze_ranking", "")))
    scores_path = ranking_path.parent / "gaze_scores.npy"
    if not scores_path.exists():
        return {
            "available": False,
            "n_heads": len(heads),
            "scores_path": str(scores_path),
        }
    scores = np.load(scores_path)
    values = np.asarray(
        [float(scores[layer, head]) for layer, head in heads], dtype=np.float64
    )
    layers: dict[str, int] = {}
    for layer, _ in heads:
        layers[str(layer)] = layers.get(str(layer), 0) + 1
    return {
        "available": True,
        "n_heads": len(heads),
        "scores_path": str(scores_path),
        "mean_gaze_score": float(values.mean()),
        "median_gaze_score": float(np.median(values)),
        "p90_gaze_score": float(np.percentile(values, 90)),
        "max_gaze_score": float(values.max()),
        "layer_counts": dict(sorted(layers.items(), key=lambda item: int(item[0]))),
    }


def _control_head_overlap(
    configs: dict[int, dict[str, Any]], seeds: list[int]
) -> dict[str, int]:
    selected = {
        seed: {
            (int(head[0]), int(head[1]))
            for head in configs[seed].get("selected_control_heads", [])
        }
        for seed in seeds
    }
    return {
        f"{left}-{right}": len(selected[left] & selected[right])
        for index, left in enumerate(seeds)
        for right in seeds[index + 1 :]
    }


def _seed_mean_t_interval(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    critical_by_df = {
        1: 12.7062047364,
        2: 4.3026527297,
        3: 3.1824463053,
        4: 2.7764451052,
        5: 2.5705818356,
        6: 2.4469118488,
        7: 2.3646242510,
        8: 2.3060041352,
        9: 2.2621571629,
    }
    df = len(values) - 1
    critical = critical_by_df.get(df, 1.96)
    center = mean(values)
    half_width = critical * stdev(values) / math.sqrt(len(values))
    return [center - half_width, center + half_width]


def _safe_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _safe_optional_correlation(
    left: list[float | None], right: list[float]
) -> float | None:
    if any(value is None for value in left):
        return None
    return _safe_correlation([float(value) for value in left if value is not None], right)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_result(out_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "aggregate_results.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result.get("valid"):
        (out_dir / "report.md").write_text(_markdown_report(result), encoding="utf-8")
    return result


def _markdown_report(result: dict[str, Any]) -> str:
    aggregate = result["aggregate"]
    rows = []
    for seed, values in result["per_seed"].items():
        rows.append(
            f"| {seed} | {values['gaze_accuracy']:.3f} | "
            f"{values['control_accuracy']:.3f} | {values['delta']:+.3f} | "
            f"[{values['delta_ci'][0]:+.3f}, {values['delta_ci'][1]:+.3f}] |"
        )
    return (
        "# Qwen3-VL static Gaze Heads: corrected paper control\n\n"
        f"- Comics: {result['n_comics']}\n"
        f"- Gaze heads: top-{result['top_k']}\n"
        f"- Judge: {result['judge']} with visible panel labels\n"
        f"- Bootstrap unit: comic strip ({result['n_bootstrap']} replicates)\n"
        f"- Gaze accuracy (counted once): "
        f"{aggregate['gaze_accuracy_reference_seed']:.3f}\n"
        f"- Mean control accuracy across seeds: "
        f"{aggregate['control_accuracy_mean_across_seeds']:.3f}\n"
        f"- Mean gaze-minus-control effect: "
        f"{aggregate['delta_mean_across_control_seeds']:+.3f}\n\n"
        "| Control seed | Gaze accuracy | Control accuracy | Delta | "
        "95% paired comic-bootstrap CI |\n"
        "|---:|---:|---:|---:|:---|\n"
        + "\n".join(rows)
        + "\n"
    )


if __name__ == "__main__":
    main()
