from __future__ import annotations

import argparse
import json
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
    rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    source_paths: dict[int, str] = {}
    for seed in seeds:
        path = (
            segment_root
            / "runs"
            / f"static_paper_replication_seed{seed}_top{top_k}_merged_0_{n_comics}"
            / "kimi_judge"
            / "judgments.jsonl"
        )
        source_paths[seed] = str(path)
        try:
            rows = _read_jsonl(path)
            config = json.loads(
                (path.parent / "judgment_config.json").read_text(encoding="utf-8")
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
            },
        )

    reference_seed = seeds[0]
    reference_gaze = _condition_map(
        rows_by_seed[reference_seed], gaze_condition
    )
    for seed in seeds[1:]:
        candidate = _condition_map(rows_by_seed[seed], gaze_condition)
        if set(candidate) != set(reference_gaze):
            errors.append(f"seed {seed} repeated gaze keys differ from seed {reference_seed}")
            continue
        disagreements = [
            key
            for key in reference_gaze
            if _gaze_signature(reference_gaze[key])
            != _gaze_signature(candidate[key])
        ]
        if disagreements:
            errors.append(
                f"seed {seed} has {len(disagreements)} repeated gaze judgment "
                f"disagreements; examples: {disagreements[:5]}"
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
        }

    control_accuracies = [
        float(seed_results[str(seed)]["control_accuracy"]) for seed in seeds
    ]
    deltas = [float(seed_results[str(seed)]["delta"]) for seed in seeds]
    gaze_accuracy = float(seed_results[str(reference_seed)]["gaze_accuracy"])
    aggregate = {
        "gaze_accuracy_counted_once": gaze_accuracy,
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
        "delta_seed_range": [min(deltas), max(deltas)],
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
            f"seed {reference_seed} counted once; other gaze rows required to match exactly"
        ),
        "bootstrap_unit": "comic strip",
        "n_bootstrap": n_bootstrap,
        "source_paths": source_paths,
        "aggregate": aggregate,
        "per_seed": seed_results,
        "errors": errors,
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


def _gaze_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    judgment = row.get("judgment") or {}
    return (
        row.get("generated_text"),
        row.get("baseline_text"),
        judgment.get("matched_panel"),
        bool(judgment.get("is_junk")),
        bool(judgment.get("correct")),
        bool(judgment.get("matches_baseline")),
        bool(judgment.get("parse_failed")),
        judgment.get("raw_judge_text"),
    )


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
        f"{aggregate['gaze_accuracy_counted_once']:.3f}\n"
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
