#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

from vlm_eval.qwen3_attention_methods import QUALIFICATION, read_jsonl
from vlm_eval.qwen3_gaze_specificity import (
    DEFAULT_REPORT_ROOT,
    DEFAULT_RUN_ROOT,
    STAGE_SPLITS,
)


METRICS = {
    "vlmbias_accuracy": ("vlmbias", "accuracy", True),
    "vlmbias_bias_aligned_fraction": (
        "vlmbias",
        "bias_aligned_fraction",
        False,
    ),
    "vlmbias_invalid_rate": ("vlmbias", "invalid_rate", False),
    "naturalbench_Acc": ("naturalbench", "Acc", True),
    "naturalbench_G_Acc": ("naturalbench", "G_Acc", True),
    "naturalbench_invalid_rate": ("naturalbench", "invalid_rate", False),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate and validate one Qwen3 gaze-specificity stage."
    )
    parser.add_argument(
        "stage", choices=["repair", "controls", "tune", "final", "robustness"]
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    report = aggregate_stage(
        stage=args.stage,
        manifest_path=args.manifest,
        run_root=args.run_root,
        report_root=args.report_root,
        n_bootstrap=args.n_bootstrap,
    )
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(2)


def aggregate_stage(
    *,
    stage: str,
    manifest_path: Path,
    run_root: Path,
    report_root: Path,
    n_bootstrap: int = 10_000,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    errors = _validate_manifest(manifest, stage)
    rows: list[dict[str, Any]] = []
    for condition in manifest.get("conditions", []):
        summary_path = run_root / stage / condition["name"] / "summary.json"
        if not summary_path.is_file():
            errors.append(f"missing summary: {summary_path}")
            continue
        summary = _read_json(summary_path)
        rows.append(summary)
        errors.extend(_validate_summary(summary, condition))
    report: dict[str, Any] = {
        "valid": not errors and len(rows) == len(manifest.get("conditions", [])),
        "stage": stage,
        "manifest": str(manifest_path),
        "split": STAGE_SPLITS[stage],
        "n_conditions_expected": len(manifest.get("conditions", [])),
        "n_conditions_found": len(rows),
        "conditions": [_compact(row) for row in rows],
        "errors": errors,
        "warnings": [],
    }
    if report["valid"]:
        if stage in {"repair", "final"}:
            report["paired_comparison"] = _paired_stage_comparison(
                rows, run_root / stage, n_bootstrap=n_bootstrap
            )
        elif stage == "controls":
            distribution, distribution_errors = _control_distribution(
                rows, run_root / stage, n_bootstrap=n_bootstrap
            )
            report["control_distribution"] = distribution
            report["errors"].extend(distribution_errors)
            report["valid"] = not report["errors"]
        elif stage == "tune":
            selection = _select_tune(rows)
            report["selection"] = selection
            report["errors"].extend(selection["errors"])
            report["warnings"].extend(selection["warnings"])
            report["valid"] = selection["valid"]
        else:
            report["robustness"] = _robustness(rows)

    stage_dir = report_root / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "aggregate_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (stage_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    if report.get("selection") is not None:
        (stage_dir / "selection.json").write_text(
            json.dumps(report["selection"], indent=2), encoding="utf-8"
        )
    return report


def _validate_manifest(manifest: dict[str, Any], stage: str) -> list[str]:
    errors: list[str] = []
    if manifest.get("stage") != stage:
        errors.append(
            f"manifest stage is {manifest.get('stage')!r}, expected {stage!r}"
        )
        return errors
    expected_split = STAGE_SPLITS[stage]
    split_manifest_path = Path(manifest["split_manifest"])
    if not split_manifest_path.is_file():
        return [f"missing split manifest: {split_manifest_path}"]
    split_manifest = _read_json(split_manifest_path)
    expected = {
        "vlmbias_dataset": split_manifest["paths"][f"{expected_split}_vlmbias"],
        "naturalbench_dataset": split_manifest["paths"][
            f"{expected_split}_naturalbench"
        ],
    }
    names: list[str] = []
    for condition in manifest.get("conditions", []):
        name = str(condition.get("name", "<unnamed>"))
        names.append(name)
        if condition.get("split") != expected_split:
            errors.append(
                f"{name}: expected stage split {expected_split!r}, found "
                f"{condition.get('split')!r}"
            )
        for key, path in expected.items():
            if condition.get(key) != path:
                errors.append(
                    f"{name}: {key} is not the authoritative {expected_split} path"
                )
    if len(names) != len(set(names)):
        errors.append("manifest contains duplicate condition names")
    return errors


def _validate_summary(summary: dict[str, Any], condition: dict[str, Any]) -> list[str]:
    name = condition["name"]
    errors: list[str] = []
    if not summary.get("valid"):
        errors.append(f"{name}: summary is not valid")
    if summary.get("condition") != condition:
        errors.append(f"{name}: summary condition does not exactly match manifest")
    expected_vlmbias = len(read_jsonl(Path(condition["vlmbias_dataset"])))
    expected_groups = len(read_jsonl(Path(condition["naturalbench_dataset"])))
    vlmbias = summary.get("vlmbias", {})
    naturalbench = summary.get("naturalbench", {})
    if vlmbias.get("n") != expected_vlmbias:
        errors.append(
            f"{name}: expected {expected_vlmbias} VLMBias rows, "
            f"found {vlmbias.get('n')}"
        )
    if vlmbias.get("n_unique") != expected_vlmbias:
        errors.append(f"{name}: VLMBias unique count is incomplete")
    if vlmbias.get("duplicate_count") != 0:
        errors.append(f"{name}: duplicate VLMBias rows")
    if naturalbench.get("n_groups") != expected_groups:
        errors.append(
            f"{name}: expected {expected_groups} NaturalBench groups, "
            f"found {naturalbench.get('n_groups')}"
        )
    if naturalbench.get("n_model_calls") != 4 * expected_groups:
        errors.append(f"{name}: NaturalBench call count is incomplete")
    if naturalbench.get("n_unique_calls") != 4 * expected_groups:
        errors.append(f"{name}: NaturalBench unique-call count is incomplete")
    if naturalbench.get("duplicate_count") != 0:
        errors.append(f"{name}: duplicate NaturalBench calls")
    heads = [tuple(head) for head in summary.get("selected_heads", [])]
    if len(heads) != int(condition["head_count"]) or len(set(heads)) != len(heads):
        errors.append(
            f"{name}: expected {condition['head_count']} unique heads, "
            f"found {len(set(heads))}"
        )
    if condition["head_count"]:
        for dataset in ("vlmbias", "naturalbench"):
            telemetry = summary.get("telemetry", {}).get(dataset, {})
            if telemetry.get("mean_boosted_head_image_attention_mass") is None:
                errors.append(f"{name}: missing {dataset} attention telemetry")
            if (
                condition["controller"] == "target_mass"
                and telemetry.get("mean_effective_alpha") is None
            ):
                errors.append(f"{name}: missing {dataset} target-mass telemetry")
            if (
                condition["controller"] == "confidence_gate"
                and telemetry.get("confidence_gate_intervention_rate") is None
            ):
                errors.append(f"{name}: missing {dataset} gate telemetry")
    return errors


def _paired_stage_comparison(
    rows: list[dict[str, Any]],
    stage_dir: Path,
    *,
    n_bootstrap: int,
) -> dict[str, Any]:
    baseline = next(row for row in rows if row["condition"]["name"] == "baseline")
    output: dict[str, Any] = {
        "baseline": _compact(baseline),
        "bootstrap_unit": {
            "vlmbias": "example",
            "naturalbench": "group",
        },
        "n_bootstrap": n_bootstrap,
        "conditions": {},
    }
    for index, row in enumerate(rows):
        if row is baseline:
            continue
        output["conditions"][row["condition"]["name"]] = {
            "metrics": _compact(row),
            "paired_delta_vs_baseline": _paired_deltas(
                reference_name=baseline["condition"]["name"],
                treatment_name=row["condition"]["name"],
                stage_dir=stage_dir,
                n_bootstrap=n_bootstrap,
                seed=7100 + index,
            ),
            "passes_guardrails": _qualification(row, baseline)["qualified"],
        }
    return output


def _control_distribution(
    rows: list[dict[str, Any]],
    stage_dir: Path,
    *,
    n_bootstrap: int,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    by_name = {row["condition"]["name"]: row for row in rows}
    baseline = by_name["baseline"]
    gaze = by_name["gaze_top50_alpha0p5"]
    layer_random = [
        row
        for row in rows
        if row["condition"]["name"].startswith("layer_matched_random50_")
    ]
    paper_random = [
        row for row in rows if row["condition"]["name"].startswith("paper_random50_")
    ]
    low = by_name["layer_matched_low50"]
    gaze_layers = Counter(head[0] for head in gaze["selected_heads"])
    gaze_head_set = {tuple(head) for head in gaze["selected_heads"]}
    layer_random_fingerprints = {
        tuple(sorted(tuple(head) for head in row["selected_heads"]))
        for row in layer_random
    }
    if len(layer_random_fingerprints) != len(layer_random):
        errors.append("layer-matched random draws contain duplicate head sets")
    paper_random_fingerprints = {
        tuple(sorted(tuple(head) for head in row["selected_heads"]))
        for row in paper_random
    }
    if len(paper_random_fingerprints) != len(paper_random):
        errors.append("paper-style random draws contain duplicate head sets")
    for row in [*layer_random, low]:
        layers = Counter(head[0] for head in row["selected_heads"])
        if layers != gaze_layers:
            errors.append(
                f"{row['condition']['name']}: head layers do not exactly match gaze_top50"
            )
        overlap = len(gaze_head_set & {tuple(head) for head in row["selected_heads"]})
        if overlap:
            errors.append(
                f"{row['condition']['name']}: layer-matched control overlaps "
                f"{overlap} gaze heads"
            )
    empirical: dict[str, Any] = {}
    for metric_name, (_, _, higher_is_better) in METRICS.items():
        gaze_value = _metric(gaze, metric_name)
        control_values = [_metric(row, metric_name) for row in layer_random]
        at_least_as_good = sum(
            value >= gaze_value if higher_is_better else value <= gaze_value
            for value in control_values
        )
        gaze_better = sum(
            gaze_value > value if higher_is_better else gaze_value < value
            for value in control_values
        )
        empirical[metric_name] = {
            "direction": "higher" if higher_is_better else "lower",
            "gaze": gaze_value,
            "control_mean": mean(control_values),
            "control_std": stdev(control_values),
            "control_range": [min(control_values), max(control_values)],
            "gaze_fraction_strictly_better": gaze_better / len(control_values),
            "n_controls_at_least_as_good_as_gaze": at_least_as_good,
            "one_sided_add_one_empirical_p": (1 + at_least_as_good)
            / (1 + len(control_values)),
        }
    comparisons = {}
    for index, row in enumerate([*layer_random, *paper_random, low]):
        name = row["condition"]["name"]
        comparisons[name] = {
            "family": _control_family(name),
            "metrics": _compact(row),
            "head_overlap_with_gaze": len(
                {tuple(head) for head in row["selected_heads"]} & gaze_head_set
            ),
            "paired_delta_gaze_minus_control": _paired_deltas(
                reference_name=name,
                treatment_name=gaze["condition"]["name"],
                stage_dir=stage_dir,
                n_bootstrap=n_bootstrap,
                seed=8200 + index,
            ),
        }
    output = {
        "question": (
            "Do globally top-50 gaze heads outperform the distribution of "
            "equally sized layer-matched random head sets?"
        ),
        "primary_control_family": "layer_matched_random",
        "n_layer_matched_random_draws": len(layer_random),
        "n_paper_random_draws": len(paper_random),
        "baseline": _compact(baseline),
        "gaze": {
            **_compact(gaze),
            "paired_delta_vs_baseline": _paired_deltas(
                reference_name="baseline",
                treatment_name="gaze_top50_alpha0p5",
                stage_dir=stage_dir,
                n_bootstrap=n_bootstrap,
                seed=8001,
            ),
            "layer_counts": dict(sorted(gaze_layers.items())),
        },
        "empirical_head_draw_tests": empirical,
        "primary_specificity_gate": {
            "gaze_passes_baseline_guardrails": _qualification(gaze, baseline)[
                "qualified"
            ],
            "gaze_reduces_bias_aligned_fraction_vs_baseline": (
                gaze["vlmbias"]["bias_aligned_fraction"]
                < baseline["vlmbias"]["bias_aligned_fraction"]
            ),
            "gaze_beats_layer_matched_null_at_p_le_0p05": (
                empirical["vlmbias_bias_aligned_fraction"][
                    "one_sided_add_one_empirical_p"
                ]
                <= 0.05
            ),
            "supports_gaze_specific_bias_mitigation": (
                _qualification(gaze, baseline)["qualified"]
                and gaze["vlmbias"]["bias_aligned_fraction"]
                < baseline["vlmbias"]["bias_aligned_fraction"]
                and empirical["vlmbias_bias_aligned_fraction"][
                    "one_sided_add_one_empirical_p"
                ]
                <= 0.05
            ),
        },
        "controls": comparisons,
        "multiple_testing_note": (
            "Endpoints are reported separately. The pre-specified primary head-draw "
            "comparison is VLMBias bias-aligned fraction, subject to accuracy, "
            "NaturalBench, and invalid-rate guardrails."
        ),
    }
    return output, errors


def _paired_deltas(
    *,
    reference_name: str,
    treatment_name: str,
    stage_dir: Path,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    reference_vlm = _keyed_vlmbias(stage_dir / reference_name / "vlmbias.jsonl")
    treatment_vlm = _keyed_vlmbias(stage_dir / treatment_name / "vlmbias.jsonl")
    reference_nb = _keyed_naturalbench(
        stage_dir / reference_name / "naturalbench.jsonl"
    )
    treatment_nb = _keyed_naturalbench(
        stage_dir / treatment_name / "naturalbench.jsonl"
    )
    if reference_vlm.keys() != treatment_vlm.keys():
        raise ValueError(
            f"VLMBias keys differ between {reference_name} and {treatment_name}"
        )
    if reference_nb.keys() != treatment_nb.keys():
        raise ValueError(
            f"NaturalBench keys differ between {reference_name} and {treatment_name}"
        )
    vlm_deltas = {
        "vlmbias_accuracy": [
            treatment_vlm[key]["accuracy"] - reference_vlm[key]["accuracy"]
            for key in sorted(reference_vlm)
        ],
        "vlmbias_bias_aligned_fraction": [
            treatment_vlm[key]["bias"] - reference_vlm[key]["bias"]
            for key in sorted(reference_vlm)
        ],
        "vlmbias_invalid_rate": [
            treatment_vlm[key]["invalid"] - reference_vlm[key]["invalid"]
            for key in sorted(reference_vlm)
        ],
    }
    nb_deltas = {
        metric: [
            treatment_nb[key][field] - reference_nb[key][field]
            for key in sorted(reference_nb)
        ]
        for metric, field in (
            ("naturalbench_Acc", "accuracy"),
            ("naturalbench_G_Acc", "g_accuracy"),
            ("naturalbench_invalid_rate", "invalid"),
        )
    }
    return {
        **_bootstrap_metrics(
            vlm_deltas,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        **_bootstrap_metrics(
            nb_deltas,
            n_bootstrap=n_bootstrap,
            seed=seed + 100_000,
        ),
    }


def _keyed_vlmbias(path: Path) -> dict[str, dict[str, float]]:
    output = {}
    for row in read_jsonl(path):
        output[str(row["example_id"])] = {
            "accuracy": float(bool(row.get("is_correct"))),
            "bias": float(bool(row.get("is_bias_aligned_error"))),
            "invalid": float(not bool(str(row.get("parsed_answer", "")).strip())),
        }
    return output


def _keyed_naturalbench(path: Path) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        groups[str(row["group_id"])].append(row)
    output = {}
    for group_id, group in groups.items():
        output[group_id] = {
            "accuracy": mean(bool(row.get("is_correct")) for row in group),
            "g_accuracy": float(
                len(group) == 4 and all(bool(row.get("is_correct")) for row in group)
            ),
            "invalid": mean(
                not bool(str(row.get("parsed_answer", "")).strip()) for row in group
            ),
        }
    return output


def _bootstrap_metrics(
    metrics: dict[str, list[float]], *, n_bootstrap: int, seed: int
) -> dict[str, Any]:
    if not metrics or any(not values for values in metrics.values()):
        raise ValueError("cannot bootstrap empty paired vectors")
    names = list(metrics)
    lengths = {len(metrics[name]) for name in names}
    if len(lengths) != 1:
        raise ValueError("paired metric vectors must have equal lengths")
    n = lengths.pop()
    matrix = np.asarray([metrics[name] for name in names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty((len(names), n_bootstrap), dtype=np.float64)
    chunk_size = 1000
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        indices = rng.integers(0, n, size=(stop - start, n))
        estimates[:, start:stop] = matrix[:, indices].mean(axis=2)
    return {
        name: {
            "delta": float(matrix[index].mean()),
            "ci_low": float(np.quantile(estimates[index], 0.025)),
            "ci_high": float(np.quantile(estimates[index], 0.975)),
            "n_clusters": n,
            "n_bootstrap": n_bootstrap,
        }
        for index, name in enumerate(names)
    }


def _select_tune(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(row for row in rows if row["condition"]["name"] == "baseline")
    selected = {}
    evaluated = []
    warnings = []
    errors = []
    for row in rows:
        if row is baseline:
            continue
        qualification = _qualification(row, baseline)
        item = {
            "name": row["condition"]["name"],
            "spec": row["condition"],
            "qualified": qualification["qualified"],
            "qualification": qualification,
            "metrics": _compact(row),
        }
        evaluated.append(item)
    for family in ("fixed", "target_mass", "confidence_gate"):
        candidates = [item for item in evaluated if _family(item["spec"]) == family]
        if not candidates:
            errors.append(f"no {family} condition was evaluated")
            continue
        qualified = [item for item in candidates if item["qualified"]]
        selected[family] = min(qualified if qualified else candidates, key=_tune_key)
        if not qualified:
            warnings.append(
                f"no {family} condition passed every development guardrail; "
                "the best family candidate is retained as a diagnostic"
            )
    return {
        "valid": not errors,
        "stage": "tune",
        "selection_policy": (
            "Within each controller family, require accuracy, NaturalBench G_Acc, "
            "and invalid-rate guardrails; minimize VLMBias bias-aligned fraction, "
            "then maximize VLMBias accuracy, minimize invalid rate, maximize "
            "NaturalBench G_Acc and Acc."
        ),
        "baseline": _compact(baseline),
        "selected_by_family": selected,
        "evaluated": evaluated,
        "errors": errors,
        "warnings": warnings,
    }


def _qualification(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    deltas = {
        "vlmbias_accuracy_drop": (
            baseline["vlmbias"]["accuracy"] - row["vlmbias"]["accuracy"]
        ),
        "naturalbench_g_acc_drop": (
            baseline["naturalbench"]["G_Acc"] - row["naturalbench"]["G_Acc"]
        ),
        "vlmbias_invalid_rate_increase": (
            row["vlmbias"]["invalid_rate"] - baseline["vlmbias"]["invalid_rate"]
        ),
        "naturalbench_invalid_rate_increase": (
            row["naturalbench"]["invalid_rate"]
            - baseline["naturalbench"]["invalid_rate"]
        ),
    }
    checks = {
        "vlmbias_accuracy": deltas["vlmbias_accuracy_drop"]
        <= QUALIFICATION["max_vlmbias_accuracy_drop"],
        "naturalbench_g_acc": deltas["naturalbench_g_acc_drop"]
        <= QUALIFICATION["max_naturalbench_g_acc_drop"],
        "vlmbias_invalid_rate": deltas["vlmbias_invalid_rate_increase"]
        <= QUALIFICATION["max_invalid_rate_increase"],
        "naturalbench_invalid_rate": deltas["naturalbench_invalid_rate_increase"]
        <= QUALIFICATION["max_invalid_rate_increase"],
    }
    return {"qualified": all(checks.values()), "checks": checks, "deltas": deltas}


def _tune_key(item: dict[str, Any]) -> tuple[float, ...]:
    metrics = item["metrics"]
    return (
        metrics["vlmbias"]["bias_aligned_fraction"],
        -metrics["vlmbias"]["accuracy"],
        metrics["vlmbias"]["invalid_rate"],
        -metrics["naturalbench"]["G_Acc"],
        -metrics["naturalbench"]["Acc"],
    )


def _robustness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    baselines: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row["condition"]["seed"])
        base_name = _base_name(row["condition"])
        grouped[base_name].append(row)
        if base_name == "baseline":
            baselines[seed] = row
    output = {}
    for name, group in sorted(grouped.items()):
        metrics = {}
        for metric_name in METRICS:
            values = [_metric(row, metric_name) for row in group]
            metrics[metric_name] = _mean_std(values)
        seed_deltas = {}
        if name != "baseline":
            for metric_name in METRICS:
                values = [
                    _metric(row, metric_name)
                    - _metric(baselines[int(row["condition"]["seed"])], metric_name)
                    for row in group
                ]
                seed_deltas[metric_name] = _mean_std(values)
        output[name] = {
            "n_seeds": len(group),
            "seeds": sorted(int(row["condition"]["seed"]) for row in group),
            "metrics": metrics,
            "delta_vs_same_seed_baseline": seed_deltas,
        }
    return {
        "interpretation": (
            "Secondary all-data sampling robustness. Held-out deterministic final "
            "and head-draw distribution tests remain primary."
        ),
        "groups": output,
    }


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row["condition"]["name"],
        "spec": row["condition"],
        "vlmbias": {
            key: row["vlmbias"][key]
            for key in (
                "n",
                "accuracy",
                "bias_aligned_fraction",
                "bias_aligned_error_rate",
                "invalid_rate",
            )
        },
        "naturalbench": {
            key: row["naturalbench"][key]
            for key in (
                "n_groups",
                "n_model_calls",
                "Acc",
                "G_Acc",
                "invalid_rate",
            )
        },
        "telemetry": row.get("telemetry", {}),
        "selected_heads": row.get("selected_heads", []),
    }


def _metric(row: dict[str, Any], name: str) -> float:
    section, key, _ = METRICS[name]
    return float(row[section][key])


def _family(spec: dict[str, Any]) -> str:
    return "fixed" if spec["controller"] == "fixed" else spec["controller"]


def _control_family(name: str) -> str:
    if name.startswith("layer_matched_random"):
        return "layer_matched_random"
    if name.startswith("paper_random"):
        return "paper_random"
    return "layer_matched_low"


def _base_name(condition: dict[str, Any]) -> str:
    name = str(condition["name"])
    marker = f"_seed{condition['seed']}"
    return name[: -len(marker)] if name.endswith(marker) else name


def _mean_std(values: list[float]) -> dict[str, Any]:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Qwen3 gaze specificity: {report['stage']}",
        "",
        f"- Valid: `{str(report['valid']).lower()}`",
        f"- Split: `{report['split']}`",
        (
            f"- Conditions: {report['n_conditions_found']}/"
            f"{report['n_conditions_expected']}"
        ),
        f"- Errors: {len(report['errors'])}",
        f"- Warnings: {len(report['warnings'])}",
        "",
        "## Condition metrics",
        "",
        (
            "| Condition | VLMBias acc | Bias fraction | VLMBias invalid | "
            "NaturalBench Acc | G_Acc | NB invalid |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["conditions"]:
        lines.append(
            "| {name} | {va:.4f} | {vb:.4f} | {vi:.4f} | "
            "{na:.4f} | {ng:.4f} | {ni:.4f} |".format(
                name=row["name"],
                va=row["vlmbias"]["accuracy"],
                vb=row["vlmbias"]["bias_aligned_fraction"],
                vi=row["vlmbias"]["invalid_rate"],
                na=row["naturalbench"]["Acc"],
                ng=row["naturalbench"]["G_Acc"],
                ni=row["naturalbench"]["invalid_rate"],
            )
        )
    if report.get("selection"):
        lines.extend(["", "## Development selections", ""])
        for family, item in report["selection"]["selected_by_family"].items():
            lines.append(f"- {family}: `{item['name']}`")
    if report.get("control_distribution"):
        tests = report["control_distribution"]["empirical_head_draw_tests"]
        lines.extend(["", "## Primary layer-matched head-draw tests", ""])
        for metric, result in tests.items():
            lines.append(
                f"- {metric}: gaze={result['gaze']:.4f}, "
                f"control mean={result['control_mean']:.4f}, "
                f"p={result['one_sided_add_one_empirical_p']:.4f}"
            )
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
