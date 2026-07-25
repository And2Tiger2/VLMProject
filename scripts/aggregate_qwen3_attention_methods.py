#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from vlm_eval.qwen3_attention_methods import (
    DEFAULT_REPORT_ROOT,
    DEFAULT_RUN_ROOT,
    QUALIFICATION,
    read_jsonl,
)

EXPECTED_STAGE_SPLIT = {
    "smoke": "smoke",
    "controller": "dev",
    "heads": "dev",
    "confirm": "confirm",
    "robustness": "all",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, aggregate, and lock one Qwen3 attention-method stage."
    )
    parser.add_argument(
        "stage", choices=["smoke", "controller", "heads", "confirm", "robustness"]
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()

    report = aggregate_stage(
        stage=args.stage,
        manifest_path=args.manifest,
        run_root=args.run_root,
        report_root=args.report_root,
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
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("stage") != stage:
        raise ValueError(
            f"manifest stage is {manifest.get('stage')!r}, expected {stage!r}"
        )
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    errors.extend(_validate_manifest_routing(manifest, stage))
    for condition in manifest["conditions"]:
        summary_path = run_root / stage / condition["name"] / "summary.json"
        if not summary_path.exists():
            errors.append(f"missing summary: {summary_path}")
            continue
        summary = _read_json(summary_path)
        rows.append(summary)
        errors.extend(_validate_summary(summary, condition))

    report: dict[str, Any] = {
        "valid": not errors,
        "stage": stage,
        "manifest": str(manifest_path),
        "qualification": QUALIFICATION,
        "n_conditions_expected": len(manifest["conditions"]),
        "n_conditions_found": len(rows),
        "conditions": rows,
        "errors": errors,
        "warnings": warnings,
    }
    if not errors:
        if stage in {"controller", "heads"}:
            selection = _select_development_setting(
                stage=stage,
                rows=rows,
                report_root=report_root,
            )
            report["selection"] = selection
            report["valid"] = bool(selection["valid"])
            report["errors"].extend(selection["errors"])
            report["warnings"].extend(selection["warnings"])
        elif stage == "confirm":
            report["comparison"] = _confirmation_comparison(rows)
        elif stage == "robustness":
            report["robustness"] = _robustness_summary(rows)
        else:
            report["mechanics_gate"] = {
                "conditions": [row["condition"]["name"] for row in rows],
                "message": (
                    "All four controller paths produced complete, unique rows "
                    "with their required telemetry."
                ),
            }

    stage_dir = report_root / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    report_path = stage_dir / "aggregate_results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (stage_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    if report.get("selection") is not None:
        (stage_dir / "selection.json").write_text(
            json.dumps(report["selection"], indent=2), encoding="utf-8"
        )
    return report


def _validate_manifest_routing(manifest: dict[str, Any], stage: str) -> list[str]:
    errors: list[str] = []
    expected_split = EXPECTED_STAGE_SPLIT[stage]
    split_manifest_path = Path(manifest["split_manifest"])
    split_manifest = _read_json(split_manifest_path)
    expected_vlmbias = split_manifest["paths"][f"{expected_split}_vlmbias"]
    expected_naturalbench = split_manifest["paths"][f"{expected_split}_naturalbench"]
    for condition in manifest.get("conditions", []):
        name = condition.get("name", "<unnamed>")
        if condition.get("split") != expected_split:
            errors.append(
                f"{name}: stage {stage} requires split {expected_split!r}, "
                f"found {condition.get('split')!r}"
            )
        if condition.get("vlmbias_dataset") != expected_vlmbias:
            errors.append(
                f"{name}: stage {stage} does not use the authoritative "
                f"{expected_split} VLMBias dataset"
            )
        if condition.get("naturalbench_dataset") != expected_naturalbench:
            errors.append(
                f"{name}: stage {stage} does not use the authoritative "
                f"{expected_split} NaturalBench dataset"
            )
    return errors


def _validate_summary(summary: dict[str, Any], condition: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    name = condition["name"]
    if not summary.get("valid"):
        errors.append(f"{name}: summary is not valid")
    if summary.get("condition") != condition:
        errors.append(f"{name}: condition does not exactly match its manifest")
    expected_vlmbias = len(read_jsonl(Path(condition["vlmbias_dataset"])))
    expected_groups = len(read_jsonl(Path(condition["naturalbench_dataset"])))
    if summary.get("vlmbias", {}).get("n") != expected_vlmbias:
        errors.append(
            f"{name}: expected {expected_vlmbias} VLMBias rows, found "
            f"{summary.get('vlmbias', {}).get('n')}"
        )
    if summary.get("vlmbias", {}).get("n_unique") != expected_vlmbias:
        errors.append(
            f"{name}: VLMBias unique-row count is "
            f"{summary.get('vlmbias', {}).get('n_unique')}, expected {expected_vlmbias}"
        )
    if summary.get("vlmbias", {}).get("duplicate_count") != 0:
        errors.append(f"{name}: VLMBias contains duplicate rows")
    naturalbench = summary.get("naturalbench", {})
    if naturalbench.get("n_groups") != expected_groups:
        errors.append(
            f"{name}: expected {expected_groups} NaturalBench groups, found "
            f"{naturalbench.get('n_groups')}"
        )
    if naturalbench.get("n_model_calls") != 4 * expected_groups:
        errors.append(
            f"{name}: expected {4 * expected_groups} NaturalBench calls, found "
            f"{naturalbench.get('n_model_calls')}"
        )
    if naturalbench.get("n_unique_calls") != 4 * expected_groups:
        errors.append(
            f"{name}: NaturalBench unique-call count is "
            f"{naturalbench.get('n_unique_calls')}, expected {4 * expected_groups}"
        )
    if naturalbench.get("duplicate_count") != 0:
        errors.append(f"{name}: NaturalBench contains duplicate calls")
    selected = summary.get("selected_heads", [])
    expected_heads = int(condition["head_count"])
    if (
        len(selected) != expected_heads
        or len({tuple(head) for head in selected}) != expected_heads
    ):
        errors.append(
            f"{name}: expected {expected_heads} unique selected heads, found "
            f"{len({tuple(head) for head in selected})}"
        )
    if expected_heads:
        for dataset in ("vlmbias", "naturalbench"):
            telemetry = summary.get("telemetry", {}).get(dataset, {})
            if telemetry.get("mean_boosted_head_image_attention_mass") is None:
                errors.append(f"{name}: missing {dataset} attention telemetry")
        if condition["controller"] == "target_mass":
            for dataset in ("vlmbias", "naturalbench"):
                telemetry = summary.get("telemetry", {}).get(dataset, {})
                if telemetry.get("mean_effective_alpha") is None:
                    errors.append(
                        f"{name}: missing {dataset} target-mass alpha telemetry"
                    )
        if condition["controller"] == "confidence_gate":
            for dataset in ("vlmbias", "naturalbench"):
                telemetry = summary.get("telemetry", {}).get(dataset, {})
                if telemetry.get("confidence_gate_intervention_rate") is None:
                    errors.append(
                        f"{name}: missing {dataset} confidence-gate telemetry"
                    )
    return errors


def _select_development_setting(
    *,
    stage: str,
    rows: list[dict[str, Any]],
    report_root: Path,
) -> dict[str, Any]:
    if stage == "controller":
        baseline = next(
            (row for row in rows if row["condition"]["name"] == "baseline"), None
        )
    else:
        controller_report = _read_json(
            report_root / "controller" / "aggregate_results.json"
        )
        baseline = next(
            row
            for row in controller_report["conditions"]
            if row["condition"]["name"] == "baseline"
        )
    if baseline is None:
        raise ValueError("development aggregation requires a baseline")

    evaluated = []
    for row in rows:
        qualification = _qualification(row, baseline)
        evaluated.append(
            {
                "name": row["condition"]["name"],
                "spec": row["condition"],
                "qualified": qualification["qualified"],
                "qualification": qualification,
                "objective": _objective(row),
                "metrics": _compact_metrics(row),
            }
        )

    warnings: list[str] = []
    errors: list[str] = []
    if stage == "controller":
        nonbaseline = [item for item in evaluated if item["spec"]["name"] != "baseline"]
        qualified = [item for item in nonbaseline if item["qualified"]]
        selected_overall = min(qualified, key=_selection_key) if qualified else None
        selected_by_family = {}
        for family in ("fixed", "target_mass", "confidence_gate"):
            candidates = [
                item
                for item in nonbaseline
                if _controller_family(item["spec"]) == family
            ]
            selected_by_family[family] = min(candidates, key=_selection_key)
            if not selected_by_family[family]["qualified"]:
                warnings.append(
                    f"best {family} development condition did not meet every guardrail"
                )
        if selected_overall is None:
            errors.append(
                "no non-baseline controller met the development-set guardrails; "
                "downstream head and confirmation stages are intentionally blocked"
            )
        selection = {
            "valid": selected_overall is not None,
            "stage": stage,
            "selection_policy": (
                "Among guardrail-qualified non-baseline controllers, minimize "
                "VLMBias bias_aligned_fraction, then maximize VLMBias accuracy, "
                "NaturalBench G_Acc, and NaturalBench Acc."
            ),
            "baseline": _selection_item(baseline),
            "selected_overall": selected_overall,
            "selected_by_family": selected_by_family,
            "evaluated": evaluated,
            "errors": errors,
            "warnings": warnings,
        }
    else:
        eligible_gaze = [
            item
            for item in evaluated
            if item["qualified"]
            and str(item["spec"]["head_selection"]).startswith("gaze_")
        ]
        selected = min(eligible_gaze, key=_selection_key) if eligible_gaze else None
        if selected is None:
            errors.append(
                "no gaze-ranked head configuration met the development-set "
                "guardrails; confirmation is intentionally blocked"
            )
        controls = [
            item
            for item in evaluated
            if not str(item["spec"]["head_selection"]).startswith("gaze_")
        ]
        selection = {
            "valid": selected is not None,
            "stage": stage,
            "selection_policy": (
                "Only gaze-ranked global or layer-band configurations are eligible "
                "to be locked. Random and low-score sets are mechanistic controls. "
                "Within qualified gaze settings, use the controller-stage objective."
            ),
            "baseline": _selection_item(baseline),
            "selected_head": selected,
            "controls": controls,
            "evaluated": evaluated,
            "errors": errors,
            "warnings": warnings,
        }
    return selection


def _qualification(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    values = {
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
        "vlmbias_accuracy": values["vlmbias_accuracy_drop"]
        <= QUALIFICATION["max_vlmbias_accuracy_drop"] + 1e-12,
        "naturalbench_g_acc": values["naturalbench_g_acc_drop"]
        <= QUALIFICATION["max_naturalbench_g_acc_drop"] + 1e-12,
        "vlmbias_invalid_rate": values["vlmbias_invalid_rate_increase"]
        <= QUALIFICATION["max_invalid_rate_increase"] + 1e-12,
        "naturalbench_invalid_rate": values["naturalbench_invalid_rate_increase"]
        <= QUALIFICATION["max_invalid_rate_increase"] + 1e-12,
    }
    return {"qualified": all(checks.values()), "checks": checks, "deltas": values}


def _objective(row: dict[str, Any]) -> dict[str, float]:
    return {
        "vlmbias_bias_aligned_fraction": row["vlmbias"]["bias_aligned_fraction"],
        "vlmbias_accuracy": row["vlmbias"]["accuracy"],
        "naturalbench_G_Acc": row["naturalbench"]["G_Acc"],
        "naturalbench_Acc": row["naturalbench"]["Acc"],
    }


def _selection_key(item: dict[str, Any]) -> tuple[float, float, float, float, str]:
    metrics = item["objective"]
    return (
        metrics["vlmbias_bias_aligned_fraction"],
        -metrics["vlmbias_accuracy"],
        -metrics["naturalbench_G_Acc"],
        -metrics["naturalbench_Acc"],
        item["name"],
    )


def _selection_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row["condition"]["name"],
        "spec": row["condition"],
        "metrics": _compact_metrics(row),
    }


def _compact_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
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
            for key in ("n_groups", "n_model_calls", "Acc", "G_Acc", "invalid_rate")
        },
        "telemetry": row.get("telemetry", {}),
    }


def _controller_family(spec: dict[str, Any]) -> str:
    if spec["controller"] == "fixed":
        return "fixed"
    return str(spec["controller"])


def _confirmation_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(row for row in rows if row["condition"]["name"] == "baseline")
    comparisons = {}
    for row in rows:
        if row is baseline:
            continue
        if row["vlmbias"]["n"] != baseline["vlmbias"]["n"]:
            raise ValueError(
                "held-out confirmation conditions have unequal VLMBias row counts"
            )
        if row["naturalbench"]["n_groups"] != baseline["naturalbench"]["n_groups"]:
            raise ValueError(
                "held-out confirmation conditions have unequal NaturalBench group counts"
            )
        comparisons[row["condition"]["name"]] = {
            "metrics": _compact_metrics(row),
            "delta_vs_baseline": {
                "vlmbias_accuracy": row["vlmbias"]["accuracy"]
                - baseline["vlmbias"]["accuracy"],
                "vlmbias_bias_aligned_fraction": row["vlmbias"]["bias_aligned_fraction"]
                - baseline["vlmbias"]["bias_aligned_fraction"],
                "naturalbench_Acc": row["naturalbench"]["Acc"]
                - baseline["naturalbench"]["Acc"],
                "naturalbench_G_Acc": row["naturalbench"]["G_Acc"]
                - baseline["naturalbench"]["G_Acc"],
            },
            "passes_guardrails": _qualification(row, baseline)["qualified"],
        }
    return {
        "primary_inference_split": "held-out confirm split",
        "baseline": _selection_item(baseline),
        "conditions": comparisons,
    }


def _robustness_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_robustness_base_name(row["condition"])].append(row)
    output = {}
    for name, group in sorted(grouped.items()):
        output[name] = {
            "n_seeds": len(group),
            "seeds": sorted(int(row["condition"]["seed"]) for row in group),
            "vlmbias_accuracy": _mean_std(
                [row["vlmbias"]["accuracy"] for row in group]
            ),
            "vlmbias_bias_aligned_fraction": _mean_std(
                [row["vlmbias"]["bias_aligned_fraction"] for row in group]
            ),
            "naturalbench_Acc": _mean_std(
                [row["naturalbench"]["Acc"] for row in group]
            ),
            "naturalbench_G_Acc": _mean_std(
                [row["naturalbench"]["G_Acc"] for row in group]
            ),
        }
    return {
        "interpretation": (
            "Multi-seed all-data robustness is secondary because it reuses the "
            "development data. Held-out deterministic confirmation remains primary."
        ),
        "groups": output,
    }


def _robustness_base_name(condition: dict[str, Any]) -> str:
    name = str(condition["name"])
    marker = f"_seed{condition['seed']}"
    return name[: -len(marker)] if name.endswith(marker) else name


def _mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Qwen3 attention methods: {report['stage']}",
        "",
        f"- Valid: `{str(report['valid']).lower()}`",
        f"- Conditions: {report['n_conditions_found']}/{report['n_conditions_expected']}",
        f"- Errors: {len(report['errors'])}",
        f"- Warnings: {len(report['warnings'])}",
        "",
        "## Condition metrics",
        "",
        "| Condition | VLMBias acc | Bias-aligned fraction | NaturalBench Acc | NaturalBench G_Acc |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["conditions"]:
        lines.append(
            "| {name} | {va:.4f} | {vb:.4f} | {na:.4f} | {ng:.4f} |".format(
                name=row["condition"]["name"],
                va=row["vlmbias"]["accuracy"],
                vb=row["vlmbias"]["bias_aligned_fraction"],
                na=row["naturalbench"]["Acc"],
                ng=row["naturalbench"]["G_Acc"],
            )
        )
    if report.get("selection"):
        selection = report["selection"]
        chosen = selection.get("selected_overall") or selection.get("selected_head")
        lines.extend(
            [
                "",
                "## Locked selection",
                "",
                (
                    f"`{chosen['name']}`"
                    if chosen
                    else "No setting passed the preregistered guardrails."
                ),
            ]
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
