#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vlm_eval.qwen3_roi_attention import (
    DEFAULT_REPORT_ROOT,
    DEFAULT_RUN_ROOT,
    read_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate one Qwen3 ROI-attention stage."
    )
    parser.add_argument("stage", choices=["smoke", "tune", "heads", "confirm"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()
    result = aggregate_stage(
        stage=args.stage,
        manifest_path=args.manifest,
        run_root=args.run_root,
        report_root=args.report_root,
    )
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


def aggregate_stage(
    *, stage: str, manifest_path: Path, run_root: Path, report_root: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    rows = []
    if manifest.get("stage") != stage:
        errors.append(f"manifest stage {manifest.get('stage')!r} != {stage!r}")
    for condition in manifest.get("conditions", []):
        path = run_root / stage / condition["name"] / "summary.json"
        if not path.is_file():
            errors.append(f"missing summary: {path}")
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append(summary)
        errors.extend(_validate_summary(summary, condition))
    report: dict[str, Any] = {
        "valid": not errors,
        "stage": stage,
        "manifest": str(manifest_path),
        "n_conditions_expected": len(manifest.get("conditions", [])),
        "n_conditions_found": len(rows),
        "conditions": rows,
        "errors": errors,
        "warnings": warnings,
    }
    if not errors and stage in {"tune", "heads"}:
        selection = _select(stage, rows)
        report["selection"] = selection
        warnings.extend(selection["warnings"])
    elif not errors and stage == "confirm":
        report["comparison"] = _compare_confirm(rows)
    elif not errors:
        report["mechanics_gate"] = {
            "message": "All ROI, full-image, and random-head smoke paths completed with valid token masks.",
            "conditions": [row["condition"]["name"] for row in rows],
        }
    stage_dir = report_root / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "aggregate_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (stage_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    if report.get("selection") is not None:
        (stage_dir / "selection.json").write_text(
            json.dumps(report["selection"], indent=2) + "\n", encoding="utf-8"
        )
    return report


def _validate_summary(summary: dict[str, Any], condition: dict[str, Any]) -> list[str]:
    errors = []
    name = condition["name"]
    if summary.get("condition") != condition:
        errors.append(f"{name}: summary condition differs from manifest")
    expected = len(read_jsonl(Path(condition["dataset"])))
    metrics = summary.get("vlmbias", {})
    if metrics.get("n") != expected or metrics.get("n_unique") != expected:
        errors.append(f"{name}: expected {expected} unique predictions")
    if metrics.get("duplicate_count") != 0:
        errors.append(f"{name}: duplicate predictions")
    selected = summary.get("selected_heads", [])
    if len({tuple(head) for head in selected}) != int(condition["head_count"]):
        errors.append(f"{name}: wrong selected-head count")
    tokens = summary.get("token_mask_telemetry", {})
    if tokens.get("n_nonempty_target_masks") != expected:
        errors.append(f"{name}: empty or missing target token masks")
    fraction = tokens.get("mean_target_token_fraction")
    if condition["region"] == "full_image" and fraction != 1.0:
        errors.append(f"{name}: full-image condition does not target every image token")
    if condition["region"] != "full_image" and (
        fraction is None or not 0.0 < fraction < 1.0
    ):
        errors.append(f"{name}: spatial condition has invalid target-token fraction")
    if int(condition["head_count"]) > 0:
        attention = summary.get("attention_telemetry", {})
        if attention.get("mean_boosted_head_image_attention_mass") is None:
            errors.append(f"{name}: missing attention telemetry")
    return errors


def _select(stage: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(row for row in rows if row["condition"]["name"] == "baseline")
    evaluated = []
    for row in rows:
        condition = row["condition"]
        accuracy_drop = baseline["vlmbias"]["accuracy"] - row["vlmbias"]["accuracy"]
        invalid_increase = (
            row["vlmbias"]["invalid_rate"] - baseline["vlmbias"]["invalid_rate"]
        )
        qualified = accuracy_drop <= 0.02 + 1e-12 and invalid_increase <= 0.03 + 1e-12
        evaluated.append(
            {
                "name": condition["name"],
                "spec": condition,
                "qualified": qualified,
                "guardrail_deltas": {
                    "accuracy_drop": accuracy_drop,
                    "invalid_rate_increase": invalid_increase,
                },
                "metrics": _metrics(row),
            }
        )
    if stage == "tune":
        eligible = [
            item
            for item in evaluated
            if item["spec"]["region"] == "roi"
            and item["spec"]["head_selection"] == "gaze_global"
            and item["spec"]["head_count"] == 50
        ]
    else:
        eligible = [
            item
            for item in evaluated
            if item["spec"]["region"] == "roi"
            and str(item["spec"]["head_selection"]).startswith("gaze_")
        ]
    qualified = [item for item in eligible if item["qualified"]]
    pool = qualified or eligible
    selected = min(pool, key=_selection_key)
    warnings = (
        []
        if qualified
        else [
            "No eligible condition met both guardrails; selected the best eligible condition for diagnostic continuation."
        ]
    )
    return {
        "valid": True,
        "stage": stage,
        "selection_policy": "Maximize VLMBias accuracy, then minimize bias-aligned fraction and invalid rate. Random/low-score heads and non-ROI regions are controls only.",
        "baseline": {"spec": baseline["condition"], "metrics": _metrics(baseline)},
        "selected": selected,
        "evaluated": evaluated,
        "warnings": warnings,
    }


def _selection_key(item: dict[str, Any]) -> tuple[float, float, float, str]:
    metrics = item["metrics"]
    return (
        -metrics["accuracy"],
        metrics["bias_aligned_fraction"],
        metrics["invalid_rate"],
        item["name"],
    )


def _metrics(row: dict[str, Any]) -> dict[str, Any]:
    result = row["vlmbias"]
    return {
        "n": result["n"],
        "accuracy": result["accuracy"],
        "bias_aligned_fraction": result["bias_aligned_fraction"],
        "bias_aligned_error_rate": result["bias_aligned_error_rate"],
        "invalid_rate": result["invalid_rate"],
        "mean_target_token_fraction": row["token_mask_telemetry"][
            "mean_target_token_fraction"
        ],
        "mean_target_attention_mass": row["attention_telemetry"][
            "mean_image_attention_mass"
        ],
        "mean_boosted_head_target_attention_mass": row["attention_telemetry"][
            "mean_boosted_head_image_attention_mass"
        ],
    }


def _compare_confirm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(row for row in rows if row["condition"]["name"] == "baseline")
    comparisons = {}
    for row in rows:
        if row is baseline:
            continue
        comparisons[row["condition"]["name"]] = {
            "condition": row["condition"],
            "metrics": _metrics(row),
            "delta_vs_baseline": {
                "accuracy": row["vlmbias"]["accuracy"]
                - baseline["vlmbias"]["accuracy"],
                "bias_aligned_fraction": row["vlmbias"]["bias_aligned_fraction"]
                - baseline["vlmbias"]["bias_aligned_fraction"],
                "invalid_rate": row["vlmbias"]["invalid_rate"]
                - baseline["vlmbias"]["invalid_rate"],
            },
        }
    return {
        "primary_split": "held-out ROI groups (no shared edited image with development)",
        "baseline": _metrics(baseline),
        "conditions": comparisons,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Qwen3 VLMBias ROI attention: {report['stage']}",
        "",
        f"- Valid: `{str(report['valid']).lower()}`",
        f"- Conditions: {report['n_conditions_found']}/{report['n_conditions_expected']}",
        f"- Errors: {len(report['errors'])}",
        "",
        "| Condition | Region | Heads | Alpha | Accuracy | Bias-aligned | Invalid | Target token frac. | Target attention |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["conditions"]:
        metrics = _metrics(row)
        condition = row["condition"]
        lines.append(
            "| {name} | {region} | {heads} | {alpha:g} | {accuracy:.4f} | {bias:.4f} | {invalid:.4f} | {tokens:.4f} | {attention:.4f} |".format(
                name=condition["name"],
                region=condition["region"],
                heads=condition["head_count"],
                alpha=condition["alpha"],
                accuracy=metrics["accuracy"],
                bias=metrics["bias_aligned_fraction"],
                invalid=metrics["invalid_rate"],
                tokens=metrics["mean_target_token_fraction"],
                attention=metrics["mean_target_attention_mass"],
            )
        )
    if report.get("selection"):
        lines.extend(
            [
                "",
                "## Locked selection",
                "",
                f"`{report['selection']['selected']['name']}`",
            ]
        )
    if report["errors"]:
        lines.extend(
            ["", "## Errors", ""] + [f"- {error}" for error in report["errors"]]
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
