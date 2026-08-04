#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vlm_eval.qwen3_roi_attention import read_jsonl
from vlm_eval.qwen3_roi_context_factorial import DEFAULT_REPORT_ROOT, DEFAULT_RUN_ROOT


OUTCOMES = (
    "accuracy",
    "bias_aligned_fraction",
    "bias_aligned_error_rate",
    "invalid_rate",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate the Qwen3 ROI/context factorial."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()
    report = aggregate(
        manifest_path=args.manifest,
        run_root=args.run_root,
        report_root=args.report_root,
    )
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(2)


def aggregate(
    *, manifest_path: Path, run_root: Path, report_root: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    conditions = manifest.get("conditions", [])
    if manifest.get("stage") != "factorial":
        errors.append("manifest is not the factorial stage")
    for condition in conditions:
        summary_path = run_root / "factorial" / condition["name"] / "summary.json"
        if not summary_path.is_file():
            errors.append(f"missing summary: {summary_path}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(summary)
        errors.extend(_validate(summary, condition))
    report: dict[str, Any] = {
        "valid": not errors,
        "stage": "factorial",
        "exploratory": True,
        "manifest": str(manifest_path),
        "n_conditions_expected": len(conditions),
        "n_conditions_found": len(rows),
        "conditions": rows,
        "errors": errors,
        "warnings": [
            "All 114 rows were reused after inspecting the earlier experiment; treat effect estimates as exploratory."
        ],
    }
    if not errors:
        report["comparisons"] = _comparisons(rows)
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "aggregate_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (report_root / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _validate(summary: dict[str, Any], condition: dict[str, Any]) -> list[str]:
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
    if len({tuple(head) for head in selected}) != 50:
        errors.append(f"{name}: expected exactly 50 unique gaze heads")
    tokens = summary.get("token_mask_telemetry", {})
    if tokens.get("n_nonempty_target_masks") != expected:
        errors.append(f"{name}: empty or missing ROI token masks")
    roi_fraction = tokens.get("mean_target_token_fraction")
    context_fraction = tokens.get("mean_context_token_fraction")
    if roi_fraction is None or not 0.0 < roi_fraction < 1.0:
        errors.append(f"{name}: invalid ROI token fraction")
    if context_fraction is None or not 0.0 < context_fraction < 1.0:
        errors.append(f"{name}: invalid context token fraction")
    elif roi_fraction is not None and abs(roi_fraction + context_fraction - 1.0) > 1e-6:
        errors.append(f"{name}: ROI and context token fractions are not complementary")
    telemetry = summary.get("attention_telemetry", {})
    for key in (
        "mean_boosted_head_image_attention_mass",
        "mean_boosted_head_context_attention_mass",
    ):
        if telemetry.get(key) is None:
            errors.append(f"{name}: missing {key}")
    return errors


def _metrics(row: dict[str, Any]) -> dict[str, Any]:
    result = row["vlmbias"]
    n = int(result["n"])
    correct = round(float(result["accuracy"]) * n)
    bias = int(
        result.get("n_bias_aligned_errors")
        or round(float(result["bias_aligned_fraction"]) * n)
    )
    invalid = round(float(result["invalid_rate"]) * n)
    return {
        "n": n,
        "n_correct": correct,
        "n_bias_aligned_errors": bias,
        "n_invalid": invalid,
        "n_other_errors": n - correct - bias - invalid,
        **{key: float(result[key]) for key in OUTCOMES},
        "by_topic": result.get("by_topic", {}),
        "mean_roi_token_fraction": row["token_mask_telemetry"][
            "mean_target_token_fraction"
        ],
        "mean_context_token_fraction": row["token_mask_telemetry"][
            "mean_context_token_fraction"
        ],
        "mean_selected_head_roi_attention": row["attention_telemetry"][
            "mean_boosted_head_image_attention_mass"
        ],
        "mean_selected_head_context_attention": row["attention_telemetry"][
            "mean_boosted_head_context_attention_mass"
        ],
    }


def _comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_factors = {
        (
            bool(row["condition"]["suppress_roi"]),
            bool(row["condition"]["boost_context"]),
        ): row
        for row in rows
    }
    expected = {(False, False), (True, False), (False, True), (True, True)}
    if set(by_factors) != expected:
        raise ValueError("factorial report does not contain one row for every 2x2 cell")
    metrics = {cell: _metrics(row) for cell, row in by_factors.items()}
    baseline = metrics[(False, False)]
    direct = {
        by_factors[cell]["condition"]["name"]: {
            key: metrics[cell][key] - baseline[key] for key in OUTCOMES
        }
        for cell in expected - {(False, False)}
    }
    effects = {}
    for outcome in OUTCOMES:
        y00 = metrics[(False, False)][outcome]
        y10 = metrics[(True, False)][outcome]
        y01 = metrics[(False, True)][outcome]
        y11 = metrics[(True, True)][outcome]
        effects[outcome] = {
            "roi_suppression_main_effect": ((y10 - y00) + (y11 - y01)) / 2,
            "context_boost_main_effect": ((y01 - y00) + (y11 - y10)) / 2,
            "interaction": y11 - y10 - y01 + y00,
        }
    return {
        "baseline": baseline,
        "delta_vs_baseline": direct,
        "factorial_effects": effects,
        "effect_signs": "For accuracy, positive is better. For bias-aligned and invalid rates, negative is better.",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen3 tight-ROI suppression × context boosting",
        "",
        f"- Valid: `{str(report['valid']).lower()}`",
        f"- Conditions: {report['n_conditions_found']}/{report['n_conditions_expected']}",
        "- Scope: exploratory, all 114 high-bias-topic rows",
        "- Heads: gaze top 50",
        "- Logit magnitude: 5 before softmax",
        "",
        "| Condition | ROI bias | Context bias | Accuracy | Bias-aligned | Bias/error | Invalid | ROI attn. | Context attn. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["conditions"]:
        condition = row["condition"]
        metrics = _metrics(row)
        lines.append(
            "| {name} | {roi:g} | {context:g} | {accuracy:.4f} | {bias:.4f} | {bias_error:.4f} | {invalid:.4f} | {roi_attn:.4f} | {context_attn:.4f} |".format(
                name=condition["name"],
                roi=condition["roi_attention_bias"],
                context=condition["context_attention_bias"],
                accuracy=metrics["accuracy"],
                bias=metrics["bias_aligned_fraction"],
                bias_error=metrics["bias_aligned_error_rate"],
                invalid=metrics["invalid_rate"],
                roi_attn=metrics["mean_selected_head_roi_attention"],
                context_attn=metrics["mean_selected_head_context_attention"],
            )
        )
    if any(row["vlmbias"].get("by_topic") for row in report["conditions"]):
        lines.extend(
            [
                "",
                "## Metrics by topic",
                "",
                "| Condition | Topic | N | Accuracy | Bias-aligned | Invalid |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in report["conditions"]:
            for topic, metrics in row["vlmbias"].get("by_topic", {}).items():
                lines.append(
                    f"| {row['condition']['name']} | {topic} | {metrics['n']} | {metrics['accuracy']:.4f} | {metrics.get('bias_aligned_fraction', 0.0):.4f} | {metrics.get('invalid_rate', 0.0):.4f} |"
                )
    if report.get("comparisons"):
        lines.extend(["", "## Factorial effects", ""])
        for outcome, effects in report["comparisons"]["factorial_effects"].items():
            lines.append(
                "- {outcome}: suppression {suppression:+.4f}; context {context:+.4f}; interaction {interaction:+.4f}".format(
                    outcome=outcome,
                    suppression=effects["roi_suppression_main_effect"],
                    context=effects["context_boost_main_effect"],
                    interaction=effects["interaction"],
                )
            )
    if report["warnings"]:
        lines.extend(
            ["", "## Warning", ""] + [f"- {item}" for item in report["warnings"]]
        )
    if report["errors"]:
        lines.extend(["", "## Errors", ""] + [f"- {item}" for item in report["errors"]])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
