from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from vlm_eval.gaze_judge import aggregate_dynamic_judgments, aggregate_judgments, spearman_rho


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-style summary tables for Qwen2.5-VL GazeHeads runs.")
    parser.add_argument("--segment-root", default="segments/gaze_heads_qwen25")
    parser.add_argument("--run-suffix", default="", help="Optional suffix such as _smoke or _100_25.")
    parser.add_argument("--out-dir", default="", help="Defaults to <segment-root>/reports.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.segment_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(root, run_suffix=args.run_suffix, seed=args.seed)
    suffix_label = args.run_suffix or ""
    json_path = out_dir / f"gaze_results_report{suffix_label}.json"
    tsv_path = out_dir / f"gaze_results_summary{suffix_label}.tsv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_summary_tsv(report, tsv_path)
    print(f"Wrote report to {json_path}")
    print(f"Wrote summary table to {tsv_path}")


def build_report(root: Path, *, run_suffix: str = "", seed: int = 42) -> dict[str, Any]:
    runs = root / "runs"
    discovery_dir = runs / f"gaze_discovery{run_suffix}"
    trajectory_path = runs / f"narration_trajectory{run_suffix}" / "trajectories.jsonl"
    static_path = runs / f"static_narration{run_suffix}" / "judgments.jsonl"
    vqa_path = runs / f"vqa_steering{run_suffix}" / "judgments.jsonl"
    dynamic_path = runs / f"dynamic_narration{run_suffix}" / "judgments.jsonl"

    report = {
        "segment_root": str(root),
        "run_suffix": run_suffix,
        "discovery": _discovery_summary(discovery_dir),
        "trajectory": _trajectory_summary(_read_jsonl(trajectory_path)),
        "static": _judgment_summary(_read_jsonl(static_path), seed=seed),
        "vqa": _judgment_summary(_read_jsonl(vqa_path), seed=seed),
        "dynamic": _dynamic_summary(_read_jsonl(dynamic_path), seed=seed),
    }
    report["stage_status"] = {
        key: _stage_has_rows(value)
        for key, value in report.items()
        if key not in {"segment_root", "run_suffix", "discovery"}
    }
    return report


def _discovery_summary(discovery_dir: Path) -> dict[str, Any]:
    summary_path = discovery_dir / "summary.json"
    ranking_path = discovery_dir / "gaze_head_ranking.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    ranking = json.loads(ranking_path.read_text(encoding="utf-8")) if ranking_path.exists() else []
    return {
        "path": str(discovery_dir),
        "exists": summary_path.exists() and ranking_path.exists(),
        "valid_samples": summary.get("valid_samples"),
        "n_layers": summary.get("n_layers"),
        "n_heads": summary.get("n_heads"),
        "top_head": summary.get("top_head") or (ranking[0] if ranking else None),
        "top_10_heads": ranking[:10],
    }


def _trajectory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row.get("condition", "")), []).append(row)

    conditions = {}
    for condition, condition_rows in sorted(by_condition.items()):
        strip_summaries = [_summarize_single_trajectory(row) for row in condition_rows]
        rhos = [item["step_panel_spearman"] for item in strip_summaries if item["n_decode_steps"] >= 3]
        conditions[condition] = {
            "n_rows": len(condition_rows),
            "mean_decode_steps": _mean([item["n_decode_steps"] for item in strip_summaries]),
            "mean_step_panel_spearman": _mean(rhos),
            "strip_summaries": strip_summaries,
        }
    return {"n_rows": len(rows), "conditions": conditions}


def _summarize_single_trajectory(row: dict[str, Any]) -> dict[str, Any]:
    steps = (row.get("trajectory") or {}).get("steps") or []
    dominant_panels = []
    for step in steps:
        panel_masses = step.get("panel_masses") or {}
        if not panel_masses:
            continue
        panel_name = max(panel_masses.items(), key=lambda item: float(item[1]))[0]
        dominant_panels.append(_panel_number(panel_name))
    valid_panels = [panel for panel in dominant_panels if panel is not None]
    step_panel_spearman = spearman_rho(list(range(1, len(valid_panels) + 1)), valid_panels) if len(valid_panels) >= 2 else 0.0
    return {
        "strip_name": row.get("strip_name"),
        "condition": row.get("condition"),
        "n_decode_steps": len(steps),
        "dominant_panels": valid_panels,
        "step_panel_spearman": step_panel_spearman,
    }


def _judgment_summary(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    aggregate = aggregate_judgments(rows, seed=seed) if rows else {}
    return {
        "n_rows": len(rows),
        "conditions": aggregate,
    }


def _dynamic_summary(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    aggregate = aggregate_dynamic_judgments(rows, seed=seed) if rows else {}
    return {
        "n_rows": len(rows),
        "conditions": aggregate,
    }


def _write_summary_tsv(report: dict[str, Any], path: Path) -> None:
    rows = []
    for stage in ["trajectory", "static", "vqa", "dynamic"]:
        stage_summary = report.get(stage) or {}
        for condition, payload in (stage_summary.get("conditions") or {}).items():
            row = {"stage": stage, "condition": condition}
            if stage == "trajectory":
                row.update(
                    {
                        "n": payload.get("n_rows", 0),
                        "metric": "mean_step_panel_spearman",
                        "value": payload.get("mean_step_panel_spearman", 0.0),
                        "ci_low": "",
                        "ci_high": "",
                    }
                )
            elif stage == "dynamic":
                acc = payload.get("per_segment_accuracy", {})
                row.update(
                    {
                        "n": acc.get("n", 0),
                        "metric": "per_segment_accuracy",
                        "value": acc.get("accuracy", 0.0),
                        "ci_low": acc.get("ci_low", 0.0),
                        "ci_high": acc.get("ci_high", 0.0),
                        "spearman_rho_mean": payload.get("spearman_rho_mean", 0.0),
                    }
                )
            else:
                acc = (payload.get("overall") or {})
                row.update(
                    {
                        "n": acc.get("n", 0),
                        "metric": "target_panel_accuracy",
                        "value": acc.get("accuracy", 0.0),
                        "ci_low": acc.get("ci_low", 0.0),
                        "ci_high": acc.get("ci_high", 0.0),
                        "junk_count": payload.get("junk_count", 0),
                        "baseline_match_count": payload.get("baseline_match_count", 0),
                    }
                )
            rows.append(row)

    fieldnames = sorted({key for row in rows for key in row}) if rows else ["stage", "condition", "metric", "value"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _stage_has_rows(value: Any) -> bool:
    return bool(isinstance(value, dict) and int(value.get("n_rows", 0)) > 0)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _panel_number(name: str) -> int | None:
    if name.startswith("panel_"):
        try:
            return int(name.removeprefix("panel_"))
        except ValueError:
            return None
    return None


def _mean(values: list[float | int]) -> float:
    return float(np.mean(values)) if values else 0.0


if __name__ == "__main__":
    main()
