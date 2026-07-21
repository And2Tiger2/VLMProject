from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts.discover_qwen25_gaze_heads import gaze_routing_diagnostics


KEY_FIELDS = ("strip_name", "condition", "target_panel")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit static GazeHeads panel-sweep artifacts for invalid results.")
    parser.add_argument("--segment-root", default="segments/gaze_heads_qwen25")
    parser.add_argument("--top-ks", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--n-comics", type=int, default=500)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.segment_root)
    report = audit_panel_sweep(root, top_ks=args.top_ks, n_comics=args.n_comics)
    out_path = Path(args.out) if args.out else root / "reports" / "panel_sweep_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote panel-sweep audit to {out_path}")
    print(f"valid={report['valid']} errors={len(report['errors'])} warnings={len(report['warnings'])}")


def audit_panel_sweep(root: Path, *, top_ks: list[int], n_comics: int) -> dict[str, Any]:
    runs = []
    errors: list[str] = []
    warnings: list[str] = []
    expected_rows = n_comics * 6 * 2

    for top_k in top_ks:
        run_dir = root / "runs" / f"static_narration_top{top_k}_merged_0_{n_comics}"
        run = audit_run(run_dir, expected_rows=expected_rows)
        run["top_k"] = top_k
        runs.append(run)
        errors.extend(f"top-{top_k}: {message}" for message in run["errors"])
        warnings.extend(f"top-{top_k}: {message}" for message in run["warnings"])

    overlap = discovery_evaluation_overlap(root, runs)
    if overlap["n_overlap"]:
        errors.append(
            f"discovery/evaluation overlap contains {overlap['n_overlap']} strips; "
            "this is not the paper's disjoint-dataset protocol"
        )

    discovery = audit_discovery(root)
    if not discovery.get("available", False):
        warnings.append("merged discovery diagnostics are unavailable")
    elif discovery.get("max_head_routing_accuracy", 1.0) <= 0.5:
        errors.append(
            "discovery did not find a head routing to the queried panel more than half the time; "
            "the raw diagonal score is dominated by fixed spatial/image attention"
        )

    return {
        "valid": not errors,
        "segment_root": str(root),
        "expected_rows_per_top_k": expected_rows,
        "errors": errors,
        "warnings": warnings,
        "discovery_evaluation_overlap": overlap,
        "discovery_diagnostics": discovery,
        "runs": runs,
    }


def audit_run(run_dir: Path, *, expected_rows: int) -> dict[str, Any]:
    generations_path = run_dir / "generations.jsonl"
    judgments_path = run_dir / "qwen_judge" / "judgments.jsonl"
    summary_path = run_dir / "summary.json"
    errors: list[str] = []
    warnings: list[str] = []

    if not generations_path.exists():
        return {"path": str(run_dir), "errors": ["missing generations.jsonl"], "warnings": []}

    generations = _read_jsonl(generations_path)
    keys = [tuple(row.get(field) for field in KEY_FIELDS) for row in generations]
    duplicate_count = len(keys) - len(set(keys))
    if len(generations) != expected_rows:
        errors.append(f"row count {len(generations)} != expected {expected_rows}")
    if duplicate_count:
        errors.append(f"found {duplicate_count} duplicate experiment keys")

    empty_by_condition = Counter(
        str(row.get("condition")) for row in generations if not str(row.get("generated_text", "") or "").strip()
    )
    count_by_condition = Counter(str(row.get("condition")) for row in generations)
    empty_rates = {
        condition: empty_by_condition[condition] / count
        for condition, count in sorted(count_by_condition.items())
    }
    if any(rate > 0.05 for rate in empty_rates.values()):
        errors.append(f"empty-generation rate exceeds 5%: {empty_rates}")

    source_summaries = []
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_summaries = summary.get("source_summaries") or [summary]
    decode_modes = [item.get("decode_only") for item in source_summaries]
    if any(mode is True for mode in decode_modes):
        errors.append(
            "one or more shards used decode-only steering; the official static protocol steers prefill and decode"
        )
    elif source_summaries and all(mode is None for mode in decode_modes):
        errors.append("shard summaries predate decode_only provenance; full-sequence compliance is unverified")

    judgment_stats: dict[str, Any] = {}
    if judgments_path.exists():
        judgments = _read_jsonl(judgments_path)
        empty_judged_nonjunk = [
            row
            for row in judgments
            if not str(row.get("generated_text", "") or "").strip()
            and not bool((row.get("judgment") or {}).get("is_junk"))
        ]
        if empty_judged_nonjunk:
            errors.append(f"judge accepted {len(empty_judged_nonjunk)} empty generations as non-junk")

        for condition in sorted({str(row.get("condition")) for row in judgments}):
            rows = [row for row in judgments if str(row.get("condition")) == condition]
            reported_correct = sum(bool((row.get("judgment") or {}).get("correct")) for row in rows)
            corrected_correct = sum(
                bool(str(row.get("generated_text", "") or "").strip())
                and bool((row.get("judgment") or {}).get("correct"))
                for row in rows
            )
            judgment_stats[condition] = {
                "n": len(rows),
                "reported_accuracy": reported_correct / len(rows) if rows else 0.0,
                "empty_corrected_accuracy_upper_bound": corrected_correct / len(rows) if rows else 0.0,
                "matched_panel_distribution": dict(
                    Counter(str((row.get("judgment") or {}).get("matched_panel")) for row in rows)
                ),
            }
    else:
        warnings.append("missing qwen_judge/judgments.jsonl")

    return {
        "path": str(run_dir),
        "n_rows": len(generations),
        "n_unique_keys": len(set(keys)),
        "duplicate_count": duplicate_count,
        "empty_generation_rates": empty_rates,
        "decode_only_values": decode_modes,
        "judgments": judgment_stats,
        "errors": errors,
        "warnings": warnings,
    }


def discovery_evaluation_overlap(root: Path, runs: list[dict[str, Any]]) -> dict[str, Any]:
    discovery_names: set[str] = set()
    for path in sorted((root / "runs").glob("gaze_discovery_*_*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        discovery_names.update(str(name) for name in summary.get("sampled_names", []))

    evaluation_names: set[str] = set()
    for run in runs:
        generations = Path(run["path"]) / "generations.jsonl"
        if generations.exists():
            evaluation_names.update(str(row.get("strip_name")) for row in _read_jsonl(generations))

    overlap = sorted(discovery_names & evaluation_names)
    return {
        "n_discovery": len(discovery_names),
        "n_evaluation": len(evaluation_names),
        "n_overlap": len(overlap),
        "examples": overlap[:10],
    }


def audit_discovery(root: Path) -> dict[str, Any]:
    discovery_dir = root / "runs" / "gaze_discovery_merged_0_500"
    attention_path = discovery_dir / "mean_panel_attention.npy"
    ranking_path = discovery_dir / "gaze_head_ranking.json"
    if not attention_path.exists() or not ranking_path.exists():
        return {"available": False, "path": str(discovery_dir)}

    attention = np.load(attention_path)
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    selectivity, routing = gaze_routing_diagnostics(attention)
    top = ranking[0]
    layer, head = int(top["layer"]), int(top["head"])
    return {
        "available": True,
        "path": str(discovery_dir),
        "top_head": {"layer": layer, "head": head, "raw_score": float(top["score"])},
        "top_head_selectivity": float(selectivity[layer, head]),
        "top_head_routing_accuracy": float(routing[layer, head]),
        "max_head_routing_accuracy": float(routing.max()),
        "chance_routing_accuracy": 1.0 / float(attention.shape[0]),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
