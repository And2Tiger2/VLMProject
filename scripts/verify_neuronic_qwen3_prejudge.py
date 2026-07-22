from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.validate_qwen3_gaze_stage import (
    validate_datasets,
    validate_discovery,
    validate_static,
)


DEFAULT_SEGMENT_ROOT = Path("segments/gaze_heads_qwen3_8b")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify datasets, discovery rankings, and every merged static run before judging."
    )
    parser.add_argument("--segment-root", type=Path, default=DEFAULT_SEGMENT_ROOT)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 10, 50, 100])
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--shard-size", type=int, default=50)
    args = parser.parse_args()

    report = verify_prejudge(
        segment_root=args.segment_root,
        seeds=args.seeds,
        base_seed=args.base_seed,
        top_ks=args.top_ks,
        shards=args.shards,
        shard_size=args.shard_size,
    )
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(2)


def verify_prejudge(
    *,
    segment_root: Path,
    seeds: int,
    base_seed: int,
    top_ks: list[int],
    shards: int,
    shard_size: int,
) -> dict[str, Any]:
    errors: list[str] = []
    dataset_report = _capture(
        errors,
        "datasets",
        validate_datasets,
        segment_root / "data/discovery_comics",
        segment_root / "data/eval_comics",
        Path("segments/vlm_bias_attention/data/vlmbias_400.jsonl"),
        Path("segments/vlm_bias_attention/data/naturalbench_100_groups.jsonl"),
    )

    discovery_reports: list[dict[str, Any]] = []
    static_reports: list[dict[str, Any]] = []
    merged_count = shards * shard_size
    for seed in range(base_seed, base_seed + seeds):
        discovery_dir = segment_root / "runs" / f"gaze_discovery_seed{seed}_merged"
        discovery_report = _capture(
            errors,
            f"discovery seed {seed}",
            validate_discovery,
            discovery_dir,
            10,
            0.5,
        )
        if discovery_report is not None:
            discovery_reports.append({"seed": seed, **discovery_report})
        for required in ("gaze_head_ranking.json", "gaze_scores.npy"):
            if not (discovery_dir / required).is_file():
                errors.append(f"discovery seed {seed}: missing {discovery_dir / required}")

        for top_k in top_ks:
            run_dir = (
                segment_root
                / "runs"
                / f"static_narration_seed{seed}_top{top_k}_merged_0_{merged_count}"
            )
            static_report = _capture(
                errors,
                f"static seed {seed} top-{top_k}",
                validate_static,
                run_dir,
                0.50,
                0.95,
                warn_empty_rate=0.05,
            )
            if static_report is not None:
                static_reports.append(
                    {
                        "seed": seed,
                        "top_k": top_k,
                        "run_dir": str(run_dir),
                        "valid": static_report["valid"],
                        "n_rows": static_report["n_rows"],
                        "expected_rows": static_report["expected_rows"],
                        "empty_rates": static_report["empty_rates"],
                        "target_attention_mass": static_report["target_attention_mass"],
                        "warnings": static_report["warnings"],
                    }
                )

    expected_static_runs = seeds * len(top_ks)
    if len(static_reports) != expected_static_runs:
        errors.append(
            f"validated {len(static_reports)} merged static runs; expected {expected_static_runs}"
        )
    total_rows = sum(int(report["n_rows"]) for report in static_reports)
    return {
        "valid": not errors,
        "stage": "prejudge",
        "datasets_valid": bool(dataset_report and dataset_report["valid"]),
        "n_discovery_seeds": len(discovery_reports),
        "n_static_runs": len(static_reports),
        "expected_static_runs": expected_static_runs,
        "n_static_rows": total_rows,
        "discovery": discovery_reports,
        "static": static_reports,
        "errors": errors,
    }


def _capture(
    errors: list[str],
    label: str,
    function: Any,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any] | None:
    try:
        report = function(*args, **kwargs)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: {exc}")
        return None
    if not report.get("valid"):
        for error in report.get("errors", []):
            errors.append(f"{label}: {error}")
    return report


if __name__ == "__main__":
    main()
