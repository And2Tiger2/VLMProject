from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGES = [
    "data",
    "discovery",
    "trajectory",
    "static",
    "vqa",
    "dynamic",
    "static_score",
    "vqa_score",
    "dynamic_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GazeHeads Qwen2.5-VL pipeline artifacts.")
    parser.add_argument("--segment-root", default="segments/gaze_heads_qwen25")
    parser.add_argument(
        "--run-suffix",
        default="",
        help="Optional suffix for run directories, for example '_smoke' validates gaze_discovery_smoke.",
    )
    parser.add_argument(
        "--discovery-suffix",
        default=None,
        help="Optional suffix for the discovery directory. Defaults to --run-suffix for backward compatibility.",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.segment_root)
    discovery_suffix = args.run_suffix if args.discovery_suffix is None else args.discovery_suffix
    report = validate(root, run_suffix=args.run_suffix, discovery_suffix=discovery_suffix)
    out_path = Path(args.out) if args.out else root / "reports" / "pipeline_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"Wrote validation report to {out_path}")
    for stage in STAGES:
        status = "OK" if report["stages"][stage]["ok"] else "MISSING"
        print(f"{stage:14s} {status}")


def validate(root: Path, run_suffix: str = "", discovery_suffix: str | None = None) -> dict[str, Any]:
    runs = root / "runs"
    discovery_suffix = run_suffix if discovery_suffix is None else discovery_suffix
    report = {
        "segment_root": str(root),
        "run_suffix": run_suffix,
        "discovery_suffix": discovery_suffix,
        "ok": True,
        "stages": {
            "data": _validate_data(root / "data" / "comics"),
            "discovery": _validate_files(
                runs / f"gaze_discovery{discovery_suffix}",
                ["gaze_head_ranking.json", "gaze_scores.npy", "mean_panel_attention.npy", "summary.json"],
            ),
            "trajectory": _validate_jsonl_stage(runs / f"narration_trajectory{run_suffix}", "trajectories.jsonl"),
            "static": _validate_jsonl_stage(runs / f"static_narration{run_suffix}", "generations.jsonl"),
            "vqa": _validate_jsonl_stage(runs / f"vqa_steering{run_suffix}", "generations.jsonl"),
            "dynamic": _validate_jsonl_stage(runs / f"dynamic_narration{run_suffix}", "generations.jsonl"),
            "static_score": _validate_score(runs / f"static_narration{run_suffix}"),
            "vqa_score": _validate_score(runs / f"vqa_steering{run_suffix}"),
            "dynamic_score": _validate_score(runs / f"dynamic_narration{run_suffix}"),
        },
    }
    report["ok"] = all(stage["ok"] for stage in report["stages"].values())
    return report


def _validate_data(comics_root: Path) -> dict[str, Any]:
    comic_dirs = []
    if comics_root.exists():
        for comic_dir in sorted(path for path in comics_root.iterdir() if path.is_dir()):
            panel_count = sum(1 for idx in range(1, 7) if (comic_dir / f"p{idx}.png").exists())
            if panel_count == 6:
                comic_dirs.append(comic_dir.name)
    return {
        "ok": bool(comic_dirs),
        "path": str(comics_root),
        "n_comics": len(comic_dirs),
        "examples": comic_dirs[:5],
    }


def _validate_files(stage_dir: Path, filenames: list[str]) -> dict[str, Any]:
    files = {filename: stage_dir / filename for filename in filenames}
    missing = [filename for filename, path in files.items() if not path.exists()]
    return {
        "ok": not missing,
        "path": str(stage_dir),
        "missing": missing,
        "files": {filename: str(path) for filename, path in files.items() if path.exists()},
    }


def _validate_jsonl_stage(stage_dir: Path, filename: str) -> dict[str, Any]:
    path = stage_dir / filename
    n_rows = _count_jsonl_rows(path) if path.exists() else 0
    summary = stage_dir / "summary.json"
    return {
        "ok": path.exists() and n_rows > 0,
        "path": str(stage_dir),
        "jsonl": str(path),
        "n_rows": n_rows,
        "summary_exists": summary.exists(),
    }


def _validate_score(stage_dir: Path) -> dict[str, Any]:
    judgments = stage_dir / "judgments.jsonl"
    aggregate = stage_dir / "aggregate_results.json"
    return {
        "ok": judgments.exists() and aggregate.exists() and _count_jsonl_rows(judgments) > 0,
        "path": str(stage_dir),
        "judgments": str(judgments),
        "aggregate": str(aggregate),
        "n_judgments": _count_jsonl_rows(judgments) if judgments.exists() else 0,
    }


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


if __name__ == "__main__":
    main()
