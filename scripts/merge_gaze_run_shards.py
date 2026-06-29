from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGE_FILES = {
    "trajectory": ("narration_trajectory", "trajectories.jsonl"),
    "static": ("static_narration", "generations.jsonl"),
    "vqa": ("vqa_steering", "generations.jsonl"),
    "dynamic": ("dynamic_narration", "generations.jsonl"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge suffixed GazeHeads run shards into one JSONL stage artifact.")
    parser.add_argument("--segment-root", default="segments/gaze_heads_qwen25")
    parser.add_argument("--stage", choices=sorted(STAGE_FILES), required=True)
    parser.add_argument("--suffixes", nargs="+", required=True, help="Shard suffixes such as _0_25 _25_25.")
    parser.add_argument("--out-dir", default="", help="Defaults to the canonical unsuffixed stage directory.")
    args = parser.parse_args()

    root = Path(args.segment_root)
    out = merge_shards(root, args.stage, args.suffixes, Path(args.out_dir) if args.out_dir else None)
    print(f"Wrote {out['n_rows']} rows to {out['jsonl']}")
    print(f"Wrote summary to {out['summary']}")


def merge_shards(root: Path, stage: str, suffixes: list[str], out_dir: Path | None = None) -> dict[str, Any]:
    stage_dir_name, filename = STAGE_FILES[stage]
    runs = root / "runs"
    out_dir = out_dir or runs / stage_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    source_dirs = [runs / f"{stage_dir_name}{suffix}" for suffix in suffixes]
    missing = [str(source_dir / filename) for source_dir in source_dirs if not (source_dir / filename).exists()]
    if missing:
        raise FileNotFoundError("Missing shard JSONL files: " + ", ".join(missing))

    rows: list[str] = []
    row_counts: dict[str, int] = {}
    summaries: list[dict[str, Any]] = []
    for source_dir in source_dirs:
        path = source_dir / filename
        shard_rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows.extend(shard_rows)
        row_counts[str(source_dir)] = len(shard_rows)
        summary_path = source_dir / "summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    jsonl_path = out_dir / filename
    jsonl_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    summary_path = out_dir / "summary.json"
    summary = {
        "stage": stage,
        "merged_from": [str(source_dir) for source_dir in source_dirs],
        "suffixes": suffixes,
        "n_rows": len(rows),
        "row_counts": row_counts,
        "source_summaries": summaries,
        "jsonl": str(jsonl_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"jsonl": str(jsonl_path), "summary": str(summary_path), "n_rows": len(rows)}


if __name__ == "__main__":
    main()
