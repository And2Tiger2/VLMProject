from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.merge_gaze_run_shards import merge_shards


DEFAULT_TOP_KS = [1, 5, 10, 20]
DEFAULT_STARTS = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge static steering top-k batch outputs.")
    parser.add_argument("--segment-root", default="segments/gaze_heads_qwen25")
    parser.add_argument("--top-ks", type=int, nargs="+", default=DEFAULT_TOP_KS)
    parser.add_argument("--starts", type=int, nargs="+", default=DEFAULT_STARTS)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--targets-per-strip", type=int, default=6)
    args = parser.parse_args()

    results = merge_static_topk_batches(
        segment_root=Path(args.segment_root),
        top_ks=args.top_ks,
        starts=args.starts,
        batch_size=args.batch_size,
        targets_per_strip=args.targets_per_strip,
    )
    print(json.dumps(results, indent=2))


def merge_static_topk_batches(
    *,
    segment_root: Path,
    top_ks: list[int],
    starts: list[int],
    batch_size: int,
    targets_per_strip: int,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for top_k in top_ks:
        suffixes = [f"_top{top_k}_{start}_{batch_size}" for start in starts]
        out_dir = segment_root / "runs" / f"static_narration_top{top_k}_merged_0_{len(starts) * batch_size}"
        result = merge_shards(segment_root, "static", suffixes, out_dir=out_dir)
        expected_rows = len(starts) * batch_size * targets_per_strip * 2
        if int(result["n_rows"]) != expected_rows:
            raise RuntimeError(
                f"top{top_k} merged row count mismatch: got {result['n_rows']}, expected {expected_rows}. "
                "Each comic should have gaze and non-gaze conditions for each target panel."
            )
        merged[f"top{top_k}"] = {
            **result,
            "expected_rows": expected_rows,
            "suffixes": suffixes,
        }

    summary_path = segment_root / "runs" / "static_narration_topk_merged_summary.json"
    summary_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    merged["summary"] = str(summary_path)
    return merged


if __name__ == "__main__":
    main()
