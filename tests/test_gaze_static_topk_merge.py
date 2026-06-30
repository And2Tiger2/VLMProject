from __future__ import annotations

import json
from pathlib import Path

from scripts.merge_gaze_static_topk_batches import merge_static_topk_batches


def test_merge_static_topk_batches_validates_expected_rows(tmp_path: Path) -> None:
    root = tmp_path / "gaze"
    for top_k in [1, 5]:
        for start in [0, 2]:
            shard = root / "runs" / f"static_narration_top{top_k}_{start}_2"
            shard.mkdir(parents=True)
            rows = [
                {"top_k": top_k, "start": start, "idx": idx}
                for idx in range(2 * 3 * 2)
            ]
            (shard / "generations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            (shard / "summary.json").write_text(json.dumps({"top_k": top_k, "start": start}), encoding="utf-8")

    result = merge_static_topk_batches(
        segment_root=root,
        top_ks=[1, 5],
        starts=[0, 2],
        batch_size=2,
        targets_per_strip=3,
    )

    assert result["top1"]["n_rows"] == 24
    assert result["top5"]["n_rows"] == 24
    assert (root / "runs" / "static_narration_top1_merged_0_4" / "generations.jsonl").exists()
    assert (root / "runs" / "static_narration_topk_merged_summary.json").exists()
