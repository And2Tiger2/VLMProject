from __future__ import annotations

import json
from pathlib import Path

from scripts.merge_gaze_run_shards import merge_shards


def test_merge_gaze_static_shards(tmp_path: Path) -> None:
    root = tmp_path / "gaze"
    for suffix, values in [("_0_2", [1, 2]), ("_2_2", [3])]:
        shard = root / "runs" / f"static_narration{suffix}"
        shard.mkdir(parents=True)
        shard_rows = [json.dumps({"idx": value}) for value in values]
        (shard / "generations.jsonl").write_text("\n".join(shard_rows) + "\n")
        (shard / "summary.json").write_text(json.dumps({"suffix": suffix}))

    out = merge_shards(root, "static", ["_0_2", "_2_2"])

    rows = [
        json.loads(line)
        for line in (root / "runs" / "static_narration" / "generations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    summary = json.loads((root / "runs" / "static_narration" / "summary.json").read_text())

    assert out["n_rows"] == 3
    assert [row["idx"] for row in rows] == [1, 2, 3]
    assert summary["suffixes"] == ["_0_2", "_2_2"]
    assert summary["n_rows"] == 3
