from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.merge_qwen25_gaze_discovery_shards import merge_discovery_shards


def _write_shard(path: Path, *, valid_samples: int, score: float, start: int) -> None:
    path.mkdir(parents=True)
    gaze_sum = np.zeros((2, 1, 2, 2), dtype=np.float64)
    gaze_sum[0, 0, 0, 0] = score * valid_samples
    gaze_sum[1, 0, 0, 1] = score * valid_samples
    gaze_sum[0, 0, 1, 0] = 0.1 * valid_samples
    gaze_sum[1, 0, 1, 1] = 0.1 * valid_samples
    np.save(path / "gaze_sum.npy", gaze_sum)
    mean = gaze_sum / valid_samples
    np.save(path / "mean_panel_attention.npy", mean)
    np.save(path / "gaze_scores.npy", np.array([[score, 0.1]], dtype=np.float64))
    ranking = [
        {"layer": 0, "head": 0, "score": score},
        {"layer": 0, "head": 1, "score": 0.1},
    ]
    (path / "gaze_head_ranking.json").write_text(json.dumps(ranking))
    (path / "summary.json").write_text(
        json.dumps(
            {
                "valid_samples": valid_samples,
                "start_comic_idx": start,
                "max_comics": valid_samples,
                "top_head": ranking[0],
            }
        )
    )


def test_merge_discovery_shards_weights_by_valid_samples(tmp_path: Path) -> None:
    shard_a = tmp_path / "gaze_discovery_0_2"
    shard_b = tmp_path / "gaze_discovery_2_4"
    _write_shard(shard_a, valid_samples=2, score=0.5, start=0)
    _write_shard(shard_b, valid_samples=4, score=0.2, start=2)

    result = merge_discovery_shards([shard_a, shard_b], tmp_path / "merged", top_k=2)

    ranking = json.loads((tmp_path / "merged" / "gaze_head_ranking.json").read_text())
    summary = json.loads((tmp_path / "merged" / "summary.json").read_text())
    stability = (tmp_path / "merged" / "top2_stability.tsv").read_text()

    assert result["valid_samples"] == 6
    assert summary["valid_samples"] == 6
    assert ranking[0]["layer"] == 0
    assert ranking[0]["head"] == 0
    assert abs(ranking[0]["score"] - ((0.5 * 2 + 0.2 * 4) / 6)) < 1e-12
    assert "L0H0" in stability
