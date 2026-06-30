from __future__ import annotations

import json
from pathlib import Path

from adapters.qwen25_vl_gaze_attention import _heads_by_layer, _load_gaze_heads
from scripts.run_vlmbias_gaze_attention_sweep import _conditions


def test_load_gaze_heads_uses_ranking_order(tmp_path: Path) -> None:
    ranking = [
        {"layer": 5, "head": 10, "score": 0.15},
        {"layer": 0, "head": 14, "score": 0.09},
        {"layer": 33, "head": 2, "score": 0.08},
    ]
    path = tmp_path / "gaze_head_ranking.json"
    path.write_text(json.dumps(ranking), encoding="utf-8")

    assert _load_gaze_heads(path, 2) == [(5, 10), (0, 14)]


def test_heads_by_layer_deduplicates_and_sorts() -> None:
    assert _heads_by_layer([(5, 10), (5, 2), (0, 14), (5, 10)]) == {
        0: [14],
        5: [2, 10],
    }


def test_conditions_include_one_baseline_and_topk_alpha_grid() -> None:
    conditions = _conditions(alphas=[0.0, 0.5, 2.0], top_ks=[1, 5])

    assert conditions[0] == {
        "condition": "baseline",
        "attention_alpha": 0.0,
        "top_k_gaze": 0,
        "is_baseline": True,
    }
    assert [condition["condition"] for condition in conditions[1:]] == [
        "gaze_top1_alpha0p5",
        "gaze_top5_alpha0p5",
        "gaze_top1_alpha2",
        "gaze_top5_alpha2",
    ]
