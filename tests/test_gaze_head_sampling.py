from __future__ import annotations

import numpy as np

from adapters.qwen25_vl_gaze import PAPER_DECODE_ONLY, Qwen25VLGazeAdapter, _weighted_record_mean
from scripts.run_qwen25_gaze_static_narration import (
    DEFAULT_DECODE_ONLY,
    sample_non_gaze_heads,
    select_control_heads,
)
from scripts.run_qwen25_gaze_vqa_steering import DEFAULT_DECODE_ONLY as VQA_DEFAULT_DECODE_ONLY
from scripts.discover_qwen25_gaze_heads import gaze_routing_diagnostics


def test_sample_non_gaze_heads_matches_paper_percentile_by_default() -> None:
    scores = np.arange(20, dtype=np.float64).reshape(4, 5)

    heads = sample_non_gaze_heads(
        n_layers=4,
        n_heads=5,
        exclude={(0, 0), (0, 1)},
        n_select=10,
        seed=0,
        scores=scores,
        max_score=1.0,
    )

    assert heads == []


def test_sample_non_gaze_heads_can_backfill_to_requested_count() -> None:
    scores = np.arange(20, dtype=np.float64).reshape(4, 5)

    heads = sample_non_gaze_heads(
        n_layers=4,
        n_heads=5,
        exclude={(0, 0), (0, 1)},
        n_select=10,
        seed=0,
        scores=scores,
        max_score=1.0,
        backfill=True,
    )

    assert len(heads) == 10
    assert len(set(heads)) == 10
    assert (0, 0) not in heads
    assert (0, 1) not in heads


def test_paper_control_is_exact_k_from_inclusive_layers_20_to_35() -> None:
    gaze_heads = [(20, 0), (24, 29), (35, 31)]
    heads, cutoff = select_control_heads(
        control_mode="paper",
        n_layers=36,
        n_heads=32,
        gaze_heads=gaze_heads,
        n_select=100,
        seed=42,
        gaze_scores=None,
        nongaze_percentile=5.0,
    )

    assert cutoff is None
    assert len(heads) == len(set(heads)) == 100
    assert all(20 <= layer <= 35 for layer, _ in heads)
    assert not set(heads).intersection(gaze_heads)


def test_paper_control_is_deterministic_for_seed() -> None:
    kwargs = dict(
        control_mode="paper",
        n_layers=36,
        n_heads=32,
        gaze_heads=[(24, 29)],
        n_select=100,
        seed=43,
        gaze_scores=None,
        nongaze_percentile=5.0,
    )
    first, _ = select_control_heads(**kwargs)
    second, _ = select_control_heads(**kwargs)
    assert first == second


def test_static_paper_protocol_steers_prefill_and_decode_by_default() -> None:
    assert DEFAULT_DECODE_ONLY is False
    assert VQA_DEFAULT_DECODE_ONLY is False
    assert PAPER_DECODE_ONLY is False
    assert Qwen25VLGazeAdapter.generate_steered.__kwdefaults__["decode_only"] is False


def test_attention_record_mean_weights_each_selected_head_equally() -> None:
    records = [
        {"panel_1_attention_mass": 1.0, "tracked_heads": 1},
        {"panel_1_attention_mass": 0.0, "tracked_heads": 3},
    ]

    assert _weighted_record_mean(records, "panel_1_attention_mass") == 0.25


def test_gaze_routing_diagnostics_distinguish_tracking_from_fixed_attention() -> None:
    tracking = np.zeros((3, 1, 1, 3), dtype=np.float64)
    fixed = np.zeros((3, 1, 1, 3), dtype=np.float64)
    for panel in range(3):
        tracking[panel, 0, 0, panel] = 0.9
        fixed[panel, 0, 0, 0] = 0.9

    tracking_selectivity, tracking_routing = gaze_routing_diagnostics(tracking)
    fixed_selectivity, fixed_routing = gaze_routing_diagnostics(fixed)

    assert np.isclose(tracking_selectivity[0, 0], 0.9)
    assert tracking_routing[0, 0] == 1.0
    assert np.isclose(fixed_selectivity[0, 0], 0.0)
    assert fixed_routing[0, 0] == 1 / 3
