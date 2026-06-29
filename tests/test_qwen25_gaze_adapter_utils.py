from __future__ import annotations

from adapters.qwen25_vl_gaze import _trajectory_from_records


def test_trajectory_from_records_groups_decode_steps() -> None:
    records = [
        {"layer_idx": 0, "query_len": 3, "key_len": 10, "panel_1_attention_mass": 0.9},
        {"layer_idx": 0, "query_len": 1, "key_len": 11, "panel_1_attention_mass": 0.8, "panel_2_attention_mass": 0.2},
        {"layer_idx": 1, "query_len": 1, "key_len": 11, "panel_1_attention_mass": 0.6, "panel_2_attention_mass": 0.4},
        {"layer_idx": 0, "query_len": 1, "key_len": 12, "panel_1_attention_mass": 0.1, "panel_2_attention_mass": 0.9},
    ]

    trajectory = _trajectory_from_records(records)

    assert trajectory["n_decode_steps"] == 2
    assert trajectory["panel_names"] == ["panel_1", "panel_2"]
    assert trajectory["steps"][0]["panel_masses"] == {"panel_1": 0.7, "panel_2": 0.30000000000000004}
    assert trajectory["steps"][1]["panel_masses"] == {"panel_1": 0.1, "panel_2": 0.9}
