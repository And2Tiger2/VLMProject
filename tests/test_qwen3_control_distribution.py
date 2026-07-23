from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.aggregate_qwen3_control_distribution import (
    aggregate_control_distribution,
)
from scripts.merge_qwen3_control_distribution import merge_control_shards


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _generation_rows(condition: str, n_comics: int) -> list[dict]:
    return [
        {
            "strip_name": f"comic{comic}",
            "condition": condition,
            "target_panel": panel,
            "generated_text": f"{condition}-{comic}-{panel}",
        }
        for comic in range(n_comics)
        for panel in range(1, 7)
    ]


def _judgment_rows(
    condition: str, n_comics: int, *, correct_panels: set[int]
) -> list[dict]:
    rows = _generation_rows(condition, n_comics)
    for row in rows:
        correct = row["target_panel"] in correct_panels
        row["judgment"] = {
            "matched_panel": row["target_panel"] if correct else 0,
            "is_junk": False,
            "correct": correct,
            "matches_baseline": False,
            "parse_failed": False,
        }
    return rows


def test_merge_layer_matched_control_checks_histogram(tmp_path: Path) -> None:
    segment = tmp_path / "segment"
    ranking_dir = segment / "runs" / "discovery"
    ranking_dir.mkdir(parents=True)
    scores = np.arange(16, dtype=np.float64).reshape(4, 4)
    np.save(ranking_dir / "gaze_scores.npy", scores)
    gaze = [[0, 0], [1, 0]]
    control = [[0, 1], [1, 1]]
    for start in (0, 1):
        run = (
            segment
            / "runs"
            / f"static_control_distribution_layer_matched_random_seed45_top2_{start}_1"
        )
        config = {
            "control_mode": "layer_matched_random",
            "condition_set": "control",
            "condition_labels": ["non_gaze_layer_matched_random_2"],
            "selected_gaze_heads": gaze,
            "selected_control_heads": control,
            "gaze_ranking": str(ranking_dir / "gaze_head_ranking.json"),
        }
        _write_json(run / "experiment_config.json", config)
        _write_json(run / "summary.json", {"n_rows": 6})
        rows = [
            {
                "strip_name": f"comic{start}",
                "condition": "non_gaze_layer_matched_random_2",
                "target_panel": panel,
            }
            for panel in range(1, 7)
        ]
        _write_jsonl(run / "generations.jsonl", rows)
    result = merge_control_shards(
        segment_root=segment,
        mode="layer_matched_random",
        seed=45,
        top_k=2,
        starts=[0, 1],
        shard_size=1,
    )
    assert result["n_rows"] == 12
    assert result["conditions"] == ["non_gaze_layer_matched_random_2"]


def test_aggregate_counts_reference_gaze_once(tmp_path: Path) -> None:
    segment = tmp_path / "segment"
    top_k = 2
    n_comics = 2
    reference = (
        segment
        / "runs"
        / "static_paper_replication_seed42_top2_merged_0_2"
        / "kimi_judge"
    )
    _write_jsonl(
        reference / "judgments.jsonl",
        _judgment_rows("gaze_top2", n_comics, correct_panels={1, 2, 3, 4, 5}),
    )
    _write_json(
        reference / "judgment_config.json",
        {"judgment_schema_version": 6, "label_panels": True},
    )
    ranking_dir = segment / "runs" / "ranking"
    ranking_dir.mkdir(parents=True)
    np.save(ranking_dir / "gaze_scores.npy", np.arange(16).reshape(4, 4))
    ranking = ranking_dir / "gaze_head_ranking.json"
    _write_json(ranking, [[0, 0], [1, 0]])

    specs = [
        ("paper", 45, {1}),
        ("paper", 46, {1, 2}),
        ("layer_matched_random", 45, {1, 2, 3}),
        ("layer_matched_random", 46, {1, 2, 3, 4}),
        ("layer_matched_low", 42, {1, 2, 3, 4, 5}),
    ]
    for mode, seed, correct_panels in specs:
        run = (
            segment
            / "runs"
            / f"static_control_distribution_{mode}_seed{seed}_top2_merged_0_2"
        )
        condition = f"non_gaze_{mode}_2"
        _write_jsonl(
            run / "kimi_judge" / "judgments.jsonl",
            _judgment_rows(condition, n_comics, correct_panels=correct_panels),
        )
        _write_json(
            run / "kimi_judge" / "judgment_config.json",
            {"judgment_schema_version": 6, "label_panels": True},
        )
        _write_json(
            run / "experiment_config.json",
            {
                "selected_gaze_heads": [[0, 0], [1, 0]],
                "selected_control_heads": [[0, 1], [1, 1]],
                "gaze_ranking": str(ranking),
            },
        )
    result = aggregate_control_distribution(
        segment_root=segment,
        paper_seeds=[45, 46],
        matched_seeds=[45, 46],
        low_seed=42,
        reference_seed=42,
        reference_comics=2,
        n_comics=2,
        top_k=2,
        n_bootstrap=100,
        bootstrap_seed=1,
        out_dir=segment / "reports" / "control",
    )
    assert result["valid"]
    assert result["independent_gaze_rows"] == 12
    assert result["gaze"]["accuracy"] == 5 / 6
    assert (
        result["groups"]["paper_uniform_layers_20_35"]["control_accuracy_mean"]
        == 0.25
    )
    assert result["groups"]["layer_matched_random"][
        "control_accuracy_mean"
    ] == pytest.approx(
        7 / 12
    )
