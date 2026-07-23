from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.aggregate_qwen3_static_paper_judgments import (
    aggregate_judgment_runs,
)


def _write_run(
    root: Path,
    *,
    seed: int,
    gaze_correct: list[bool],
    control_correct: list[bool],
    alter_gaze_match: bool = False,
) -> None:
    judge_dir = (
        root
        / "runs"
        / f"static_paper_replication_seed{seed}_top100_merged_0_1"
        / "kimi_judge"
    )
    judge_dir.mkdir(parents=True)
    (judge_dir / "judgment_config.json").write_text(
        json.dumps({"judgment_schema_version": 6, "label_panels": True}),
        encoding="utf-8",
    )
    rows = []
    for panel in range(1, 7):
        for condition, outcomes in (
            ("gaze_top100", gaze_correct),
            ("non_gaze_paper_100", control_correct),
        ):
            matched = panel if outcomes[panel - 1] else (panel % 6) + 1
            if alter_gaze_match and condition == "gaze_top100" and panel == 1:
                matched = 6
            rows.append(
                {
                    "strip_name": "comic1",
                    "condition": condition,
                    "target_panel": panel,
                    "generated_text": f"shared gaze {panel}"
                    if condition == "gaze_top100"
                    else f"control {seed} {panel}",
                    "baseline_text": "baseline",
                    "judgment": {
                        "matched_panel": matched,
                        "correct": outcomes[panel - 1],
                        "is_junk": False,
                        "matches_baseline": False,
                        "parse_failed": False,
                        "raw_judge_text": str(matched),
                    },
                }
            )
    (judge_dir / "judgments.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_aggregate_counts_repeated_gaze_once_and_reports_control_seed_effects(
    tmp_path: Path,
) -> None:
    gaze = [True, True, True, True, True, False]
    _write_run(
        tmp_path, seed=42, gaze_correct=gaze, control_correct=[True] + [False] * 5
    )
    _write_run(
        tmp_path, seed=43, gaze_correct=gaze, control_correct=[True, True] + [False] * 4
    )
    _write_run(
        tmp_path, seed=44, gaze_correct=gaze, control_correct=[False] * 6
    )

    result = aggregate_judgment_runs(
        segment_root=tmp_path,
        seeds=[42, 43, 44],
        top_k=100,
        n_comics=1,
        n_bootstrap=100,
        bootstrap_seed=0,
        out_dir=tmp_path / "report",
    )

    assert result["valid"] is True
    assert result["independent_gaze_rows"] == 6
    assert result["aggregate"]["gaze_accuracy_counted_once"] == 5 / 6
    assert result["aggregate"]["control_accuracy_mean_across_seeds"] == 1 / 6
    assert np.isclose(
        result["aggregate"]["delta_mean_across_control_seeds"], 4 / 6
    )
    assert (tmp_path / "report" / "report.md").exists()


def test_aggregate_rejects_repeated_gaze_judgment_disagreement(
    tmp_path: Path,
) -> None:
    gaze = [True] * 6
    for seed in (42, 43):
        _write_run(
            tmp_path,
            seed=seed,
            gaze_correct=gaze,
            control_correct=[False] * 6,
            alter_gaze_match=seed == 43,
        )

    result = aggregate_judgment_runs(
        segment_root=tmp_path,
        seeds=[42, 43],
        top_k=100,
        n_comics=1,
        n_bootstrap=10,
        bootstrap_seed=0,
        out_dir=tmp_path / "report",
    )

    assert result["valid"] is False
    assert any("disagreements" in error for error in result["errors"])
