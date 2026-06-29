from __future__ import annotations

import json
from pathlib import Path

from scripts.report_qwen25_gaze_results import build_report


def test_build_gaze_report_from_synthetic_outputs(tmp_path: Path) -> None:
    root = tmp_path / "gaze"

    discovery = root / "runs" / "gaze_discovery_smoke"
    discovery.mkdir(parents=True)
    (discovery / "summary.json").write_text(
        json.dumps({"valid_samples": 1, "n_layers": 2, "n_heads": 3, "top_head": {"layer": 1, "head": 2}})
    )
    (discovery / "gaze_head_ranking.json").write_text(json.dumps([{"layer": 1, "head": 2, "score": 0.9}]))

    trajectory = root / "runs" / "narration_trajectory_smoke"
    trajectory.mkdir(parents=True)
    _write_jsonl(
        trajectory / "trajectories.jsonl",
        [
            {
                "strip_name": "comic1",
                "condition": "gaze_top2",
                "trajectory": {
                    "steps": [
                        {"panel_masses": {"panel_1": 0.8, "panel_2": 0.1}},
                        {"panel_masses": {"panel_1": 0.2, "panel_2": 0.7}},
                        {"panel_masses": {"panel_1": 0.1, "panel_2": 0.9}},
                    ]
                },
            }
        ],
    )

    static = root / "runs" / "static_narration_smoke"
    static.mkdir(parents=True)
    _write_jsonl(
        static / "judgments.jsonl",
        [
            _judged_row("gaze_top2", target=1, correct=True),
            _judged_row("gaze_top2", target=2, correct=False),
            _judged_row("non_gaze_2", target=1, correct=False, matches_baseline=True),
        ],
    )

    vqa = root / "runs" / "vqa_steering_smoke"
    vqa.mkdir(parents=True)
    _write_jsonl(vqa / "judgments.jsonl", [_judged_row("gaze_top2", target=1, correct=True, task="vqa_steering")])

    dynamic = root / "runs" / "dynamic_narration_smoke"
    dynamic.mkdir(parents=True)
    _write_jsonl(
        dynamic / "judgments.jsonl",
        [
            {
                "task": "dynamic_narration",
                "condition": "gaze_top2",
                "segment_judgments": [
                    {"target_panel": 1, "matched_panel": 1, "correct": True},
                    {"target_panel": 2, "matched_panel": 2, "correct": True},
                    {"target_panel": 3, "matched_panel": 1, "correct": False},
                ],
            }
        ],
    )

    report = build_report(root, run_suffix="_smoke", seed=0)

    assert report["discovery"]["top_head"] == {"layer": 1, "head": 2}
    assert report["trajectory"]["conditions"]["gaze_top2"]["n_rows"] == 1
    assert report["static"]["conditions"]["gaze_top2"]["overall"]["n"] == 2
    assert report["static"]["conditions"]["non_gaze_2"]["baseline_match_count"] == 1
    assert report["vqa"]["conditions"]["gaze_top2"]["overall"]["accuracy"] == 1.0
    assert report["dynamic"]["conditions"]["gaze_top2"]["per_segment_accuracy"]["n"] == 3


def _judged_row(
    condition: str,
    *,
    target: int,
    correct: bool,
    matches_baseline: bool = False,
    task: str = "static_narration",
) -> dict:
    return {
        "task": task,
        "condition": condition,
        "target_panel": target,
        "judgment": {
            "matched_panel": target if correct else None,
            "correct": correct,
            "is_junk": False,
            "matches_baseline": matches_baseline,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
