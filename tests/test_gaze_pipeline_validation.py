from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_gaze_pipeline import validate


def test_validate_gaze_pipeline_synthetic_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "gaze"
    comics = root / "data" / "comics" / "comic1"
    comics.mkdir(parents=True)
    for idx in range(1, 7):
        (comics / f"p{idx}.png").write_bytes(b"fake")

    discovery = root / "runs" / "gaze_discovery"
    discovery.mkdir(parents=True)
    for name in ["gaze_head_ranking.json", "gaze_scores.npy", "mean_panel_attention.npy", "summary.json"]:
        (discovery / name).write_text("{}")

    static = root / "runs" / "static_narration"
    static.mkdir(parents=True)
    (static / "generations.jsonl").write_text(json.dumps({"x": 1}) + "\n")
    (static / "summary.json").write_text("{}")
    (static / "judgments.jsonl").write_text(json.dumps({"x": 1}) + "\n")
    (static / "aggregate_results.json").write_text("{}")

    report = validate(root)

    assert report["stages"]["data"]["ok"]
    assert report["stages"]["discovery"]["ok"]
    assert report["stages"]["static"]["ok"]
    assert report["stages"]["static_score"]["ok"]
    assert not report["stages"]["vqa"]["ok"]
    assert not report["ok"]


def test_validate_gaze_pipeline_run_suffix(tmp_path: Path) -> None:
    root = tmp_path / "gaze"
    comics = root / "data" / "comics" / "comic1"
    comics.mkdir(parents=True)
    for idx in range(1, 7):
        (comics / f"p{idx}.png").write_bytes(b"fake")

    discovery = root / "runs" / "gaze_discovery_smoke"
    discovery.mkdir(parents=True)
    for name in ["gaze_head_ranking.json", "gaze_scores.npy", "mean_panel_attention.npy", "summary.json"]:
        (discovery / name).write_text("{}")

    trajectory = root / "runs" / "narration_trajectory_smoke"
    trajectory.mkdir(parents=True)
    (trajectory / "trajectories.jsonl").write_text(json.dumps({"x": 1}) + "\n")

    report = validate(root, run_suffix="_smoke")

    assert report["run_suffix"] == "_smoke"
    assert report["stages"]["discovery"]["ok"]
    assert report["stages"]["trajectory"]["ok"]
    assert not report["stages"]["static"]["ok"]


def test_validate_gaze_pipeline_separate_discovery_suffix(tmp_path: Path) -> None:
    root = tmp_path / "gaze"
    comics = root / "data" / "comics" / "comic1"
    comics.mkdir(parents=True)
    for idx in range(1, 7):
        (comics / f"p{idx}.png").write_bytes(b"fake")

    discovery = root / "runs" / "gaze_discovery"
    discovery.mkdir(parents=True)
    for name in ["gaze_head_ranking.json", "gaze_scores.npy", "mean_panel_attention.npy", "summary.json"]:
        (discovery / name).write_text("{}")

    static = root / "runs" / "static_narration_0_25"
    static.mkdir(parents=True)
    (static / "generations.jsonl").write_text(json.dumps({"x": 1}) + "\n")

    report = validate(root, run_suffix="_0_25", discovery_suffix="")

    assert report["run_suffix"] == "_0_25"
    assert report["discovery_suffix"] == ""
    assert report["stages"]["discovery"]["ok"]
    assert report["stages"]["static"]["ok"]
