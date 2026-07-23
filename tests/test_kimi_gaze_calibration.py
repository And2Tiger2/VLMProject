from __future__ import annotations

from scripts.calibrate_kimi_gaze_judge import build_calibration_rows


def test_caption_calibration_rows_are_panel_balanced_and_deterministic() -> None:
    manifest = [
        {
            "comic_id": comic,
            "comic_dir": f"/comics/comic{comic}",
            "captions": [f"comic {comic} panel {panel}" for panel in range(1, 7)],
        }
        for comic in range(20)
    ]

    first = build_calibration_rows(manifest, limit=60, seed=42)
    second = build_calibration_rows(manifest, limit=60, seed=42)

    assert first == second
    assert len(first) == 60
    assert {
        panel: sum(row["target_panel"] == panel for row in first)
        for panel in range(1, 7)
    } == {panel: 10 for panel in range(1, 7)}
    assert all(row["condition"] == "source_caption" for row in first)
