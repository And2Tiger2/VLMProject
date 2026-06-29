from __future__ import annotations

import json
from pathlib import Path

from scripts.score_qwen25_gaze_generations import score_generations


def test_score_generations_resume_skips_static_rows(tmp_path: Path) -> None:
    generations = tmp_path / "generations.jsonl"
    out_dir = tmp_path / "out"
    rows = [
        _generation_row("comic1", "gaze_top5", 1, "same", "same"),
        _generation_row("comic1", "gaze_top5", 2, "new text", "baseline"),
    ]
    _write_jsonl(generations, rows)
    out_dir.mkdir()
    existing = dict(rows[0])
    existing["judgment"] = {"correct": False, "matches_baseline": True}
    _write_jsonl(out_dir / "judgments.jsonl", [existing])

    result = score_generations(generations_path=generations, out_dir=out_dir, resume=True, seed=0)

    judged = _read_jsonl(out_dir / "judgments.jsonl")
    aggregate = json.loads((out_dir / "aggregate_results.json").read_text())
    assert result["n_rows"] == 2
    assert result["n_new_rows"] == 1
    assert len(judged) == 2
    assert aggregate["n_rows"] == 2
    assert aggregate["n_new_rows"] == 1
    assert aggregate["n_skipped_existing_rows"] == 1


def test_score_generations_resume_skips_dynamic_rows(tmp_path: Path) -> None:
    generations = tmp_path / "dynamic.jsonl"
    out_dir = tmp_path / "out"
    rows = [
        _dynamic_row("comic1", "gaze_top5"),
        _dynamic_row("comic2", "gaze_top5"),
    ]
    _write_jsonl(generations, rows)
    out_dir.mkdir()
    existing = dict(rows[0])
    existing["segments"] = ["one", "two"]
    existing["segment_judgments"] = [
        {"target_panel": 1, "matched_panel": None, "correct": False},
        {"target_panel": 2, "matched_panel": None, "correct": False},
    ]
    _write_jsonl(out_dir / "judgments.jsonl", [existing])

    score_generations(generations_path=generations, out_dir=out_dir, resume=True, seed=0)

    judged = _read_jsonl(out_dir / "judgments.jsonl")
    aggregate = json.loads((out_dir / "aggregate_results.json").read_text())
    assert len(judged) == 2
    assert aggregate["n_rows"] == 2
    assert aggregate["n_new_rows"] == 1
    assert aggregate["n_skipped_existing_rows"] == 1
    assert "dynamic_aggregate" in aggregate


def _generation_row(strip_name: str, condition: str, target_panel: int, generated: str, baseline: str) -> dict:
    return {
        "strip_name": strip_name,
        "comic_dir": "/tmp/no-image-needed",
        "condition": condition,
        "target_panel": target_panel,
        "baseline_text": baseline,
        "generated_text": generated,
    }


def _dynamic_row(strip_name: str, condition: str) -> dict:
    return {
        "task": "dynamic_narration",
        "strip_name": strip_name,
        "comic_dir": "/tmp/no-image-needed",
        "condition": condition,
        "generated_text": "Panel 1: one. Panel 2: two.",
        "schedule": [
            {"start_decode_step": 0, "target_panel": 1},
            {"start_decode_step": 5, "target_panel": 2},
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
