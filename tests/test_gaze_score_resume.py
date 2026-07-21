from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.score_qwen25_gaze_generations as scorer
from scripts.score_qwen25_gaze_generations import DEFAULT_JUDGE_MODEL, _score_config, score_generations
from vlm_eval.gaze_resume import ensure_resume_config


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
    ensure_resume_config(
        out_dir,
        _score_config(generations, "baseline-only", DEFAULT_JUDGE_MODEL, 6, 0),
        resume=False,
        artifact_name="judgments.jsonl",
        config_name="judgment_config.json",
    )
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
    ensure_resume_config(
        out_dir,
        _score_config(generations, "baseline-only", DEFAULT_JUDGE_MODEL, 6, 0),
        resume=False,
        artifact_name="judgments.jsonl",
        config_name="judgment_config.json",
    )
    _write_jsonl(out_dir / "judgments.jsonl", [existing])

    score_generations(generations_path=generations, out_dir=out_dir, resume=True, seed=0)

    judged = _read_jsonl(out_dir / "judgments.jsonl")
    aggregate = json.loads((out_dir / "aggregate_results.json").read_text())
    assert len(judged) == 2
    assert aggregate["n_rows"] == 2
    assert aggregate["n_new_rows"] == 1
    assert aggregate["n_skipped_existing_rows"] == 1
    assert "dynamic_aggregate" in aggregate


def test_score_generations_flushes_each_judgment_before_failure(tmp_path: Path, monkeypatch) -> None:
    generations = tmp_path / "generations.jsonl"
    out_dir = tmp_path / "out"
    rows = [
        _generation_row("comic1", "gaze_top5", 1, "first", "baseline"),
        _generation_row("comic1", "gaze_top5", 2, "second", "baseline"),
    ]
    _write_jsonl(generations, rows)
    calls = 0

    def fail_on_second(row, args, strip_cache):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated API failure")
        judged = dict(row)
        judged["judgment"] = {"correct": False}
        return judged

    monkeypatch.setattr(scorer, "_score_static_or_vqa_row", fail_on_second)

    with pytest.raises(RuntimeError, match="simulated API failure"):
        score_generations(generations_path=generations, out_dir=out_dir)

    assert _read_jsonl(out_dir / "judgments.jsonl") == [
        {**rows[0], "judgment": {"correct": False}}
    ]


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
