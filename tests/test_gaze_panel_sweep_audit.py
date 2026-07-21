from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_gaze_panel_sweep import audit_run


def test_audit_rejects_empty_outputs_and_missing_steering_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    judge_dir = run_dir / "qwen_judge"
    judge_dir.mkdir(parents=True)
    generation = {
        "strip_name": "comic1",
        "condition": "gaze_top1",
        "target_panel": 2,
        "generated_text": "",
    }
    (run_dir / "generations.jsonl").write_text(json.dumps(generation) + "\n")
    (run_dir / "summary.json").write_text(json.dumps({"source_summaries": [{}]}))
    judged = {**generation, "judgment": {"matched_panel": 2, "is_junk": False, "correct": True}}
    (judge_dir / "judgments.jsonl").write_text(json.dumps(judged) + "\n")

    report = audit_run(run_dir, expected_rows=1)

    assert any("empty-generation rate" in message for message in report["errors"])
    assert any("decode_only provenance" in message for message in report["errors"])
    assert any("accepted 1 empty" in message for message in report["errors"])
    assert report["judgments"]["gaze_top1"]["reported_accuracy"] == 1.0
    assert report["judgments"]["gaze_top1"]["empty_corrected_accuracy_upper_bound"] == 0.0


def test_audit_accepts_complete_full_sequence_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    judge_dir = run_dir / "qwen_judge"
    judge_dir.mkdir(parents=True)
    generation = {
        "strip_name": "comic1",
        "condition": "gaze_top1",
        "target_panel": 2,
        "generated_text": "A hero jumps.",
    }
    (run_dir / "generations.jsonl").write_text(json.dumps(generation) + "\n")
    (run_dir / "summary.json").write_text(json.dumps({"source_summaries": [{"decode_only": False}]}))
    judged = {**generation, "judgment": {"matched_panel": 2, "is_junk": False, "correct": True}}
    (judge_dir / "judgments.jsonl").write_text(json.dumps(judged) + "\n")

    report = audit_run(run_dir, expected_rows=1)

    assert report["errors"] == []


def test_audit_rejects_decode_only_static_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    generation = {
        "strip_name": "comic1",
        "condition": "gaze_top1",
        "target_panel": 2,
        "generated_text": "A hero jumps.",
    }
    (run_dir / "generations.jsonl").write_text(json.dumps(generation) + "\n")
    (run_dir / "summary.json").write_text(json.dumps({"source_summaries": [{"decode_only": True}]}))

    report = audit_run(run_dir, expected_rows=1)

    assert any("official static protocol" in message for message in report["errors"])
