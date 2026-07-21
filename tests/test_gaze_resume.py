from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_eval.gaze_resume import ensure_resume_config, load_completed_keys, row_key


def test_load_completed_keys_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    rows = [
        {"strip_name": "comic1", "condition": "gaze_top5", "target_panel": 1},
        {"strip_name": "comic1", "condition": "non_gaze_5", "target_panel": 2},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    completed = load_completed_keys(path, ["strip_name", "condition", "target_panel"])

    assert completed == {
        ("comic1", "gaze_top5", 1),
        ("comic1", "non_gaze_5", 2),
    }


def test_row_key_uses_missing_fields_as_none() -> None:
    assert row_key({"strip_name": "comic1"}, ["strip_name", "condition"]) == ("comic1", None)


def test_resume_config_rejects_legacy_artifact_without_provenance(tmp_path: Path) -> None:
    (tmp_path / "generations.jsonl").write_text("{}\n")

    with pytest.raises(RuntimeError, match="experiment_config.json is missing"):
        ensure_resume_config(tmp_path, {"decode_only": True}, resume=True, artifact_name="generations.jsonl")


def test_resume_config_rejects_changed_configuration(tmp_path: Path) -> None:
    config = {"decode_only": True, "top_k": 10}
    ensure_resume_config(tmp_path, config, resume=False, artifact_name="generations.jsonl")
    (tmp_path / "generations.jsonl").write_text("{}\n")

    with pytest.raises(RuntimeError, match="decode_only"):
        ensure_resume_config(
            tmp_path,
            {"decode_only": False, "top_k": 10},
            resume=True,
            artifact_name="generations.jsonl",
        )


def test_resume_config_accepts_identical_configuration(tmp_path: Path) -> None:
    config = {"decode_only": True, "top_k": 10}
    ensure_resume_config(tmp_path, config, resume=False, artifact_name="generations.jsonl")
    (tmp_path / "generations.jsonl").write_text("{}\n")

    ensure_resume_config(tmp_path, config, resume=True, artifact_name="generations.jsonl")
