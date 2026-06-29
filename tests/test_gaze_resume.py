from __future__ import annotations

import json
from pathlib import Path

from vlm_eval.gaze_resume import load_completed_keys, row_key


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
