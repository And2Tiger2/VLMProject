from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_kimi_calibration import diagnose_calibration


def test_calibration_diagnostics_build_confusion_and_failed_examples(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "strip_name": "comic1",
            "target_panel": 4,
            "generated_text": "A robot paints.",
            "judgment": {
                "matched_panel": 3,
                "is_junk": False,
                "raw_judge_text": "3",
            },
        },
        {
            "strip_name": "comic2",
            "target_panel": 2,
            "generated_text": "A dog runs.",
            "judgment": {
                "matched_panel": 2,
                "is_junk": False,
                "raw_judge_text": "2",
            },
        },
    ]
    path = tmp_path / "judgments.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = diagnose_calibration(path)

    assert report["confusion_matrix"]["4"]["3"] == 1
    assert report["confusion_matrix"]["2"]["2"] == 1
    assert report["predicted_panel_counts"]["3"] == 1
    assert report["panel_4_failures"][0]["caption"] == "A robot paints."
