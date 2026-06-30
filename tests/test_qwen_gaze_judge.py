from __future__ import annotations

import json
from pathlib import Path

from scripts import judge_qwen25_gaze_generations as judge


def test_parse_qwen_panel_judgment_from_json() -> None:
    parsed = judge.parse_qwen_panel_judgment(
        '{"matched_panel": 4, "is_junk": false, "reasoning": "door scene"}',
        n_panels=6,
    )

    assert parsed["matched_panel"] == 4
    assert parsed["is_junk"] is False


def test_parse_qwen_panel_judgment_marks_junk() -> None:
    parsed = judge.parse_qwen_panel_judgment(
        '{"matched_panel": null, "is_junk": true, "reasoning": "empty"}',
        n_panels=6,
    )

    assert parsed["matched_panel"] is None
    assert parsed["is_junk"] is True


def test_parse_qwen_panel_judgment_fallback_panel_text() -> None:
    parsed = judge.parse_qwen_panel_judgment("The answer best matches panel 2.", n_panels=6)

    assert parsed["matched_panel"] == 2
    assert parsed["is_junk"] is False


def test_judge_row_with_qwen_uses_forced_choice_prompt() -> None:
    row = {
        "strip_name": "comic1",
        "condition": "gaze_top20",
        "target_panel": 3,
        "generated_text": "A dog plays near the fire.",
        "baseline_text": "A dog runs through the woods.",
    }
    adapter = _FakeAdapter('{"matched_panel": 3, "is_junk": false, "reasoning": "fire scene"}')

    judgment = judge.judge_row_with_qwen(adapter, row, strip_image=object(), n_panels=6)

    assert judgment["correct"] is True
    assert judgment["matched_panel"] == 3
    assert "Answer to judge" in adapter.prompts[0]
    assert adapter.example_ids == ["comic1::gaze_top20::3"]


class _FakeAdapter:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []
        self.example_ids: list[str] = []

    def generate(self, example) -> str:
        self.example_ids.append(example.id)
        self.prompts.append(example.prompt)
        return self.response
