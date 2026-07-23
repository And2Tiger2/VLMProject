from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image

from scripts import judge_kimi_gaze_generations as judge


def test_parse_kimi_panel_judgment_from_json() -> None:
    parsed = judge.parse_panel_judgment(
        '```json\n{"matched_panel": 4, "is_junk": false, "reasoning": "door scene"}\n```',
        n_panels=6,
    )
    assert parsed == {
        "matched_panel": 4,
        "is_junk": False,
        "parse_failed": False,
        "reasoning": "door scene",
    }


def test_parse_kimi_panel_judgment_flags_unparseable_output() -> None:
    parsed = judge.parse_panel_judgment("I am uncertain.", n_panels=6)
    assert parsed["matched_panel"] is None
    assert parsed["is_junk"] is True
    assert parsed["parse_failed"] is True


def test_parse_kimi_single_token_classes() -> None:
    panel = judge.parse_panel_judgment("4", n_panels=6)
    junk = judge.parse_panel_judgment("0", n_panels=6)

    assert panel["matched_panel"] == 4
    assert panel["parse_failed"] is False
    assert junk["matched_panel"] is None
    assert junk["is_junk"] is True
    assert junk["parse_failed"] is False


def test_judge_row_with_kimi_uses_image_and_forced_choice() -> None:
    row = {
        "strip_name": "comic1",
        "condition": "gaze_top1",
        "target_panel": 3,
        "generated_text": "A dog sits beside a fire.",
        "baseline_text": "A person opens a door.",
    }
    generator = _FakeGenerator(
        '{"matched_panel": 3, "is_junk": false, "reasoning": "fire scene"}'
    )
    image = object()

    result = judge.judge_row_with_kimi(generator, row, image, n_panels=6)

    assert result["correct"] is True
    assert result["parse_failed"] is False
    assert generator.images == [image]
    assert "Answer to judge" in generator.prompts[0]
    assert "Baseline answer" not in generator.prompts[0]
    assert row["baseline_text"] not in generator.prompts[0]
    assert "large visible number" in generator.prompts[0]


def test_judge_row_with_kimi_skips_empty_output() -> None:
    row = {
        "strip_name": "comic1",
        "condition": "gaze_top1",
        "target_panel": 3,
        "generated_text": "  ",
    }
    generator = _FakeGenerator("unused")

    result = judge.judge_row_with_kimi(generator, row, object(), n_panels=6)

    assert result["correct"] is False
    assert result["is_junk"] is True
    assert generator.prompts == []


def test_judge_rows_with_kimi_batches_model_calls() -> None:
    rows = [
        {
            "strip_name": f"comic{panel}",
            "condition": "gaze_top1",
            "target_panel": panel,
            "generated_text": f"Description of panel {panel}",
            "baseline_text": "Different baseline text",
        }
        for panel in (1, 2, 3, 4)
    ]
    generator = _FakeBatchGenerator(["1", "2", "3", "4"])
    images = [object() for _ in rows]

    results = judge.judge_rows_with_kimi(
        generator, rows, images, n_panels=6
    )

    assert all(result["correct"] for result in results)
    assert generator.batch_sizes == [4]


def test_kimi_judge_writes_complete_valid_artifacts(tmp_path: Path) -> None:
    comic_dir = tmp_path / "comic1"
    comic_dir.mkdir()
    for panel in range(1, 7):
        Image.new("RGB", (20, 20), (panel * 20, 0, 0)).save(comic_dir / f"p{panel}.png")
    generations = tmp_path / "generations.jsonl"
    generations.write_text(
        json.dumps(
            {
                "strip_name": "comic1",
                "comic_dir": str(comic_dir),
                "condition": "gaze_top1",
                "target_panel": 3,
                "generated_text": "A dog sits beside a fire.",
                "baseline_text": "A person opens a door.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = judge.judge_generations_kimi(
        generations_path=generations,
        out_dir=tmp_path / "judge",
        generator=_FakeGenerator(
            '{"matched_panel": 3, "is_junk": false, "reasoning": "fire scene"}'
        ),
    )

    assert result["valid"] is True
    assert result["n_rows"] == 1
    aggregate = json.loads((tmp_path / "judge" / "aggregate_results.json").read_text())
    assert aggregate["aggregate"]["gaze_top1"]["overall"]["accuracy"] == 1.0


def test_resolve_local_snapshot_uses_configured_hub_cache(
    tmp_path: Path, monkeypatch
) -> None:
    revision = "abc123"
    snapshot = (
        tmp_path
        / "models--moonshotai--Kimi-VL-A3B-Instruct"
        / "snapshots"
        / revision
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "wrong-cache"))

    resolved = judge.resolve_local_snapshot(
        "moonshotai/Kimi-VL-A3B-Instruct", revision
    )

    assert resolved == os.fspath(snapshot)


def test_representative_limit_balances_conditions_and_panels() -> None:
    rows = [
        {
            "strip_name": f"comic{comic}",
            "condition": condition,
            "target_panel": panel,
        }
        for comic in range(20)
        for condition in ("gaze_top100", "non_gaze_paper_100")
        for panel in range(1, 7)
    ]

    selected = judge.select_representative_rows(rows, limit=24, seed=42)

    counts: dict[tuple[str, int], int] = {}
    for row in selected:
        key = (row["condition"], row["target_panel"])
        counts[key] = counts.get(key, 0) + 1
    assert len(selected) == 24
    assert set(counts.values()) == {2}
    assert len({row["strip_name"] for row in selected}) > 2


class _FakeGenerator:
    def __init__(self, response: str) -> None:
        self.response = response
        self.images: list[object] = []
        self.prompts: list[str] = []

    def generate(self, image: object, prompt: str) -> str:
        self.images.append(image)
        self.prompts.append(prompt)
        return self.response


class _FakeBatchGenerator:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.batch_sizes: list[int] = []

    def generate_many(self, images: list[object], prompts: list[str]) -> list[str]:
        assert len(images) == len(prompts)
        self.batch_sizes.append(len(images))
        return self.responses
