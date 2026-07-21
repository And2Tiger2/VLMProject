import json
from pathlib import Path

from scripts.validate_qwen3_gaze_stage import validate_benchmark, validate_static


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_benchmark_validation_requires_boosted_head_telemetry(tmp_path: Path) -> None:
    for benchmark in ("vlmbias", "naturalbench"):
        _write_json(
            tmp_path / benchmark / "baseline.summary.json",
            {"run_config": {"condition": "baseline", "seed": 0}},
        )
        _write_json(
            tmp_path / benchmark / "boost.summary.json",
            {
                "run_config": {"condition": "gaze_top10_alpha1", "seed": 0},
                "mean_boosted_head_image_attention_mass": 0.8,
            },
        )
    assert validate_benchmark(tmp_path)["valid"] is True

    broken = tmp_path / "naturalbench" / "boost.summary.json"
    _write_json(broken, {"run_config": {"condition": "gaze_top10_alpha1", "seed": 0}})
    report = validate_benchmark(tmp_path)
    assert report["valid"] is False
    assert any("telemetry" in error for error in report["errors"])


def test_static_validation_rejects_duplicate_keys(tmp_path: Path) -> None:
    row = {
        "strip_name": "comic1",
        "condition": "gaze_top10",
        "target_panel": 1,
        "generated_text": "A person runs.",
        "metadata": {"attention": {"mean_decode_panel_1_attention_mass": 1.0}},
    }
    (tmp_path / "generations.jsonl").write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
    )
    report = validate_static(tmp_path, max_empty_rate=0.05, min_target_mass=0.95)
    assert report["valid"] is False
    assert report["duplicate_count"] == 1


def test_static_validation_rejects_incomplete_configured_run(tmp_path: Path) -> None:
    row = {
        "strip_name": "comic1",
        "condition": "gaze_top10",
        "target_panel": 1,
        "generated_text": "A person runs.",
        "metadata": {"attention": {"mean_decode_panel_1_attention_mass": 1.0}},
    }
    (tmp_path / "generations.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    _write_json(
        tmp_path / "experiment_config.json",
        {"targets_per_strip": 6, "include_all_heads": False},
    )
    _write_json(tmp_path / "summary.json", {"n_comics": 1})

    report = validate_static(tmp_path, max_empty_rate=0.05, min_target_mass=0.95)
    assert report["valid"] is False
    assert report["expected_rows"] == 12
    assert any("row count" in error for error in report["errors"])
