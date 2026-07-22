from __future__ import annotations

import json
from pathlib import Path

from scripts.assemble_qwen3_static_paper_control import assemble_paper_control


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_assembly_reuses_gaze_and_accepts_only_exact_paper_control(tmp_path: Path) -> None:
    gaze_run = tmp_path / "gaze"
    control_run = tmp_path / "control"
    out = tmp_path / "paired"
    heads = [[20 + index % 16, index // 16] for index in range(100)]
    gaze_heads = [[0, 0], [1, 1]]
    config = {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "comics_root": "eval",
        "gaze_ranking": "ranking.json",
        "control_mode": "paper",
        "condition_set": "control",
        "condition_labels": ["non_gaze_paper_100"],
        "paper_control_layers": [20, 35],
        "selected_gaze_heads": gaze_heads,
        "selected_control_heads": heads,
        "max_new_tokens": 100,
        "swap_bias": 10000.0,
        "decode_only": False,
        "seed": 42,
        "prompt": "What is happening?",
        "git_commit": "abc",
    }
    _write_json(control_run / "experiment_config.json", config)
    gaze_rows = []
    control_rows = []
    for panel in range(1, 7):
        common = {"strip_name": "comic1", "target_panel": panel}
        gaze_rows.append({**common, "condition": "gaze_top100", "generated_text": "gaze"})
        control_rows.append(
            {**common, "condition": "non_gaze_paper_100", "generated_text": "control"}
        )
    _write_jsonl(gaze_run / "generations.jsonl", gaze_rows)
    _write_jsonl(control_run / "generations.jsonl", control_rows)

    result = assemble_paper_control(
        existing_gaze_run=gaze_run,
        control_shard_runs=[control_run],
        out_dir=out,
        top_k=100,
        expected_comics=1,
    )

    assert result["n_rows"] == 12
    rows = [json.loads(line) for line in (out / "generations.jsonl").read_text().splitlines()]
    assert {row["condition"] for row in rows} == {"gaze_top100", "non_gaze_paper_100"}
    assert len(json.loads((out / "experiment_config.json").read_text())["selected_control_heads"]) == 100
