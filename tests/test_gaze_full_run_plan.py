from __future__ import annotations

from scripts.plan_gaze_full_run import build_commands, render_shell, shard_suffixes


def test_shard_suffixes_handles_partial_final_shard() -> None:
    assert shard_suffixes(10, 4) == [
        (0, 4, "_0_4"),
        (4, 4, "_4_4"),
        (8, 2, "_8_2"),
    ]


def test_build_full_run_plan_includes_shards_merges_and_final_scoring() -> None:
    commands = build_commands(
        segment_root="segments/gaze_heads_qwen25",
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        device_map="auto",
        total_comics=10,
        shard_size=4,
        n_samples=10,
        top_k_gaze=100,
        top_k_random=100,
        targets_per_strip=1,
        judge="anthropic",
        include_all_heads=True,
        include_download=False,
        include_discover=True,
        resume=True,
    )

    command_text = "\n".join(" ".join(command) for command in commands)
    assert "--stages discover" in command_text
    assert "--run-suffix _0_4" in command_text
    assert "--run-suffix _8_2" in command_text
    assert "--stage static --suffixes _0_4 _4_4 _8_2" in command_text
    assert "--stages score_static score_vqa score_dynamic report validate" in command_text
    assert "--judge anthropic --resume" in command_text
    assert "--include-all-heads" in command_text


def test_render_shell_quotes_commands() -> None:
    text = render_shell([["uv", "run", "python", "script.py", "--arg", "value with spaces"]])
    assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "'value with spaces'" in text
