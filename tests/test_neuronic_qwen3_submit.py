from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "submit_neuronic_qwen3.py"


def _run(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "--skip-preflight", *args],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def test_discovery_seed_count_expands_gpu_and_merge_arrays() -> None:
    output = _run("discovery", "--seeds", "3", "--shards", "10")
    assert "--array=0-29" in output
    assert "--array=0-2%3" in output
    assert "--dependency=afterok:DRYRUN1" in output


def test_static_seed_count_expands_seed_topk_shard_product() -> None:
    output = _run(
        "static", "--seeds", "2", "--shards", "3", "--top-ks", "10", "100", "--judge", "none"
    )
    assert "--array=0-11" in output
    assert "--array=0-1%2" in output
    assert "TOP_KS_COLON=10:100" in output


def test_benchmark_seed_count_and_smoke_output_are_automatic() -> None:
    output = _run("benchmark", "--seeds", "5", "--limit", "20", "--naturalbench-limit-groups", "5")
    assert "--array=0-4" in output
    assert "OUT_DIR=segments/gaze_heads_qwen3_8b/runs/benchmark_attention_smoke" in output
    assert "--dependency=afterok:DRYRUN1" in output


def test_static_judge_only_submits_no_gpu_or_merge_jobs() -> None:
    output = _run("static", "--seeds", "2", "--top-ks", "10", "100", "--judge-only")
    assert "slurm_neuronic_qwen3_judge_static_kimi.sh" in output
    assert "slurm_neuronic_qwen3_static_full.sh" not in output
    assert "slurm_neuronic_qwen3_merge_static.sh" not in output
    assert "--array=0-3%2" in output
    assert "KIMI_MIN_GPU_MEMORY_GB=40.0" in output


def test_static_kimi_smoke_uses_separate_limit() -> None:
    output = _run(
        "static", "--top-ks", "1", "--judge-only", "--judge-limit", "24"
    )
    assert "JUDGE_LIMIT=24" in output
    assert "--array=0-0%2" in output


def test_static_merge_only_submits_cpu_merge_without_gpu_or_judge() -> None:
    output = _run(
        "static", "--seeds", "3", "--top-ks", "1", "10", "50", "100", "--merge-only"
    )
    assert "slurm_neuronic_qwen3_merge_static.sh" in output
    assert "slurm_neuronic_qwen3_static_full.sh" not in output
    assert "slurm_neuronic_qwen3_judge_static_kimi.sh" not in output
    assert "--array=0-2%3" in output


def test_static_paper_control_pilot_reruns_only_control_then_assembles() -> None:
    output = _run(
        "static-paper-control", "--seeds", "1", "--shards", "1", "--shard-size", "50"
    )
    assert "slurm_neuronic_qwen3_static_paper_control.sh" in output
    assert "slurm_neuronic_qwen3_assemble_paper_control.sh" in output
    assert "slurm_neuronic_qwen3_static_full.sh" not in output
    assert "slurm_neuronic_qwen3_judge_static_kimi.sh" not in output
    assert "TOP_KS_COLON=100" in output
    assert "--array=0-0" in output


def test_static_paper_control_full_has_thirty_gpu_tasks() -> None:
    output = _run(
        "static-paper-control", "--seeds", "3", "--shards", "10", "--shard-size", "50"
    )
    assert "--array=0-29" in output
    assert "--array=0-2%3" in output
    assert "SOURCE_GAZE_COMICS=500" in output


def test_static_paper_control_judge_only_submits_no_qwen_or_assembly() -> None:
    output = _run(
        "static-paper-control",
        "--seeds",
        "3",
        "--shards",
        "10",
        "--judge",
        "kimi",
        "--judge-only",
        "--judge-limit",
        "120",
    )
    assert "slurm_neuronic_qwen3_judge_static_kimi.sh" in output
    assert "slurm_neuronic_qwen3_static_paper_control.sh" not in output
    assert "slurm_neuronic_qwen3_assemble_paper_control.sh" not in output
    assert "--array=0-2%2" in output
    assert "JUDGE_LIMIT=120" in output


def test_control_distribution_submits_complete_dependency_chain() -> None:
    output = _run("control-distribution")
    assert "slurm_neuronic_qwen3_control_distribution.sh" in output
    assert "slurm_neuronic_qwen3_merge_control_distribution.sh" in output
    assert "slurm_neuronic_qwen3_judge_control_distribution_kimi.sh" in output
    assert "slurm_neuronic_qwen3_aggregate_control_distribution.sh" in output
    assert "--array=0-41%8" in output
    assert "--array=0-20%21" in output
    assert "--array=0-20%2" in output
    assert "paper@45:paper@46" in output
    assert "layer_matched_random@45" in output
    assert "layer_matched_low@42" in output
    assert "--dependency=afterok:DRYRUN3" in output
