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
    assert "slurm_neuronic_qwen3_judge_static.sh" in output
    assert "slurm_neuronic_qwen3_static_full.sh" not in output
    assert "slurm_neuronic_qwen3_merge_static.sh" not in output
    assert "--array=0-3%2" in output
