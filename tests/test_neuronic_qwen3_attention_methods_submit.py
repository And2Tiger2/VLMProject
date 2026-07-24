from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "submit_neuronic_qwen3_attention_methods.py"


def _run(*args: str) -> str:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            str(ROOT),
            "--dry-run",
            "--skip-preflight",
            *args,
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def test_overnight_submits_every_stage_in_one_afterok_chain() -> None:
    output = _run("overnight")
    assert output.count("scripts/slurm_neuronic_qwen3_prepare_attention_methods.sh") == 10
    assert output.count("scripts/slurm_neuronic_qwen3_attention_method_condition.sh") == 10
    assert output.count("scripts/slurm_neuronic_qwen3_aggregate_attention_methods.sh") == 10
    assert "--array 0-3%2" in output
    assert "--array 0-9%8" in output
    assert "--array 0-3%4" in output
    assert "--array 0-11%4" in output
    assert "STAGE=smoke" in output
    assert "STAGE=controller" in output
    assert "STAGE=heads" in output
    assert "STAGE=confirm" in output
    assert "STAGE=robustness" in output
    assert "--dependency afterok:DRYRUN14" in output
    assert '"final_job": "DRYRUN15"' in output


def test_overnight_seed_count_expands_robustness_array() -> None:
    output = _run("overnight", "--seeds", "7", "8")
    assert "SEEDS_COLON=7:8" in output
    assert "--array 0-7%4" in output
