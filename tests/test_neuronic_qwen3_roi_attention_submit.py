from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "submit_neuronic_qwen3_roi_attention.py"
WRAPPER = ROOT / "scripts" / "run_neuronic_qwen3_roi_attention.sh"
BUNDLE = ROOT / "segments/vlm_bias_attention/assets/vlmbias_roi_masks_v1_runtime.tar.gz"


def _run(*args: str) -> str:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *args,
            "--repo",
            str(ROOT),
            "--dry-run",
            "--skip-preflight",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def test_full_submission_is_one_afterok_chain() -> None:
    output = _run("full")
    assert output.count("scripts/slurm_neuronic_qwen3_roi_prepare.sh") == 8
    assert output.count("scripts/slurm_neuronic_qwen3_roi_condition.sh") == 8
    assert output.count("scripts/slurm_neuronic_qwen3_roi_aggregate.sh") == 8
    assert "--array 0-3%2" in output
    assert "--array 0-6%4" in output
    assert "--array 0-11%4" in output
    assert "--array 0-4%4" in output
    assert "STAGE=smoke" in output
    assert "STAGE=tune" in output
    assert "STAGE=heads" in output
    assert "STAGE=confirm" in output
    assert "--dependency afterok:DRYRUN11" in output
    assert '"final_job": "DRYRUN12"' in output


def test_single_stage_has_expected_array_size() -> None:
    output = _run("heads", "--max-parallel", "3")
    assert "--array 0-11%3" in output
    assert '"final_job": "DRYRUN3"' in output


def test_wrapper_bootstraps_all_runtime_masks_from_tracked_bundle(tmp_path) -> None:
    destination = (
        tmp_path
        / "segments/vlm_bias_attention/assets/vlmbias_roi_masks_v1_runtime.tar.gz"
    )
    destination.parent.mkdir(parents=True)
    shutil.copyfile(BUNDLE, destination)
    result = subprocess.run(
        ["bash", str(WRAPPER), "prepare-inputs"],
        cwd=ROOT,
        env={**os.environ, "REPO": str(tmp_path)},
        check=True,
        text=True,
        capture_output=True,
    )
    roi_root = tmp_path / "segments/vlm_bias_attention/data/vlmbias_roi_masks_v1"
    assert "Prepared ROI runtime inputs: 141 masks" in result.stdout
    assert len((roi_root / "accepted.jsonl").read_text().splitlines()) == 141
    assert len(list((roi_root / "masks").glob("*.png"))) == 141
