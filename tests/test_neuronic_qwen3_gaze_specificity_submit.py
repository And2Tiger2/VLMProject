from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "submit_neuronic_qwen3_gaze_specificity.py"


def _run(*args: str) -> str:
    return subprocess.run(
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
    ).stdout


def test_overnight_is_one_strict_five_stage_chain() -> None:
    output = _run("overnight")
    assert "--array 0-3%4" in output
    assert "--array 0-32%8" in output
    assert "--array 0-20%6" in output
    assert "--array 0-4%5" in output
    assert "--array 0-14%4" in output
    for stage in ("repair", "controls", "tune", "final", "robustness"):
        assert f"STAGE={stage}" in output
    assert "--dependency afterok:DRYRUN14" in output
    assert '"final_job": "DRYRUN15"' in output


def test_robustness_count_tracks_seed_count() -> None:
    output = _run("robustness", "--seeds", "7", "8")
    assert "SEEDS_COLON=7:8" in output
    assert "--array 0-9%4" in output
