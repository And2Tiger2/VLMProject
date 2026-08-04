from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import sys

from vlm_eval.qwen3_high_bias_roi_attention import (
    canonical_group_id,
    confirm_conditions,
    head_conditions,
    smoke_conditions,
    tune_conditions,
)


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts" / "submit_neuronic_qwen3_high_bias_roi_attention.py"
WRAPPER = ROOT / "scripts" / "run_neuronic_qwen3_high_bias_roi_attention.sh"
BUNDLE = ROOT / "segments/vlm_bias_attention/assets/vlmbias_high_bias_roi_masks_v1_runtime.tar.gz"


def test_conditions_cover_both_mask_variants_and_controls() -> None:
    assert len(smoke_conditions()) == 6
    tune = tune_conditions()
    assert len(tune) == 12
    roi_sweep = [
        row
        for row in tune
        if row["region"] == "roi" and row["head_selection"] == "gaze_global"
    ]
    assert {(row["mask_variant"], row["alpha"]) for row in roi_sweep} == {
        (variant, alpha)
        for variant in ("tight", "broad")
        for alpha in (0.5, 1.0, 2.0, 5.0)
    }
    heads = head_conditions(5.0, "broad")
    assert len(heads) == 12
    assert all(row["mask_variant"] == "broad" for row in heads)
    confirm = confirm_conditions(tune[1], heads[1])
    assert len(confirm) == 6
    assert {row["mask_variant"] for row in confirm} == {"tight", "broad"}
    assert any(row["head_selection"] == "layer_matched_random" for row in confirm)


def test_canonical_group_collapses_prompt_and_resolution_variants() -> None:
    assert canonical_group_id("Poggendorff_011_Q2_notitle_px384") == "Poggendorff_011_notitle"
    assert (
        canonical_group_id("chess_grid_04_row_add_last_row_notitle_px768_Q2")
        == "chess_grid_04_row_add_last_row_notitle"
    )


def test_full_submission_uses_parallel_arrays_in_one_dependency_chain() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SUBMIT),
            "full",
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
    output = result.stdout
    assert "--array 0-5%2" in output
    assert output.count("--array 0-11%4") == 4
    assert "--array 0-5%4" in output
    assert "--dependency afterok:DRYRUN11" in output
    assert '"final_job": "DRYRUN12"' in output


def test_wrapper_extracts_reviewed_tight_and_broad_masks(tmp_path) -> None:
    destination = (
        tmp_path
        / "segments/vlm_bias_attention/assets/vlmbias_high_bias_roi_masks_v1_runtime.tar.gz"
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
    roi_root = tmp_path / "segments/vlm_bias_attention/data/vlmbias_high_bias_roi_masks_v1"
    assert "Prepared high-bias ROI runtime inputs" in result.stdout
    assert len((roi_root / "accepted.jsonl").read_text().splitlines()) == 114
    assert len(
        (roi_root / "dataset/vlmbias_high_bias_114.jsonl").read_text().splitlines()
    ) == 114
    assert len(list((roi_root / "dataset/images").glob("*.png"))) == 114
    assert len(list((roi_root / "masks_tight").glob("*.png"))) == 114
    assert len(list((roi_root / "masks_broad").glob("*.png"))) == 114
