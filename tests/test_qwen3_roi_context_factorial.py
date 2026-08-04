from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from adapters.qwen3_vl_roi_attention import Qwen3VLROIAttentionAdapter
from scripts.aggregate_qwen3_roi_context_factorial import _comparisons
from vlm_eval.qwen3_roi_context_factorial import factorial_conditions


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts/submit_neuronic_qwen3_roi_context_factorial.py"


def test_factorial_is_exact_two_by_two_with_tight_roi_and_gaze50() -> None:
    rows = factorial_conditions()
    assert len(rows) == 4
    assert {(row["suppress_roi"], row["boost_context"]) for row in rows} == {
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    }
    assert all(row["mask_variant"] == "tight" for row in rows)
    assert all(row["head_selection"] == "gaze_global" for row in rows)
    assert all(row["head_count"] == 50 for row in rows)
    assert {row["roi_attention_bias"] for row in rows} == {0.0, -5.0}
    assert {row["context_attention_bias"] for row in rows} == {0.0, 5.0}


def test_roi_adapter_installs_complementary_signed_token_bias() -> None:
    class Inputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

    layers = [SimpleNamespace(self_attn=SimpleNamespace()) for _ in range(2)]
    adapter = object.__new__(Qwen3VLROIAttentionAdapter)
    adapter._torch = torch
    adapter._model = SimpleNamespace(
        config=SimpleNamespace(image_token_id=99),
        model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)),
    )
    adapter._processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
    adapter._attention_records = []
    adapter._token_confidence_metadata = {}
    adapter.last_generation_metadata = None
    array = np.zeros((40, 40), dtype=np.uint8)
    array[:20, :20] = 255
    adapter._pending_roi_mask = Image.fromarray(array)
    adapter._attention_region = "roi"
    adapter._roi_token_metadata = {}
    adapter.roi_token_min_coverage = 0.05
    adapter.roi_attention_bias = -5.0
    adapter.context_attention_bias = 5.0
    inputs = Inputs(
        input_ids=torch.tensor([[10, 99, 99, 99, 99, 11]]),
        image_grid_thw=torch.tensor([[1, 4, 4]]),
    )

    adapter._before_generate(inputs, include_image=True)

    expected_roi = torch.tensor([False, True, False, False, False, False])
    expected_context = torch.tensor([False, False, True, True, True, False])
    expected_bias = torch.tensor([0.0, -5.0, 5.0, 5.0, 5.0, 0.0])
    for layer in layers:
        assert torch.equal(layer.self_attn._vlm_gaze_image_token_mask, expected_roi)
        assert torch.equal(
            layer.self_attn._vlm_gaze_image_context_token_mask, expected_context
        )
        assert torch.equal(layer.self_attn._vlm_gaze_image_token_bias, expected_bias)
    assert adapter._roi_token_metadata["n_target_tokens"] == 1
    assert adapter._roi_token_metadata["n_context_tokens"] == 3


def test_submission_runs_four_conditions_concurrently() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SUBMIT),
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
    assert "--array 0-3%4" in result.stdout
    assert "--dependency afterok:DRYRUN1" in result.stdout
    assert "--dependency afterok:DRYRUN2" in result.stdout
    assert '"final_job": "DRYRUN3"' in result.stdout


def test_factorial_effects_use_standard_two_by_two_contrasts() -> None:
    outcomes = {
        (False, False): (0.20, 0.40, 0.50, 0.30),
        (True, False): (0.30, 0.20, 0.25, 0.20),
        (False, True): (0.25, 0.30, 0.40, 0.25),
        (True, True): (0.40, 0.10, 0.15, 0.10),
    }
    rows = []
    for index, ((suppress, context), values) in enumerate(outcomes.items()):
        accuracy, bias, bias_error, invalid = values
        rows.append(
            {
                "condition": {
                    "name": f"condition_{index}",
                    "suppress_roi": suppress,
                    "boost_context": context,
                },
                "vlmbias": {
                    "n": 100,
                    "n_bias_aligned_errors": round(bias * 100),
                    "accuracy": accuracy,
                    "bias_aligned_fraction": bias,
                    "bias_aligned_error_rate": bias_error,
                    "invalid_rate": invalid,
                    "by_topic": {},
                },
                "token_mask_telemetry": {
                    "mean_target_token_fraction": 0.2,
                    "mean_context_token_fraction": 0.8,
                },
                "attention_telemetry": {
                    "mean_boosted_head_image_attention_mass": 0.1,
                    "mean_boosted_head_context_attention_mass": 0.2,
                },
            }
        )
    effects = _comparisons(rows)["factorial_effects"]
    assert effects["accuracy"]["roi_suppression_main_effect"] == pytest.approx(0.125)
    assert effects["accuracy"]["context_boost_main_effect"] == pytest.approx(0.075)
    assert effects["accuracy"]["interaction"] == pytest.approx(0.05)
    assert effects["bias_aligned_fraction"][
        "roi_suppression_main_effect"
    ] == pytest.approx(-0.2)
    assert effects["bias_aligned_fraction"][
        "context_boost_main_effect"
    ] == pytest.approx(-0.1)
