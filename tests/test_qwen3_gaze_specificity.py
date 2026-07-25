from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_qwen3_gaze_specificity import aggregate_stage
from vlm_eval.qwen3_attention_methods import write_condition_manifest
from vlm_eval.qwen3_gaze_specificity import (
    STAGE_COUNTS,
    control_conditions,
    final_conditions,
    tune_conditions,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_specificity_design_has_preregistered_condition_counts() -> None:
    controls = control_conditions()
    tune = tune_conditions()
    assert len(controls) == STAGE_COUNTS["controls"] == 33
    assert len(tune) == STAGE_COUNTS["tune"] == 21
    assert (
        sum(row["head_selection"] == "layer_matched_random" for row in controls) == 20
    )
    assert sum(row["head_selection"] == "paper_random" for row in controls) == 10
    gates = [row for row in tune if row["controller"] == "confidence_gate"]
    assert len(gates) == 12
    assert {row["confidence_threshold"] for row in gates} == {
        0.7,
        0.8,
        0.85,
        0.9,
    }
    assert {row["alpha"] for row in gates} == {0.5, 1.0, 2.0}


def test_final_conditions_lock_one_candidate_per_family() -> None:
    selected = {}
    for condition in tune_conditions():
        family = (
            "fixed" if condition["controller"] == "fixed" else condition["controller"]
        )
        if (
            family in {"fixed", "target_mass", "confidence_gate"}
            and condition["name"] != "baseline"
        ):
            selected.setdefault(family, {"spec": condition})
    final = final_conditions({"selected_by_family": selected})
    assert len(final) == STAGE_COUNTS["final"] == 5
    assert final[0]["name"] == "baseline"
    assert final[1]["name"] == "anchor_gaze_top50_fixed_alpha0p5"
    assert {row["controller"] for row in final[2:]} == {
        "fixed",
        "target_mass",
        "confidence_gate",
    }
    assert all(row["head_count"] == 50 for row in final[1:])


def test_control_distribution_uses_layer_matched_draws_and_add_one_p(
    tmp_path: Path,
) -> None:
    vlmbias = tmp_path / "data" / "confirm_v.jsonl"
    naturalbench = tmp_path / "data" / "confirm_n.jsonl"
    _write_jsonl(vlmbias, [{"id": "v0"}, {"id": "v1"}])
    _write_jsonl(naturalbench, [{"id": "n0"}])
    split_manifest = {
        "paths": {
            "confirm_vlmbias": str(vlmbias),
            "confirm_naturalbench": str(naturalbench),
        }
    }
    split_path = tmp_path / "experiment" / "split_manifest.json"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps(split_manifest), encoding="utf-8")
    manifest_path = tmp_path / "experiment" / "controls_manifest.json"
    manifest = write_condition_manifest(
        stage="controls",
        conditions=control_conditions(),
        split_manifest=split_manifest,
        out_path=manifest_path,
        split="confirm",
    )
    manifest["split_manifest"] = str(split_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    run_root = tmp_path / "runs"
    gaze_heads = [[0, index] for index in range(50)]
    for condition_index, condition in enumerate(manifest["conditions"]):
        name = condition["name"]
        if name == "baseline":
            heads = []
            correct = [False, False]
            bias = [False, True]
        elif name == "gaze_top50_alpha0p5":
            heads = gaze_heads
            correct = [True, True]
            bias = [False, False]
        else:
            # Exact layer histogram, distinct head IDs.
            heads = [[0, 1000 + condition_index * 100 + index] for index in range(50)]
            correct = [True, False]
            bias = [False, True]
        out = run_root / "controls" / name
        _write_jsonl(
            out / "vlmbias.jsonl",
            [
                {
                    "example_id": f"v{index}",
                    "is_correct": correct[index],
                    "is_bias_aligned_error": bias[index],
                    "parsed_answer": "A",
                }
                for index in range(2)
            ],
        )
        nb_correct = name == "gaze_top50_alpha0p5"
        _write_jsonl(
            out / "naturalbench.jsonl",
            [
                {
                    "group_id": "n0",
                    "call_id": call_id,
                    "is_correct": nb_correct,
                    "parsed_answer": "Yes",
                }
                for call_id in ("q0_i0", "q0_i1", "q1_i0", "q1_i1")
            ],
        )
        summary = _summary(
            condition,
            heads=heads,
            accuracy=sum(correct) / 2,
            bias_fraction=sum(bias) / 2,
            nb_acc=float(nb_correct),
            nb_g_acc=float(nb_correct),
        )
        (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    report = aggregate_stage(
        stage="controls",
        manifest_path=manifest_path,
        run_root=run_root,
        report_root=tmp_path / "reports",
        n_bootstrap=100,
    )
    assert report["valid"]
    tests = report["control_distribution"]["empirical_head_draw_tests"]
    assert tests["vlmbias_accuracy"]["one_sided_add_one_empirical_p"] == pytest.approx(
        1 / 21
    )
    assert tests["vlmbias_bias_aligned_fraction"][
        "one_sided_add_one_empirical_p"
    ] == pytest.approx(1 / 21)


def test_aggregator_rejects_stage_split_mismatch(tmp_path: Path) -> None:
    dev_v = tmp_path / "dev_v.jsonl"
    dev_n = tmp_path / "dev_n.jsonl"
    confirm_v = tmp_path / "confirm_v.jsonl"
    confirm_n = tmp_path / "confirm_n.jsonl"
    for path in (dev_v, dev_n, confirm_v, confirm_n):
        _write_jsonl(path, [{"id": path.stem}])
    split = {
        "paths": {
            "dev_vlmbias": str(dev_v),
            "dev_naturalbench": str(dev_n),
            "confirm_vlmbias": str(confirm_v),
            "confirm_naturalbench": str(confirm_n),
        }
    }
    split_path = tmp_path / "split_manifest.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    condition = {
        **control_conditions()[0],
        "seed": 0,
        "do_sample": False,
        "temperature": None,
        "split": "dev",
        "vlmbias_dataset": str(dev_v),
        "naturalbench_dataset": str(dev_n),
    }
    manifest = tmp_path / "repair_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "stage": "repair",
                "split_manifest": str(split_path),
                "conditions": [condition],
            }
        ),
        encoding="utf-8",
    )
    report = aggregate_stage(
        stage="repair",
        manifest_path=manifest,
        run_root=tmp_path / "runs",
        report_root=tmp_path / "reports",
        n_bootstrap=10,
    )
    assert not report["valid"]
    assert any("expected stage split 'confirm'" in error for error in report["errors"])


def _summary(
    condition: dict,
    *,
    heads: list[list[int]],
    accuracy: float,
    bias_fraction: float,
    nb_acc: float,
    nb_g_acc: float,
) -> dict:
    telemetry = {
        "mean_boosted_head_image_attention_mass": (
            0.2 if condition["head_count"] else None
        ),
        "mean_effective_alpha": (
            1.0 if condition["controller"] == "target_mass" else None
        ),
        "confidence_gate_intervention_rate": (
            0.5 if condition["controller"] == "confidence_gate" else None
        ),
    }
    return {
        "valid": True,
        "stage": "controls",
        "condition": condition,
        "selected_heads": heads,
        "vlmbias": {
            "n": 2,
            "n_unique": 2,
            "duplicate_count": 0,
            "accuracy": accuracy,
            "bias_aligned_fraction": bias_fraction,
            "bias_aligned_error_rate": bias_fraction,
            "invalid_rate": 0.0,
        },
        "naturalbench": {
            "n_groups": 1,
            "n_model_calls": 4,
            "n_unique_calls": 4,
            "duplicate_count": 0,
            "Acc": nb_acc,
            "G_Acc": nb_g_acc,
            "invalid_rate": 0.0,
        },
        "telemetry": {"vlmbias": telemetry, "naturalbench": telemetry},
    }
