import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adapters.qwen25_vl_gaze_attention import (
    Qwen25VLGazeAttentionAdapter,
    select_image_boost_heads,
)
from scripts.aggregate_qwen3_attention_methods import aggregate_stage
from scripts.run_qwen3_attention_method_condition import _generate
from vlm_eval.qwen3_attention_methods import (
    controller_conditions,
    head_conditions,
    prepare_stratified_splits,
    read_jsonl,
)
from vlm_eval.types import EvalExample


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_stratified_splits_are_exact_disjoint_and_deterministic(tmp_path: Path) -> None:
    vlmbias = tmp_path / "source" / "vlmbias.jsonl"
    naturalbench = tmp_path / "source" / "naturalbench.jsonl"
    _write_jsonl(
        vlmbias,
        [
            {
                "id": f"v{index}",
                "topic": f"topic{index % 4}",
                "image_path": f"images/{index}.png",
            }
            for index in range(20)
        ],
    )
    _write_jsonl(
        naturalbench,
        [
            {
                "id": f"n{index}",
                "question_type": "yes_no" if index % 2 else "multiple_choice",
                "source": f"source{index % 3}",
                "image_0_path": f"images/{index}_0.png",
                "image_1_path": f"images/{index}_1.png",
            }
            for index in range(12)
        ],
    )
    first = prepare_stratified_splits(
        vlmbias_path=vlmbias,
        naturalbench_path=naturalbench,
        out_dir=tmp_path / "splits",
        vlmbias_dev=5,
        naturalbench_dev=3,
        vlmbias_smoke=2,
        naturalbench_smoke=1,
        seed=9,
    )
    second = prepare_stratified_splits(
        vlmbias_path=vlmbias,
        naturalbench_path=naturalbench,
        out_dir=tmp_path / "splits_again",
        vlmbias_dev=5,
        naturalbench_dev=3,
        vlmbias_smoke=2,
        naturalbench_smoke=1,
        seed=9,
    )
    assert first["counts"] == {
        "smoke_vlmbias": 2,
        "dev_vlmbias": 5,
        "confirm_vlmbias": 15,
        "all_vlmbias": 20,
        "smoke_naturalbench": 1,
        "dev_naturalbench": 3,
        "confirm_naturalbench": 9,
        "all_naturalbench": 12,
    }
    dev = read_jsonl(Path(first["paths"]["dev_vlmbias"]))
    confirm = read_jsonl(Path(first["paths"]["confirm_vlmbias"]))
    assert {row["id"] for row in dev}.isdisjoint(
        {row["id"] for row in confirm}
    )
    assert [row["id"] for row in dev] == [
        row["id"] for row in read_jsonl(Path(second["paths"]["dev_vlmbias"]))
    ]
    assert all(Path(row["image_path"]).is_absolute() for row in dev)


def test_condition_families_and_head_sweep_are_explicit() -> None:
    controllers = controller_conditions()
    assert len(controllers) == 10
    assert [row["controller"] for row in controllers].count("target_mass") == 3
    assert [row["controller"] for row in controllers].count("confidence_gate") == 2
    heads = head_conditions(controllers[1])
    assert len(heads) == 10
    assert {row["head_selection"] for row in heads} >= {
        "gaze_global",
        "gaze_early",
        "gaze_middle",
        "gaze_late",
        "layer_matched_random",
        "layer_matched_low",
        "paper_random",
    }


def test_head_controls_are_exact_reproducible_and_layer_matched(
    tmp_path: Path,
) -> None:
    ranking_path = tmp_path / "gaze_head_ranking.json"
    ranking = [
        {"layer": layer, "head": head, "score": 1.0 / (1 + layer + head)}
        for head in range(6)
        for layer in range(4)
    ]
    ranking_path.write_text(json.dumps(ranking), encoding="utf-8")
    np.save(
        ranking_path.with_name("gaze_scores.npy"),
        np.arange(24, dtype=float).reshape(4, 6),
    )
    global_heads = select_image_boost_heads(
        ranking_path=ranking_path,
        n_select=8,
        selection="gaze_global",
        seed=0,
        n_layers=4,
        n_heads=6,
    )
    random_a = select_image_boost_heads(
        ranking_path=ranking_path,
        n_select=8,
        selection="layer_matched_random",
        seed=55,
        n_layers=4,
        n_heads=6,
    )
    random_b = select_image_boost_heads(
        ranking_path=ranking_path,
        n_select=8,
        selection="layer_matched_random",
        seed=55,
        n_layers=4,
        n_heads=6,
    )
    low = select_image_boost_heads(
        ranking_path=ranking_path,
        n_select=8,
        selection="layer_matched_low",
        seed=0,
        n_layers=4,
        n_heads=6,
    )
    assert random_a == random_b
    assert len(set(random_a)) == len(set(low)) == 8
    assert _layer_counts(global_heads) == _layer_counts(random_a)
    assert _layer_counts(global_heads) == _layer_counts(low)
    assert set(global_heads).isdisjoint(random_a)
    assert set(global_heads).isdisjoint(low)


def test_controller_aggregation_locks_only_qualified_nonbaseline(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    vlmbias = data / "v.jsonl"
    naturalbench = data / "n.jsonl"
    _write_jsonl(vlmbias, [{"id": f"v{i}"} for i in range(4)])
    _write_jsonl(naturalbench, [{"id": f"n{i}"} for i in range(2)])
    conditions = [
        {
            **controller_conditions()[0],
            "seed": 0,
            "do_sample": False,
            "temperature": None,
            "split": "dev",
            "vlmbias_dataset": str(vlmbias),
            "naturalbench_dataset": str(naturalbench),
        },
        {
            **controller_conditions()[1],
            "seed": 0,
            "do_sample": False,
            "temperature": None,
            "split": "dev",
            "vlmbias_dataset": str(vlmbias),
            "naturalbench_dataset": str(naturalbench),
        },
        {
            **controller_conditions()[5],
            "seed": 0,
            "do_sample": False,
            "temperature": None,
            "split": "dev",
            "vlmbias_dataset": str(vlmbias),
            "naturalbench_dataset": str(naturalbench),
        },
        {
            **controller_conditions()[8],
            "seed": 0,
            "do_sample": False,
            "temperature": None,
            "split": "dev",
            "vlmbias_dataset": str(vlmbias),
            "naturalbench_dataset": str(naturalbench),
        },
    ]
    manifest = tmp_path / "experiment" / "controller_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"stage": "controller", "conditions": conditions}),
        encoding="utf-8",
    )
    run_root = tmp_path / "runs"
    metrics = [
        (0.75, 0.30, 0.75, 0.50),
        (0.75, 0.20, 0.75, 0.50),
        (0.60, 0.05, 0.75, 0.50),
        (0.75, 0.25, 0.75, 0.50),
    ]
    for condition, values in zip(conditions, metrics):
        summary = _summary(condition, *values)
        path = run_root / "controller" / condition["name"] / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(summary), encoding="utf-8")
    report = aggregate_stage(
        stage="controller",
        manifest_path=manifest,
        run_root=run_root,
        report_root=tmp_path / "reports",
    )
    assert report["valid"]
    assert report["selection"]["selected_overall"]["name"] == "fixed_alpha0p5"
    target = next(
        item
        for item in report["selection"]["evaluated"]
        if item["spec"]["controller"] == "target_mass"
    )
    assert not target["qualified"]


def test_confidence_gate_runs_second_pass_only_below_threshold() -> None:
    condition = {
        "controller": "confidence_gate",
        "confidence_threshold": 0.6,
        "alpha": 2.0,
    }
    low = _FakeAdapter(confidence=0.3)
    response, metadata = _generate(
        low, EvalExample(id="low", prompt="question", ground_truth="A"), condition
    )
    assert response == "boosted"
    assert low.alphas == [0.0, 2.0]
    assert metadata["confidence_gate"]["intervened"]
    assert metadata["confidence_gate"]["baseline_response"] == "baseline"

    high = _FakeAdapter(confidence=0.8)
    response, metadata = _generate(
        high, EvalExample(id="high", prompt="question", ground_truth="A"), condition
    )
    assert response == "baseline"
    assert high.alphas == [0.0]
    assert not metadata["confidence_gate"]["intervened"]


def test_generated_token_confidence_uses_chosen_token_probabilities() -> None:
    adapter = object.__new__(Qwen25VLGazeAttentionAdapter)
    adapter.record_token_confidence = True
    adapter._torch = torch
    adapter._token_confidence_metadata = {}
    probabilities = [
        torch.tensor([[0.1, 0.8, 0.1]]).log(),
        torch.tensor([[0.5, 0.25, 0.25]]).log(),
    ]
    generated = SimpleNamespace(
        sequences=torch.tensor([[9, 9, 1, 0]]),
        scores=probabilities,
    )
    inputs = SimpleNamespace(input_ids=torch.tensor([[9, 9]]))
    adapter._after_generate_output(generated, inputs)
    confidence = adapter._token_confidence_metadata
    assert confidence["n_tokens"] == 2
    assert confidence["first_token_probability"] == pytest.approx(0.8)
    assert confidence["minimum_token_probability"] == pytest.approx(0.5)
    assert confidence["geometric_mean_probability"] == pytest.approx(
        (0.8 * 0.5) ** 0.5
    )


def _summary(
    condition: dict,
    vlmbias_accuracy: float,
    bias_fraction: float,
    naturalbench_acc: float,
    naturalbench_g_acc: float,
) -> dict:
    head_count = condition["head_count"]
    attention = {
        "mean_boosted_head_image_attention_mass": 0.5 if head_count else None,
        "mean_effective_alpha": (
            1.0 if condition["controller"] == "target_mass" else None
        ),
        "confidence_gate_intervention_rate": (
            0.5 if condition["controller"] == "confidence_gate" else None
        ),
    }
    return {
        "valid": True,
        "stage": "controller",
        "condition": condition,
        "selected_heads": [[index // 32, index % 32] for index in range(head_count)],
        "vlmbias": {
            "n": 4,
            "n_unique": 4,
            "duplicate_count": 0,
            "accuracy": vlmbias_accuracy,
            "bias_aligned_fraction": bias_fraction,
            "bias_aligned_error_rate": bias_fraction,
            "invalid_rate": 0.0,
        },
        "naturalbench": {
            "n_groups": 2,
            "n_model_calls": 8,
            "n_unique_calls": 8,
            "duplicate_count": 0,
            "Acc": naturalbench_acc,
            "G_Acc": naturalbench_g_acc,
            "invalid_rate": 0.0,
        },
        "telemetry": {"vlmbias": attention, "naturalbench": attention},
    }


def _layer_counts(heads: list[tuple[int, int]]) -> dict[int, int]:
    output = {}
    for layer, _ in heads:
        output[layer] = output.get(layer, 0) + 1
    return output


class _FakeAdapter:
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence
        self.attention_alpha = 0.0
        self.attention_controller = "fixed"
        self.last_generation_metadata = None
        self.alphas: list[float] = []

    def _configure_attention_modules(self) -> None:
        return None

    def generate(self, example: EvalExample) -> str:
        self.alphas.append(self.attention_alpha)
        self.last_generation_metadata = {
            "token_confidence": {
                "geometric_mean_probability": self.confidence
            }
        }
        return "boosted" if self.attention_alpha else "baseline"
