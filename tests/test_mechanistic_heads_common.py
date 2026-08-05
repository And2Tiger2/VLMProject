from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vlm_eval.mechanistic_heads.ablations import image_attention_knockout
from vlm_eval.mechanistic_heads.atlas import initialize_head_atlas
from vlm_eval.mechanistic_heads.controls import (
    layer_matched_control_draws,
    multivariate_matched_control_draws,
)
from vlm_eval.mechanistic_heads.likelihood import (
    append_answer_tokens,
    score_answer_from_logits,
)
from vlm_eval.mechanistic_heads.patching import (
    batched_single_head_patches,
    patch_projected_head,
    projected_head_contributions,
    reconstruct_attention_output,
    transplant_attention_map,
)
from vlm_eval.mechanistic_heads.reproducibility import write_run_manifest
from vlm_eval.mechanistic_heads.schema import PairedExample, assert_no_group_leakage
from vlm_eval.mechanistic_heads.token_spans import trace_qwen3_token_spans
from vlm_eval.mechanistic_heads.token_spans import locate_subsequence
from vlm_eval.mechanistic_heads.checkpoint import JsonlCheckpoint
from vlm_eval.mechanistic_heads.causal import (
    batched_visual_attention_map_patch_many,
    projected_head_patch,
    projected_head_set_replacement,
    repeat_model_inputs,
)
from vlm_eval.mechanistic_heads.config import parse_layer_spec, prepare_output_directory
from vlm_eval.mechanistic_heads.capture import (
    MECHANISTIC_ATTENTION_IMPL,
    get_language_layers,
    register_qwen3_mechanistic_attention,
)
from vlm_eval.mechanistic_heads.qwen3_runtime import runtime_from_config


def test_projected_heads_reconstruct_attention_output_including_bias() -> None:
    generator = torch.Generator().manual_seed(3)
    raw = torch.randn(2, 5, 4, 3, generator=generator)
    weight = torch.randn(7, 12, generator=generator)
    bias = torch.randn(7, generator=generator)
    projected = projected_head_contributions(raw, weight)
    expected = torch.nn.functional.linear(raw.flatten(-2), weight, bias)
    actual = reconstruct_attention_output(projected, bias)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_identity_and_self_subtraction_patches_are_noops() -> None:
    raw = torch.randn(1, 3, 2, 4)
    weight = torch.randn(8, 8)
    projected = projected_head_contributions(raw, weight)
    output = projected.sum(-2)
    for head in range(2):
        patched = patch_projected_head(output, projected[..., head, :], projected[..., head, :])
        assert torch.equal(patched, output)


def test_batched_and_serial_single_head_patching_agree() -> None:
    recipient = torch.randn(1, 3, 6)
    donor_heads = torch.randn(1, 3, 4, 6)
    recipient_heads = torch.randn(1, 3, 4, 6)
    batched = batched_single_head_patches(recipient, donor_heads, recipient_heads)
    serial = torch.stack(
        [
            patch_projected_head(
                recipient, donor_heads[..., head, :], recipient_heads[..., head, :]
            )
            for head in range(4)
        ]
    )
    assert torch.equal(batched, serial)


def test_serial_projected_head_hook_supports_identical_microbatch_rows() -> None:
    attention = torch.nn.Identity()
    layer = SimpleNamespace(self_attn=attention)
    model = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(layers=[layer]))
    )
    current = torch.zeros(2, 3, 4)
    recipient = torch.zeros(1, 3, 2, 4)
    donor = recipient.clone()
    donor[:, 2, 1, :] = 3
    with projected_head_patch(
        model,
        layer_idx=0,
        head_idx=1,
        donor_projected=donor,
        recipient_projected=recipient,
        positions=[2],
    ):
        actual = attention(current)
    assert torch.equal(actual[0], actual[1])
    assert torch.equal(actual[:, 2, :], torch.full((2, 4), 3.0))


def test_sequence_log_probability_matches_manual_teacher_forcing() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [3.0, 1.0, 0.0],
                [0.0, 2.0, 1.0],
            ]
        ]
    )
    answers = torch.tensor([[2, 0]])
    score = score_answer_from_logits(logits, answers, prompt_length=2)
    expected = (
        torch.log_softmax(logits[0, 1], dim=-1)[2]
        + torch.log_softmax(logits[0, 2], dim=-1)[0]
    )
    assert score.total_log_probability.item() == pytest.approx(expected.item())


def test_answer_extension_keeps_qwen_multimodal_token_types_aligned() -> None:
    inputs = {
        "input_ids": torch.tensor([[10, 99, 99, 20]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[0, 1, 1, 0]]),
        "position_ids": torch.zeros(3, 1, 4, dtype=torch.long),
    }
    kwargs, answers, prompt_length = append_answer_tokens(
        inputs, torch.tensor([[7, 8]])
    )
    assert prompt_length == 4
    assert answers.tolist() == [[7, 8]]
    assert kwargs["input_ids"].shape == (1, 6)
    assert kwargs["attention_mask"].shape == (1, 6)
    assert kwargs["mm_token_type_ids"].tolist() == [[0, 1, 1, 0, 0, 0]]
    assert "position_ids" not in kwargs


def test_token_spans_are_asserted_and_complete() -> None:
    ids = [10, 11, 20, 99, 99, 99, 100, 30, 31, 40, 41, 7, 8]
    spans = trace_qwen3_token_spans(
        ids,
        image_token_id=99,
        image_end_token_id=100,
        system_end=2,
        user_start=7,
        final_prompt_start=9,
        prompt_length=11,
        answer_length=2,
    )
    assert spans.visual.indices() == [3, 4, 5]
    assert spans.image_end.indices() == [6]
    assert spans.generated_answer.indices() == [11, 12]


def test_attention_transplant_and_knockout_are_normalized() -> None:
    recipient = torch.tensor([[[[0.2, 0.3, 0.5]]]])
    donor = torch.tensor([[[[0.1, 0.7, 0.2]]]])
    full = transplant_attention_map(recipient, donor)
    assert torch.equal(full.probabilities, donor)
    sliced = transplant_attention_map(
        recipient,
        donor,
        recipient_key_indices=[0, 1],
        donor_key_indices=[1, 2],
    )
    assert sliced.probabilities.sum().item() == pytest.approx(1.0)
    knocked = image_attention_knockout(recipient, torch.tensor([False, True, False]))
    assert knocked.sum().item() == pytest.approx(1.0)
    assert knocked[..., 1].item() == 0.0


def _pair(group: str, split: str, source: str) -> PairedExample:
    return PairedExample(
        pair_id=f"{group}-{split}",
        group_id=group,
        donor_image="donor.png",
        recipient_image="recipient.png",
        donor_prompt="count",
        recipient_prompt="count",
        donor_answer="4",
        recipient_answer="5",
        split=split,
        generator_seed=1,
        source_id=source,
    )


def test_group_source_and_template_leakage_refuses_split_overlap() -> None:
    assert_no_group_leakage([_pair("a", "train", "s1"), _pair("b", "test", "s2")])
    with pytest.raises(ValueError, match="split leakage"):
        assert_no_group_leakage([_pair("a", "train", "s1"), _pair("a", "test", "s2")])


def test_control_draws_are_repeated_layer_matched_and_reproducible() -> None:
    selected = [(0, 0), (0, 1), (1, 0)]
    first = layer_matched_control_draws(
        selected, n_layers=2, n_heads=8, n_draws=20, seed=9
    )
    second = layer_matched_control_draws(
        selected, n_layers=2, n_heads=8, n_draws=20, seed=9
    )
    assert first == second
    assert all(sum(layer == 0 for layer, _ in draw) == 2 for draw in first)
    assert all(sum(layer == 1 for layer, _ in draw) == 1 for draw in first)


def test_multivariate_controls_match_layer_and_use_all_diagnostics() -> None:
    features = {
        (layer, head): {
            "image_attention": head / 10,
            "projected_output_norm": 1 + head / 20,
            "attention_entropy": 2 - head / 30,
            "gaze_score": head / 40,
            "general_causal_importance": head / 50,
        }
        for layer in range(2)
        for head in range(10)
    }
    selected = [(0, 0), (0, 1), (1, 0)]
    draws = multivariate_matched_control_draws(selected, features, n_draws=20, seed=3)
    assert len(draws) == 20
    assert all(len(set(draw)) == 3 for draw in draws)
    assert all([layer for layer, _ in draw].count(0) == 2 for draw in draws)


def test_manifest_has_hashes_environment_seed_git_and_resume_marker(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("input", encoding="utf-8")
    output = tmp_path / "out" / "rows.jsonl"
    output.parent.mkdir()
    output.write_text("{}\n", encoding="utf-8")
    path = write_run_manifest(
        output.parent,
        config={"smoke": True},
        seeds={"global": 3},
        inputs=[source],
        outputs=[output],
        status="complete",
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["input_sha256"][str(source)]
    assert data["output_sha256"][str(output)]
    assert data["environment"]["python"]
    assert data["seeds"] == {"global": 3}
    assert Path(data["resume_marker"]).is_file()


def test_atlas_has_every_runtime_head_slot() -> None:
    rows = initialize_head_atlas(36, 32)
    assert len(rows) == 1152
    assert rows[0]["layer"] == 0 and rows[0]["head"] == 0
    assert rows[-1]["layer"] == 35 and rows[-1]["head"] == 31


def test_language_layer_discovery_unwraps_peft_style_containers() -> None:
    layers = [object(), object()]
    qwen = SimpleNamespace(language_model=SimpleNamespace(layers=layers))
    wrapped = SimpleNamespace(base_model=SimpleNamespace(model=qwen))
    assert get_language_layers(wrapped) is layers


def test_custom_attention_registers_matching_eager_causal_mask() -> None:
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    register_qwen3_mechanistic_attention()
    assert MECHANISTIC_ATTENTION_IMPL in modeling_qwen3_vl.ALL_ATTENTION_FUNCTIONS
    assert MECHANISTIC_ATTENTION_IMPL in ALL_MASK_ATTENTION_FUNCTIONS
    assert ALL_MASK_ATTENTION_FUNCTIONS[MECHANISTIC_ATTENTION_IMPL] is eager_mask


def test_prompt_subsequence_must_be_unique() -> None:
    assert locate_subsequence([1, 2, 3, 4], [2, 3]) == (1, 3)
    with pytest.raises(ValueError, match="found 2"):
        locate_subsequence([1, 2, 1, 2], [1, 2])


def test_multimodal_inputs_repeat_with_qwen_position_layout() -> None:
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "pixel_values": torch.randn(4, 8),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "position_ids": torch.arange(3).reshape(1, 1, 3).repeat(3, 1, 1),
    }
    repeated = repeat_model_inputs(inputs, 2)
    assert repeated["input_ids"].shape == (2, 3)
    assert repeated["pixel_values"].shape == (8, 8)
    assert repeated["image_grid_thw"].shape == (2, 3)
    assert repeated["position_ids"].shape == (3, 2, 3)


def test_jsonl_checkpoint_is_resumable_and_deduplicated(tmp_path: Path) -> None:
    path = tmp_path / "rows.checkpoint.jsonl"
    first = JsonlCheckpoint(path, key=lambda row: (row["id"],), resume=False)
    first.append([{"id": "a", "value": 1}, {"id": "a", "value": 2}])
    resumed = JsonlCheckpoint(path, key=lambda row: (row["id"],), resume=True)
    assert len(resumed.rows) == 1
    assert not resumed.missing(("a",))


def test_multihead_exact_replacement_keeps_shared_bias_and_sums_deltas() -> None:
    attention = torch.nn.Identity()
    layer = SimpleNamespace(self_attn=attention)
    model = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(layers=[layer]))
    )
    current = torch.randn(1, 3, 5)
    contributions = torch.randn(1, 3, 2, 5)
    replacement = torch.randn(1, 5)
    with projected_head_set_replacement(
        model,
        replacements={(0, 0): None, (0, 1): replacement},
        recipient_projected={0: contributions},
        positions=[2],
    ):
        actual = attention(current)
    expected = current.clone()
    expected[:, 2, :] += (
        -contributions[:, 2, 0, :]
        - contributions[:, 2, 1, :]
        + replacement
    )
    assert torch.allclose(actual, expected)


def test_batched_visual_map_transplant_assigns_one_head_per_batch_row() -> None:
    attention = torch.nn.Identity()
    layer = SimpleNamespace(self_attn=attention)
    model = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(layers=[layer]))
    )
    recipient = torch.full((2, 2, 1, 4), 0.25)
    donor = torch.tensor(
        [[[[0.7, 0.1, 0.1, 0.1]], [[0.1, 0.7, 0.1, 0.1]]]]
    )
    with batched_visual_attention_map_patch_many(
        model,
        layer_idx=0,
        head_indices=[0, 1],
        donor_probabilities=donor,
        donor_query_positions=[0],
        recipient_query_positions=[0],
        donor_visual_positions=[0, 1],
        recipient_visual_positions=[0, 1],
    ):
        patched = attention._vlm_mechanistic_attention_replacement(recipient)
    assert torch.allclose(patched.sum(-1), torch.ones_like(patched.sum(-1)))
    assert patched[0, 0, 0, 0] > patched[0, 0, 0, 1]
    assert patched[1, 1, 0, 1] > patched[1, 1, 0, 0]
    assert torch.equal(patched[0, 1], recipient[0, 1])
    assert torch.equal(patched[1, 0], recipient[1, 0])


def test_layer_parser_and_safe_output_default(tmp_path: Path) -> None:
    assert parse_layer_spec("0,2-4", n_layers=5) == [0, 2, 3, 4]
    with pytest.raises(ValueError):
        parse_layer_spec("5", n_layers=5)
    output = tmp_path / "run"
    output.mkdir(); (output / "rows.tsv").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_output_directory(output, resume=False, overwrite=False, known_outputs=("rows.tsv",))


def test_point_runtime_refuses_missing_local_adapter_before_model_load() -> None:
    with pytest.raises(FileNotFoundError, match="training calibration first"):
        runtime_from_config(
            {
                "model_id": "Qwen/Qwen3-VL-8B-Instruct",
                "adapter_path": "segments/mechanistic_heads_qwen3_8b/checkpoints/missing-test-adapter",
            },
            device_map="cpu",
        )


def test_required_scientific_clis_use_uniform_arguments() -> None:
    required = [
        "prepare_mmmc.py", "generate_counting_data.py", "run_counting_vap.py",
        "run_counting_head_scan.py", "generate_point_search_data.py",
        "train_point_search.py", "run_search_head_scan.py",
        "run_verification_head_scan.py", "run_maci_head_scan.py",
        "run_maci_ablation.py", "run_vlmbias_signed_head_scan.py",
        "render_mechanistic_head_reports.py",
    ]
    for name in required:
        source = (Path("scripts") / name).read_text(encoding="utf-8")
        assert "add_standard_run_arguments" in source, name
