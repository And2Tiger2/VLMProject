from __future__ import annotations

import importlib.util
from pathlib import Path

from scripts.submit_neuronic_mechanistic_overnight import Submitter
from vlm_eval.mechanistic_heads.base_search import (
    CUE_MODES,
    assert_base_only,
    assert_unmodified_runtime,
    build_search_probe,
    find_exemplars,
    seeded_group_sample,
)
from vlm_eval.mechanistic_heads.patching import (
    LazyProjectedHeads,
    projected_head_contributions,
)
from vlm_eval.mechanistic_heads.synthetic import render_waldo_like_scene


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def waldo_row(tmp_path: Path) -> dict:
    scene = render_waldo_like_scene(
        seed=17,
        scene_id="base-search-unit",
        target_present=True,
        similarity=3,
        clutter=8,
        target_cell=55,
    )
    image_path = tmp_path / "scene.png"
    scene.image.save(image_path)
    masks = {}
    for name, mask in scene.masks.items():
        path = tmp_path / f"{name}.png"
        mask.save(path)
        masks[name] = str(path)
    target = next(obj for obj in scene.objects if obj["class"] == "target")
    decoys = []
    for obj in scene.objects:
        if obj["class"].startswith("distractor") and obj["cell"] not in decoys and obj["cell"] != target["cell"]:
            decoys.append(obj["cell"])
    return {
        "id": "base-search-unit",
        "group_id": "base-search-unit-group",
        "split": "train",
        "image_path": str(image_path),
        "target_present": True,
        "target_cell": target["cell"],
        "masks": masks,
        "objects": scene.objects,
        "metadata": {"four_candidate_cells": [decoys[0], target["cell"], decoys[1], decoys[2]]},
    }


def test_base_search_probe_has_exemplar_and_matched_candidate_rois(tmp_path: Path) -> None:
    row = waldo_row(tmp_path)
    exemplars = find_exemplars([row])
    probes = {
        cue: build_search_probe(row, cue_mode=cue, exemplars=exemplars)
        for cue in CUE_MODES
    }
    assert probes["text"].masks["reference"].getbbox() is None
    assert probes["target_exemplar"].masks["reference"].getbbox() is not None
    assert probes["impostor_exemplar"].masks["reference"].getbbox() is not None
    for probe in probes.values():
        assert probe.image.size == (580, 440)
        assert probe.answer == "candidate=1"
        assert probe.masks["target_object"].getbbox() is not None
        assert probe.masks["target_candidate"].getbbox() is not None
        assert probe.masks["distractor_candidates"].getbbox() is not None
        assert all(probe.masks[f"candidate_{index}"].getbbox() is not None for index in range(4))


def test_base_search_config_refuses_any_trained_checkpoint() -> None:
    assert_base_only({"adapter_path": None, "checkpoint": None, "training": False})
    for key in ("adapter_path", "checkpoint", "lora"):
        try:
            assert_base_only({key: "trained-artifact"})
        except ValueError as exc:
            assert "forbid trained checkpoints" in str(exc)
        else:
            raise AssertionError(f"{key} should have been rejected")
    base_runtime = type("Runtime", (), {"adapter_path": None, "adapter_merged": False})()
    assert_unmodified_runtime(base_runtime)
    adapted_runtime = type("Runtime", (), {"adapter_path": "adapter", "adapter_merged": True})()
    try:
        assert_unmodified_runtime(adapted_runtime)
    except RuntimeError as exc:
        assert "adapter" in str(exc)
    else:
        raise AssertionError("an adapted runtime should have been rejected")


def test_seeded_group_sample_is_reproducible_balanced_and_seed_specific() -> None:
    rows = [
        {"id": f"row-{group}-{variant}", "group_id": f"group-{group}"}
        for group in range(20)
        for variant in range(2)
    ]
    first = seeded_group_sample(rows, limit=8, seed=101, purpose="discovery")
    repeat = seeded_group_sample(rows, limit=8, seed=101, purpose="discovery")
    second = seeded_group_sample(rows, limit=8, seed=202, purpose="discovery")

    assert [row["id"] for row in first] == [row["id"] for row in repeat]
    assert [row["id"] for row in first] != [row["id"] for row in second]
    assert len({row["group_id"] for row in first}) == len(first)


def test_search_ranking_prefers_cue_invariant_selectivity_and_builds_controls() -> None:
    module = load_script("analyze_base_search_heads.py")
    rows = []
    for layer, head, text, exemplar, image_attention in (
        (0, 0, 3.0, 2.5, 0.2),
        (0, 1, 4.0, -1.0, 0.9),
        (1, 0, 0.1, 0.2, 0.8),
    ):
        for cue, score in (("text", text), ("target_exemplar", exemplar)):
            for example_id in ("a", "b"):
                rows.append(
                    {
                        "id": example_id,
                        "cue_mode": cue,
                        "layer": str(layer),
                        "head": str(head),
                        "target_selectivity": str(score),
                        "routing_correct": "1",
                        "target_object_density": "1.0",
                        "image_attention": str(image_attention),
                        "projected_output_norm": "1.0",
                    }
                )
    ranking = module.rank_heads(rows, ranking_cues=("text", "target_exemplar"))
    assert (ranking[0]["layer"], ranking[0]["head"]) == (0, 0)
    controls = module.select_controls(ranking, top_k=1, seed=7)
    assert controls["search_heads"] == [{"layer": 0, "head": 0}]
    assert controls["high_image_attention_control"] == [{"layer": 0, "head": 1}]
    assert controls["random_control"] == [{"layer": 1, "head": 0}]


def test_search_scan_materializes_only_selected_lazy_query_heads() -> None:
    import torch

    module = load_script("run_base_search_head_scan.py")
    raw = torch.arange(1 * 4 * 2 * 3, dtype=torch.float32).reshape(1, 4, 2, 3)
    weight = torch.arange(6 * 6, dtype=torch.float32).reshape(6, 6) / 10
    lazy = LazyProjectedHeads(raw, weight)

    actual = module.materialize_projected_query_heads(lazy, [1, 3])
    expected = projected_head_contributions(raw, weight)[0, [1, 3], :, :]

    assert actual.shape == (2, 2, 6)
    assert torch.equal(actual, expected)


def test_base_search_smoke_retries_overwrite_stale_checkpoints() -> None:
    gpu_source = (ROOT / "scripts/slurm_neuronic_base_search.sh").read_text()
    post_source = (ROOT / "scripts/slurm_neuronic_base_search_postprocess.sh").read_text()

    assert "RUN_POLICY=(--resume)" in gpu_source
    assert "RUN_POLICY=(--overwrite)" in gpu_source
    assert "RUN_POLICY=(--overwrite)" in post_source


def test_base_search_submission_has_no_training_or_adapter_jobs(tmp_path: Path) -> None:
    module = load_script("submit_neuronic_base_search.py")
    submitter = Submitter(repo=tmp_path, dry_run=True)
    terminal = module.submit_base_search(submitter, profile="full")
    assert terminal == submitter.jobs["base_search_locked_validation"]
    serialized = " ".join(" ".join(command) for command in submitter.commands).casefold()
    assert "train" not in serialized
    assert "adapter" not in serialized
    assert "0-35%4" in serialized


def test_three_seed_submission_is_namespaced_and_has_final_stability_job(
    tmp_path: Path,
) -> None:
    module = load_script("submit_neuronic_base_search.py")
    submitter = Submitter(repo=tmp_path, dry_run=True)
    seeds = (101, 202, 303)
    terminal = module.submit_base_search(submitter, profile="full", seeds=seeds)

    assert terminal == submitter.jobs["base_search_seed_summary"]
    assert sum("--array=0-35%4" in command for command in submitter.commands) == 3
    serialized = "\n".join(" ".join(command) for command in submitter.commands)
    for seed in seeds:
        assert f"SEED={seed}" in serialized
        assert f"RUN_NAMESPACE=seed-{seed}" in serialized
        assert f"base_search_locked_validation_seed_{seed}" in submitter.jobs
    assert "SEEDS=101:202:303" in serialized


def test_seed_stability_summary_requires_positive_absolute_causal_harm() -> None:
    module = load_script("analyze_base_search_seed_stability.py")
    seeds = [1, 2, 3]
    rankings = {
        seed: [
            {"layer": 0, "head": 0},
            {"layer": 0, "head": 1},
            {"layer": 1, "head": 0},
        ]
        for seed in seeds
    }
    ranking_summaries = {
        seed: {"top_heads": [{"layer": 0, "head": 0}]} for seed in seeds
    }
    validation_summaries = {
        seed: {
            "by_condition": {
                "text:intact": {"n": 4, "accuracy": 0.5},
                "text:search_heads": {"n": 4, "mean_ablation_harm": 0.2},
            },
            "contrasts": {
                "search_minus_high_image_attention_harm": 0.1,
                "search_minus_random_harm": 0.1,
            },
        }
        for seed in seeds
    }
    result, rows = module.summarize_seed_stability(
        seeds,
        rankings=rankings,
        ranking_summaries=ranking_summaries,
        validation_summaries=validation_summaries,
        behavior={"valid": True},
    )

    assert result["all_seeds_pass_claim_gate"] is True
    assert result["mean_top_k_overlap"] == 1
    assert result["mean_rank_spearman"] == 1
    assert all(row["passes_seed_claim_gate"] for row in rows)
