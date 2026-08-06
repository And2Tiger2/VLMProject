from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from vlm_eval.mechanistic_heads.io import write_tsv
from vlm_eval.mechanistic_heads.config import enforce_smoke_layer_limit, partitioned_limit
from vlm_eval.mechanistic_heads.preflight import (
    require_calibration_report,
    require_completed_manifest,
    require_current_artifact,
    require_scientific_validation,
)
from vlm_eval.mechanistic_heads.reproducibility import hash_paths, referenced_image_paths
from vlm_eval.mechanistic_heads.checkpoint import JsonlCheckpoint


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_layer_limit_cannot_be_overridden() -> None:
    class Args:
        smoke = True

    assert enforce_smoke_layer_limit(Args(), [0, 35]) == [0, 35]
    with pytest.raises(ValueError, match="at most 2 layers"):
        enforce_smoke_layer_limit(Args(), [0, 1, 2])

    Args.smoke = False
    assert enforce_smoke_layer_limit(Args(), list(range(36))) == list(range(36))


def test_total_smoke_budget_partitions_across_groups() -> None:
    assert [partitioned_limit(8, groups=3, index=index) for index in range(3)] == [3, 3, 2]
    assert sum(partitioned_limit(8, groups=3, index=index) or 0 for index in range(3)) == 8
    assert partitioned_limit(None, groups=3, index=0) is None
    with pytest.raises(ValueError, match="invalid group partition"):
        partitioned_limit(8, groups=0, index=0)


def test_data_smoke_caps_cover_expanding_generators() -> None:
    counting = (ROOT / "scripts/generate_counting_data.py").read_text(
        encoding="utf-8"
    )
    point = (ROOT / "scripts/generate_point_search_data.py").read_text(
        encoding="utf-8"
    )
    vlmbias = (ROOT / "scripts/prepare_vlmbias_signed_contrasts.py").read_text(
        encoding="utf-8"
    )
    assert "constant_pairs = min(constant_pairs, 1)" in counting
    assert "train_n = min(train_n, 2)" in point
    assert "ood_per_condition = min(ood_per_condition, 1)" in point
    assert "accepted[: min(int(limit or 8), 2)]" in vlmbias


def test_direct_smoke_paths_use_total_example_budgets() -> None:
    counting_validation = (ROOT / "scripts/run_counting_head_validation.py").read_text(encoding="utf-8")
    point_ablation = (ROOT / "scripts/run_point_head_ablation.py").read_text(encoding="utf-8")
    detector = (ROOT / "scripts/train_maci_conflict_detector.py").read_text(encoding="utf-8")
    maci_ablation = (ROOT / "scripts/run_maci_ablation.py").read_text(encoding="utf-8")
    maci_gated = (ROOT / "scripts/run_maci_gated_intervention.py").read_text(encoding="utf-8")
    vlmbias_validation = (ROOT / "scripts/run_vlmbias_head_validation.py").read_text(encoding="utf-8")
    mmmc = (ROOT / "scripts/prepare_mmmc.py").read_text(encoding="utf-8")
    assert "effective_limit(args, smoke_max=4)" in counting_validation
    assert "partitioned_limit(total_smoke_limit" in point_ablation
    assert "partitioned_limit(limit, groups=len(split_names)" in detector
    assert "resisting_k = min(2" in detector
    assert "allowed_layers: list[int] = []" in maci_ablation
    assert "driving_k=min(2" in maci_gated
    assert "if args.smoke:\n        allowed_layers=[]" in vlmbias_validation
    assert 'smoke_splits = ("prototype", "validation", "locked_test")' in mmmc


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_union_tsv_schema_accepts_normal_and_exclusion_rows(tmp_path: Path) -> None:
    path = tmp_path / "mixed.tsv"
    write_tsv(
        path,
        [
            {"pair_id": "excluded", "note": "no system tokens"},
            {"pair_id": "normal", "layer": 0, "score": 1.25},
        ],
        fallback="pair_id",
    )
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert set(rows[0]) == {"layer", "note", "pair_id", "score"}
    assert rows[0]["note"] == "no system tokens"
    assert rows[1]["score"] == "1.25"


def test_manifest_hashing_refuses_missing_declared_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="declared files are missing"):
        hash_paths([tmp_path / "missing.tsv"])


def test_referenced_image_collection_is_recursive_and_strict(tmp_path: Path) -> None:
    image = tmp_path / "scene.png"
    image.write_bytes(b"png")
    assert referenced_image_paths([{"nested": [str(image)]}]) == [image]
    with pytest.raises(FileNotFoundError, match="referenced image is missing"):
        referenced_image_paths([{"image_path": str(tmp_path / "missing.png")}])


def test_full_dag_orders_general_importance_before_matched_validations(tmp_path: Path) -> None:
    module = load_script("submit_neuronic_mechanistic_overnight.py")
    submitter = module.Submitter(repo=tmp_path, dry_run=True)
    module.submit_full_suite(submitter, smoke_barrier=["smoke-ok"])
    assert len(submitter.jobs) == 38
    assert len(submitter.commands) == 38
    assert len(set(submitter.jobs.values())) == 38

    def dependencies_by_mode(job_name: str) -> dict[str, set[str]]:
        command = submitter.commands[list(submitter.jobs).index(job_name)]
        argument = next(value for value in command if value.startswith("--dependency="))
        result = {}
        for specification in argument.removeprefix("--dependency=").split(","):
            mode, values = specification.split(":", 1)
            result[mode] = set(values.split(":"))
        return result

    def dependency_ids(job_name: str) -> set[str]:
        return dependencies_by_mode(job_name).get("afterok", set())

    general = submitter.jobs["general_importance"]
    assert general in dependency_ids("point_ablation")
    assert general in dependency_ids("maci_confirmation")
    assert general in dependency_ids("vlmbias_validation")
    assert submitter.jobs["maci_stability"] in dependencies_by_mode("head_atlas")["afterany"]
    assert submitter.jobs["maci_stability"] in dependency_ids("maci_detector")
    assert submitter.jobs["maci_ablation"] in dependency_ids("maci_detector")
    assert submitter.jobs["full_point_behavior"] in dependency_ids("point_centroids_layers")
    assert submitter.jobs["maci_heads_aggregate"] in dependencies_by_mode(
        "vlmbias_heads_layers"
    )["afterany"]
    assert submitter.jobs["counting_heads_repeat2_aggregate"] in dependencies_by_mode(
        "point_centroids_layers"
    )["afterany"]
    assert submitter.jobs["point_centroids_aggregate"] in dependencies_by_mode(
        "search_heads_layers"
    )["afterany"]
    assert submitter.jobs["search_heads_aggregate"] in dependencies_by_mode(
        "verification_heads_layers"
    )["afterany"]
    assert submitter.jobs["verification_heads_aggregate"] in dependencies_by_mode(
        "distractor_heads_layers"
    )["afterany"]
    assert submitter.jobs["distractor_heads_aggregate"] in dependencies_by_mode(
        "maci_heads_layers"
    )["afterany"]
    assert submitter.jobs["counting_vap_aggregate"] in dependencies_by_mode(
        "counting_heads_layers"
    )["afterany"]
    assert submitter.jobs["full_point_training"] in dependency_ids(
        "point_centroids_layers"
    )
    assert submitter.jobs["full_waldo_behavior"] in dependency_ids(
        "search_heads_layers"
    )
    assert submitter.jobs["counting_heads_repeat1_aggregate"] in dependency_ids(
        "counting_controls"
    )
    assert submitter.jobs["counting_heads_repeat2_aggregate"] in dependency_ids(
        "counting_controls"
    )
    assert submitter.jobs["vlmbias_heads_aggregate"] in dependency_ids(
        "maci_heads_aligned_layers"
    )
    atlas_command = submitter.commands[list(submitter.jobs).index("head_atlas")]
    assert any(value.startswith("--dependency=afterany:") for value in atlas_command)
    assert all("--kill-on-invalid-dep=yes" in command for command in submitter.commands)


def test_submission_graph_refuses_duplicate_receipt_keys(tmp_path: Path) -> None:
    module = load_script("submit_neuronic_mechanistic_overnight.py")
    submitter = module.Submitter(repo=tmp_path, dry_run=True)
    submitter.submit("same", "job.sh", exports={})
    with pytest.raises(RuntimeError, match="duplicate Slurm job name"):
        submitter.submit("same", "job.sh", exports={})


def test_prepared_reuse_requires_current_generator_source(tmp_path: Path) -> None:
    module = load_script("submit_neuronic_mechanistic_overnight.py")
    source = tmp_path / "generator.py"
    source.write_text("version = 1\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {"input_sha256": {str(source): digest}}
    module._require_manifest_sources(manifest, (source,))

    source.write_text("version = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="current generator/preparer source"):
        module._require_manifest_sources(manifest, (source,))

    with pytest.raises(RuntimeError, match="current generator/preparer source"):
        module._require_manifest_sources({"input_sha256": {}}, (source,))


def test_every_slurm_task_uses_an_existing_config_and_runner() -> None:
    source = (ROOT / "scripts/slurm_neuronic_mechanistic_heads.sh").read_text(
        encoding="utf-8"
    )
    config_paths = []
    runner_paths = []
    for token in source.replace("\\\n", " ").split():
        clean = token.strip('"')
        if clean.startswith("segments/mechanistic_heads_qwen3_8b/configs/"):
            config_paths.append(clean)
        if clean.startswith("scripts/") and clean.endswith(".py"):
            runner_paths.append(clean)
    assert config_paths and runner_paths
    assert all((ROOT / value).is_file() for value in config_paths)
    assert all((ROOT / value).is_file() for value in runner_paths)


def test_mechanistic_extras_and_frozen_sync_cover_training_runtime() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    wrapper = (ROOT / "scripts/run_neuronic_mechanistic_heads.sh").read_text(
        encoding="utf-8"
    )
    assert '"peft>=0.13.0"' in pyproject
    assert "uv sync --frozen --extra qwen --extra mechanistic --extra dev" in wrapper


def test_paper_style_maci_sets_require_signed_head_counts() -> None:
    module = load_script("run_maci_ablation.py")
    incomplete = [((0, head), 1.0) for head in range(29)] + [
        ((1, head), -1.0) for head in range(32)
    ]
    with pytest.raises(RuntimeError, match="at least 50 positive driving"):
        module.make_conditions(
            incomplete,
            n_layers=2,
            n_heads=32,
            seed=1,
            require_full_sets=True,
        )

    complete = [((0, head), 1.0) for head in range(32)] + [
        ((1, head), 1.0) for head in range(18)
    ] + [((2, head), -1.0) for head in range(32)] + [
        ((3, head), -1.0) for head in range(18)
    ]
    conditions = module.make_conditions(
        complete,
        n_layers=4,
        n_heads=32,
        seed=1,
        require_full_sets=True,
    )
    assert len(conditions["driving_top30"]) == 30
    assert len(conditions["resisting_top40"]) == 40


def test_detector_refuses_unsigned_resisting_heads(tmp_path: Path) -> None:
    module = load_script("train_maci_conflict_detector.py")
    score_path = tmp_path / "scores.tsv"
    write_tsv(
        score_path,
        [
            {"layer": 0, "head": head, "mean_signed_intervention_score": -1.0}
            for head in range(3)
        ]
        + [
            {"layer": 1, "head": head, "mean_signed_intervention_score": 1.0}
            for head in range(3)
        ],
    )
    assert len(module.load_resisting_heads(score_path, k=3)) == 3
    with pytest.raises(RuntimeError, match="4 negative resisting heads"):
        module.load_resisting_heads(score_path, k=4)


def test_count_claim_gate_requires_all_scientific_controls() -> None:
    module = load_script("run_counting_head_validation.py")
    rows = []
    real_specs = [
        ("real-color", "color", "standard", 101),
        ("real-shape", "shape", "target_relocation", 102),
    ]
    for pair_id, variant, position, renderer_seed in real_specs:
        for intervention in (
            "zero",
            "mean",
            "resample",
            "donor_patch",
            "reverse_donor_patch",
        ):
            rows.append(
                {
                    "pair_id": pair_id,
                    "pair_type": "constant-complexity",
                    "variant": variant,
                    "position_variant": position,
                    "renderer_seed": renderer_seed,
                    "head_set": "count_top10",
                    "intervention": intervention,
                    "margin_shift": -2.0,
                }
            )
    rows.extend(
        [
            {
                "pair_id": "code",
                "pair_type": "randomized-answer-code",
                "variant": "color",
                "position_variant": "standard",
                "renderer_seed": 101,
                "head_set": "count_top10",
                "intervention": "donor_patch",
                "margin_shift": -1.5,
            },
            {
                "pair_id": "sham",
                "pair_type": "matched-sham",
                "variant": "color",
                "position_variant": "standard",
                "renderer_seed": 101,
                "head_set": "count_top10",
                "intervention": "donor_patch",
                "margin_shift": -0.1,
            },
        ]
    )
    for draw in range(20):
        rows.append(
            {
                "pair_id": f"control-{draw}",
                "pair_type": "constant-complexity",
                "variant": "color",
                "position_variant": "standard",
                "renderer_seed": 101,
                "head_set": f"fully_matched_k10_draw{draw:02d}",
                "intervention": "donor_patch",
                "margin_shift": -0.2,
            }
        )
    passed = module.count_claim_checks(
        rows,
        stability_passed=True,
        matched_control_quantile=0.95,
    )
    assert passed["all_pass"]
    assert passed["per_k"]["10"]["n_matched_control_draws"] == 20
    assert not module.count_claim_checks(
        rows,
        stability_passed=False,
        matched_control_quantile=0.95,
    )["all_pass"]


def test_distractor_validation_parses_and_reports_decoy_selections() -> None:
    module = load_script("run_point_head_ablation.py")
    assert module.parse_cell("The answer is cell=07.") == 7
    assert module.parse_cell("cell=100") is None
    assert module.should_generate_selection(
        "distractor_suppression", "distractor_suppression_top", {}
    )
    assert module.should_generate_selection(
        "distractor_suppression", "distractor_suppression_fully_random_04", {}
    )
    assert not module.should_generate_selection(
        "distractor_suppression", "distractor_suppression_fully_random_05", {}
    )
    aggregate = module.summarize(
        [
            {
                "study": "distractor_suppression",
                "head_set": "baseline",
                "baseline_margin": 1.0,
                "ablated_margin": 1.0,
                "margin_change": 0.0,
                "preference_flip": 0,
                "generation_scored": 1,
                "selected_target": 0,
                "selected_decoy": 1,
                "selection_state": "decoy",
            }
        ]
    )
    assert aggregate[0]["decoy_selection_rate"] == 1.0


def test_point_locked_claim_gate_requires_double_dissociation_and_decoy_effect() -> None:
    module = load_script("run_point_head_ablation.py")
    studies = ("search", "verification", "distractor_suppression")
    aggregate = []
    for study in studies:
        aggregate.extend(
            [
                {
                    "study": study,
                    "head_set": "baseline",
                    "mean_margin_change": 0.0,
                    **({"decoy_selection_rate": 0.1} if study == "distractor_suppression" else {}),
                },
                {
                    "study": study,
                    "head_set": f"{study}_top",
                    "mean_margin_change": -2.0,
                    **({"decoy_selection_rate": 0.6} if study == "distractor_suppression" else {}),
                },
                {
                    "study": study,
                    "head_set": f"{study}_bottom",
                    "mean_margin_change": -0.1,
                },
            ]
        )
        for other in studies:
            if other != study:
                aggregate.append(
                    {
                        "study": study,
                        "head_set": f"{other}_top",
                        "mean_margin_change": -0.3,
                    }
                )
        for draw in range(20):
            aggregate.append(
                {
                    "study": study,
                    "head_set": f"{study}_fully_random_{draw:02d}",
                    "mean_margin_change": -0.5,
                    **(
                        {"decoy_selection_rate": 0.2}
                        if study == "distractor_suppression" and draw < 5
                        else {}
                    ),
                }
            )
    checks = module.point_claim_checks(aggregate)
    assert checks["all_pass"]
    failed = [dict(row) for row in aggregate]
    next(
        row
        for row in failed
        if row["study"] == "search" and row["head_set"] == "search_top"
    )["mean_margin_change"] = 0.1
    assert not module.point_claim_checks(failed)["all_pass"]


def test_vlmbias_locked_claim_gate_requires_direction_controls_and_retention() -> None:
    module = load_script("run_vlmbias_head_validation.py")
    summaries = {
        "baseline": {
            "unconditional_bias_answer_rate": 0.4,
            "mean_margin_shift": 0.0,
        },
        "driving_suppress": {
            "unconditional_bias_answer_rate": 0.2,
            "mean_margin_shift": -2.0,
        },
        "resisting_amplify": {
            "unconditional_bias_answer_rate": 0.2,
            "mean_margin_shift": -2.0,
        },
        "joint_role_aware": {
            "unconditional_bias_answer_rate": 0.1,
            "mean_margin_shift": -2.5,
        },
    }
    for role in ("driving", "resisting"):
        for draw in range(20):
            summaries[f"control_{role}_fully_{draw:02d}"] = {
                "unconditional_bias_answer_rate": 0.4,
                "mean_margin_shift": -0.2,
            }
    transitions = {
        condition: {"bias->correct": 4, "correct->bias": 1}
        for condition in ("driving_suppress", "resisting_amplify", "joint_role_aware")
    }
    naturalbench = {
        "baseline": {"Acc": 0.7, "G_Acc": 0.5},
        "driving_suppress": {"Acc": 0.69, "G_Acc": 0.48},
        "resisting_amplify": {"Acc": 0.68, "G_Acc": 0.47},
        "joint_role_aware": {"Acc": 0.67, "G_Acc": 0.46},
    }
    checks = module.vlmbias_claim_checks(
        summaries, transitions, naturalbench, naturalbench_tolerance=0.05
    )
    assert checks["all_pass"]
    naturalbench["joint_role_aware"]["G_Acc"] = 0.1
    assert not module.vlmbias_claim_checks(
        summaries, transitions, naturalbench, naturalbench_tolerance=0.05
    )["all_pass"]


def test_maci_gated_claim_requires_budget_matched_detector_benefit() -> None:
    module = load_script("run_maci_gated_intervention.py")
    conditions = {
        "never": {"mean_hallucination_advantage": 1.0, "intervention_rate": 0.0},
        "always": {"mean_hallucination_advantage": 0.3, "intervention_rate": 1.0},
        "detector_gated": {"mean_hallucination_advantage": 0.4, "intervention_rate": 0.4},
        "confidence_gated": {"mean_hallucination_advantage": 0.7, "intervention_rate": 0.5},
        "random_budget_matched": {"mean_hallucination_advantage": 0.8, "intervention_rate": 0.4},
    }
    assert module.gated_claim_checks(conditions)["all_pass"]
    conditions["random_budget_matched"]["intervention_rate"] = 0.3
    assert not module.gated_claim_checks(conditions)["all_pass"]


def test_mechanistic_shell_entrypoints_parse() -> None:
    scripts = [
        "run_neuronic_mechanistic_heads.sh",
        "slurm_neuronic_mechanistic_prepare.sh",
        "slurm_neuronic_mechanistic_heads.sh",
        "slurm_neuronic_mechanistic_aggregate.sh",
        "slurm_neuronic_mechanistic_postprocess.sh",
    ]
    for name in scripts:
        subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], check=True)


def test_run_verifier_rejects_stale_or_tampered_outputs(tmp_path: Path) -> None:
    module = load_script("validate_mechanistic_run.py")
    output = tmp_path / "result.tsv"
    source = tmp_path / "config.json"
    source.write_text("{}\n", encoding="utf-8")
    output.write_text("value\nok\n", encoding="utf-8")
    manifest = {
        "status": "complete",
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
        ).stdout.strip(),
        "input_sha256": {str(source): module.sha256_file(source)},
        "output_sha256": {str(output): module.sha256_file(output)},
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert module.validate_run(run_dir, repo=ROOT, newer_than_epoch=None)["valid"]
    output.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash-mismatch"):
        module.validate_run(run_dir, repo=ROOT, newer_than_epoch=None)


def test_run_verifier_rejects_empty_data_and_changed_inputs(tmp_path: Path) -> None:
    module = load_script("validate_mechanistic_run.py")
    source = tmp_path / "config.json"
    output = tmp_path / "scores.tsv"
    source.write_text("{}\n", encoding="utf-8")
    output.write_text("score\n", encoding="utf-8")
    manifest = {
        "status": "complete",
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
        ).stdout.strip(),
        "input_sha256": {str(source): module.sha256_file(source)},
        "output_sha256": {str(output): module.sha256_file(output)},
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="empty-data"):
        module.validate_run(tmp_path, repo=ROOT, newer_than_epoch=None)
    output.write_text("score\n1\n", encoding="utf-8")
    manifest["output_sha256"][str(output)] = module.sha256_file(output)
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    source.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="input-hash-mismatch"):
        module.validate_run(tmp_path, repo=ROOT, newer_than_epoch=None)


def test_scientific_preflight_requires_current_git_instrumentation(tmp_path: Path) -> None:
    report = tmp_path / "instrumentation_validation.json"
    required = {
        name: True
        for name in (
            "identity_patch",
            "projected_head_reconstruction",
            "self_subtraction_noop",
            "batched_serial_agreement",
            "teacher_forcing_likelihood",
            "token_spans",
            "attention_normalization",
            "generator_determinism",
            "split_leakage",
            "backend_equivalence",
            "cached_uncached_equivalence",
            "reproducibility_manifest",
        )
    }
    report.write_text(json.dumps({"valid": True, "checks": required}), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"status": "complete", "git_sha": "stale"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="current Git SHA"):
        require_scientific_validation(report)


def test_completed_manifest_binds_current_sha_inputs_and_outputs(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    output = tmp_path / "summary.json"
    source.write_text("input", encoding="utf-8")
    output.write_text("output", encoding="utf-8")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "status": "complete",
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip(),
        "input_sha256": {str(source): digest(source)},
        "output_sha256": {str(output): digest(output)},
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    require_completed_manifest(
        tmp_path, expected_outputs=(output,), require_current_git=True
    )
    source.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="input is missing or changed"):
        require_completed_manifest(
            tmp_path, expected_outputs=(output,), require_current_git=True
        )


def test_current_artifact_requires_parent_manifest_and_exact_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "scores.tsv"
    artifact.write_text("layer\thead\n0\t0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="complete output of the current Git SHA"):
        require_current_artifact(artifact)

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    current_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "git_sha": current_sha,
                "input_sha256": {},
                "output_sha256": {str(artifact): digest},
            }
        ),
        encoding="utf-8",
    )
    require_current_artifact(artifact)
    artifact.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing or changed"):
        require_current_artifact(artifact)


def test_afterany_atlas_refuses_stale_scientific_artifacts() -> None:
    source = (ROOT / "scripts/render_mechanistic_head_reports.py").read_text(
        encoding="utf-8"
    )
    assert "is_current_artifact(path)" in source
    assert "stale_path_ignored" in source


def test_preparation_regenerates_outputs_after_generator_changes() -> None:
    source = (
        ROOT / "scripts/slurm_neuronic_mechanistic_prepare.sh"
    ).read_text(encoding="utf-8")
    assert source.count("--overwrite") == 4
    assert "--resume" not in source


def test_optional_real_waldo_hashes_source_and_derived_images() -> None:
    source = (ROOT / "scripts/prepare_real_waldo_transfer.py").read_text(
        encoding="utf-8"
    )
    assert "referenced_image_paths(records)" in source
    assert "*source_images" in source
    assert "*derived" in source


def test_point_smoke_and_full_checkpoints_are_isolated() -> None:
    source = (ROOT / "scripts/slurm_neuronic_mechanistic_heads.sh").read_text(
        encoding="utf-8"
    )
    assert 'POINT_CHECKPOINT_ROOT="$POINT_CHECKPOINT_ROOT/smoke"' in source
    assert '--checkpoint "$POINT_ANSWER_CHECKPOINT"' in source


def test_point_adapter_is_merged_before_exact_head_projection() -> None:
    source = (
        ROOT / "vlm_eval/mechanistic_heads/qwen3_runtime.py"
    ).read_text(encoding="utf-8")
    assert "merge_and_unload(safe_merge=True)" in source
    assert "self.adapter_merged = True" in source


def test_waldo_head_discovery_and_locked_validation_families_are_disjoint(
    tmp_path: Path,
) -> None:
    module = load_script("generate_point_search_data.py")
    search, verification, distractor = module._write_waldo_pairs(
        tmp_path, seed=7, n_groups=10, resume=False
    )
    for family in (search, verification, distractor):
        by_split = {
            split: {row.group_id for row in family if row.split == split}
            for split in ("prototype", "validation", "locked_test")
        }
        assert all(by_split.values())
        assert not (by_split["prototype"] & by_split["validation"])
        assert not (by_split["prototype"] & by_split["locked_test"])
        assert not (by_split["validation"] & by_split["locked_test"])
        assert all(Path(row.donor_mask).is_file() for row in family)
        assert all(Path(row.recipient_mask).is_file() for row in family)

    verification_pair = verification[0]
    true_row = next(row for row in search + verification + distractor if row.pair_id == verification_pair.pair_id)
    assert true_row.metadata["matched_distractor_centers"] is True


def test_counting_answer_codebooks_vary_and_shams_are_declared_controls(
    tmp_path: Path,
) -> None:
    module = load_script("generate_counting_data.py")
    result = module.generate_counting_datasets(
        tmp_path,
        config={
            "syndot_train": 1,
            "syndot_test": 1,
            "mechanistic_pairs": 1,
            "mechanistic_repeat_seeds": [20, 21],
            "constant_complexity_pairs": 12,
        },
        seed=19,
        smoke=False,
        limit=None,
        resume=False,
    )
    assert result["valid"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "constant_complexity_pairs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    codebooks = {
        tuple(sorted(row["metadata"]["codebook"].items()))
        for row in rows
        if row["metadata"]["pair_type"] == "randomized-answer-code"
    }
    assert len(codebooks) > 1
    standard = next(
        row for row in rows if row["metadata"]["pair_type"] == "constant-complexity"
    )
    assert standard["metadata"]["renderer_seed"] != 19
    primary = json.loads(
        (tmp_path / "mechanistic_pairs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    repeats = [
        json.loads(
            (tmp_path / f"mechanistic_pairs_repeat{index}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        for index in (1, 2)
    ]
    assert {row["generator_seed"] for row in repeats} == {20, 21}
    assert all(row["donor_image"] != primary["donor_image"] for row in repeats)


def test_waldo_four_candidate_target_position_is_not_constant(tmp_path: Path) -> None:
    module = load_script("generate_point_search_data.py")
    result = module.generate_point_search_datasets(
        tmp_path,
        config={"training_scenes": 2, "ood_scenes_per_condition": 1, "waldo_like_scenes": 24},
        seed=17,
        smoke=False,
        limit=None,
        resume=False,
    )
    assert result["valid"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "waldo_like.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    present = [row for row in rows if row["target_present"]]
    candidate_indices = {
        int(row["tasks"]["four_candidate_selection"].split("=")[1]) for row in present
    }
    assert len(candidate_indices) > 1
    assert all(len(row["masks"]) == len(row["objects"]) + 1 for row in rows)


def test_mmmc_prompt_contrast_holds_image_fixed() -> None:
    source = (ROOT / "scripts/prepare_mmmc.py").read_text(encoding="utf-8")
    assert 'donor_image=f"hf://{DATASET_ID}/{row[\'source_split\']}/{row[\'conflict_index\']}"' in source
    assert 'recipient_image=f"hf://{DATASET_ID}/{row[\'source_split\']}/{row[\'conflict_index\']}"' in source
    assert '"same_image_prompt_contrast": True' in source


def test_mmmc_smoke_selects_complete_pairs_and_exercises_splits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_script("prepare_mmmc.py")

    class FakeRows(list):
        _fingerprint = "fake-fingerprint"

        def cast_column(self, _name, _feature):
            return self

        def select(self, indices):
            return FakeRows(self[index] for index in indices)

    rows = FakeRows()
    for index in range(10):
        rows.extend(
            [
                {
                    "image_id": f"image-{index}",
                    "image": {"path": f"image-{index}.png"},
                    "conflict_type": "clean",
                    "question": f"clean question {index}",
                    "answer": f"prior-{index}",
                    "key_component": "object",
                },
                {
                    "image_id": f"image-{index}",
                    "image": {"path": f"image-{index}.png"},
                    "conflict_type": "object",
                    "question": f"conflict question {index}",
                    "answer": f"fact-{index}",
                    "key_component": "object",
                },
            ]
        )

    class FakeImage:
        def __init__(self, *, decode):
            self.decode = decode

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Image = FakeImage
    fake_datasets.load_dataset = lambda *_args, **_kwargs: {"train": rows}
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    audit, pairs = module.prepare_mmmc(
        output_dir=tmp_path,
        cache_dir=None,
        config={"clean_conflict_values": ["clean"]},
        seed=7,
        smoke=True,
        limit=8,
        skip_tokenization=True,
    )
    assert audit["valid"]
    assert audit["n_object_conflict_pairs"] == 8
    assert audit["n_object_conflict_pairs_available"] == 10
    assert audit["pairing_success_rate"] == 1.0
    assert set(audit["split_pair_counts"]) == {
        "prototype",
        "validation",
        "locked_test",
    }
    assert len(pairs) == 8
    assert all(pair.donor_image == pair.recipient_image for pair in pairs)


def test_checkpoint_resume_rejects_changed_run_context(tmp_path: Path) -> None:
    path = tmp_path / "scan.checkpoint.jsonl"
    first = JsonlCheckpoint(
        path, key=lambda row: (row["id"],), resume=False, context={"seed": 1}
    )
    first.append([{"id": "a"}])
    with pytest.raises(RuntimeError, match="context does not match"):
        JsonlCheckpoint(
            path, key=lambda row: (row["id"],), resume=True, context={"seed": 2}
        )


def test_locked_configs_bound_expensive_validation_work() -> None:
    maci = json.loads(
        (ROOT / "segments/mechanistic_heads_qwen3_8b/configs/maci_ablation_locked.json").read_text()
    )
    detector = json.loads(
        (ROOT / "segments/mechanistic_heads_qwen3_8b/configs/maci_detector.json").read_text()
    )
    point = json.loads(
        (ROOT / "segments/mechanistic_heads_qwen3_8b/configs/point_head_ablation.json").read_text()
    )
    counting = json.loads(
        (ROOT / "segments/mechanistic_heads_qwen3_8b/configs/counting_validation.json").read_text()
    )
    assert maci["max_examples"] == 500
    assert maci["control_draws"] >= 20
    assert maci["diagnostic_single_feature_controls"] is False
    assert detector["max_pairs_by_split"]["locked_test"] == 500
    assert point["max_examples_per_study"] == 50
    assert point["random_draws"] >= 20
    assert counting["max_examples"] == 40
    assert counting["validation_control_families"] == ["fully_matched"]


def test_all_long_head_scans_have_context_bound_checkpoints() -> None:
    names = (
        "run_counting_vap.py",
        "run_counting_head_scan.py",
        "run_maci_head_scan.py",
        "run_vlmbias_signed_head_scan.py",
        "run_search_head_scan.py",
        "run_verification_head_scan.py",
        "run_distractor_head_scan.py",
        "run_point_attention_centroids.py",
    )
    for name in names:
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "JsonlCheckpoint" in source
        assert "context=" in source
        assert '"input_sha256"' in source


def test_final_atlas_reports_missing_branches_as_pending(tmp_path: Path) -> None:
    module = load_script("render_mechanistic_head_reports.py")
    missing = tmp_path / "missing-summary.json"
    inputs: list[Path] = []
    statuses = module.load_validation_statuses(
        {"status_sources": {"missing_branch": str(missing)}}, inputs
    )
    assert statuses["missing_branch"]["valid"] is None
    assert statuses["missing_branch"]["label"] == "computationally pending"
    assert inputs == []

    rows = [{"layer": 0, "head": 0, **{column: None for column in module.ATLAS_COLUMNS[2:]}}]
    markdown = module._status_markdown(rows, [], inputs, statuses)
    assert "## Computationally pending" in markdown
    assert "missing_branch: computationally pending" in markdown
    assert "## Failed\n\n- None reported" in markdown

    overlap = tmp_path / "overlap.tsv"
    correlation = tmp_path / "correlation.tsv"
    double = tmp_path / "double.tsv"
    module._write_overlaps(rows, overlap, k=50)
    module._write_correlations(rows, correlation)
    module._write_double_dissociation(rows, double, k=50)
    for path in (overlap, correlation, double):
        with path.open(encoding="utf-8") as handle:
            table = list(csv.DictReader(handle, delimiter="\t"))
        assert table and "pending" in table[0].values()


def test_failed_calibration_blocks_downstream_scientific_stage(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"valid": True, "calibration_result": "failed calibration"}))
    with pytest.raises(RuntimeError, match="did not pass"):
        require_calibration_report(path)
    path.write_text(json.dumps({"valid": True, "passes_stability_gate": False}))
    with pytest.raises(RuntimeError, match="did not pass"):
        require_calibration_report(path, boolean_key="passes_stability_gate")
