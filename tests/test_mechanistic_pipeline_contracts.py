from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from vlm_eval.mechanistic_heads.io import write_tsv
from vlm_eval.mechanistic_heads.preflight import (
    require_calibration_report,
    require_completed_manifest,
    require_current_artifact,
    require_scientific_validation,
)
from vlm_eval.mechanistic_heads.reproducibility import hash_paths, referenced_image_paths
from vlm_eval.mechanistic_heads.checkpoint import JsonlCheckpoint


ROOT = Path(__file__).resolve().parents[1]


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
    assert len(submitter.jobs) == 34
    assert len(submitter.commands) == 34
    assert len(set(submitter.jobs.values())) == 34

    def dependency_ids(job_name: str) -> set[str]:
        job_id = submitter.jobs[job_name]
        command = submitter.commands[list(submitter.jobs).index(job_name)]
        del job_id
        argument = next(value for value in command if value.startswith("--dependency="))
        return set(argument.split(":", 1)[1].split(":"))

    general = submitter.jobs["general_importance"]
    assert general in dependency_ids("point_ablation")
    assert general in dependency_ids("maci_confirmation")
    assert general in dependency_ids("vlmbias_validation")
    assert submitter.jobs["maci_stability"] in dependency_ids("head_atlas")
    assert submitter.jobs["maci_stability"] in dependency_ids("maci_detector")
    assert submitter.jobs["maci_ablation"] in dependency_ids("maci_detector")
    assert submitter.jobs["full_point_behavior"] in dependency_ids("point_centroids_layers")
    assert submitter.jobs["maci_heads_aggregate"] in dependency_ids(
        "vlmbias_heads_layers"
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
    with pytest.raises(RuntimeError, match="30 positive driving heads"):
        module.make_conditions(
            incomplete,
            n_layers=2,
            n_heads=32,
            seed=1,
            require_full_sets=True,
        )

    complete = [((0, head), 1.0) for head in range(30)] + [
        ((1, head), -1.0) for head in range(32)
    ] + [((2, head), -1.0) for head in range(8)]
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
