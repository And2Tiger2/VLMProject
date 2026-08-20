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
from PIL import Image

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
    assert "if args.smoke:" in vlmbias_validation
    assert "allowed_layers=[]" in vlmbias_validation
    assert "len(driving) < min(2, driving_k)" in vlmbias_validation
    assert 'smoke_splits = ("prototype", "validation", "locked_test")' in mmmc


def test_long_prompt_signed_scans_offload_captures_to_cpu() -> None:
    for script_name in ("run_maci_head_scan.py", "run_vlmbias_signed_head_scan.py"):
        source = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert source.count("to_cpu=True") >= 2


def test_distractor_scan_uses_matched_low_decoy_difference() -> None:
    source = (ROOT / "scripts/run_distractor_head_scan.py").read_text(
        encoding="utf-8"
    )
    assert "low_baseline" in source
    assert "low_ablated_values" in source
    assert (
        '"distractor_suppression_score": high_decoy_ablation_harm - '
        "low_decoy_ablation_harm"
    ) in source


def test_functional_head_rankings_preserve_expected_causal_sign() -> None:
    count_controls = (ROOT / "scripts/analyze_count_head_controls.py").read_text(
        encoding="utf-8"
    )
    count_validation = (ROOT / "scripts/run_counting_head_validation.py").read_text(
        encoding="utf-8"
    )
    point_validation = (ROOT / "scripts/run_point_head_ablation.py").read_text(
        encoding="utf-8"
    )
    assert 'key=lambda row: row["count_causal_score"], reverse=True' in count_controls
    assert '"spearman_rho": signed' in count_controls
    assert '"bidirectional_positive": int(forward > 0 and reverse > 0)' in count_controls
    assert 'key=lambda row: float(row["count_causal_score"]), reverse=True' in count_validation
    assert 'row.get("bidirectional_positive", 0)' in count_validation
    assert "key=lambda head: ranking[head], reverse=True" in point_validation
    assert "key=lambda head: abs(ranking[head]), reverse=True" not in point_validation


def test_point_diagnostics_normalize_post_wo_attention_ratio_schema() -> None:
    module = load_script("run_point_head_ablation.py")
    head = (3, 7)
    result = module.aggregate_diagnostics(
        [
            {
                "layer": "3",
                "head": "7",
                "image_attention_ratio": "0.25",
                "projected_output_norm": "2.0",
                "attention_entropy": "1.5",
            }
        ],
        {head: 0.75},
        {head: 0.5},
    )
    assert result[head] == {
        "image_attention": 0.25,
        "projected_output_norm": 2.0,
        "attention_entropy": 1.5,
        "gaze_score": 0.75,
        "general_causal_importance": 0.5,
    }


def test_count_stability_uses_average_ranks_for_ties() -> None:
    module = load_script("analyze_count_head_controls.py")
    assert module.ranks([2.0, 1.0, 1.0, 3.0]) == [2.0, 0.5, 0.5, 3.0]


def test_maci_stability_uses_average_ranks_for_ties() -> None:
    module = load_script("analyze_maci_head_stability.py")
    assert module.ranks([2.0, 1.0, 1.0, 3.0]) == [2.0, 0.5, 0.5, 3.0]


def test_general_importance_uses_average_percentiles_for_ties() -> None:
    module = load_script("build_general_head_importance.py")
    values = {(0, 0): 1.0, (0, 1): -1.0, (0, 2): 3.0}
    percentiles = module.percentile_abs(values, list(values))
    assert percentiles[(0, 0)] == percentiles[(0, 1)] == 0.25
    assert percentiles[(0, 2)] == 1.0


def test_vlmbias_controls_use_selection_contrast_diagnostics() -> None:
    source = (ROOT / "scripts/run_vlmbias_head_validation.py").read_text(
        encoding="utf-8"
    )
    assert "build_conditions(driving,resisting,selected_rows" in source


def test_point_report_refuses_failed_locked_ablation() -> None:
    source = (ROOT / "scripts/render_point_search_reports.py").read_text(
        encoding="utf-8"
    )
    assert "require_calibration_report(ablation_summary)" in source


def test_point_head_selection_requires_bidirectional_effects() -> None:
    module = load_script("run_point_head_ablation.py")
    rows = [
        {"layer": "0", "head": "0", "forward_margin_shift": "2", "reverse_margin_shift": "1"},
        {"layer": "0", "head": "1", "forward_margin_shift": "4", "reverse_margin_shift": "-1"},
    ]
    eligible = module.bidirectional_positive_heads(rows)
    assert eligible == {(0, 0)}
    selected = module.select_positive_function_heads(
        {(0, 0): 1.5, (0, 1): 3.0},
        requested_k=2,
        allow_unsigned_fallback=False,
        eligible_heads=eligible,
    )
    assert selected == [(0, 0)]
    assert module.bidirectional_positive_heads(
        [{"layer": "0", "head": "0", "distractor_suppression_score": "1"}]
    ) is None


def test_point_rmse_uses_global_assignment_not_greedy_matching() -> None:
    module = load_script("evaluate_point_search.py")
    # Greedy matching assigns (1, 0) to (0, 0), leaving a very poor second
    # match.  The globally optimal assignment has squared costs 4 and 1.
    assert module.point_rmse([(1, 0), (-2, 0)], [(0, 0), (2, 0)]) == pytest.approx(
        (5 / 2) ** 0.5
    )


def test_waldo_grid_error_is_spatial_and_candidate_labels_have_no_distance() -> None:
    module = load_script("run_waldo_behavior.py")
    _, correct, error = module.parse_result("invisible_grid", "cell=11", "cell=00")
    assert not correct
    assert error == pytest.approx(2**0.5)
    _, correct, error = module.parse_result(
        "four_candidate_selection", "candidate=3", "candidate=1"
    )
    assert not correct
    assert error == ""


def test_atlas_head_sets_do_not_merge_opposite_causal_roles() -> None:
    module = load_script("render_mechanistic_head_reports.py")
    rows = [
        {"layer": 0, "head": 0, "count_causal_score": 2.0, "mmmc_signed_score": 3.0},
        {"layer": 0, "head": 1, "count_causal_score": -9.0, "mmmc_signed_score": -4.0},
        {"layer": 0, "head": 2, "count_causal_score": 1.0, "mmmc_signed_score": 0.5},
    ]
    sets = module._functional_head_sets(
        rows,
        columns=["count_causal_score", "mmmc_signed_score"],
        k=2,
    )
    assert [row["head"] for row in sets["count_causal_score"]] == [0, 2]
    assert [row["head"] for row in sets["mmmc_signed_score_positive"]] == [0, 2]
    assert [row["head"] for row in sets["mmmc_signed_score_negative"]] == [1]


def test_point_training_honors_requested_device_mapping() -> None:
    source = (ROOT / "scripts/train_point_search.py").read_text(encoding="utf-8")
    assert "device_map=args.device_map" in source
    assert "device_map=_resolve_device_map(device_map, torch)" in source
    assert 'device_map="auto"' not in source


def test_failed_behavioral_calibrations_fail_slurm_prerequisites() -> None:
    expectations = {
        "run_counting_behavior.py": "counting behavioral calibration failed",
        "evaluate_point_search.py": "Point-Answer behavioral calibration failed",
        "run_waldo_behavior.py": "Waldo-like behavioral calibration failed",
    }
    for script_name, message in expectations.items():
        source = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert "if not args.smoke and not calibration_passed:" in source
        assert message in source


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vlmbias_direct_attribution_uses_complete_answer_sequences(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    module = load_script("run_vlmbias_signed_head_scan.py")

    class Materialized:
        def __init__(self, value):
            self.value = value

        def materialize(self):
            return self.value

    class Projected:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, key):
            _batch, positions, _heads, _width = key
            return Materialized(self.value[positions])

    class Capture:
        def __init__(self, value, answer_length):
            self.prompt_length = 3
            self.store = types.SimpleNamespace(projected_heads={0: Projected(value)})
            self.answer_length = answer_length

    # Two heads, two-dimensional model width. Correct has two tokens and bias
    # has one; the second correct token must affect the returned attribution.
    correct_projected = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 2.0], [3.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    bias_projected = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[2.0, 0.0], [0.0, 2.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    captures = {
        "correct": Capture(correct_projected, 2),
        "bias": Capture(bias_projected, 1),
    }

    monkeypatch.setattr(
        module,
        "capture_teacher_forced",
        lambda runtime, **kwargs: captures["correct" if kwargs["answer"] == "correct" else "bias"],
    )
    runtime = types.SimpleNamespace(
        torch=torch,
        answer_token_ids=lambda answer: torch.tensor([[0, 1]]) if answer == "correct" else torch.tensor([[2]]),
        model=types.SimpleNamespace(
            lm_head=types.SimpleNamespace(
                weight=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            )
        ),
    )
    result = module.full_sequence_direct_attributions(
        runtime,
        image_path="unused",
        prompt="prompt",
        correct_answer="correct",
        bias_answer="bias",
        layers=[0],
    )
    assert result[0] == pytest.approx([1.0, -2.0])


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
    assert submitter.jobs["full_counting_behavior"] in dependency_ids(
        "counting_vap_layers"
    )
    assert submitter.jobs["full_counting_behavior"] in dependency_ids(
        "counting_heads_layers"
    )
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


def test_point_recovery_archives_only_invalid_point_branch(tmp_path: Path) -> None:
    module = load_script("submit_neuronic_point_recovery.py")
    segment = tmp_path / "segments/mechanistic_heads_qwen3_8b"
    invalid = (
        segment / "reports/instrumentation",
        segment / "checkpoints/point-answer-lora",
        segment / "runs/point_behavior/point_answer",
    )
    for path in invalid:
        path.mkdir(parents=True)
        (path / "invalid.txt").write_text("invalid\n", encoding="utf-8")
    preserved = segment / "runs/counting_head_scan/full"
    preserved.mkdir(parents=True)
    (preserved / "keep.txt").write_text("keep\n", encoding="utf-8")

    result = module.archive_invalid_point_outputs(
        tmp_path,
        execute=True,
        stamp="20260809T000000Z",
        revision="a" * 40,
    )

    archive = tmp_path / result["archive_root"]
    assert result["n_archived"] == len(invalid)
    assert all(not path.exists() for path in invalid)
    assert (archive / "reports/instrumentation/invalid.txt").is_file()
    assert (archive / "checkpoints/point-answer-lora/invalid.txt").is_file()
    assert (archive / "runs/point_behavior/point_answer/invalid.txt").is_file()
    assert (preserved / "keep.txt").read_text(encoding="utf-8") == "keep\n"


def test_point_recovery_dry_run_is_minimal_and_dependency_safe(tmp_path: Path) -> None:
    overnight = load_script("submit_neuronic_mechanistic_overnight.py")
    module = load_script("submit_neuronic_point_recovery.py")
    submitter = overnight.Submitter(repo=tmp_path, dry_run=True)
    terminal_jobs = module.submit_point_recovery(submitter)

    assert set(submitter.jobs) == {
        "instrumentation",
        "smoke_point_training",
        "smoke_point_behavior",
        "smoke_waldo_behavior",
        "smoke_point_centroids",
        "smoke_search_heads",
        "smoke_verification_heads",
        "smoke_distractor_heads",
        "full_point_training",
        "full_point_behavior",
        "full_waldo_behavior",
        "point_centroids_layers",
        "point_centroids_aggregate",
        "search_heads_layers",
        "search_heads_aggregate",
        "verification_heads_layers",
        "verification_heads_aggregate",
        "distractor_heads_layers",
        "distractor_heads_aggregate",
    }
    assert not any(
        value in " ".join(command)
        for command in submitter.commands
        for value in ("counting-heads", "maci-heads", "vlmbias-heads")
    )
    assert terminal_jobs == [
        submitter.jobs["point_centroids_aggregate"],
        submitter.jobs["search_heads_aggregate"],
        submitter.jobs["verification_heads_aggregate"],
        submitter.jobs["distractor_heads_aggregate"],
    ]
    full_train = submitter.commands[list(submitter.jobs).index("full_point_training")]
    dependency = next(value for value in full_train if value.startswith("--dependency="))
    assert submitter.jobs["smoke_point_behavior"] in dependency
    assert submitter.jobs["smoke_waldo_behavior"] in dependency
    for name in (
        "point_centroids_layers",
        "search_heads_layers",
        "verification_heads_layers",
        "distractor_heads_layers",
    ):
        command = submitter.commands[list(submitter.jobs).index(name)]
        dependency = next(value for value in command if value.startswith("--dependency="))
        assert submitter.jobs["full_point_behavior"] in dependency
        if name == "point_centroids_layers":
            assert submitter.jobs["full_waldo_behavior"] not in dependency
        else:
            assert submitter.jobs["full_waldo_behavior"] in dependency
    assert all("--kill-on-invalid-dep=yes" in command for command in submitter.commands)


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
    assert "overnight-smoke-resume" in wrapper
    assert "--profile smoke --reuse-prepared" in wrapper
    assert "refresh-generated-data" in wrapper
    assert "generate_counting_data.py" in wrapper
    assert "--seed 260318523 --resume" in wrapper
    assert "generate_point_search_data.py" in wrapper
    assert "--seed 260525427 --overwrite" in wrapper
    assert "export UV_NO_SYNC=1" in wrapper
    assert "export UV_FROZEN=1" in wrapper
    for script in (
        "slurm_neuronic_mechanistic_prepare.sh",
        "slurm_neuronic_mechanistic_heads.sh",
        "slurm_neuronic_mechanistic_aggregate.sh",
        "slurm_neuronic_mechanistic_postprocess.sh",
    ):
        source = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert "export UV_NO_SYNC=1" in source
        assert "export UV_FROZEN=1" in source


def test_preparation_uses_study_specific_generator_seeds() -> None:
    source = (ROOT / "scripts/slurm_neuronic_mechanistic_prepare.sh").read_text(
        encoding="utf-8"
    )
    assert 'COUNT_SEED="${COUNT_SEED:-${SEED:-260318523}}"' in source
    assert 'POINT_SEED="${POINT_SEED:-260525427}"' in source
    assert 'VLMBIAS_SEED="${VLMBIAS_SEED:-260519250}"' in source
    assert 'MMMC_SEED="${MMMC_SEED:-260519250}"' in source


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


def test_maci_ablation_report_cannot_label_failed_gate_as_reproduction() -> None:
    source = (ROOT / "scripts/run_maci_ablation.py").read_text(encoding="utf-8")
    assert '"failed calibration"\n                if not claim_checks["all_pass"]' in source


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


def test_point_function_head_selection_requires_positive_scores() -> None:
    module = load_script("run_point_head_ablation.py")
    ranking = {(0, 0): 2.0, (0, 1): -8.0, (1, 0): 1.0}
    assert module.select_positive_function_heads(
        ranking, requested_k=3, allow_unsigned_fallback=False
    ) == [(0, 0), (1, 0)]
    assert module.select_positive_function_heads(
        ranking, requested_k=3, allow_unsigned_fallback=True
    ) == [(0, 0), (1, 0), (0, 1)]


def test_point_claim_gate_requires_complete_positive_head_set() -> None:
    module = load_script("run_point_head_ablation.py")
    aggregate = [
        {
            "study": study,
            "head_set": f"{study}_top",
            "n_heads": 2,
            "mean_margin_change": -1.0,
        }
        for study in ("search", "verification", "distractor_suppression")
    ]
    checks = module.point_claim_checks(
        aggregate,
        required_head_counts={
            "search": 30,
            "verification": 30,
            "distractor_suppression": 30,
        },
    )
    assert not checks["all_pass"]
    assert not checks["per_study"]["search"][
        "required_positive_head_count_available"
    ]


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


def test_vlmbias_matched_controls_are_likelihood_only() -> None:
    module = load_script("run_vlmbias_head_validation.py")
    assert module.should_generate_vlmbias("baseline")
    assert module.should_generate_vlmbias("joint_role_aware")
    assert not module.should_generate_vlmbias("control_driving_fully_00")
    assert not module.should_generate_vlmbias("control_resisting_fully_19")


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


def test_maci_gated_does_not_retain_device_inputs_between_examples() -> None:
    source = (ROOT / "scripts/run_maci_gated_intervention.py").read_text(
        encoding="utf-8"
    )
    assert "prepared.append((pair, inputs" not in source
    assert "prepared.append((pair, margin, probability))" in source
    assert "del capture, inputs, image" in source


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


def test_backend_equivalence_requires_capture_fidelity_and_behavioral_parity() -> None:
    module = load_script("validate_mechanistic_instrumentation.py")

    passed, ordering = module.backend_equivalence_passes(
        eager_custom_error=0.0,
        eager_custom_tolerance=1e-5,
        custom_candidate_margin=2.0,
        sdpa_candidate_margin=1.25,
        greedy_agreement=True,
    )
    assert passed
    assert ordering

    # A large SDPA/eager magnitude difference is retained as telemetry, but
    # is not itself evidence that the custom capture diverges from eager.
    passed, ordering = module.backend_equivalence_passes(
        eager_custom_error=0.0,
        eager_custom_tolerance=1e-5,
        custom_candidate_margin=2.0,
        sdpa_candidate_margin=0.2,
        greedy_agreement=True,
    )
    assert passed
    assert ordering

    for overrides in (
        {"eager_custom_error": 1e-3},
        {"sdpa_candidate_margin": -0.2},
        {"greedy_agreement": False},
    ):
        arguments = {
            "eager_custom_error": 0.0,
            "eager_custom_tolerance": 1e-5,
            "custom_candidate_margin": 2.0,
            "sdpa_candidate_margin": 0.2,
            "greedy_agreement": True,
            **overrides,
        }
        passed, _ = module.backend_equivalence_passes(**arguments)
        assert not passed


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


def test_completed_manifest_can_skip_unrequested_stale_outputs(tmp_path: Path) -> None:
    current = tmp_path / "current.jsonl"
    stale = tmp_path / "stale.png"
    current.write_text("{}\n", encoding="utf-8")
    stale.write_bytes(b"original")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "input_sha256": {},
                "output_sha256": {
                    str(current): digest(current),
                    str(stale): digest(stale),
                },
            }
        ),
        encoding="utf-8",
    )
    stale.write_bytes(b"changed but unreferenced")
    require_completed_manifest(
        tmp_path,
        expected_outputs=(current,),
        validate_all_outputs=False,
    )
    with pytest.raises(RuntimeError, match="output is missing or changed"):
        require_completed_manifest(tmp_path, expected_outputs=(current,))


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
    assert 'MODE="${MODE:-full}"' in source
    assert 'prepare_mode_args+=(--smoke)' in source


def test_full_transition_archives_current_smoke_results_but_not_data(
    tmp_path: Path,
) -> None:
    module = load_script("archive_stale_mechanistic_runs.py")
    segment = tmp_path / "segments/mechanistic_heads_qwen3_8b"
    smoke_run = segment / "runs/example/smoke"
    full_run = segment / "runs/example/full"
    smoke_data = segment / "data/generated/example"
    for path, smoke in (
        (smoke_run, True),
        (full_run, False),
        (smoke_data, True),
    ):
        path.mkdir(parents=True)
        (path / "run_manifest.json").write_text(
            json.dumps({"config": {"smoke": smoke}}), encoding="utf-8"
        )

    assert module.smoke_output_dirs(tmp_path) == [smoke_run]
    source = (ROOT / "scripts/run_neuronic_mechanistic_heads.sh").read_text(
        encoding="utf-8"
    )
    assert 'if [[ "$ACTION" == "overnight-all" ]]' in source
    assert "archive_args+=(--include-current-smoke)" in source


def test_refresh_generated_data_rebuilds_every_source_bound_changed_dataset() -> None:
    source = (ROOT / "scripts/run_neuronic_mechanistic_heads.sh").read_text(
        encoding="utf-8"
    )
    refresh = source.split("refresh-generated-data)", 1)[1].split(";;", 1)[0]
    assert "generate_counting_data.py" in refresh
    assert "generate_point_search_data.py" in refresh
    assert "prepare_vlmbias_signed_contrasts.py" in refresh


def test_smoke_profile_submits_bounded_preparation(tmp_path: Path) -> None:
    module = load_script("submit_neuronic_mechanistic_overnight.py")
    source = (ROOT / "scripts/submit_neuronic_mechanistic_overnight.py").read_text(
        encoding="utf-8"
    )
    assert 'exports={"MODE": "smoke" if args.profile == "smoke" else "full"}' in source
    submitter = module.Submitter(repo=tmp_path, dry_run=True)
    submitter.submit(
        "prepare_data",
        module.PREP_SCRIPT,
        exports={"MODE": "smoke"},
    )
    command = submitter.commands[0]
    export_arg = next(value for value in command if value.startswith("--export="))
    assert "MODE=smoke" in export_arg


def test_prepared_reuse_applies_profile_appropriate_size_gates() -> None:
    module = load_script("submit_neuronic_mechanistic_overnight.py")
    smoke = {
        "counting": {
            "counts": {
                "mechanistic_pairs": 4,
                "mechanistic_repeat_pairs_each": 4,
                "mechanistic_repeat_seeds": [11, 12],
            }
        },
        "point": {
            "counts": {"point_search_train": 2},
            "waldo_pair_split_group_counts": {
                family: {"prototype": 2}
                for family in ("search", "verification", "distractor")
            },
        },
        "vlmbias": {
            "semantic_source_rows": 2,
            "context_detail_source_rows": 2,
            "counts_by_contrast": {"semantic_prior": 2},
            "split_group_counts": {"prototype": 1, "validation": 1},
        },
        "mmmc": {
            "split_pair_counts": {
                "prototype": 3,
                "validation": 3,
                "locked_test": 2,
            }
        },
    }
    module.validate_prepared_dataset_contracts(**smoke, profile="smoke")
    with pytest.raises(RuntimeError, match="100 mechanistic pairs"):
        module.validate_prepared_dataset_contracts(**smoke, profile="all")

    full = {
        "counting": {
            "counts": {
                "mechanistic_pairs": 100,
                "mechanistic_repeat_pairs_each": 100,
                "mechanistic_repeat_seeds": [11, 12],
            }
        },
        "point": {
            "counts": {"point_search_train": 2000},
            "waldo_pair_split_group_counts": {
                family: {
                    "prototype": 300,
                    "validation": 100,
                    "locked_test": 100,
                }
                for family in ("search", "verification", "distractor")
            },
        },
        "vlmbias": {
            "semantic_source_rows": 400,
            "context_detail_source_rows": 114,
            "counts_by_contrast": {"semantic_prior": 400},
            "split_group_counts": {
                "prototype": 1,
                "validation": 1,
                "locked_test": 1,
            },
        },
        "mmmc": {
            "split_pair_counts": {
                "prototype": 256,
                "validation": 512,
                "locked_test": 500,
            }
        },
    }
    module.validate_prepared_dataset_contracts(**full, profile="smoke")
    module.validate_prepared_dataset_contracts(**full, profile="all")
    with pytest.raises(ValueError, match="unsupported prepared-data profile"):
        module.validate_prepared_dataset_contracts(**full, profile="invalid")


def test_resume_passes_requested_profile_to_prepared_data_validator() -> None:
    source = (ROOT / "scripts/submit_neuronic_mechanistic_overnight.py").read_text(
        encoding="utf-8"
    )
    assert "require_valid_prepared_data(args.repo, profile=args.profile)" in source


def test_smoke_dag_exercises_every_downstream_consumer(tmp_path: Path) -> None:
    module = load_script("submit_neuronic_mechanistic_overnight.py")
    submitter = module.Submitter(repo=tmp_path, dry_run=True)
    instrumentation = submitter.gpu(
        "instrumentation", "instrumentation", "smoke", dependencies=[]
    )
    primary = module.submit_smokes(submitter, instrumentation=instrumentation)
    finals = module.submit_downstream_smokes(
        submitter, primary_smokes=primary
    )

    assert len(primary) == 13
    assert finals == [submitter.jobs["smoke_head_atlas"]]
    expected = {
        "smoke_general_importance",
        "smoke_maci_stability",
        "smoke_counting_controls",
        "smoke_counting_validation",
        "smoke_point_ablation",
        "smoke_point_reports",
        "smoke_maci_ablation",
        "smoke_maci_detector",
        "smoke_maci_gated",
        "smoke_maci_confirmation",
        "smoke_vlmbias_validation",
        "smoke_head_atlas",
    }
    assert expected <= set(submitter.jobs)
    for name in expected:
        command = submitter.commands[list(submitter.jobs).index(name)]
        export_arg = next(value for value in command if value.startswith("--export="))
        assert "MODE=smoke" in export_arg

    atlas_command = submitter.commands[
        list(submitter.jobs).index("smoke_head_atlas")
    ]
    atlas_dependency = next(
        value for value in atlas_command if value.startswith("--dependency=")
    )
    assert submitter.jobs["smoke_vlmbias_validation"] in atlas_dependency
    assert submitter.jobs["smoke_point_reports"] in atlas_dependency
    assert submitter.jobs["smoke_counting_validation"] in atlas_dependency


def test_smoke_configs_only_consume_smoke_discovery_artifacts() -> None:
    config_names = (
        "smoke_general_head_importance.json",
        "smoke_counting_controls.json",
        "smoke_counting_validation.json",
        "smoke_point_head_ablation.json",
        "smoke_maci_stability.json",
        "smoke_maci_ablation.json",
        "smoke_maci_confirmation.json",
        "smoke_maci_detector.json",
        "smoke_maci_gated_intervention.json",
        "smoke_vlmbias_head_validation.json",
        "smoke_point_search_reports.json",
        "smoke_head_atlas.json",
    )
    for name in config_names:
        path = ROOT / "segments/mechanistic_heads_qwen3_8b/configs" / name
        config = json.loads(path.read_text(encoding="utf-8"))
        encoded = json.dumps(config)
        assert "/full/" not in encoded

    general = json.loads(
        (ROOT / "segments/mechanistic_heads_qwen3_8b/configs/smoke_general_head_importance.json").read_text(
            encoding="utf-8"
        )
    )
    assert general["expected_heads"] == 64


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
    assert result["four_candidate_sets_valid"] is True
    rows = [
        json.loads(line)
        for line in (tmp_path / "waldo_like.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    present = [row for row in rows if row["target_present"]]
    candidate_indices = {
        int(row["tasks"]["four_candidate_selection"].split("=")[1]) for row in present
    }
    assert len(candidate_indices) > 1
    assert all(len(row["metadata"]["four_candidate_cells"]) == 4 for row in rows)
    assert all(
        len(set(row["metadata"]["four_candidate_cells"])) == 4 for row in rows
    )
    assert all(len(row["masks"]) == len(row["objects"]) + 1 for row in rows)


def test_generated_point_search_rows_declare_condition_specific_prompts(
    tmp_path: Path,
) -> None:
    module = load_script("generate_point_search_data.py")
    module.generate_point_search_datasets(
        tmp_path,
        config={"training_scenes": 2, "ood_scenes_per_condition": 1, "waldo_like_scenes": 4},
        seed=23,
        smoke=True,
        limit=None,
        resume=False,
    )
    row = json.loads(
        (tmp_path / "point_search.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert row["prompt"] == row["prompts"]["direct"]
    assert "number only" in row["prompts"]["direct"]
    assert "do not give coordinates" in row["prompts"]["direct_length_matched"]
    assert "points=[" in row["prompts"]["point"]
    assert len(set(row["prompts"].values())) == 3


def test_point_behavior_budget_can_emit_fifty_coordinates() -> None:
    config = json.loads(
        (ROOT / "segments/mechanistic_heads_qwen3_8b/configs/point_search_behavior.json")
        .read_text(encoding="utf-8")
    )
    assert int(config["max_new_tokens"]) >= 512


def test_point_behavior_count_parser_handles_every_declared_protocol() -> None:
    module = load_script("evaluate_point_search.py")
    assert module.parse_count_answer("3") == 3
    assert module.parse_count_answer("3 neutral evidence seen") == 3
    assert module.parse_count_answer("answer=3 evidence seen") == 3
    assert module.parse_count_answer("points=[(037,064)]; answer=1") == 1
    assert module.parse_count_answer("points=[(037,064)]") is None


def test_point_calibration_requires_actual_coordinate_quality() -> None:
    module = load_script("evaluate_point_search.py")
    config = {
        "minimum_calibration_count_accuracy": 0.8,
        "minimum_calibration_point_parse_rate": 0.8,
        "maximum_calibration_point_rmse": 40.0,
    }
    passed, checks, thresholds = module.point_calibration_result(
        condition="point_answer",
        calibration={
            "count_accuracy": 0.96,
            "point_parse_rate": 0.96,
            "point_rmse": 32.9,
        },
        config=config,
    )
    assert passed
    assert all(checks.values())
    assert thresholds["maximum_point_rmse"] == 40.0

    failed, checks, _ = module.point_calibration_result(
        condition="point_answer",
        calibration={
            "count_accuracy": 0.96,
            "point_parse_rate": 0.96,
            "point_rmse": 40.1,
        },
        config=config,
    )
    assert not failed
    assert checks["point_rmse"] is False


def test_point_centroid_trace_uses_the_point_answer_instruction() -> None:
    source = (ROOT / "scripts/run_point_attention_centroids.py").read_text(
        encoding="utf-8"
    )
    assert 'point_condition_prompt(row, "point_answer")' in source


def test_point_lora_training_uses_bounded_activation_memory() -> None:
    source = (ROOT / "scripts/train_point_search.py").read_text(encoding="utf-8")
    assert "model.config.use_cache = False" in source
    assert "model.gradient_checkpointing_enable" in source
    assert "gradient_checkpointing=True" in source
    assert '"use_reentrant": False' in source


def test_point_training_arguments_match_locked_transformers_api(
    tmp_path: Path,
) -> None:
    from transformers import TrainingArguments

    module = load_script("train_point_search.py")
    arguments = module.make_training_arguments(
        TrainingArguments,
        output_dir=str(tmp_path),
        seed=7,
        config={
            "learning_rate": 1e-5,
            "warmup_steps": 200,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "epochs": 1,
            "save_steps": 50,
            "save_total_limit": 2,
        },
        smoke=True,
        max_steps=2,
        bf16=False,
    )
    assert arguments.max_steps == 2
    assert arguments.save_strategy.value == "no"
    assert arguments.gradient_checkpointing
    assert arguments.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert "overwrite_output_dir=" not in (
        ROOT / "scripts/train_point_search.py"
    ).read_text(encoding="utf-8")


def test_point_training_labels_only_verified_assistant_suffix() -> None:
    import torch

    module = load_script("train_point_search.py")
    full_ids = torch.tensor([[0, 0, 11, 12, 13, 21, 22], [31, 32, 41, 42, 0, 0, 0]])
    full_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]])
    prompt_ids = torch.tensor([[0, 11, 12, 13], [31, 32, 0, 0]])
    prompt_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]])

    labels = module.labels_after_prompt_prefix(
        full_ids, full_mask, prompt_ids, prompt_mask
    )

    assert labels.tolist() == [
        [-100, -100, -100, -100, -100, 21, 22],
        [-100, -100, 41, 42, -100, -100, -100],
    ]
    assert labels.ne(-100).sum(dim=1).tolist() == [2, 2]


def test_point_training_labels_reject_empty_or_mismatched_suffix() -> None:
    import torch

    module = load_script("train_point_search.py")
    mask = torch.ones((1, 3), dtype=torch.long)
    with pytest.raises(RuntimeError, match="no supervised tokens"):
        module.labels_after_prompt_prefix(
            torch.tensor([[1, 2, 3]]), mask, torch.tensor([[1, 2, 3]]), mask
        )
    with pytest.raises(RuntimeError, match="not a prefix"):
        module.labels_after_prompt_prefix(
            torch.tensor([[1, 2, 3, 4]]),
            torch.ones((1, 4), dtype=torch.long),
            torch.tensor([[1, 9, 3]]),
            mask,
        )


def test_point_training_aligns_spatial_contracts_without_waldo_leakage() -> None:
    module = load_script("train_point_search.py")
    row = {
        "id": "train-0",
        "split": "train",
        "image_path": "unused.png",
        "target_count": 1,
        "target": {"color": "red", "shape": "L"},
        "objects": [
            {"class": "target", "center": [112, 56]},
            {"class": "distractor", "center": [20, 200]},
        ],
        "prompts": {
            "direct": "count",
            "direct_length_matched": "count with filler",
            "point": "report points",
        },
        "answers": {
            "base": "1",
            "direct": "1",
            "direct_length_matched": "1",
            "point": "points=[(112,056)]; answer=1",
            "shuffled_point": "points=[(020,200)]; answer=1",
        },
    }
    point = module.build_training_examples(
        [row],
        condition="point_answer",
        auxiliary_examples_per_task=1,
        image_size=224,
    )
    shuffled = module.build_training_examples(
        [row],
        condition="shuffled_point_answer",
        auxiliary_examples_per_task=1,
        image_size=224,
    )
    direct = module.build_training_examples(
        [row],
        condition="direct_answer",
        auxiliary_examples_per_task=1,
        image_size=224,
    )

    assert [example["format"] for example in point] == [
        "standard",
        "normalized_point",
        "grid_cell",
        "presence",
    ]
    point_answers = {example["format"]: example["answer"] for example in point}
    shuffled_answers = {
        example["format"]: example["answer"] for example in shuffled
    }
    assert point_answers["normalized_point"] == "point=(0.502,0.251)"
    assert point_answers["grid_cell"] == "cell=25"
    assert point_answers["presence"] == "present"
    assert shuffled_answers["normalized_point"] == "point=(0.090,0.897)"
    assert shuffled_answers["grid_cell"] == "cell=80"
    assert shuffled_answers["presence"] == "present"
    assert len(direct) == len(point) == len(shuffled) == 4
    assert {example["format"] for example in direct} == {"standard"}

    locked = dict(row, split="locked_test")
    with pytest.raises(RuntimeError, match="only training rows"):
        module.spatial_contract_example(
            locked,
            condition="point_answer",
            task="normalized_point",
            image_size=224,
        )


def test_point_training_contract_alignment_keeps_optimizer_exposure_matched() -> None:
    config = json.loads(
        (
            ROOT
            / "segments/mechanistic_heads_qwen3_8b/configs/point_search_lora.json"
        ).read_text(encoding="utf-8")
    )
    total_examples = 2000 + 3 * int(config["spatial_contract_examples_per_task"])
    consumed_slots = int(config["max_steps"]) * int(
        config["gradient_accumulation_steps"]
    )
    assert total_examples == 3500
    assert total_examples <= consumed_slots < total_examples + int(
        config["gradient_accumulation_steps"]
    )


def test_every_mechanistic_slurm_entrypoint_refuses_tracked_dirty_code() -> None:
    for name in (
        "slurm_neuronic_mechanistic_heads.sh",
        "slurm_neuronic_mechanistic_prepare.sh",
        "slurm_neuronic_mechanistic_aggregate.sh",
        "slurm_neuronic_mechanistic_postprocess.sh",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "git diff --quiet -- ." in source
        assert "git diff --cached --quiet -- ." in source


def test_vlmbias_semantic_contrast_uses_all_rows_without_requiring_masks(
    tmp_path: Path,
) -> None:
    module = load_script("prepare_vlmbias_signed_contrasts.py")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    dataset_rows = []
    for index in range(3):
        image = image_dir / f"subject_{index}_px384_Q1.png"
        Image.new("RGB", (16, 16), (index * 20, 0, 0)).save(image)
        dataset_rows.append(
            {
                "id": f"subject_{index}_px384_Q1",
                "prompt": "What is shown?",
                "ground_truth": "correct",
                "expected_bias": "bias",
                "topic": "Synthetic",
                "image_path": str(image.relative_to(tmp_path)),
            }
        )
    dataset = tmp_path / "vlmbias.jsonl"
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in dataset_rows), encoding="utf-8"
    )
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    tight = mask_dir / "tight.png"
    Image.new("L", (16, 16), 255).save(tight)
    accepted = tmp_path / "accepted.jsonl"
    accepted.write_text(
        json.dumps(
            {
                "id": dataset_rows[0]["id"],
                "group_id": "subject_0",
                "artifacts": {"tight_mask": str(tight.relative_to(tmp_path))},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pairs, audit = module.prepare_contrasts(
        config={
            "vlmbias_dataset": str(dataset),
            "accepted_masks": str(accepted),
            "candidate_root": str(tmp_path / "candidates"),
            "context_removal": "whiten",
        },
        output_dir=tmp_path / "out",
        seed=7,
        limit=None,
        smoke=False,
    )
    semantic = [pair for pair in pairs if pair.metadata["contrast"] == "semantic_prior"]
    context = [pair for pair in pairs if pair.metadata["contrast"] == "context"]
    assert len(semantic) == 3
    assert len(context) == 1
    assert audit["semantic_source_rows"] == 3
    assert audit["context_detail_source_rows"] == 1
    assert audit["n_groups"] == 3
    assert sum(audit["split_group_counts"].values()) == 3
    assert all(pair.metadata["mask_required"] is False for pair in semantic)


def test_point_training_overwrite_removes_only_numeric_checkpoint_dirs(
    tmp_path: Path,
) -> None:
    module = load_script("train_point_search.py")
    removable = tmp_path / "checkpoint-12"
    removable.mkdir()
    (removable / "state.json").write_text("{}", encoding="utf-8")
    retained = tmp_path / "checkpoint-notes"
    retained.mkdir()
    module.remove_declared_training_checkpoints(tmp_path)
    assert not removable.exists()
    assert retained.is_dir()


def test_point_training_resume_uses_highest_numeric_checkpoint(tmp_path: Path) -> None:
    module = load_script("train_point_search.py")
    for name in ("checkpoint-9", "checkpoint-100", "checkpoint-20"):
        (tmp_path / name).mkdir()
    (tmp_path / "checkpoint-notes").mkdir()
    assert module.args_resume_checkpoint(tmp_path) == str(tmp_path / "checkpoint-100")


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


def test_atlas_overlaps_expand_signed_scores_into_role_specific_sets(
    tmp_path: Path,
) -> None:
    module = load_script("render_mechanistic_head_reports.py")
    rows = [
        {
            "layer": 0,
            "head": head,
            "count_causal_score": float(head + 1),
            "mmmc_signed_score": value,
            **{
                column: None
                for column in module.ATLAS_COLUMNS[2:]
                if column not in {"count_causal_score", "mmmc_signed_score"}
            },
        }
        for head, value in enumerate((2.0, 1.0, -1.0, -2.0))
    ]
    output = tmp_path / "overlaps.tsv"
    module._write_overlaps(rows, output, k=2)
    with output.open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle, delimiter="\t"))
    labels = {row["left"] for row in table} | {row["right"] for row in table}
    assert "mmmc_signed_score_positive" in labels
    assert "mmmc_signed_score_negative" in labels
    assert "count_causal_score" in labels


def test_failed_calibration_blocks_downstream_scientific_stage(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"valid": True, "calibration_result": "failed calibration"}))
    with pytest.raises(RuntimeError, match="did not pass"):
        require_calibration_report(path)
    path.write_text(json.dumps({"valid": True, "passes_stability_gate": False}))
    with pytest.raises(RuntimeError, match="did not pass"):
        require_calibration_report(path, boolean_key="passes_stability_gate")
