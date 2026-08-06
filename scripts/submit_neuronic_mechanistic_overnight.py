#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Iterable

from vlm_eval.mechanistic_heads.preflight import require_completed_manifest
from vlm_eval.mechanistic_heads.reproducibility import sha256_file


REPO_DEFAULT = "/n/fs/pvl-memory/at7979/VLMProject"
GPU_SCRIPT = "scripts/slurm_neuronic_mechanistic_heads.sh"
AGG_SCRIPT = "scripts/slurm_neuronic_mechanistic_aggregate.sh"
PREP_SCRIPT = "scripts/slurm_neuronic_mechanistic_prepare.sh"
POST_SCRIPT = "scripts/slurm_neuronic_mechanistic_postprocess.sh"


class Submitter:
    def __init__(self, *, repo: Path, dry_run: bool) -> None:
        self.repo = repo
        self.dry_run = dry_run
        self.commands: list[list[str]] = []
        self.jobs: dict[str, str] = {}
        self._dry_counter = 9000000

    def submit(
        self,
        name: str,
        script: str,
        *,
        exports: dict[str, str],
        dependencies: Iterable[str] = (),
        afterany_dependencies: Iterable[str] = (),
        array: str | None = None,
        dependency_mode: str = "afterok",
    ) -> str:
        if name in self.jobs:
            raise RuntimeError(f"duplicate Slurm job name in submission graph: {name}")
        command = ["sbatch", "--parsable"]
        if dependency_mode not in {"afterok", "afterany"}:
            raise ValueError(f"unsupported dependency mode: {dependency_mode}")
        command.append("--kill-on-invalid-dep=yes")
        deps = [str(value) for value in dependencies if value]
        afterany = [str(value) for value in afterany_dependencies if value]
        dependency_specs = []
        if deps:
            dependency_specs.append(f"{dependency_mode}:" + ":".join(deps))
        if afterany:
            dependency_specs.append("afterany:" + ":".join(afterany))
        if dependency_specs:
            command.append("--dependency=" + ",".join(dependency_specs))
        if array is not None:
            command.append(f"--array={array}")
        exported = {"REPO": str(self.repo), **exports}
        command.append("--export=ALL," + ",".join(f"{key}={value}" for key, value in exported.items()))
        command.append(script)
        self.commands.append(command)
        print(" ".join(command), flush=True)
        if self.dry_run:
            self._dry_counter += 1
            job_id = str(self._dry_counter)
        else:
            completed = subprocess.run(
                command,
                cwd=self.repo,
                check=True,
                text=True,
                capture_output=True,
            )
            job_id = completed.stdout.strip().split(";", 1)[0]
            if not job_id:
                raise RuntimeError(f"sbatch returned no job ID for {name}")
        self.jobs[name] = job_id
        print(f"submitted {name}: {job_id}", flush=True)
        return job_id

    def gpu(
        self,
        name: str,
        task: str,
        mode: str,
        *,
        dependencies: Iterable[str],
        afterany_dependencies: Iterable[str] = (),
        array: str | None = None,
    ) -> str:
        return self.submit(
            name,
            GPU_SCRIPT,
            exports={"TASK": task, "MODE": mode},
            dependencies=dependencies,
            afterany_dependencies=afterany_dependencies,
            array=array,
        )

    def scan(
        self,
        name: str,
        task: str,
        *,
        dependencies: Iterable[str],
        afterany_dependencies: Iterable[str] = (),
    ) -> str:
        run = self.gpu(
            f"{name}_layers",
            task,
            "full",
            dependencies=dependencies,
            afterany_dependencies=afterany_dependencies,
            array="0-35%4",
        )
        return self.submit(
            f"{name}_aggregate",
            AGG_SCRIPT,
            exports={"SOURCE_TASK": task},
            dependencies=[run],
        )


def submit_smokes(submitter: Submitter, *, instrumentation: str) -> list[str]:
    independent = [
        submitter.gpu("smoke_counting_behavior", "counting-behavior", "smoke", dependencies=[instrumentation]),
        submitter.gpu("smoke_counting_vap", "counting-vap", "smoke", dependencies=[instrumentation]),
        submitter.gpu("smoke_counting_heads", "counting-heads", "smoke", dependencies=[instrumentation]),
        submitter.gpu("smoke_maci_heads", "maci-heads", "smoke", dependencies=[instrumentation]),
        submitter.gpu("smoke_maci_aligned", "maci-heads-aligned", "smoke", dependencies=[instrumentation]),
        submitter.gpu("smoke_vlmbias_heads", "vlmbias-heads", "smoke", dependencies=[instrumentation]),
    ]
    point_train = submitter.gpu(
        "smoke_point_training",
        "point-train-all",
        "smoke",
        dependencies=[instrumentation],
        array="0-3%4",
    )
    point_smokes = [
        submitter.gpu("smoke_point_behavior", "point-behavior-all", "smoke", dependencies=[point_train], array="0-4%4"),
        submitter.gpu("smoke_waldo_behavior", "waldo-behavior", "smoke", dependencies=[point_train]),
        submitter.gpu("smoke_point_centroids", "point-centroids", "smoke", dependencies=[point_train]),
        submitter.gpu("smoke_search_heads", "search-heads", "smoke", dependencies=[point_train]),
        submitter.gpu("smoke_verification_heads", "verification-heads", "smoke", dependencies=[point_train]),
        submitter.gpu("smoke_distractor_heads", "distractor-heads", "smoke", dependencies=[point_train]),
    ]
    return independent + [point_train] + point_smokes


def submit_full_suite(
    submitter: Submitter,
    *,
    smoke_barrier: list[str],
) -> list[str]:
    point_train = submitter.gpu(
        "full_point_training",
        "point-train-all",
        "full",
        dependencies=smoke_barrier,
        array="0-3%4",
    )
    point_behavior = submitter.gpu(
        "full_point_behavior",
        "point-behavior-all",
        "full",
        dependencies=[point_train],
        array="0-4%4",
    )
    counting_behavior = submitter.gpu(
        "full_counting_behavior",
        "counting-behavior",
        "full",
        dependencies=smoke_barrier,
    )
    waldo_behavior = submitter.gpu(
        "full_waldo_behavior",
        "waldo-behavior",
        "full",
        dependencies=[point_train],
    )

    # Discovery arrays are resource-serialized so this suite requests at most
    # four scan GPUs at once. Cross-study serialization uses ``afterany``:
    # failure in one scientifically independent study must not invalidate all
    # later studies. True data/calibration prerequisites remain ``afterok``.
    count_vap = submitter.scan(
        "counting_vap",
        "counting-vap",
        dependencies=[*smoke_barrier, counting_behavior],
        afterany_dependencies=[point_train],
    )
    count_heads = submitter.scan(
        "counting_heads",
        "counting-heads",
        dependencies=[*smoke_barrier, counting_behavior],
        afterany_dependencies=[count_vap],
    )
    count_heads_repeat1 = submitter.scan(
        "counting_heads_repeat1",
        "counting-heads-repeat1",
        dependencies=[count_heads],
    )
    count_heads_repeat2 = submitter.scan(
        "counting_heads_repeat2",
        "counting-heads-repeat2",
        dependencies=[count_heads_repeat1],
    )

    point_centroids = submitter.scan(
        "point_centroids",
        "point-centroids",
        dependencies=[point_train, point_behavior],
        afterany_dependencies=[count_heads_repeat2],
    )
    search = submitter.scan(
        "search_heads",
        "search-heads",
        dependencies=[point_train, point_behavior, waldo_behavior],
        afterany_dependencies=[point_centroids],
    )
    verification = submitter.scan(
        "verification_heads",
        "verification-heads",
        dependencies=[point_train, point_behavior, waldo_behavior],
        afterany_dependencies=[search],
    )
    distractor = submitter.scan(
        "distractor_heads",
        "distractor-heads",
        dependencies=[point_train, point_behavior, waldo_behavior],
        afterany_dependencies=[verification],
    )
    maci = submitter.scan(
        "maci_heads",
        "maci-heads",
        dependencies=smoke_barrier,
        afterany_dependencies=[distractor],
    )
    maci_stability = submitter.submit(
        "maci_stability",
        POST_SCRIPT,
        exports={"TASK": "maci-stability"},
        dependencies=[maci],
    )
    maci_ablation = submitter.gpu("maci_ablation", "maci-ablation", "full", dependencies=[maci])
    maci_detector = submitter.gpu("maci_detector", "maci-detector", "full", dependencies=[maci, maci_stability, maci_ablation])
    maci_gated = submitter.gpu("maci_gated", "maci-gated", "full", dependencies=[maci_detector])

    # VLMBias does not consume the optional equal-length all-prefill MACI
    # branch. Run the core VLMBias scan first, then serialize the secondary
    # aligned scan behind it so a legitimate unequal-length exclusion cannot
    # invalidate the core path or increase scan concurrency above four GPUs.
    vlmbias = submitter.scan(
        "vlmbias_heads",
        "vlmbias-heads",
        dependencies=smoke_barrier,
        afterany_dependencies=[maci],
    )
    maci_aligned = submitter.scan(
        "maci_heads_aligned", "maci-heads-aligned", dependencies=[vlmbias]
    )

    general = submitter.submit(
        "general_importance",
        POST_SCRIPT,
        exports={"TASK": "general-importance"},
        dependencies=[count_heads, search, verification, distractor, maci, vlmbias],
    )
    # These locked validations require the cross-task importance table for
    # matched controls. Scheduling them before `general` makes them fail even
    # when every upstream scan succeeded.
    point_ablation = submitter.gpu(
        "point_ablation",
        "point-ablation",
        "full",
        dependencies=[search, verification, distractor, general],
    )
    point_reports = submitter.submit(
        "point_reports",
        POST_SCRIPT,
        exports={"TASK": "point-reports"},
        dependencies=[point_ablation, point_behavior, waldo_behavior],
    )
    maci_confirm = submitter.gpu(
        "maci_confirmation",
        "maci-confirm",
        "full",
        dependencies=[maci_ablation, maci_stability, general],
    )
    vlmbias_validation = submitter.gpu(
        "vlmbias_validation",
        "vlmbias-validation",
        "full",
        dependencies=[vlmbias, maci_detector, general],
    )
    # Counting controls wait for cross-task general importance so the required
    # general-causal-importance-matched distribution is not silently omitted.
    count_controls = submitter.submit(
        "counting_controls",
        POST_SCRIPT,
        exports={"TASK": "counting-controls"},
        dependencies=[general, count_heads_repeat1, count_heads_repeat2],
    )
    count_validation = submitter.gpu(
        "counting_validation",
        "counting-validation",
        "full",
        dependencies=[count_controls],
    )
    atlas = submitter.submit(
        "head_atlas",
        POST_SCRIPT,
        exports={"TASK": "atlas"},
        dependencies=[general, count_validation, point_reports, maci_stability, maci_gated, maci_confirm, vlmbias_validation],
        dependency_mode="afterany",
    )
    return [
        point_train,
        point_behavior,
        counting_behavior,
        waldo_behavior,
        count_validation,
        point_ablation,
        point_reports,
        maci_aligned,
        maci_stability,
        maci_gated,
        maci_confirm,
        vlmbias_validation,
        general,
        atlas,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit the dependency-safe Qwen3 mechanistic overnight suite."
    )
    parser.add_argument("--repo", type=Path, default=Path(REPO_DEFAULT))
    parser.add_argument("--profile", choices=("smoke", "all"), default="smoke")
    parser.add_argument("--confirm-full", action="store_true")
    parser.add_argument(
        "--reuse-prepared",
        action="store_true",
        help="Reuse an already valid completed preparation stage.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.profile == "all" and not args.confirm_full:
        parser.error("--profile all requires explicit --confirm-full")

    args.repo.joinpath("segments/mechanistic_heads_qwen3_8b/runs/slurm").mkdir(
        parents=True, exist_ok=True
    )
    submitter = Submitter(repo=args.repo, dry_run=args.dry_run)
    prepare_dependencies: list[str] = []
    if args.reuse_prepared:
        require_valid_prepared_data(args.repo)
    else:
        prepare = submitter.submit(
            "prepare_data",
            PREP_SCRIPT,
            exports={"MODE": "smoke" if args.profile == "smoke" else "full"},
            dependencies=[],
        )
        prepare_dependencies = [prepare]
    instrumentation = submitter.gpu(
        "instrumentation",
        "instrumentation",
        "smoke",
        dependencies=prepare_dependencies,
    )
    smokes = submit_smokes(submitter, instrumentation=instrumentation)
    finals: list[str] = smokes
    if args.profile == "all":
        finals = submit_full_suite(submitter, smoke_barrier=smokes)

    receipt = {
        "profile": args.profile,
        "reuse_prepared": args.reuse_prepared,
        "dry_run": args.dry_run,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": submitter.jobs,
        "terminal_jobs": finals,
        "commands": submitter.commands,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=args.repo,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip(),
    }
    receipt_path = args.repo / "segments/mechanistic_heads_qwen3_8b/runs/overnight_submission.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


def require_valid_prepared_data(repo: Path) -> None:
    groups = {
        "segments/mechanistic_heads_qwen3_8b/data/generated/counting": (
            "mechanistic_pairs.jsonl", "mechanistic_pairs_repeat1.jsonl",
            "mechanistic_pairs_repeat2.jsonl", "constant_complexity_pairs.jsonl",
            "syndot.jsonl", "dataset_manifest.json"
        ),
        "segments/mechanistic_heads_qwen3_8b/data/generated/point_search": (
            "point_search.jsonl", "waldo_like.jsonl", "search_pairs.jsonl",
            "verification_pairs.jsonl", "distractor_pairs.jsonl", "dataset_manifest.json",
        ),
        "segments/mechanistic_heads_qwen3_8b/data/generated/vlmbias_contrasts": (
            "vlmbias_signed_contrasts.jsonl", "audit.json"
        ),
        "segments/mechanistic_heads_qwen3_8b/data/mmmc/prepared": (
            "object_pairs.jsonl", "audit.json"
        ),
    }
    missing = []
    manifests: dict[str, dict] = {}
    for root_value, names in groups.items():
        root = repo / root_value
        outputs = tuple(root / name for name in names)
        missing.extend(str(path.relative_to(repo)) for path in outputs if not path.is_file())
        if all(path.is_file() for path in outputs):
            manifests[root_value] = require_completed_manifest(
                root, expected_outputs=outputs
            )
    audit_path = repo / "segments/mechanistic_heads_qwen3_8b/data/mmmc/prepared/audit.json"
    if audit_path.is_file() and not json.loads(audit_path.read_text(encoding="utf-8")).get("valid"):
        raise RuntimeError(f"MMMC preparation audit is not valid: {audit_path}")
    if missing:
        raise FileNotFoundError(
            "cannot reuse preparation; missing required artifacts: " + ", ".join(missing)
        )
    # Prepared datasets may intentionally survive a Git revision, but only if
    # the exact generator/preparer implementation that produced them is still
    # current. Older manifests did not bind source files and must be rebuilt
    # instead of silently reusing scientifically different controls.
    expected_sources = {
        "segments/mechanistic_heads_qwen3_8b/data/generated/counting": (
            repo / "scripts/generate_counting_data.py",
            repo / "vlm_eval/mechanistic_heads/synthetic.py",
        ),
        "segments/mechanistic_heads_qwen3_8b/data/generated/point_search": (
            repo / "scripts/generate_point_search_data.py",
            repo / "vlm_eval/mechanistic_heads/synthetic.py",
        ),
        "segments/mechanistic_heads_qwen3_8b/data/generated/vlmbias_contrasts": (
            repo / "scripts/prepare_vlmbias_signed_contrasts.py",
        ),
        "segments/mechanistic_heads_qwen3_8b/data/mmmc/prepared": (
            repo / "scripts/prepare_mmmc.py",
        ),
    }
    for root_value, source_paths in expected_sources.items():
        _require_manifest_sources(manifests[root_value], source_paths)
    for root_value in (
        "segments/mechanistic_heads_qwen3_8b/data/generated/counting",
        "segments/mechanistic_heads_qwen3_8b/data/generated/point_search",
    ):
        root = repo / root_value
        declared = {
            str(Path(value).resolve())
            for value in manifests[root_value].get("output_sha256", {})
        }
        referenced: set[str] = set()

        def collect_paths(value: object) -> None:
            if isinstance(value, str) and value.lower().endswith(".png"):
                referenced.add(str(Path(value).resolve()))
            elif isinstance(value, dict):
                for child in value.values():
                    collect_paths(child)
            elif isinstance(value, list):
                for child in value:
                    collect_paths(child)

        for jsonl_path in root.glob("*.jsonl"):
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    collect_paths(json.loads(line))
        if not referenced:
            raise RuntimeError(f"prepared dataset contains no referenced PNGs: {root}")
        untracked = sorted(referenced - declared)
        if untracked:
            raise RuntimeError(
                f"prepared dataset has {len(untracked)} unhashed referenced PNGs under {root}; "
                "rerun preparation with the current pipeline"
            )
    counting = json.loads((repo / "segments/mechanistic_heads_qwen3_8b/data/generated/counting/dataset_manifest.json").read_text(encoding="utf-8"))
    point = json.loads((repo / "segments/mechanistic_heads_qwen3_8b/data/generated/point_search/dataset_manifest.json").read_text(encoding="utf-8"))
    mmmc = json.loads(audit_path.read_text(encoding="utf-8"))
    if counting.get("counts", {}).get("mechanistic_pairs") != 100:
        raise RuntimeError("prepared counting data does not contain the required 100 mechanistic pairs")
    if counting.get("counts", {}).get("mechanistic_repeat_pairs_each") != 100 or len(
        counting.get("counts", {}).get("mechanistic_repeat_seeds", [])
    ) < 2:
        raise RuntimeError(
            "prepared counting data does not contain two 100-pair cross-seed repeats"
        )
    if point.get("counts", {}).get("point_search_train") != 2000:
        raise RuntimeError("prepared point-search data does not contain the required 2,000 training scenes")
    expected_pair_splits = {"prototype": 300, "validation": 100, "locked_test": 100}
    pair_split_counts = point.get("waldo_pair_split_group_counts")
    if not isinstance(pair_split_counts, dict) or any(
        family_counts != expected_pair_splits
        for family_counts in pair_split_counts.values()
    ) or set(pair_split_counts) != {"search", "verification", "distractor"}:
        raise RuntimeError(
            "prepared Waldo-like head pairs do not have the required disjoint "
            "300/100/100 prototype/validation/locked split; rerun preparation"
        )
    if int(mmmc.get("split_pair_counts", {}).get("locked_test", 0)) < 500:
        raise RuntimeError("prepared MMMC data does not contain at least 500 locked examples")
    if not isinstance(mmmc.get("dataset_fingerprints"), dict) or not mmmc[
        "dataset_fingerprints"
    ]:
        raise RuntimeError(
            "prepared MMMC data predates source-fingerprint validation; rerun preparation"
        )
    if mmmc.get("same_image_prompt_contrast") is not True:
        raise RuntimeError(
            "prepared MMMC pairs do not hold the conflict image fixed across prompts; rerun preparation"
        )
    mmmc_pairs = repo / "segments/mechanistic_heads_qwen3_8b/data/mmmc/prepared/object_pairs.jsonl"
    for line in mmmc_pairs.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["donor_image"] != row["recipient_image"]:
                raise RuntimeError(
                    "prepared MMMC contains a clean/conflict pair with different images; rerun preparation"
                )
    if point.get("waldo_pair_masks_complete") is not True:
        raise RuntimeError("prepared Waldo-like causal pairs lack exact target/decoy masks")
    if point.get("waldo_relocation_distractors_matched") is not True:
        raise RuntimeError("prepared Waldo-like relocation pairs do not hold distractors fixed")
    if len(point.get("four_candidate_target_indices", [])) < 2:
        raise RuntimeError("prepared four-candidate task has a constant target slot")


def _require_manifest_sources(manifest: dict, source_paths: Iterable[Path]) -> None:
    declared = {
        str(Path(path).resolve()): digest
        for path, digest in manifest.get("input_sha256", {}).items()
    }
    stale = []
    for source in source_paths:
        resolved = source.resolve()
        expected = declared.get(str(resolved))
        if expected is None or not resolved.is_file() or sha256_file(resolved) != expected:
            stale.append(str(source))
    if stale:
        raise RuntimeError(
            "prepared data is not bound to the current generator/preparer source; "
            "rerun preparation: " + ", ".join(stale)
        )


if __name__ == "__main__":
    main()
