#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Iterable


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
        array: str | None = None,
    ) -> str:
        command = ["sbatch", "--parsable"]
        deps = [str(value) for value in dependencies if value]
        if deps:
            command.append("--dependency=afterok:" + ":".join(deps))
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
        array: str | None = None,
    ) -> str:
        return self.submit(
            name,
            GPU_SCRIPT,
            exports={"TASK": task, "MODE": mode},
            dependencies=dependencies,
            array=array,
        )

    def scan(
        self,
        name: str,
        task: str,
        *,
        dependencies: Iterable[str],
    ) -> str:
        run = self.gpu(
            f"{name}_layers",
            task,
            "full",
            dependencies=dependencies,
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

    # Discovery arrays are serialized so this suite requests at most four scan
    # GPUs at once. Each array still evaluates four layers concurrently.
    count_vap = submitter.scan("counting_vap", "counting-vap", dependencies=[point_train])
    count_heads = submitter.scan("counting_heads", "counting-heads", dependencies=[count_vap])

    point_centroids = submitter.scan("point_centroids", "point-centroids", dependencies=[count_heads, point_train])
    search = submitter.scan("search_heads", "search-heads", dependencies=[point_centroids])
    verification = submitter.scan("verification_heads", "verification-heads", dependencies=[search])
    distractor = submitter.scan("distractor_heads", "distractor-heads", dependencies=[verification])
    point_ablation = submitter.gpu(
        "point_ablation",
        "point-ablation",
        "full",
        dependencies=[search, verification, distractor],
    )
    point_reports = submitter.submit(
        "point_reports",
        POST_SCRIPT,
        exports={"TASK": "point-reports"},
        dependencies=[point_ablation, point_behavior, waldo_behavior],
    )

    maci = submitter.scan("maci_heads", "maci-heads", dependencies=[distractor])
    maci_aligned = submitter.scan("maci_heads_aligned", "maci-heads-aligned", dependencies=[maci])
    maci_ablation = submitter.gpu("maci_ablation", "maci-ablation", "full", dependencies=[maci])
    maci_detector = submitter.gpu("maci_detector", "maci-detector", "full", dependencies=[maci])
    maci_gated = submitter.gpu("maci_gated", "maci-gated", "full", dependencies=[maci_detector])
    maci_confirm = submitter.gpu("maci_confirmation", "maci-confirm", "full", dependencies=[maci_ablation])

    vlmbias = submitter.scan("vlmbias_heads", "vlmbias-heads", dependencies=[maci_aligned])
    vlmbias_validation = submitter.gpu(
        "vlmbias_validation",
        "vlmbias-validation",
        "full",
        dependencies=[vlmbias, maci_detector],
    )

    general = submitter.submit(
        "general_importance",
        POST_SCRIPT,
        exports={"TASK": "general-importance"},
        dependencies=[count_heads, search, verification, distractor, maci, vlmbias],
    )
    # Counting controls wait for cross-task general importance so the required
    # general-causal-importance-matched distribution is not silently omitted.
    count_controls = submitter.submit(
        "counting_controls",
        POST_SCRIPT,
        exports={"TASK": "counting-controls"},
        dependencies=[general],
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
        dependencies=[general, count_validation, point_reports, maci_gated, maci_confirm, vlmbias_validation],
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.profile == "all" and not args.confirm_full:
        parser.error("--profile all requires explicit --confirm-full")

    args.repo.joinpath("segments/mechanistic_heads_qwen3_8b/runs/slurm").mkdir(
        parents=True, exist_ok=True
    )
    submitter = Submitter(repo=args.repo, dry_run=args.dry_run)
    prepare = submitter.submit("prepare_data", PREP_SCRIPT, exports={}, dependencies=[])
    instrumentation = submitter.gpu(
        "instrumentation", "instrumentation", "smoke", dependencies=[prepare]
    )
    smokes = submit_smokes(submitter, instrumentation=instrumentation)
    finals: list[str] = smokes
    if args.profile == "all":
        finals = submit_full_suite(submitter, smoke_barrier=smokes)

    receipt = {
        "profile": args.profile,
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


if __name__ == "__main__":
    main()
