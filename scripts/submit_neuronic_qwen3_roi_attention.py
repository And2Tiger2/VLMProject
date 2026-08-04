#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any

from vlm_eval.qwen3_roi_attention import (
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_GAZE_RANKING,
    DEFAULT_REPORT_ROOT,
    DEFAULT_ROI_ROOT,
    DEFAULT_RUN_ROOT,
    DEFAULT_VLMBIAS,
    read_jsonl,
)


REPO_DEFAULT = Path("/n/fs/pvl-memory/at7979/VLMProject")
STAGE_COUNTS = {"smoke": 4, "tune": 7, "heads": 12, "confirm": 5}
STAGE_ORDER = ("smoke", "tune", "heads", "confirm")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit the staged Qwen3 localized-ROI attention experiment."
    )
    parser.add_argument("stage", choices=["full", *STAGE_ORDER])
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not args.skip_preflight:
        _preflight(repo, args.stage)
    if not args.dry_run:
        (repo / "segments/vlm_bias_attention/runs/slurm").mkdir(
            parents=True, exist_ok=True
        )

    exports = {
        "REPO": str(repo),
        "EXPERIMENT_ROOT": str(DEFAULT_EXPERIMENT_ROOT),
        "RUN_ROOT": str(DEFAULT_RUN_ROOT),
        "REPORT_ROOT": str(DEFAULT_REPORT_ROOT),
        "ROI_ROOT": str(DEFAULT_ROI_ROOT),
        "VLMBIAS_DATASET": str(DEFAULT_VLMBIAS),
        "GAZE_RANKING": str(DEFAULT_GAZE_RANKING),
        "MIN_GPU_MEMORY_GB": "20.0",
    }
    submitter = Submitter(dry_run=args.dry_run)
    stages = STAGE_ORDER if args.stage == "full" else (args.stage,)
    dependency = None
    jobs: dict[str, dict[str, str]] = {}
    for stage in stages:
        stage_jobs = _submit_stage(
            submitter,
            stage=stage,
            count=STAGE_COUNTS[stage],
            max_parallel=2 if stage == "smoke" else args.max_parallel,
            dependency=dependency,
            exports=exports,
        )
        jobs[stage] = stage_jobs
        dependency = stage_jobs["aggregate"]

    result: dict[str, Any] = {
        "stage": args.stage,
        "jobs": jobs,
        "final_job": dependency,
        "dry_run": args.dry_run,
        "commands": submitter.commands,
    }
    if not args.dry_run:
        receipt = repo / DEFAULT_EXPERIMENT_ROOT / "last_submission.json"
        result.update(
            {
                "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                ).strip(),
                "receipt": str(receipt),
            }
        )
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def _submit_stage(
    submitter: "Submitter",
    *,
    stage: str,
    count: int,
    max_parallel: int,
    dependency: str | None,
    exports: dict[str, str],
) -> dict[str, str]:
    stage_exports = {**exports, "STAGE": stage}
    prepare = submitter.sbatch(
        "scripts/slurm_neuronic_qwen3_roi_prepare.sh",
        exports=stage_exports,
        dependency=dependency,
    )
    run = submitter.sbatch(
        "scripts/slurm_neuronic_qwen3_roi_condition.sh",
        exports=stage_exports,
        dependency=prepare,
        array=f"0-{count - 1}%{min(count, max_parallel)}",
    )
    aggregate = submitter.sbatch(
        "scripts/slurm_neuronic_qwen3_roi_aggregate.sh",
        exports=stage_exports,
        dependency=run,
    )
    return {"prepare": prepare, "run": run, "aggregate": aggregate}


class Submitter:
    def __init__(self, *, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.counter = 0
        self.commands: list[str] = []

    def sbatch(
        self,
        script: str,
        *,
        exports: dict[str, str],
        dependency: str | None,
        array: str | None = None,
    ) -> str:
        command = ["sbatch", "--parsable"]
        if array is not None:
            command.extend(["--array", array])
        if dependency is not None:
            command.extend(["--dependency", f"afterok:{dependency}"])
        export_value = "ALL," + ",".join(
            f"{key}={value}" for key, value in sorted(exports.items())
        )
        command.extend(["--export", export_value, script])
        rendered = shlex.join(command)
        self.commands.append(rendered)
        print(rendered, flush=True)
        if self.dry_run:
            self.counter += 1
            return f"DRYRUN{self.counter}"
        job_id = subprocess.check_output(command, text=True).strip()
        print(f"submitted {job_id}: {script}", flush=True)
        return job_id


def _preflight(repo: Path, stage: str) -> None:
    required = [
        repo / DEFAULT_GAZE_RANKING,
        repo / DEFAULT_GAZE_RANKING.with_name("gaze_scores.npy"),
        repo / DEFAULT_VLMBIAS,
        repo / DEFAULT_ROI_ROOT / "accepted.jsonl",
        repo / "scripts/slurm_neuronic_qwen3_roi_prepare.sh",
        repo / "scripts/slurm_neuronic_qwen3_roi_condition.sh",
        repo / "scripts/slurm_neuronic_qwen3_roi_aggregate.sh",
    ]
    if stage in {"heads", "confirm"}:
        required.append(repo / DEFAULT_REPORT_ROOT / "tune/selection.json")
    if stage == "confirm":
        required.append(repo / DEFAULT_REPORT_ROOT / "heads/selection.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing required inputs:\n- " + "\n- ".join(missing))

    ranking = json.loads((repo / DEFAULT_GAZE_RANKING).read_text(encoding="utf-8"))
    if not isinstance(ranking, list) or len(ranking) < 100:
        raise SystemExit("The Qwen3 gaze ranking must contain at least 100 heads")
    vlmbias = read_jsonl(repo / DEFAULT_VLMBIAS)
    if len(vlmbias) != 400 or len({str(row["id"]) for row in vlmbias}) != 400:
        raise SystemExit("Expected exactly 400 unique VLMBias rows")
    for row in vlmbias:
        image = Path(str(row.get("image_path") or "__missing__"))
        if not image.is_absolute():
            image = repo / DEFAULT_VLMBIAS.parent / image
        if not image.is_file():
            raise SystemExit(f"Missing VLMBias image: {image}")
    accepted = read_jsonl(repo / DEFAULT_ROI_ROOT / "accepted.jsonl")
    if len(accepted) < 100:
        raise SystemExit("Expected at least 100 reviewed ROI masks")
    covered = {
        str(example_id)
        for row in accepted
        for example_id in row.get("covered_local_ids", [row["id"]])
    }
    source_ids = {str(row["id"]) for row in vlmbias}
    if not covered <= source_ids:
        raise SystemExit("ROI manifest contains IDs absent from the VLMBias slice")
    for row in accepted:
        mask = repo / DEFAULT_ROI_ROOT / row["artifacts"]["mask_path"]
        if not mask.is_file():
            raise SystemExit(f"Missing ROI mask: {mask}")


if __name__ == "__main__":
    main()
