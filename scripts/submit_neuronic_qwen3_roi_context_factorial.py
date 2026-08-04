#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any

from vlm_eval.qwen3_high_bias_roi_attention import DEFAULT_ROI_ROOT, DEFAULT_VLMBIAS
from vlm_eval.qwen3_roi_attention import DEFAULT_GAZE_RANKING, read_jsonl
from vlm_eval.qwen3_roi_context_factorial import (
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_REPORT_ROOT,
    DEFAULT_RUN_ROOT,
)


REPO_DEFAULT = Path("/n/fs/pvl-memory/at7979/VLMProject")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit the four-arm Qwen3 ROI/context factorial."
    )
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not args.skip_preflight:
        _preflight(repo)
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
    prepare = submitter.sbatch(
        "scripts/slurm_neuronic_qwen3_roi_context_factorial_prepare.sh",
        exports=exports,
    )
    run = submitter.sbatch(
        "scripts/slurm_neuronic_qwen3_roi_context_factorial_condition.sh",
        exports=exports,
        dependency=prepare,
        array="0-3%4",
    )
    aggregate = submitter.sbatch(
        "scripts/slurm_neuronic_qwen3_roi_context_factorial_aggregate.sh",
        exports=exports,
        dependency=run,
    )
    result: dict[str, Any] = {
        "experiment": "qwen3_roi_context_factorial_v1",
        "jobs": {"prepare": prepare, "run": run, "aggregate": aggregate},
        "final_job": aggregate,
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
        dependency: str | None = None,
        array: str | None = None,
    ) -> str:
        command = ["sbatch", "--parsable"]
        if array is not None:
            command.extend(["--array", array])
        if dependency is not None:
            command.extend(["--dependency", f"afterok:{dependency}"])
        command.extend(
            [
                "--export",
                "ALL,"
                + ",".join(f"{key}={value}" for key, value in sorted(exports.items())),
                script,
            ]
        )
        rendered = shlex.join(command)
        self.commands.append(rendered)
        print(rendered, flush=True)
        if self.dry_run:
            self.counter += 1
            return f"DRYRUN{self.counter}"
        job_id = subprocess.check_output(command, text=True).strip()
        print(f"submitted {job_id}: {script}", flush=True)
        return job_id


def _preflight(repo: Path) -> None:
    required = [
        repo / DEFAULT_GAZE_RANKING,
        repo / DEFAULT_VLMBIAS,
        repo / DEFAULT_ROI_ROOT / "accepted.jsonl",
        repo / "scripts/slurm_neuronic_qwen3_roi_context_factorial_prepare.sh",
        repo / "scripts/slurm_neuronic_qwen3_roi_context_factorial_condition.sh",
        repo / "scripts/slurm_neuronic_qwen3_roi_context_factorial_aggregate.sh",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing required inputs:\n- " + "\n- ".join(missing))
    ranking = json.loads((repo / DEFAULT_GAZE_RANKING).read_text(encoding="utf-8"))
    if not isinstance(ranking, list) or len(ranking) < 50:
        raise SystemExit("The Qwen3 gaze ranking must contain at least 50 heads")
    rows = read_jsonl(repo / DEFAULT_VLMBIAS)
    if len(rows) != 114 or len({str(row["id"]) for row in rows}) != 114:
        raise SystemExit("Expected exactly 114 unique high-bias VLMBias rows")
    masks = read_jsonl(repo / DEFAULT_ROI_ROOT / "accepted.jsonl")
    if len(masks) != 114:
        raise SystemExit("Expected exactly 114 accepted ROI masks")
    for row in masks:
        mask = repo / DEFAULT_ROI_ROOT / row["artifacts"]["tight_mask"]
        if not mask.is_file():
            raise SystemExit(f"Missing tight mask: {mask}")


if __name__ == "__main__":
    main()
