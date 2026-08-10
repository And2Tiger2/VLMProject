#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

try:
    from submit_neuronic_mechanistic_overnight import Submitter
except ModuleNotFoundError:
    from scripts.submit_neuronic_mechanistic_overnight import Submitter


REPO_DEFAULT = "/n/fs/pvl-memory/at7979/VLMProject"
SEGMENT = Path("segments/mechanistic_heads_qwen3_8b")
GPU_SCRIPT = "scripts/slurm_neuronic_base_search.sh"
POST_SCRIPT = "scripts/slurm_neuronic_base_search_postprocess.sh"
AGG_SCRIPT = "scripts/slurm_neuronic_mechanistic_aggregate.sh"


def submit_base_search(submitter: Submitter, *, profile: str) -> str:
    instrumentation = submitter.submit(
        "base_search_instrumentation",
        GPU_SCRIPT,
        exports={"TASK": "instrumentation", "MODE": profile},
    )
    behavior = submitter.submit(
        "base_search_behavior",
        GPU_SCRIPT,
        exports={"TASK": "behavior", "MODE": profile},
        dependencies=[instrumentation],
    )
    if profile == "full":
        layers = submitter.submit(
            "base_search_discovery_layers",
            GPU_SCRIPT,
            exports={"TASK": "discovery", "MODE": "full"},
            dependencies=[instrumentation],
            array="0-35%4",
        )
        discovery = submitter.submit(
            "base_search_discovery_aggregate",
            AGG_SCRIPT,
            exports={"SOURCE_TASK": "base-search-heads"},
            dependencies=[layers],
        )
    elif profile == "smoke":
        discovery = submitter.submit(
            "base_search_discovery",
            GPU_SCRIPT,
            exports={"TASK": "discovery", "MODE": "smoke"},
            dependencies=[instrumentation],
        )
    else:
        raise ValueError(f"unknown base-search profile: {profile}")
    ranking = submitter.submit(
        "base_search_ranking",
        POST_SCRIPT,
        exports={"MODE": profile},
        dependencies=[discovery],
    )
    return submitter.submit(
        "base_search_locked_validation",
        GPU_SCRIPT,
        exports={"TASK": "validation", "MODE": profile},
        dependencies=[behavior, ranking],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit the frozen-base visual-search head DAG (no training or adapters).")
    parser.add_argument("--repo", type=Path, default=Path(REPO_DEFAULT))
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dataset = args.repo / SEGMENT / "data/generated/point_search/waldo_like.jsonl"
    if not dataset.is_file():
        raise FileNotFoundError(f"prepared Waldo-like dataset is missing: {dataset}; run prepare-synthetic once")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    submitter = Submitter(repo=args.repo, dry_run=args.dry_run)
    validation = submit_base_search(submitter, profile=args.profile)
    receipt = {
        "label": "frozen-base visual-search head diagnosis",
        "profile": args.profile,
        "base_model_only": True,
        "training_jobs": [],
        "adapter_checkpoints": [],
        "dry_run": args.dry_run,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": revision,
        "jobs": submitter.jobs,
        "terminal_jobs": [validation],
        "commands": submitter.commands,
    }
    receipt_path = args.repo / SEGMENT / "runs/base_search_submission.json"
    if not args.dry_run:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
