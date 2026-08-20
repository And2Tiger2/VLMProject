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
DEFAULT_SEED = 260809001


def submit_base_search(
    submitter: Submitter,
    *,
    profile: str,
    seeds: tuple[int, ...] = (DEFAULT_SEED,),
) -> str:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("base-search seeds must be a non-empty unique sequence")
    instrumentation = submitter.submit(
        "base_search_instrumentation",
        GPU_SCRIPT,
        exports={"TASK": "instrumentation", "MODE": profile, "SEED": str(seeds[0])},
    )
    behavior = submitter.submit(
        "base_search_behavior",
        GPU_SCRIPT,
        exports={"TASK": "behavior", "MODE": profile, "SEED": str(seeds[0])},
        dependencies=[instrumentation],
    )
    if profile not in {"full", "smoke"}:
        raise ValueError(f"unknown base-search profile: {profile}")

    validations: list[str] = []
    multi_seed = len(seeds) > 1
    for seed in seeds:
        suffix = f"_seed_{seed}" if multi_seed else ""
        namespace = f"seed-{seed}" if multi_seed else ""
        common_exports = {
            "MODE": profile,
            "SEED": str(seed),
            "RUN_NAMESPACE": namespace,
        }
        if profile == "full":
            layers = submitter.submit(
                f"base_search_discovery_layers{suffix}",
                GPU_SCRIPT,
                exports={"TASK": "discovery", **common_exports},
                dependencies=[instrumentation],
                array="0-35%4",
            )
            root = "segments/mechanistic_heads_qwen3_8b/runs/base_search_head_scan/full"
            if namespace:
                root = f"{root}/{namespace}"
            discovery = submitter.submit(
                f"base_search_discovery_aggregate{suffix}",
                AGG_SCRIPT,
                exports={
                    "SOURCE_TASK": "base-search-heads",
                    "INPUT_ROOT": root,
                    "OUTPUT_DIR": root,
                    "SEED": str(seed),
                },
                dependencies=[layers],
            )
        else:
            discovery = submitter.submit(
                f"base_search_discovery{suffix}",
                GPU_SCRIPT,
                exports={"TASK": "discovery", **common_exports},
                dependencies=[instrumentation],
            )
        ranking = submitter.submit(
            f"base_search_ranking{suffix}",
            POST_SCRIPT,
            exports={"TASK": "ranking", **common_exports},
            dependencies=[discovery],
        )
        validations.append(
            submitter.submit(
                f"base_search_locked_validation{suffix}",
                GPU_SCRIPT,
                exports={"TASK": "validation", **common_exports},
                dependencies=[behavior, ranking],
            )
        )

    if not multi_seed:
        return validations[0]
    return submitter.submit(
        "base_search_seed_summary",
        POST_SCRIPT,
        exports={
            "TASK": "seed-summary",
            "MODE": profile,
            # Slurm uses commas to delimit --export entries, so encode this
            # list with colons and decode it in the postprocess wrapper.
            "SEEDS": ":".join(str(seed) for seed in seeds),
        },
        dependencies=validations,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit the frozen-base visual-search head DAG (no training or adapters).")
    parser.add_argument("--repo", type=Path, default=Path(REPO_DEFAULT))
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_SEED])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dataset = args.repo / SEGMENT / "data/generated/point_search/waldo_like.jsonl"
    if not dataset.is_file():
        raise FileNotFoundError(f"prepared Waldo-like dataset is missing: {dataset}; run prepare-synthetic once")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    submitter = Submitter(repo=args.repo, dry_run=args.dry_run)
    seeds = tuple(args.seeds)
    terminal = submit_base_search(submitter, profile=args.profile, seeds=seeds)
    receipt = {
        "label": "frozen-base visual-search head diagnosis",
        "profile": args.profile,
        "seeds": list(seeds),
        "seed_namespaces": {
            str(seed): (f"seed-{seed}" if len(seeds) > 1 else "") for seed in seeds
        },
        "base_model_only": True,
        "training_jobs": [],
        "adapter_checkpoints": [],
        "dry_run": args.dry_run,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": revision,
        "jobs": submitter.jobs,
        "terminal_jobs": [terminal],
        "commands": submitter.commands,
    }
    receipt_path = args.repo / SEGMENT / "runs/base_search_submission.json"
    if not args.dry_run:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
