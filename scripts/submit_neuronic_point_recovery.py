#!/usr/bin/env python3
"""Submit the minimal recovery DAG after the point-label training fix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

try:
    from submit_neuronic_mechanistic_overnight import Submitter
except ModuleNotFoundError:  # Imported as a module from the repository root.
    from scripts.submit_neuronic_mechanistic_overnight import Submitter


REPO_DEFAULT = "/n/fs/pvl-memory/at7979/VLMProject"
SEGMENT = Path("segments/mechanistic_heads_qwen3_8b")
RECOVERY_TARGETS = (
    Path("reports/instrumentation"),
    Path("checkpoints/smoke/direct-answer-lora"),
    Path("checkpoints/smoke/direct-length-matched-lora"),
    Path("checkpoints/smoke/point-answer-lora"),
    Path("checkpoints/smoke/shuffled-point-answer-lora"),
    Path("checkpoints/direct-answer-lora"),
    Path("checkpoints/direct-length-matched-lora"),
    Path("checkpoints/point-answer-lora"),
    Path("checkpoints/shuffled-point-answer-lora"),
    Path("runs/point_behavior_smoke/base"),
    Path("runs/point_behavior_smoke/direct_answer"),
    Path("runs/point_behavior_smoke/direct_length_matched"),
    Path("runs/point_behavior_smoke/point_answer"),
    Path("runs/point_behavior_smoke/shuffled_point_answer"),
    Path("runs/point_behavior/base"),
    Path("runs/point_behavior/direct_answer"),
    Path("runs/point_behavior/direct_length_matched"),
    Path("runs/point_behavior/point_answer"),
    Path("runs/point_behavior/shuffled_point_answer"),
    Path("runs/waldo_behavior/smoke"),
    Path("runs/waldo_behavior/full"),
    Path("runs/point_attention_centroids/smoke"),
    Path("runs/point_attention_centroids/full"),
    Path("runs/search_head_scan/smoke"),
    Path("runs/search_head_scan/full"),
    Path("runs/verification_head_scan/smoke"),
    Path("runs/verification_head_scan/full"),
    Path("runs/distractor_head_scan/smoke"),
    Path("runs/distractor_head_scan/full"),
)


def git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def archive_invalid_point_outputs(
    repo: Path,
    *,
    execute: bool,
    stamp: str | None = None,
    revision: str | None = None,
) -> dict:
    """Recoverably move only outputs invalidated by the point-label bug."""

    revision = revision or git_commit(repo)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    segment_root = repo / SEGMENT
    archive_root = segment_root / "archive" / f"point-label-recovery-{stamp}-{revision[:8]}"
    existing = [relative for relative in RECOVERY_TARGETS if (segment_root / relative).exists()]
    moved: list[dict[str, str]] = []
    for relative in existing:
        source = segment_root / relative
        destination = archive_root / relative
        print(f"archive {source} -> {destination}", flush=True)
        if execute:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        moved.append(
            {
                "source": str(source.relative_to(repo)),
                "destination": str(destination.relative_to(repo)),
            }
        )
    return {
        "execute": execute,
        "n_archived": len(moved),
        "archive_root": str(archive_root.relative_to(repo)) if moved else None,
        "paths": moved,
    }


def submit_point_recovery(submitter: Submitter) -> list[str]:
    """Build a fail-closed smoke-to-full DAG for the affected branch only."""

    instrumentation = submitter.gpu(
        "instrumentation", "instrumentation", "smoke", dependencies=[]
    )
    smoke_training = submitter.gpu(
        "smoke_point_training",
        "point-train-all",
        "smoke",
        dependencies=[instrumentation],
        array="0-3%4",
    )
    smoke_behavior = submitter.gpu(
        "smoke_point_behavior",
        "point-behavior-all",
        "smoke",
        dependencies=[smoke_training],
        array="0-4%4",
    )
    smoke_waldo = submitter.gpu(
        "smoke_waldo_behavior",
        "waldo-behavior",
        "smoke",
        dependencies=[smoke_training],
    )
    smoke_scans = [
        submitter.gpu(
            f"smoke_{name}", task, "smoke", dependencies=[smoke_training]
        )
        for name, task in (
            ("point_centroids", "point-centroids"),
            ("search_heads", "search-heads"),
            ("verification_heads", "verification-heads"),
            ("distractor_heads", "distractor-heads"),
        )
    ]

    smoke_barrier = [smoke_behavior, smoke_waldo, *smoke_scans]
    full_training = submitter.gpu(
        "full_point_training",
        "point-train-all",
        "full",
        dependencies=smoke_barrier,
        array="0-3%4",
    )
    full_behavior = submitter.gpu(
        "full_point_behavior",
        "point-behavior-all",
        "full",
        dependencies=[full_training],
        array="0-4%4",
    )
    full_waldo = submitter.gpu(
        "full_waldo_behavior",
        "waldo-behavior",
        "full",
        dependencies=[full_training],
    )

    # These four layer scans are independent once both behavioral calibration
    # gates pass, so run them concurrently instead of serializing them.
    calibration_barrier = [full_behavior, full_waldo]
    full_scans = [
        submitter.scan(name, task, dependencies=calibration_barrier)
        for name, task in (
            ("point_centroids", "point-centroids"),
            ("search_heads", "search-heads"),
            ("verification_heads", "verification-heads"),
            ("distractor_heads", "distractor-heads"),
        )
    ]
    return full_scans


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit only the point/Waldo recovery jobs invalidated by the label bug."
    )
    parser.add_argument("--repo", type=Path, default=Path(REPO_DEFAULT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    revision = git_commit(args.repo)
    archive = archive_invalid_point_outputs(
        args.repo, execute=not args.dry_run, revision=revision
    )
    submitter = Submitter(repo=args.repo, dry_run=args.dry_run)
    terminal_jobs = submit_point_recovery(submitter)
    receipt = {
        "label": "point-label recovery",
        "profile": "point-recovery",
        "reuse_prepared": True,
        "preserved_completed_branches": ["counting", "maci", "vlmbias"],
        "dry_run": args.dry_run,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": archive,
        "jobs": submitter.jobs,
        "terminal_jobs": terminal_jobs,
        "commands": submitter.commands,
        "git_commit": revision,
    }
    receipt_path = args.repo / SEGMENT / "runs/point_recovery_submission.json"
    if not args.dry_run:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
