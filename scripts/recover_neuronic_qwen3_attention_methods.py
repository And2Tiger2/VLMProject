#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_DEFAULT = Path("/n/fs/pvl-memory/at7979/VLMProject")
EXPERIMENT_ROOT = Path(
    "segments/gaze_heads_qwen3_8b/experiments/attention_methods_v1"
)
RUN_ROOT = Path("segments/gaze_heads_qwen3_8b/runs/attention_methods_v1")
REPORT_ROOT = Path("segments/gaze_heads_qwen3_8b/reports/attention_methods_v1")
JOB_ID_PATTERN = re.compile(r"^[0-9]+(?:_[0-9]+)?$")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Archive a failed Qwen3 attention-method chain and submit a clean "
            "replacement after validating its recorded final job ID."
        )
    )
    parser.add_argument("expected_final_job")
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--robustness-max-parallel", type=int, default=2)
    args = parser.parse_args()

    repo = args.repo.resolve()
    receipt_path = repo / EXPERIMENT_ROOT / "last_submission.json"
    receipt = _read_json(receipt_path)
    final_job = str(receipt.get("final_job", ""))
    if final_job != args.expected_final_job:
        raise SystemExit(
            f"Refusing recovery: receipt final job is {final_job!r}, not "
            f"{args.expected_final_job!r}."
        )
    if receipt.get("stage") != "overnight" or receipt.get("dry_run"):
        raise SystemExit("Refusing recovery: receipt is not a real overnight run.")

    # Validate the required cluster safeguard before changing the old chain.
    # The replacement submitter checks it again, and each GPU worker checks it
    # once more before model loading.
    subprocess.run(
        ["uv", "run", "python", "scripts/check_neuronic_overheat.py"],
        cwd=repo,
        env={**os.environ, "VLM_REQUIRE_OVERHEAT_CHECK": "1"},
        check=True,
    )

    job_ids = _receipt_job_ids(receipt)
    invalid = [job_id for job_id in job_ids if not JOB_ID_PATTERN.fullmatch(job_id)]
    if invalid:
        raise SystemExit(f"Refusing recovery: invalid receipt job IDs: {invalid}")
    cancellation = subprocess.run(
        ["scancel", *job_ids],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if cancellation.stdout.strip():
        print(cancellation.stdout.strip())
    if cancellation.stderr.strip():
        print(cancellation.stderr.strip(), file=sys.stderr)
    print(
        f"Requested cancellation of {len(job_ids)} recorded jobs "
        f"(scancel exit {cancellation.returncode})."
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = {}
    for relative in (EXPERIMENT_ROOT, RUN_ROOT, REPORT_ROOT):
        source = repo / relative
        if not source.exists():
            continue
        target = source.with_name(
            f"{source.name}_failed_{args.expected_final_job}_{stamp}"
        )
        if target.exists():
            raise SystemExit(f"Archive destination already exists: {target}")
        source.rename(target)
        archived[str(relative)] = str(target.relative_to(repo))
        print(f"Archived {relative} -> {target.relative_to(repo)}")

    command = [
        sys.executable,
        str(repo / "scripts/submit_neuronic_qwen3_attention_methods.py"),
        "overnight",
        "--repo",
        str(repo),
        "--max-parallel",
        str(args.max_parallel),
        "--robustness-max-parallel",
        str(args.robustness_max_parallel),
        "--seeds",
        *(str(seed) for seed in args.seeds),
    ]
    print(
        json.dumps(
            {
                "recovered_final_job": final_job,
                "archived": archived,
                "replacement_command": command,
            },
            indent=2,
        )
    )
    subprocess.run(command, cwd=repo, check=True)


def _receipt_job_ids(receipt: dict[str, Any]) -> list[str]:
    ids = []
    for stage_jobs in receipt.get("jobs", {}).values():
        for key in ("prepare", "run", "aggregate"):
            job_id = str(stage_jobs.get(key, ""))
            if job_id and job_id not in ids:
                ids.append(job_id)
    if not ids:
        raise SystemExit("Refusing recovery: receipt contains no job IDs.")
    return ids


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Missing submission receipt: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
