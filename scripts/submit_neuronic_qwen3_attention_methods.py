#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


REPO_DEFAULT = Path("/n/fs/pvl-memory/at7979/VLMProject")
EXPERIMENT_ROOT = Path(
    "segments/gaze_heads_qwen3_8b/experiments/attention_methods_v1"
)
RUN_ROOT = Path("segments/gaze_heads_qwen3_8b/runs/attention_methods_v1")
REPORT_ROOT = Path("segments/gaze_heads_qwen3_8b/reports/attention_methods_v1")
GAZE_RANKING = Path(
    "segments/gaze_heads_qwen3_8b/runs/gaze_discovery_seed42_merged/"
    "gaze_head_ranking.json"
)
STAGE_COUNTS = {"smoke": 4, "controller": 10, "heads": 10, "confirm": 4}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit the staged Qwen3 gaze-attention method experiment."
    )
    parser.add_argument(
        "stage",
        choices=[
            "full",
            "overnight",
            "smoke",
            "controller",
            "heads",
            "confirm",
            "robustness",
        ],
        help=(
            "full runs development controller -> head sweep -> held-out confirmation. "
            "overnight appends the post-selection multi-seed all-data robustness run."
        ),
    )
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--robustness-max-parallel", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not args.skip_preflight:
        _preflight(repo, args.stage)
    if not args.dry_run:
        (repo / "segments/gaze_heads_qwen3_8b/runs/slurm").mkdir(
            parents=True, exist_ok=True
        )
    submitter = Submitter(dry_run=args.dry_run)
    exports = {
        "REPO": str(repo),
        "EXPERIMENT_ROOT": str(EXPERIMENT_ROOT),
        "RUN_ROOT": str(RUN_ROOT),
        "REPORT_ROOT": str(REPORT_ROOT),
        "GAZE_RANKING": str(GAZE_RANKING),
        "SEEDS_COLON": ":".join(str(seed) for seed in args.seeds),
        "MIN_GPU_MEMORY_GB": "20.0",
    }

    if args.stage in {"full", "overnight"}:
        dependency = None
        jobs = {}
        chain = ["smoke", "controller", "heads", "confirm"]
        if args.stage == "overnight":
            chain.append("robustness")
        for stage in chain:
            count = (
                4 * len(args.seeds)
                if stage == "robustness"
                else STAGE_COUNTS[stage]
            )
            stage_jobs = _submit_stage(
                submitter,
                stage=stage,
                count=count,
                max_parallel=(
                    2
                    if stage == "smoke"
                    else args.robustness_max_parallel
                    if stage == "robustness"
                    else args.max_parallel
                ),
                dependency=dependency,
                exports=exports,
            )
            jobs[stage] = stage_jobs
            dependency = stage_jobs["aggregate"]
    else:
        count = (
            4 * len(args.seeds)
            if args.stage == "robustness"
            else STAGE_COUNTS[args.stage]
        )
        jobs = {
            args.stage: _submit_stage(
                submitter,
                stage=args.stage,
                count=count,
                max_parallel=(
                    args.robustness_max_parallel
                    if args.stage == "robustness"
                    else args.max_parallel
                ),
                dependency=None,
                exports=exports,
            )
        }
    result = {
        "stage": args.stage,
        "jobs": jobs,
        "final_job": list(jobs.values())[-1]["aggregate"],
        "dry_run": args.dry_run,
        "commands": submitter.commands,
    }
    if not args.dry_run:
        receipt_path = repo / EXPERIMENT_ROOT / "last_submission.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        result["submitted_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        result["receipt"] = str(receipt_path)
        receipt_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
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
    prep = submitter.sbatch(
        "scripts/slurm_neuronic_qwen3_prepare_attention_methods.sh",
        exports=stage_exports,
        dependency=dependency,
    )
    run = submitter.sbatch(
        "scripts/slurm_neuronic_qwen3_attention_method_condition.sh",
        exports=stage_exports,
        dependency=prep,
        array=f"0-{count - 1}%{min(count, max_parallel)}",
    )
    aggregate = submitter.sbatch(
        "scripts/slurm_neuronic_qwen3_aggregate_attention_methods.sh",
        exports=stage_exports,
        dependency=run,
    )
    return {"prepare": prep, "run": run, "aggregate": aggregate}


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
        repo / GAZE_RANKING,
        repo / GAZE_RANKING.with_name("gaze_scores.npy"),
        repo / "segments/vlm_bias_attention/data/vlmbias_400.jsonl",
        repo / "segments/vlm_bias_attention/data/naturalbench_100_groups.jsonl",
        repo / "scripts/slurm_neuronic_qwen3_prepare_attention_methods.sh",
        repo / "scripts/slurm_neuronic_qwen3_attention_method_condition.sh",
        repo / "scripts/slurm_neuronic_qwen3_aggregate_attention_methods.sh",
    ]
    if stage in {"heads", "confirm", "robustness"}:
        required.append(repo / REPORT_ROOT / "controller/selection.json")
    if stage in {"confirm", "robustness"}:
        required.append(repo / REPORT_ROOT / "heads/selection.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required inputs:\n- " + "\n- ".join(missing))
    ranking = json.loads((repo / GAZE_RANKING).read_text(encoding="utf-8"))
    if not isinstance(ranking, list) or len(ranking) < 100:
        raise SystemExit(
            f"Gaze ranking must contain at least 100 heads: {repo / GAZE_RANKING}"
        )
    _validate_vlmbias(
        repo / "segments/vlm_bias_attention/data/vlmbias_400.jsonl"
    )
    _validate_naturalbench(
        repo / "segments/vlm_bias_attention/data/naturalbench_100_groups.jsonl"
    )


def _validate_vlmbias(path: Path) -> None:
    rows = _read_jsonl(path)
    if len(rows) != 400 or len({str(row.get("id")) for row in rows}) != 400:
        raise SystemExit(f"Expected 400 unique VLMBias rows in {path}")
    for row in rows:
        image = _resolve_input_path(path, row.get("image_path"))
        if not image.is_file():
            raise SystemExit(f"Missing VLMBias image: {image}")


def _validate_naturalbench(path: Path) -> None:
    rows = _read_jsonl(path)
    if len(rows) != 100 or len({str(row.get("id")) for row in rows}) != 100:
        raise SystemExit(f"Expected 100 unique NaturalBench groups in {path}")
    required_answers = {"q0_i0", "q0_i1", "q1_i0", "q1_i1"}
    for row in rows:
        if not required_answers <= set(row.get("answers", {})):
            raise SystemExit(f"Incomplete NaturalBench answers for {row.get('id')}")
        for key in ("image_0_path", "image_1_path"):
            image = _resolve_input_path(path, row.get(key))
            if not image.is_file():
                raise SystemExit(f"Missing NaturalBench image: {image}")


def _resolve_input_path(dataset: Path, value: Any) -> Path:
    if not value:
        return Path("__missing__")
    path = Path(str(value))
    return path if path.is_absolute() else dataset.parent / path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
