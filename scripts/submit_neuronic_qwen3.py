#!/usr/bin/env python3
"""Submit complete Qwen3-VL Gaze Heads experiments to Neuronic.

Each subcommand is one login-node command. GPU work is submitted as Slurm
arrays; ``--seeds N`` expands to N independent seed jobs without changing the
worker scripts. Dependent merge, validation, judging, and aggregation jobs are
submitted automatically.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_SEGMENT_ROOT = "segments/gaze_heads_qwen3_8b"


@dataclass
class Submitter:
    dry_run: bool
    scheduler_args: list[str]
    counter: int = 0

    def submit(
        self,
        script: str,
        *,
        array_count: int | None = None,
        max_parallel: int | None = None,
        dependency: str | None = None,
        exports: dict[str, str | int] | None = None,
    ) -> str:
        command = ["sbatch", "--parsable", *self.scheduler_args]
        if array_count is not None:
            if array_count <= 0:
                raise ValueError("Slurm array_count must be positive")
            array = f"0-{array_count - 1}"
            if max_parallel:
                array += f"%{max_parallel}"
            command.append(f"--array={array}")
        if dependency:
            command.append(f"--dependency=afterok:{dependency}")
        if exports:
            encoded = ["ALL", *[f"{key}={value}" for key, value in sorted(exports.items())]]
            command.append("--export=" + ",".join(encoded))
        command.append(script)

        print(" ".join(_shell_quote(part) for part in command), flush=True)
        if self.dry_run:
            self.counter += 1
            return f"DRYRUN{self.counter}"
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(
                f"sbatch returned an invalid job id for {script}: stdout={result.stdout!r} "
                f"stderr={result.stderr!r}"
            )
        print(f"submitted {job_id}: {script}", flush=True)
        return job_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting jobs.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip local data/key checks (mainly for CI).")
    parser.add_argument("--account", default="", help="Optional Slurm account.")
    parser.add_argument("--partition", default="", help="Optional Slurm partition.")
    parser.add_argument("--qos", default="", help="Optional Slurm QoS.")
    parser.add_argument("--constraint", default="", help="Optional Slurm node/GPU constraint.")
    parser.add_argument("--dependency", default="", help="Optional existing job ID that must finish successfully first.")
    parser.add_argument(
        "--min-gpu-memory-gb", type=_positive_float, default=20.0,
        help="Fail workers early below this allocated-GPU memory (default: 20 GiB).",
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    discovery = subparsers.add_parser("discovery", help="Discover, merge, and validate gaze heads.")
    _seed_args(discovery, default_base_seed=42)
    discovery.add_argument("--shards", type=_positive_int, default=10)
    discovery.add_argument("--shard-size", type=_positive_int, default=50)
    discovery.add_argument(
        "--max-parallel", type=_nonnegative_int, default=0,
        help="Maximum concurrent GPU array tasks; 0 leaves all tasks eligible (default).",
    )
    discovery.add_argument("--comics-root", default=f"{DEFAULT_SEGMENT_ROOT}/data/discovery_comics")

    static = subparsers.add_parser("static", help="Generate, merge, validate, and optionally judge static steering.")
    _seed_args(static, default_base_seed=42)
    static.add_argument("--ranking-seed", type=_nonnegative_int, default=42)
    static.add_argument("--shards", type=_positive_int, default=10)
    static.add_argument("--shard-size", type=_positive_int, default=50)
    static.add_argument("--top-ks", nargs="+", type=_positive_int, default=[1, 10, 50, 100])
    static.add_argument(
        "--max-parallel", type=_nonnegative_int, default=0,
        help="Maximum concurrent GPU array tasks; 0 leaves all tasks eligible (default).",
    )
    static.add_argument("--judge-parallel", type=_positive_int, default=2)
    static.add_argument("--judge", choices=["anthropic", "none"], default="anthropic")
    static_recovery = static.add_mutually_exclusive_group()
    static_recovery.add_argument(
        "--judge-only", action="store_true",
        help="Retry paper-style judging from existing merged generations without allocating GPUs.",
    )
    static_recovery.add_argument(
        "--merge-only", action="store_true",
        help="Merge and validate existing shard generations without allocating GPUs.",
    )
    static.add_argument("--control-mode", choices=["paper", "matched"], default="paper")
    static.add_argument("--eval-root", default=f"{DEFAULT_SEGMENT_ROOT}/data/eval_comics")
    static.add_argument("--discovery-root", default=f"{DEFAULT_SEGMENT_ROOT}/data/discovery_comics")
    static.add_argument("--ranking", default="", help="Override the merged gaze-head ranking path.")

    benchmark = subparsers.add_parser("benchmark", help="Run and aggregate VLMBias + NaturalBench alpha sweeps.")
    _seed_args(benchmark, default_base_seed=0)
    benchmark.add_argument("--ranking-seed", type=_nonnegative_int, default=42)
    benchmark.add_argument("--top-ks", nargs="+", type=_positive_int, default=[10, 50, 100])
    benchmark.add_argument("--alphas", nargs="+", type=_positive_float, default=[0.25, 0.5, 1, 2, 5, 10])
    benchmark.add_argument("--attention-mode", choices=["full", "decode"], default="full")
    benchmark.add_argument(
        "--max-parallel", type=_nonnegative_int, default=0,
        help="Maximum concurrent seed jobs; 0 leaves every seed eligible in parallel (default).",
    )
    benchmark.add_argument("--ranking", default="", help="Override the merged gaze-head ranking path.")
    benchmark.add_argument("--vlmbias", default="segments/vlm_bias_attention/data/vlmbias_400.jsonl")
    benchmark.add_argument("--naturalbench", default="segments/vlm_bias_attention/data/naturalbench_100_groups.jsonl")
    benchmark.add_argument("--limit", type=_nonnegative_int, default=0)
    benchmark.add_argument("--naturalbench-limit-groups", type=_nonnegative_int, default=0)
    benchmark.add_argument("--run-name", default="", help="Output directory name under the segment runs folder.")

    args = parser.parse_args()
    if hasattr(args, "top_ks"):
        _require_unique(args.top_ks, "--top-ks")
    if hasattr(args, "alphas"):
        _require_unique(args.alphas, "--alphas")
    if getattr(args, "run_name", ""):
        run_name = str(args.run_name)
        if Path(run_name).name != run_name or run_name in {".", ".."} or "," in run_name:
            parser.error("--run-name must be one simple directory name without commas or path separators")
    repo = Path.cwd()
    _base_preflight(repo, args)
    scheduler_args = _scheduler_args(args)
    submitter = Submitter(dry_run=args.dry_run, scheduler_args=scheduler_args)
    dependency = args.dependency or None

    if args.experiment == "discovery":
        final_job = submit_discovery(args, submitter, dependency)
    elif args.experiment == "static":
        final_job = submit_static(args, submitter, dependency)
    else:
        final_job = submit_benchmark(args, submitter, dependency)
    print(f"final_job={final_job}", flush=True)


def submit_discovery(args: argparse.Namespace, submitter: Submitter, dependency: str | None) -> str:
    _require_path(Path(args.comics_root), "raw COMICS root", args)
    if not args.skip_preflight:
        _require_eligible_raw_page(Path(args.comics_root))
    exports = _common_exports(args) | {
        "DISCOVERY_COMICS_ROOT": args.comics_root,
        "N_SEEDS": args.seeds,
        "BASE_SEED": args.base_seed,
        "N_SHARDS": args.shards,
        "SHARD_SIZE": args.shard_size,
    }
    workers = submitter.submit(
        "scripts/slurm_neuronic_qwen3_discovery.sh",
        array_count=args.seeds * args.shards,
        max_parallel=args.max_parallel,
        dependency=dependency,
        exports=exports,
    )
    return submitter.submit(
        "scripts/slurm_neuronic_qwen3_merge_discovery.sh",
        array_count=args.seeds,
        max_parallel=args.seeds,
        dependency=workers,
        exports=exports,
    )


def submit_static(args: argparse.Namespace, submitter: Submitter, dependency: str | None) -> str:
    _require_path(Path(args.eval_root), "OpenAI evaluation comics root", args)
    _require_path(Path(args.discovery_root), "raw COMICS discovery root", args)
    if not args.skip_preflight:
        _require_disjoint_roots(Path(args.discovery_root), Path(args.eval_root))
        _require_eval_comics(Path(args.eval_root), args.shards * args.shard_size)
    if (
        args.judge == "anthropic"
        and not args.merge_only
        and not args.dry_run
        and not os.environ.get("ANTHROPIC_API_KEY")
    ):
        raise SystemExit("ANTHROPIC_API_KEY is required for --judge anthropic; export it or pass --judge none.")
    ranking = args.ranking or (
        f"{DEFAULT_SEGMENT_ROOT}/runs/gaze_discovery_seed{args.ranking_seed}_merged/gaze_head_ranking.json"
    )
    if not dependency:
        _require_path(Path(ranking), "merged gaze-head ranking", args)
        if not args.skip_preflight:
            _require_ranking(Path(ranking), max(args.top_ks))
    exports = _common_exports(args) | {
        "EVAL_COMICS_ROOT": args.eval_root,
        "GAZE_RANKING": ranking,
        "RANKING_SEED": args.ranking_seed,
        "N_SEEDS": args.seeds,
        "BASE_SEED": args.base_seed,
        "N_SHARDS": args.shards,
        "SHARD_SIZE": args.shard_size,
        "TOP_KS_COLON": _colon(args.top_ks),
        "CONTROL_MODE": args.control_mode,
    }
    if args.merge_only:
        if not args.skip_preflight:
            for seed_index in range(args.seeds):
                seed = args.base_seed + seed_index
                for top_k in args.top_ks:
                    for shard in range(args.shards):
                        start = shard * args.shard_size
                        generations = Path(
                            f"{DEFAULT_SEGMENT_ROOT}/runs/static_narration_seed{seed}_top{top_k}_"
                            f"{start}_{args.shard_size}/generations.jsonl"
                        )
                        if not generations.exists():
                            raise SystemExit(f"Missing shard generations for --merge-only: {generations}")
        return submitter.submit(
            "scripts/slurm_neuronic_qwen3_merge_static.sh",
            array_count=args.seeds,
            max_parallel=args.seeds,
            dependency=dependency,
            exports=exports,
        )
    if args.judge_only:
        if args.judge != "anthropic":
            raise SystemExit("--judge-only requires --judge anthropic.")
        if not args.skip_preflight:
            for seed_index in range(args.seeds):
                seed = args.base_seed + seed_index
                for top_k in args.top_ks:
                    generations = Path(
                        f"{DEFAULT_SEGMENT_ROOT}/runs/static_narration_seed{seed}_top{top_k}"
                        f"_merged_0_{args.shards * args.shard_size}/generations.jsonl"
                    )
                    if not generations.exists():
                        raise SystemExit(f"Missing merged generations for --judge-only: {generations}")
        return submitter.submit(
            "scripts/slurm_neuronic_qwen3_judge_static.sh",
            array_count=args.seeds * len(args.top_ks),
            max_parallel=args.judge_parallel,
            dependency=dependency,
            exports=exports,
        )
    worker_count = args.seeds * args.shards * len(args.top_ks)
    workers = submitter.submit(
        "scripts/slurm_neuronic_qwen3_static_full.sh",
        array_count=worker_count,
        max_parallel=args.max_parallel,
        dependency=dependency,
        exports=exports,
    )
    merged = submitter.submit(
        "scripts/slurm_neuronic_qwen3_merge_static.sh",
        array_count=args.seeds,
        max_parallel=args.seeds,
        dependency=workers,
        exports=exports,
    )
    if args.judge == "none":
        return merged
    return submitter.submit(
        "scripts/slurm_neuronic_qwen3_judge_static.sh",
        array_count=args.seeds * len(args.top_ks),
        max_parallel=args.judge_parallel,
        dependency=merged,
        exports=exports,
    )


def submit_benchmark(args: argparse.Namespace, submitter: Submitter, dependency: str | None) -> str:
    ranking = args.ranking or (
        f"{DEFAULT_SEGMENT_ROOT}/runs/gaze_discovery_seed{args.ranking_seed}_merged/gaze_head_ranking.json"
    )
    if not dependency:
        _require_path(Path(ranking), "merged gaze-head ranking", args)
        if not args.skip_preflight:
            _require_ranking(Path(ranking), max(args.top_ks))
    _require_path(Path(args.vlmbias), "VLMBias dataset", args)
    _require_path(Path(args.naturalbench), "NaturalBench dataset", args)
    if not args.skip_preflight:
        _require_vlmbias_slice(Path(args.vlmbias), args.limit or 400)
        _require_naturalbench_slice(Path(args.naturalbench), args.naturalbench_limit_groups or 100)
    default_name = "benchmark_attention_smoke" if args.limit > 0 or args.naturalbench_limit_groups > 0 else f"benchmark_attention_{args.attention_mode}"
    run_name = args.run_name or default_name
    exports = _common_exports(args) | {
        "GAZE_RANKING": ranking,
        "RANKING_SEED": args.ranking_seed,
        "DATASET": args.vlmbias,
        "NATURALBENCH_DATASET": args.naturalbench,
        "N_SEEDS": args.seeds,
        "BASE_SEED": args.base_seed,
        "TOP_KS_COLON": _colon(args.top_ks),
        "ALPHAS_COLON": _colon(args.alphas),
        "ATTENTION_MODE": args.attention_mode,
        "LIMIT": args.limit,
        "NATURALBENCH_LIMIT_GROUPS": args.naturalbench_limit_groups,
        "OUT_DIR": f"{DEFAULT_SEGMENT_ROOT}/runs/{run_name}",
    }
    workers = submitter.submit(
        "scripts/slurm_neuronic_qwen3_benchmark_full.sh",
        array_count=args.seeds,
        max_parallel=args.max_parallel,
        dependency=dependency,
        exports=exports,
    )
    return submitter.submit(
        "scripts/slurm_neuronic_qwen3_aggregate_benchmark.sh",
        dependency=workers,
        exports=exports,
    )


def _seed_args(parser: argparse.ArgumentParser, *, default_base_seed: int) -> None:
    parser.add_argument("--seeds", type=_positive_int, default=1, help="Number of parallel seeds.")
    parser.add_argument(
        "--base-seed", type=_nonnegative_int, default=default_base_seed,
        help="First seed; uses consecutive values.",
    )


def _base_preflight(repo: Path, args: argparse.Namespace) -> None:
    if not (repo / "pyproject.toml").exists() or not (repo / ".git").exists():
        raise SystemExit("Run this command from the VLMProject repository root.")
    (repo / DEFAULT_SEGMENT_ROOT / "runs" / "slurm").mkdir(parents=True, exist_ok=True)
    if not args.dry_run and shutil.which("sbatch") is None:
        raise SystemExit("sbatch is not available. Run this launcher on a Neuronic login node.")
    cpu_only_static_recovery = bool(
        getattr(args, "judge_only", False) or getattr(args, "merge_only", False)
    )
    if not args.dry_run and not args.skip_preflight and not cpu_only_static_recovery:
        _require_model_cache(repo)


def _require_path(path: Path, label: str, args: argparse.Namespace) -> None:
    if not args.skip_preflight and not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def _require_disjoint_roots(discovery_root: Path, eval_root: Path) -> None:
    if discovery_root.resolve() == eval_root.resolve():
        raise SystemExit("Discovery and evaluation roots must be different directories.")
    discovery_ids = {path.name for path in discovery_root.iterdir() if path.is_dir()}
    evaluation_ids = {path.name for path in eval_root.iterdir() if path.is_dir()}
    overlap = sorted(discovery_ids & evaluation_ids)
    if overlap:
        raise SystemExit(
            f"Discovery/evaluation comic IDs overlap ({len(overlap)}); examples: {', '.join(overlap[:5])}"
        )


def _require_eligible_raw_page(root: Path) -> None:
    for comic_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        counts: dict[int, int] = {}
        for path in comic_dir.iterdir():
            parts = path.stem.split("_")
            if not path.is_file() or len(parts) != 2:
                continue
            try:
                page = int(parts[0])
                int(parts[1])
            except ValueError:
                continue
            counts[page] = counts.get(page, 0) + 1
        if any(count >= 6 for count in counts.values()):
            return
    raise SystemExit(f"Raw COMICS root has no page containing at least six panels: {root}")


def _require_eval_comics(root: Path, required: int) -> None:
    suffixes = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    valid = 0
    for comic_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if all(any((comic_dir / f"p{panel}{suffix}").exists() for suffix in suffixes) for panel in range(1, 7)):
            valid += 1
    if valid < required:
        raise SystemExit(f"Need {required} complete evaluation strips under {root}; found {valid}.")


def _require_ranking(path: Path, top_k: int) -> None:
    try:
        ranking = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read gaze-head ranking {path}: {exc}") from exc
    if not isinstance(ranking, list) or len(ranking) < top_k:
        raise SystemExit(f"Ranking {path} contains fewer than the requested {top_k} heads.")
    if not path.with_name("gaze_scores.npy").exists():
        raise SystemExit(f"Ranking control sampling requires sibling gaze_scores.npy: {path.parent}")


def _require_vlmbias_slice(path: Path, required: int) -> None:
    rows = _read_jsonl(path)
    if len(rows) < required:
        raise SystemExit(f"Need {required} VLMBias rows in {path}; found {len(rows)}.")
    selected = rows[:required]
    ids = [str(row.get("id", "")) for row in selected]
    if any(not identifier for identifier in ids) or len(ids) != len(set(ids)):
        raise SystemExit(f"VLMBias slice must have {required} non-empty unique IDs: {path}")
    for row in selected:
        image = Path(str(row.get("image_path", "")))
        resolved = image if image.is_absolute() else path.parent / image
        if not resolved.exists():
            raise SystemExit(f"VLMBias image referenced by {path} is missing: {resolved}")


def _require_naturalbench_slice(path: Path, required_groups: int) -> None:
    rows = _read_jsonl(path)
    if len(rows) < required_groups:
        raise SystemExit(f"Need {required_groups} NaturalBench groups in {path}; found {len(rows)}.")
    selected = rows[:required_groups]
    ids = [str(row.get("id", "")) for row in selected]
    if any(not identifier for identifier in ids) or len(ids) != len(set(ids)):
        raise SystemExit(f"NaturalBench slice must have {required_groups} non-empty unique group IDs: {path}")
    for row in selected:
        answers = row.get("answers") or {}
        if set(answers) != {"q0_i0", "q0_i1", "q1_i0", "q1_i1"}:
            raise SystemExit(f"NaturalBench group {row.get('id')} has incomplete answer keys in {path}")
        for key in ("image_0_path", "image_1_path"):
            image = Path(str(row.get(key, "")))
            resolved = image if image.is_absolute() else path.parent / image
            if not resolved.exists():
                raise SystemExit(f"NaturalBench image referenced by {path} is missing: {resolved}")


def _require_model_cache(repo: Path) -> None:
    cache_root = Path(os.environ.get("CACHE_ROOT", str(repo / ".cache" / "vlmproject")))
    snapshots = cache_root / "huggingface" / "hub" / "models--Qwen--Qwen3-VL-8B-Instruct" / "snapshots"
    snapshot_dirs = sorted(path for path in snapshots.iterdir() if path.is_dir()) if snapshots.exists() else []
    complete = [
        path
        for path in snapshot_dirs
        if (path / "config.json").exists()
        and (
            (path / "model.safetensors").exists()
            or (path / "model.safetensors.index.json").exists()
        )
    ]
    if not complete:
        raise SystemExit(
            f"A complete Qwen3-VL 8B snapshot is not present in the worker cache {cache_root}. "
            "Run `bash scripts/setup_neuronic_qwen3.sh` first."
        )


def _read_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read dataset JSONL {path}: {exc}") from exc


def _scheduler_args(args: argparse.Namespace) -> list[str]:
    values = {
        "account": args.account,
        "partition": args.partition,
        "qos": args.qos,
        "constraint": args.constraint,
    }
    return [f"--{key}={value}" for key, value in values.items() if value]


def _common_exports(args: argparse.Namespace) -> dict[str, str]:
    return {
        "REPO": str(Path.cwd()),
        "SEGMENT_ROOT": DEFAULT_SEGMENT_ROOT,
        "MIN_GPU_MEMORY_GB": str(args.min_gpu_memory_gb),
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not parsed > 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _require_unique(values: Sequence[object], label: str) -> None:
    if len(values) != len(set(values)):
        raise SystemExit(f"{label} cannot contain duplicates; duplicate conditions would write the same files.")


def _colon(values: Sequence[object]) -> str:
    return ":".join(str(value) for value in values)


def _shell_quote(value: str) -> str:
    if value and all(character.isalnum() or character in "_./:=,%+-" for character in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()
