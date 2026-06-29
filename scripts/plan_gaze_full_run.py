from __future__ import annotations

import argparse
import math
import shlex
from pathlib import Path


MERGE_STAGES = ["trajectory", "static", "vqa", "dynamic"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit a full Qwen2.5-VL GazeHeads replication command plan.")
    parser.add_argument("--segment-root", default="segments/gaze_heads_qwen25")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--total-comics", type=int, default=500)
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--top-k-gaze", type=int, default=100)
    parser.add_argument("--top-k-random", type=int, default=100)
    parser.add_argument("--targets-per-strip", type=int, default=1)
    parser.add_argument("--judge", choices=["baseline-only", "anthropic"], default="anthropic")
    parser.add_argument("--include-all-heads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-download", action="store_true")
    parser.add_argument("--skip-discover", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--out", default="", help="Optional shell script path to write.")
    args = parser.parse_args()

    commands = build_commands(
        segment_root=args.segment_root,
        model_id=args.model_id,
        device_map=args.device_map,
        total_comics=args.total_comics,
        shard_size=args.shard_size,
        n_samples=args.n_samples,
        top_k_gaze=args.top_k_gaze,
        top_k_random=args.top_k_random,
        targets_per_strip=args.targets_per_strip,
        judge=args.judge,
        include_all_heads=args.include_all_heads,
        include_download=args.include_download,
        include_discover=not args.skip_discover,
        resume=not args.no_resume,
    )
    text = render_shell(commands)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote full-run plan to {out_path}")
    else:
        print(text)


def build_commands(
    *,
    segment_root: str,
    model_id: str,
    device_map: str,
    total_comics: int,
    shard_size: int,
    n_samples: int,
    top_k_gaze: int,
    top_k_random: int,
    targets_per_strip: int,
    judge: str,
    include_all_heads: bool,
    include_download: bool,
    include_discover: bool,
    resume: bool,
) -> list[list[str]]:
    if total_comics <= 0:
        raise ValueError("--total-comics must be positive.")
    if shard_size <= 0:
        raise ValueError("--shard-size must be positive.")

    commands: list[list[str]] = []
    if include_download:
        commands.append(
            [
                "uv",
                "run",
                "python",
                "scripts/download_gaze_comics.py",
                "--out",
                f"{segment_root}/data/comics",
            ]
        )
    if include_discover:
        commands.append(
            [
                "uv",
                "run",
                "python",
                "scripts/run_gaze_pipeline.py",
                "--segment-root",
                segment_root,
                "--model-id",
                model_id,
                "--device-map",
                device_map,
                "--stages",
                "discover",
                "--n-samples",
                str(n_samples),
            ]
        )

    suffixes = shard_suffixes(total_comics, shard_size)
    for start_idx, count, suffix in suffixes:
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/run_gaze_pipeline.py",
            "--segment-root",
            segment_root,
            "--model-id",
            model_id,
            "--device-map",
            device_map,
            "--stages",
            "trajectory",
            "static",
            "vqa",
            "dynamic",
            "validate",
            "--start-comic-idx",
            str(start_idx),
            "--max-comics",
            str(count),
            "--trajectory-comics",
            str(count),
            "--run-suffix",
            suffix,
            "--top-k-gaze",
            str(top_k_gaze),
            "--top-k-random",
            str(top_k_random),
            "--targets-per-strip",
            str(targets_per_strip),
        ]
        if include_all_heads:
            cmd.append("--include-all-heads")
        if resume:
            cmd.append("--resume")
        commands.append(cmd)

    suffix_values = [suffix for _, _, suffix in suffixes]
    for stage in MERGE_STAGES:
        commands.append(
            [
                "uv",
                "run",
                "python",
                "scripts/merge_gaze_run_shards.py",
                "--segment-root",
                segment_root,
                "--stage",
                stage,
                "--suffixes",
                *suffix_values,
            ]
        )

    final_cmd = [
        "uv",
        "run",
        "python",
        "scripts/run_gaze_pipeline.py",
        "--segment-root",
        segment_root,
        "--model-id",
        model_id,
        "--device-map",
        device_map,
        "--stages",
        "score_static",
        "score_vqa",
        "score_dynamic",
        "report",
        "validate",
        "--judge",
        judge,
    ]
    if resume:
        final_cmd.append("--resume")
    commands.append(final_cmd)
    return commands


def shard_suffixes(total_comics: int, shard_size: int) -> list[tuple[int, int, str]]:
    n_shards = int(math.ceil(total_comics / shard_size))
    shards = []
    for shard_idx in range(n_shards):
        start_idx = shard_idx * shard_size
        count = min(shard_size, total_comics - start_idx)
        shards.append((start_idx, count, f"_{start_idx}_{count}"))
    return shards


def render_shell(commands: list[list[str]]) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.extend(shlex.join(command) for command in commands)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
