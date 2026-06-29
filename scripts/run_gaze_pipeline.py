from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


STAGE_ORDER = [
    "download",
    "discover",
    "trajectory",
    "static",
    "vqa",
    "dynamic",
    "score_static",
    "score_vqa",
    "score_dynamic",
    "report",
    "validate",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Qwen2.5-VL GazeHeads replication pipeline.")
    parser.add_argument("--segment-root", default="segments/gaze_heads_qwen25")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--stages", nargs="+", choices=STAGE_ORDER, default=STAGE_ORDER)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--max-comics", type=int, default=0)
    parser.add_argument("--start-comic-idx", type=int, default=0)
    parser.add_argument("--trajectory-comics", type=int, default=5)
    parser.add_argument("--top-k-gaze", type=int, default=100)
    parser.add_argument("--top-k-random", type=int, default=100)
    parser.add_argument("--targets-per-strip", type=int, default=6)
    parser.add_argument("--include-all-heads", action="store_true")
    parser.add_argument("--run-suffix", default="", help="Optional suffix for run directories, such as '_100_25'.")
    parser.add_argument(
        "--discovery-suffix",
        default="",
        help="Optional suffix for the gaze_discovery directory used as the ranking source.",
    )
    parser.add_argument("--judge", choices=["baseline-only", "anthropic"], default="baseline-only")
    parser.add_argument("--smoke", action="store_true", help="Run the verified one-comic smoke pipeline.")
    parser.add_argument("--resume", action="store_true", help="Append generation outputs and skip existing rows.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.segment_root)
    data_dir = root / "data" / "comics"
    run_suffix = "_smoke" if args.smoke else args.run_suffix
    discovery_suffix = "_smoke" if args.smoke else args.discovery_suffix
    discovery_dir = root / "runs" / f"gaze_discovery{discovery_suffix}"
    ranking = discovery_dir / "gaze_head_ranking.json"
    n_samples = 1 if args.smoke else args.n_samples
    trajectory_comics = 1 if args.smoke else args.trajectory_comics
    max_comics = 1 if args.smoke else args.max_comics
    start_comic_idx = 0 if args.smoke else args.start_comic_idx
    top_k_gaze = 5 if args.smoke else args.top_k_gaze
    top_k_random = 5 if args.smoke else args.top_k_random
    targets_per_strip = 1 if args.smoke else args.targets_per_strip
    include_all_heads = [] if not args.include_all_heads or args.smoke else ["--include-all-heads"]
    smoke_static_extra = ["--max-new-tokens", "20"] if args.smoke else []
    smoke_vqa_extra = ["--max-new-tokens", "10"] if args.smoke else []
    smoke_dynamic_extra = ["--segment-tokens", "5", "--max-new-tokens", "30"] if args.smoke else []
    smoke_trajectory_extra = ["--max-new-tokens", "20"] if args.smoke else []
    resume_extra = ["--resume"] if args.resume else []

    commands = {
        "download": [
            "scripts/download_gaze_comics.py",
            "--out",
            str(data_dir),
        ],
        "discover": [
            "scripts/discover_qwen25_gaze_heads.py",
            "--comics-root",
            str(data_dir),
            "--out-dir",
            str(discovery_dir),
            "--model-id",
            args.model_id,
            "--device-map",
            args.device_map,
            "--n-samples",
            str(n_samples),
        ],
        "trajectory": [
            "scripts/run_qwen25_gaze_narration_trajectory.py",
            "--comics-root",
            str(data_dir),
            "--gaze-ranking",
            str(ranking),
            "--out-dir",
            str(root / "runs" / f"narration_trajectory{run_suffix}"),
            "--model-id",
            args.model_id,
            "--device-map",
            args.device_map,
            "--max-comics",
            str(trajectory_comics),
            "--start-comic-idx",
            str(start_comic_idx),
            "--top-k-gaze",
            str(top_k_gaze),
            "--control-heads",
            str(top_k_random),
            *resume_extra,
            *smoke_trajectory_extra,
        ],
        "static": [
            "scripts/run_qwen25_gaze_static_narration.py",
            "--comics-root",
            str(data_dir),
            "--gaze-ranking",
            str(ranking),
            "--out-dir",
            str(root / "runs" / f"static_narration{run_suffix}"),
            "--model-id",
            args.model_id,
            "--device-map",
            args.device_map,
            "--max-comics",
            str(max_comics),
            "--start-comic-idx",
            str(start_comic_idx),
            "--top-k-gaze",
            str(top_k_gaze),
            "--top-k-random",
            str(top_k_random),
            "--targets-per-strip",
            str(targets_per_strip),
            *include_all_heads,
            *resume_extra,
            *smoke_static_extra,
        ],
        "vqa": [
            "scripts/run_qwen25_gaze_vqa_steering.py",
            "--comics-root",
            str(data_dir),
            "--gaze-ranking",
            str(ranking),
            "--out-dir",
            str(root / "runs" / f"vqa_steering{run_suffix}"),
            "--model-id",
            args.model_id,
            "--device-map",
            args.device_map,
            "--max-comics",
            str(max_comics),
            "--start-comic-idx",
            str(start_comic_idx),
            "--top-k-gaze",
            str(top_k_gaze),
            "--top-k-random",
            str(top_k_random),
            *include_all_heads,
            *resume_extra,
            *smoke_vqa_extra,
        ],
        "dynamic": [
            "scripts/run_qwen25_gaze_dynamic_narration.py",
            "--comics-root",
            str(data_dir),
            "--gaze-ranking",
            str(ranking),
            "--out-dir",
            str(root / "runs" / f"dynamic_narration{run_suffix}"),
            "--model-id",
            args.model_id,
            "--device-map",
            args.device_map,
            "--max-comics",
            str(max_comics),
            "--start-comic-idx",
            str(start_comic_idx),
            "--top-k-gaze",
            str(top_k_gaze),
            *resume_extra,
            *smoke_dynamic_extra,
        ],
        "score_static": _score_cmd(
            root / "runs" / f"static_narration{run_suffix}" / "generations.jsonl",
            args.judge,
            resume=args.resume,
        ),
        "score_vqa": _score_cmd(
            root / "runs" / f"vqa_steering{run_suffix}" / "generations.jsonl",
            args.judge,
            resume=args.resume,
        ),
        "score_dynamic": _score_cmd(
            root / "runs" / f"dynamic_narration{run_suffix}" / "generations.jsonl",
            args.judge,
            resume=args.resume,
        ),
        "report": [
            "scripts/report_qwen25_gaze_results.py",
            "--segment-root",
            str(root),
            *(
                [
                    "--run-suffix",
                    run_suffix,
                ]
                if run_suffix
                else []
            ),
        ],
        "validate": [
            "scripts/validate_gaze_pipeline.py",
            "--segment-root",
            str(root),
            "--discovery-suffix",
            discovery_suffix,
            *(
                [
                    "--run-suffix",
                    run_suffix,
                    "--out",
                    str(root / "reports" / f"pipeline_validation{run_suffix}.json"),
                ]
                if run_suffix
                else []
            ),
        ],
    }

    for stage in args.stages:
        cmd = [sys.executable, *commands[stage]]
        print("$ " + shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


def _score_cmd(generations: Path, judge: str, *, resume: bool = False) -> list[str]:
    cmd = [
        "scripts/score_qwen25_gaze_generations.py",
        "--generations",
        str(generations),
        "--judge",
        judge,
    ]
    if resume:
        cmd.append("--resume")
    return cmd


if __name__ == "__main__":
    main()
