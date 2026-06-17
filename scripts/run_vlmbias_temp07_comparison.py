from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLMBias image/no-image temperature 0.7 comparison.")
    parser.add_argument("--dataset", default="data/vlmbias_400.jsonl")
    parser.add_argument("--image-out", default="runs/qwen25vl_3b_vlmbias_400_temp07_image.jsonl")
    parser.add_argument("--no-image-out", default="runs/qwen25vl_3b_vlmbias_400_temp07_no_image.jsonl")
    parser.add_argument("--report-out", default="runs/qwen25vl_3b_vlmbias_400_temp07_comparison.txt")
    args = parser.parse_args()

    _run(
        [
            sys.executable,
            "-m",
            "vlm_eval.cli",
            "--dataset",
            args.dataset,
            "--adapter",
            "adapters.qwen25_vl_temp07_eval:make_adapter",
            "--out",
            args.image_out,
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "vlm_eval.cli",
            "--dataset",
            args.dataset,
            "--adapter",
            "adapters.qwen25_vl_no_image_temp07_eval:make_adapter",
            "--out",
            args.no_image_out,
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/compare_vlmbias_runs.py",
            "--image-run",
            args.image_out,
            "--no-image-run",
            args.no_image_out,
            "--out",
            args.report_out,
        ]
    )


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

