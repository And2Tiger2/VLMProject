from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_PROMPT_MODES = ["baseline", "visual_first", "counterfactual_warning", "describe_then_answer"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLMBias prompt-forcing ablations.")
    parser.add_argument("--dataset", default="data/vlmbias_400.jsonl")
    parser.add_argument("--out-dir", default="runs/vlmbias_prompt_ablation")
    parser.add_argument("--prompt-modes", nargs="+", default=DEFAULT_PROMPT_MODES)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for prompt_mode in args.prompt_modes:
        run_name = f"qwen25vl_3b_prompt_{prompt_mode}"
        out_path = out_dir / f"{run_name}.jsonl"
        adapter = (
            "adapters.qwen25_vl:make_adapter,"
            f"max_pixels={args.max_pixels},"
            "max_new_tokens=16,"
            "do_sample=true,"
            f"temperature={args.temperature},"
            f"seed={args.seed},"
            f"prompt_mode={prompt_mode}"
        )
        _run_eval(args.dataset, adapter, out_path)
        summaries.append((prompt_mode, out_path.with_suffix(".summary.json")))

    _write_report(summaries, out_dir / "summary.tsv")


def _run_eval(dataset: str, adapter: str, out_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "vlm_eval.cli",
        "--dataset",
        dataset,
        "--adapter",
        adapter,
        "--out",
        str(out_path),
    ]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _write_report(summaries: list[tuple[str, Path]], out_path: Path) -> None:
    lines = ["condition\tn\taccuracy\tbias_aligned_fraction\tbias_aligned_error_rate\terror_rate"]
    for condition, summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        lines.append(
            "\t".join(
                [
                    condition,
                    str(summary["n"]),
                    f"{summary['accuracy']:.6f}",
                    f"{summary['bias_aligned_fraction']:.6f}",
                    f"{summary['bias_aligned_error_rate']:.6f}",
                    f"{summary['error_rate']:.6f}",
                ]
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path.read_text(encoding="utf-8"))
    print(f"Wrote ablation report to {out_path}")


if __name__ == "__main__":
    main()
