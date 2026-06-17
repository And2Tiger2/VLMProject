from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


CONDITIONS = [
    ("baseline", 0.0, "baseline"),
    ("alpha025_last25", 0.25, "last25"),
    ("alpha025_last50", 0.25, "last50"),
    ("alpha025_all", 0.25, "all"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLMBias image-token attention logging/bias ablation.")
    parser.add_argument("--dataset", default="data/vlmbias_400.jsonl")
    parser.add_argument("--out-dir", default="runs/vlmbias_attention_ablation")
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", choices=[name for name, _, _ in CONDITIONS], default=None)
    args = parser.parse_args()

    selected = CONDITIONS
    if args.conditions:
        selected = [condition for condition in CONDITIONS if condition[0] in set(args.conditions)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_paths = []
    for condition_name, alpha, layer_selection in selected:
        out_path = out_dir / f"qwen25vl_3b_attention_{condition_name}.jsonl"
        adapter = (
            "adapters.qwen25_vl_attention:make_adapter,"
            f"max_pixels={args.max_pixels},"
            "max_new_tokens=16,"
            "do_sample=true,"
            f"temperature={args.temperature},"
            f"seed={args.seed},"
            f"attention_alpha={alpha},"
            f"layer_selection={layer_selection},"
            "prompt_mode=baseline"
        )
        _run_eval(args.dataset, adapter, out_path, args.limit)
        run_paths.append((condition_name, out_path))

    _write_report(run_paths, out_dir / "summary.tsv")


def _run_eval(dataset: str, adapter: str, out_path: Path, limit: int | None) -> None:
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
    if limit is not None:
        command.extend(["--limit", str(limit)])
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _write_report(run_paths: list[tuple[str, Path]], out_path: Path) -> None:
    header = [
        "condition",
        "n",
        "accuracy",
        "bias_aligned_fraction",
        "bias_aligned_error_rate",
        "mean_attn_all",
        "mean_attn_correct",
        "mean_attn_bias",
        "mean_attn_other_wrong",
        "mean_attn_boosted_layers",
    ]
    lines = ["\t".join(header)]
    for condition_name, jsonl_path in run_paths:
        summary = json.loads(jsonl_path.with_suffix(".summary.json").read_text(encoding="utf-8"))
        rows = _load_rows(jsonl_path)
        attn_all = [_attention_mass(row, "mean_image_attention_mass") for row in rows]
        attn_boosted = [_attention_mass(row, "mean_boosted_layer_image_attention_mass") for row in rows]
        lines.append(
            "\t".join(
                [
                    condition_name,
                    str(summary["n"]),
                    f"{summary['accuracy']:.6f}",
                    f"{summary['bias_aligned_fraction']:.6f}",
                    f"{summary['bias_aligned_error_rate']:.6f}",
                    _fmt_mean(attn_all),
                    _fmt_mean([_attention_mass(row, "mean_image_attention_mass") for row in rows if row["is_correct"]]),
                    _fmt_mean(
                        [
                            _attention_mass(row, "mean_image_attention_mass")
                            for row in rows
                            if row["is_bias_aligned_error"]
                        ]
                    ),
                    _fmt_mean(
                        [
                            _attention_mass(row, "mean_image_attention_mass")
                            for row in rows
                            if (not row["is_correct"]) and (not row["is_bias_aligned_error"])
                        ]
                    ),
                    _fmt_mean(attn_boosted),
                ]
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path.read_text(encoding="utf-8"))
    print(f"Wrote attention ablation report to {out_path}")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _attention_mass(row: dict[str, Any], key: str) -> float | None:
    attention = (((row.get("metadata") or {}).get("generation") or {}).get("attention") or {})
    value = attention.get(key)
    return float(value) if value is not None else None


def _fmt_mean(values: list[float | None]) -> str:
    clean = [value for value in values if value is not None]
    return f"{sum(clean) / len(clean):.6f}" if clean else ""


if __name__ == "__main__":
    main()
