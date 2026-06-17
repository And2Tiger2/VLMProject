from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


VARIANTS = ["neutral", "counterfactual", "structured", "verification"]
ANSWER_MODES = [
    ("image_again", True),
    ("description_only", False),
]
CONDITIONS = [
    (f"{variant}_{mode_name}", variant, answer_include_image)
    for variant in VARIANTS
    for mode_name, answer_include_image in ANSWER_MODES
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLMBias two-pass prompt variants with description-attention logging.")
    parser.add_argument("--dataset", default="data/vlmbias_400.jsonl")
    parser.add_argument("--out-dir", default="runs/vlmbias_two_pass_prompt_attention")
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--description-max-new-tokens", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", choices=[name for name, _, _ in CONDITIONS], default=None)
    args = parser.parse_args()

    selected = CONDITIONS
    if args.conditions:
        selected_names = set(args.conditions)
        selected = [condition for condition in CONDITIONS if condition[0] in selected_names]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_paths = []
    for condition_name, variant, answer_include_image in selected:
        out_path = out_dir / f"qwen25vl_3b_two_pass_{condition_name}.jsonl"
        adapter = (
            "adapters.qwen25_vl_two_pass_attention:make_adapter,"
            f"max_pixels={args.max_pixels},"
            "max_new_tokens=16,"
            "do_sample=true,"
            f"temperature={args.temperature},"
            f"seed={args.seed},"
            f"description_max_new_tokens={args.description_max_new_tokens},"
            f"description_prompt_variant={variant},"
            f"answer_include_image={str(answer_include_image).lower()}"
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
        "error_rate",
        "mean_description_attn",
        "mean_description_attn_correct",
        "mean_description_attn_bias",
        "mean_description_attn_other_wrong",
        "mean_image_attn",
        "mean_image_attn_correct",
        "mean_image_attn_bias",
        "mean_image_attn_other_wrong",
        "mean_question_attn",
    ]
    lines = ["\t".join(header)]
    for condition_name, jsonl_path in run_paths:
        summary = json.loads(jsonl_path.with_suffix(".summary.json").read_text(encoding="utf-8"))
        rows = _load_rows(jsonl_path)
        lines.append(
            "\t".join(
                [
                    condition_name,
                    str(summary["n"]),
                    f"{summary['accuracy']:.6f}",
                    f"{summary['bias_aligned_fraction']:.6f}",
                    f"{summary['bias_aligned_error_rate']:.6f}",
                    f"{summary['error_rate']:.6f}",
                    _fmt_metric(rows, "mean_description_attention_mass"),
                    _fmt_metric(rows, "mean_description_attention_mass", "correct"),
                    _fmt_metric(rows, "mean_description_attention_mass", "bias"),
                    _fmt_metric(rows, "mean_description_attention_mass", "other_wrong"),
                    _fmt_metric(rows, "mean_image_attention_mass"),
                    _fmt_metric(rows, "mean_image_attention_mass", "correct"),
                    _fmt_metric(rows, "mean_image_attention_mass", "bias"),
                    _fmt_metric(rows, "mean_image_attention_mass", "other_wrong"),
                    _fmt_metric(rows, "mean_question_attention_mass"),
                ]
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path.read_text(encoding="utf-8"))
    print(f"Wrote two-pass prompt attention report to {out_path}")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fmt_metric(rows: list[dict[str, Any]], key: str, category: str | None = None) -> str:
    selected = rows if category is None else [row for row in rows if _category(row) == category]
    values = [_attention_value(row, key) for row in selected]
    clean = [value for value in values if value is not None]
    return f"{sum(clean) / len(clean):.6f}" if clean else ""


def _attention_value(row: dict[str, Any], key: str) -> float | None:
    attention = (((row.get("metadata") or {}).get("generation") or {}).get("attention") or {})
    value = attention.get(key)
    return float(value) if value is not None else None


def _category(row: dict[str, Any]) -> str:
    if row["is_correct"]:
        return "correct"
    if row["is_bias_aligned_error"]:
        return "bias"
    return "other_wrong"


if __name__ == "__main__":
    main()
