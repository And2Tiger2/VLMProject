from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from tqdm import tqdm

from adapters.qwen25_vl_gaze_attention import make_adapter
from vlm_eval.datasets import load_examples
from vlm_eval.metrics import prediction_to_dict, score_response, summarize
from vlm_eval.overheat import maybe_pause


VLMBIAS_METRICS = ("accuracy", "bias_aligned_fraction", "bias_aligned_error_rate", "error_rate")
ATTENTION_METRICS = (
    "mean_image_attention_mass",
    "mean_boosted_layer_image_attention_mass",
    "mean_boosted_head_image_attention_mass",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep image-token attention boosts on discovered Qwen gaze heads.")
    parser.add_argument("--dataset", default="segments/vlm_bias_attention/data/vlmbias_400.jsonl")
    parser.add_argument("--out-dir", default="segments/vlm_bias_attention/runs/vlmbias_gaze_attention_sweep")
    parser.add_argument(
        "--gaze-ranking",
        default="segments/gaze_heads_qwen25/runs/gaze_discovery_merged_0_500/gaze_head_ranking.json",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--min-pixels", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prompt-mode", default="baseline")
    parser.add_argument("--limit", type=int, default=None, help="Use 0 or omit for the full dataset.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--decode-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a condition when both its prediction JSONL and summary JSON already exist.",
    )
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "dataset": args.dataset,
        "limit": limit,
        "model_id": args.model_id,
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "device_map": args.device_map,
        "max_new_tokens": args.max_new_tokens,
        "prompt_mode": args.prompt_mode,
        "gaze_ranking": args.gaze_ranking,
        "seeds": args.seeds,
        "top_ks": args.top_ks,
        "alphas": args.alphas,
        "decode_only": args.decode_only,
        "conditions": _conditions(args.alphas, args.top_ks),
    }
    _write_json(out_dir / "experiment_config.json", run_config)

    examples = load_examples(args.dataset, limit=limit)
    adapter = make_adapter(
        model_id=args.model_id,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seeds[0] if args.seeds else 0,
        device_map=args.device_map,
        prompt_mode=args.prompt_mode,
        attention_alpha=0.0,
        gaze_ranking_path=args.gaze_ranking,
        top_k_gaze=0,
        decode_only=args.decode_only,
    )

    rows = []
    for seed in args.seeds:
        for condition in _conditions(args.alphas, args.top_ks):
            _configure_adapter(
                adapter,
                seed=seed,
                attention_alpha=condition["attention_alpha"],
                top_k_gaze=condition["top_k_gaze"],
                decode_only=args.decode_only,
            )
            rows.append(
                _run_condition(
                    adapter=adapter,
                    examples=examples,
                    out_dir=out_dir,
                    condition=condition,
                    seed=seed,
                    run_config=run_config,
                    resume=args.resume,
                )
            )

    _write_summary_tsv(rows, out_dir / "summary_by_seed.tsv")
    _write_aggregate_tsv(rows, out_dir / "summary_aggregate.tsv")


def _conditions(alphas: list[float], top_ks: list[int]) -> list[dict[str, Any]]:
    conditions = [
        {
            "condition": "baseline",
            "attention_alpha": 0.0,
            "top_k_gaze": 0,
            "is_baseline": True,
        }
    ]
    for alpha in alphas:
        if alpha == 0:
            continue
        for top_k in top_ks:
            conditions.append(
                {
                    "condition": f"gaze_top{top_k}_alpha{_alpha_label(alpha)}",
                    "attention_alpha": float(alpha),
                    "top_k_gaze": int(top_k),
                    "is_baseline": False,
                }
            )
    return conditions


def _configure_adapter(adapter, *, seed: int, attention_alpha: float, top_k_gaze: int, decode_only: bool) -> None:
    adapter.seed = seed
    adapter.attention_alpha = float(attention_alpha)
    adapter.top_k_gaze = int(top_k_gaze)
    adapter.decode_only = bool(decode_only)
    adapter.name = (
        f"{adapter.model_id}-gaze-attention-top{top_k_gaze}-alpha{attention_alpha}"
        f"-decode_only{int(decode_only)}-seed{seed}"
    )
    adapter._configure_attention_modules()


def _run_condition(
    *,
    adapter,
    examples,
    out_dir: Path,
    condition: dict[str, Any],
    seed: int,
    run_config: dict,
    resume: bool,
) -> dict:
    condition_name = condition["condition"]
    out_path = out_dir / f"qwen25vl_3b_vlmbias_gaze_attention_{condition_name}_seed{seed}.jsonl"
    summary_path = out_path.with_suffix(".summary.json")
    if resume and out_path.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return _summary_row(condition, seed, summary, out_path)

    predictions = []
    with out_path.open("w", encoding="utf-8") as handle:
        for example in tqdm(examples, desc=f"VLMBias {condition_name} seed{seed}"):
            maybe_pause()
            raw_response = adapter.generate(example)
            maybe_pause()
            prediction = score_response(example, raw_response)
            generation_metadata = getattr(adapter, "last_generation_metadata", None)
            if generation_metadata:
                prediction = replace(
                    prediction,
                    metadata={
                        **(prediction.metadata or {}),
                        "generation": generation_metadata,
                    },
                )
            predictions.append(prediction)
            handle.write(json.dumps(prediction_to_dict(prediction), ensure_ascii=False) + "\n")
            handle.flush()

    rows = [prediction_to_dict(prediction) for prediction in predictions]
    summary = summarize(predictions)
    summary.update(_attention_summary(rows))
    summary["run_config"] = _condition_run_config(run_config, condition, seed)
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return _summary_row(condition, seed, summary, out_path)


def _attention_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    return {metric: _mean_clean([_attention_value(row, metric) for row in rows]) for metric in ATTENTION_METRICS}


def _attention_value(row: dict[str, Any], key: str) -> float | None:
    attention = (((row.get("metadata") or {}).get("generation") or {}).get("attention") or {})
    value = attention.get(key)
    return float(value) if value is not None else None


def _mean_clean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _condition_run_config(run_config: dict, condition: dict[str, Any], seed: int) -> dict:
    return {
        **run_config,
        "condition": condition["condition"],
        "seed": seed,
        "attention_alpha": condition["attention_alpha"],
        "top_k_gaze": condition["top_k_gaze"],
    }


def _summary_row(condition: dict[str, Any], seed: int, summary: dict, out_path: Path) -> dict:
    row = {
        "condition": condition["condition"],
        "seed": seed,
        "attention_alpha": condition["attention_alpha"],
        "top_k_gaze": condition["top_k_gaze"],
        "is_baseline": condition["is_baseline"],
        "out_path": str(out_path),
    }
    for key, value in summary.items():
        if isinstance(value, int | float) or value is None:
            row[key] = value
    return row


def _write_summary_tsv(rows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "condition",
        "seed",
        "attention_alpha",
        "top_k_gaze",
        "is_baseline",
        *VLMBIAS_METRICS,
        *ATTENTION_METRICS,
        "out_path",
    ]
    _write_tsv(rows, out_path, fieldnames)
    print(out_path.read_text(encoding="utf-8"))
    print(f"Wrote per-seed summary to {out_path}")


def _write_aggregate_tsv(rows: list[dict], out_path: Path) -> None:
    grouped: dict[tuple[str, float, int], list[dict]] = {}
    for row in rows:
        key = (row["condition"], float(row["attention_alpha"]), int(row["top_k_gaze"]))
        grouped.setdefault(key, []).append(row)

    aggregate_rows = []
    for (condition, attention_alpha, top_k_gaze), group_rows in sorted(grouped.items()):
        aggregate_row = {
            "condition": condition,
            "attention_alpha": attention_alpha,
            "top_k_gaze": top_k_gaze,
            "seeds": ",".join(str(row["seed"]) for row in sorted(group_rows, key=lambda item: item["seed"])),
            "runs": len(group_rows),
        }
        for metric in (*VLMBIAS_METRICS, *ATTENTION_METRICS):
            values = [row.get(metric) for row in group_rows if row.get(metric) is not None]
            aggregate_row[f"{metric}_mean"] = mean(values) if values else None
            aggregate_row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else None
        aggregate_rows.append(aggregate_row)

    fieldnames = ["condition", "attention_alpha", "top_k_gaze", "seeds", "runs"]
    for metric in (*VLMBIAS_METRICS, *ATTENTION_METRICS):
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    _write_tsv(aggregate_rows, out_path, fieldnames)
    print(out_path.read_text(encoding="utf-8"))
    print(f"Wrote aggregate summary to {out_path}")


def _write_tsv(rows: list[dict], out_path: Path, fieldnames: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_value(row.get(field)) for field in fieldnames})


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else str(value)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _alpha_label(alpha: float) -> str:
    text = f"{alpha:g}"
    return text.replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    main()
