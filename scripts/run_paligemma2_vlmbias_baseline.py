from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from tqdm import tqdm

from adapters.paligemma2 import DEFAULT_MODEL_ID, PaliGemma2Adapter
from vlm_eval.datasets import load_examples
from vlm_eval.metrics import prediction_to_dict, score_response, summarize
from vlm_eval.overheat import maybe_pause


CONDITIONS = {
    "image": True,
    "blank_image": False,
}
SUMMARY_METRICS = ("accuracy", "bias_aligned_fraction", "bias_aligned_error_rate", "error_rate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vanilla PaliGemma2 VLMBias baselines.")
    parser.add_argument("--dataset", default="data/vlmbias_400.jsonl")
    parser.add_argument("--out-dir", default="runs/paligemma2_vlmbias_baseline")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--blank-image-size", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prompt-mode", default="baseline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(CONDITIONS))
    args = parser.parse_args()

    examples = load_examples(args.dataset, limit=args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "dataset": args.dataset,
        "limit": args.limit,
        "model_id": args.model_id,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "device_map": args.device_map,
        "blank_image_size": args.blank_image_size,
        "max_new_tokens": args.max_new_tokens,
        "prompt_mode": args.prompt_mode,
        "seeds": args.seeds,
        "conditions": args.conditions,
        "blank_image_behavior": "white RGB image used because PaliGemma requires an image input",
    }
    _write_json(out_dir / "experiment_config.json", run_config)

    adapter = PaliGemma2Adapter(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        include_image=True,
        seed=args.seeds[0] if args.seeds else 0,
        device_map=args.device_map,
        prompt_mode=args.prompt_mode,
        blank_image_size=args.blank_image_size,
    )

    summary_rows = []
    for seed in args.seeds:
        for condition in args.conditions:
            include_image = CONDITIONS[condition]
            adapter.seed = seed
            adapter.include_image = include_image
            adapter.name = f"{adapter.model_id}-{condition}-seed{seed}"
            out_path = out_dir / f"paligemma2_3b_vlmbias_baseline_{condition}_seed{seed}.jsonl"
            predictions = _run_condition(adapter, examples, out_path, condition, seed)
            summary = summarize(predictions)
            summary["run_config"] = {
                **run_config,
                "condition": condition,
                "seed": seed,
                "include_image": include_image,
                "adapter_name": adapter.name,
                "adapter_config": adapter.eval_config,
            }
            _write_json(out_path.with_suffix(".summary.json"), summary)
            summary_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "include_image": include_image,
                    "n": summary["n"],
                    **{metric: summary[metric] for metric in SUMMARY_METRICS},
                    "out_path": str(out_path),
                }
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            print(f"Wrote predictions to {out_path}")
            print(f"Wrote summary to {out_path.with_suffix('.summary.json')}")

    _write_per_seed_summary(summary_rows, out_dir / "summary_by_seed.tsv")
    _write_aggregate_summary(summary_rows, out_dir / "summary_aggregate.tsv")


def _run_condition(adapter, examples, out_path: Path, condition: str, seed: int):
    predictions = []
    with out_path.open("w", encoding="utf-8") as handle:
        for example in tqdm(examples, desc=f"PaliGemma2 {condition} seed{seed}"):
            maybe_pause()
            raw_response = adapter.generate(example)
            maybe_pause()
            prediction = score_response(example, raw_response)
            predictions.append(prediction)
            handle.write(json.dumps(prediction_to_dict(prediction), ensure_ascii=False) + "\n")
            handle.flush()
    return predictions


def _write_per_seed_summary(rows: list[dict], out_path: Path) -> None:
    fieldnames = ["condition", "seed", "include_image", "n", *SUMMARY_METRICS, "out_path"]
    _write_tsv(rows, out_path, fieldnames)
    print(out_path.read_text(encoding="utf-8"))
    print(f"Wrote per-seed summary to {out_path}")


def _write_aggregate_summary(rows: list[dict], out_path: Path) -> None:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["condition"], []).append(row)

    aggregate_rows = []
    for condition, group_rows in sorted(grouped.items()):
        aggregate_row = {
            "condition": condition,
            "include_image": group_rows[0]["include_image"],
            "seeds": ",".join(str(row["seed"]) for row in sorted(group_rows, key=lambda item: item["seed"])),
            "runs": len(group_rows),
            "n_per_run": group_rows[0]["n"],
        }
        for metric in SUMMARY_METRICS:
            values = [row[metric] for row in group_rows]
            aggregate_row[f"{metric}_mean"] = mean(values)
            aggregate_row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(aggregate_row)

    fieldnames = ["condition", "include_image", "seeds", "runs", "n_per_run"]
    for metric in SUMMARY_METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    _write_tsv(aggregate_rows, out_path, fieldnames)
    print(out_path.read_text(encoding="utf-8"))
    print(f"Wrote aggregate summary to {out_path}")


def _write_tsv(rows: list[dict], out_path: Path, fieldnames: list[str]) -> None:
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
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
