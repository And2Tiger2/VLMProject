from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from PIL import Image
from tqdm import tqdm

from adapters.paligemma2 import DEFAULT_MODEL_ID
from adapters.paligemma2_attention import make_adapter
from vlm_eval.datasets import load_examples
from vlm_eval.metrics import prediction_to_dict, score_response, summarize
from vlm_eval.naturalbench import (
    NaturalBenchPrediction,
    extract_naturalbench_answer,
    load_naturalbench_calls,
    normalize_naturalbench_answer,
    prediction_to_dict as naturalbench_prediction_to_dict,
    summarize_naturalbench,
)
from vlm_eval.overheat import maybe_pause
from vlm_eval.types import EvalExample


LAYER_SELECTIONS = ("last25", "last50", "all")
VLMBIAS_METRICS = ("accuracy", "bias_aligned_fraction", "bias_aligned_error_rate", "error_rate")
NATURALBENCH_METRICS = ("Acc", "Q_Acc", "I_Acc", "G_Acc")
ATTENTION_METRICS = ("mean_image_attention_mass", "mean_boosted_layer_image_attention_mass")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep PaliGemma2 image-token attention alpha.")
    parser.add_argument("--vlmbias-dataset", default="data/vlmbias_400.jsonl")
    parser.add_argument("--naturalbench-dataset", default="data/naturalbench_100_groups.jsonl")
    parser.add_argument("--out-dir", default="runs/paligemma2_attention_alpha_sweep")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--blank-image-size", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prompt-mode", default="baseline")
    parser.add_argument("--vlmbias-limit", type=int, default=None)
    parser.add_argument("--naturalbench-limit-groups", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0],
    )
    parser.add_argument("--layer-selections", nargs="+", choices=LAYER_SELECTIONS, default=list(LAYER_SELECTIONS))
    parser.add_argument(
        "--skip-naturalbench",
        action="store_true",
        help="Run only VLMBias. Intended for debugging; full experiment should include NaturalBench.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "vlmbias_dataset": args.vlmbias_dataset,
        "naturalbench_dataset": args.naturalbench_dataset,
        "vlmbias_limit": args.vlmbias_limit,
        "naturalbench_limit_groups": args.naturalbench_limit_groups,
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
        "alphas": args.alphas,
        "layer_selections": args.layer_selections,
        "conditions": _conditions(args.alphas, args.layer_selections),
    }
    _write_json(out_dir / "experiment_config.json", run_config)

    vlmbias_examples = load_examples(args.vlmbias_dataset, limit=args.vlmbias_limit)
    naturalbench_calls = [] if args.skip_naturalbench else load_naturalbench_calls(
        args.naturalbench_dataset,
        limit_groups=args.naturalbench_limit_groups,
    )

    adapter = make_adapter(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seeds[0] if args.seeds else 0,
        device_map=args.device_map,
        prompt_mode=args.prompt_mode,
        blank_image_size=args.blank_image_size,
        attention_alpha=0.0,
        layer_selection="baseline",
    )

    vlmbias_rows = []
    naturalbench_rows = []
    for seed in args.seeds:
        for condition in _conditions(args.alphas, args.layer_selections):
            _configure_adapter(
                adapter,
                seed=seed,
                attention_alpha=condition["attention_alpha"],
                layer_selection=condition["layer_selection"],
            )
            vlmbias_rows.append(
                _run_vlmbias_condition(
                    adapter,
                    vlmbias_examples,
                    out_dir,
                    condition,
                    seed,
                    run_config,
                )
            )
            if not args.skip_naturalbench:
                naturalbench_rows.append(
                    _run_naturalbench_condition(
                        adapter,
                        naturalbench_calls,
                        out_dir,
                        condition,
                        seed,
                        run_config,
                    )
                )

    _write_summary_tsv(vlmbias_rows, out_dir / "vlmbias_summary_by_seed.tsv", benchmark="vlmbias")
    _write_aggregate_tsv(vlmbias_rows, out_dir / "vlmbias_summary_aggregate.tsv", VLMBIAS_METRICS)
    if naturalbench_rows:
        _write_summary_tsv(naturalbench_rows, out_dir / "naturalbench_summary_by_seed.tsv", benchmark="naturalbench")
        _write_aggregate_tsv(naturalbench_rows, out_dir / "naturalbench_summary_aggregate.tsv", NATURALBENCH_METRICS)


def _conditions(alphas: list[float], layer_selections: list[str]) -> list[dict[str, Any]]:
    conditions = [
        {
            "condition": "baseline",
            "attention_alpha": 0.0,
            "layer_selection": "baseline",
            "is_baseline": True,
        }
    ]
    for alpha in alphas:
        if alpha == 0:
            continue
        for layer_selection in layer_selections:
            conditions.append(
                {
                    "condition": f"alpha{_alpha_label(alpha)}_{layer_selection}",
                    "attention_alpha": alpha,
                    "layer_selection": layer_selection,
                    "is_baseline": False,
                }
            )
    return conditions


def _configure_adapter(adapter, *, seed: int, attention_alpha: float, layer_selection: str) -> None:
    adapter.seed = seed
    adapter.attention_alpha = attention_alpha
    adapter.layer_selection = layer_selection
    adapter.name = f"{adapter.model_id}-attention-{layer_selection}-alpha{attention_alpha}-seed{seed}"
    adapter._configure_attention_modules()


def _run_vlmbias_condition(adapter, examples, out_dir: Path, condition: dict[str, Any], seed: int, run_config: dict):
    condition_name = condition["condition"]
    out_path = out_dir / "vlmbias" / f"paligemma2_3b_attention_{condition_name}_seed{seed}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
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

    summary = summarize(predictions)
    rows = [prediction_to_dict(prediction) for prediction in predictions]
    summary.update(_attention_summary(rows))
    summary["run_config"] = _condition_run_config(run_config, condition, seed, "vlmbias")
    _write_json(out_path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return _summary_row("vlmbias", condition, seed, summary, out_path)


def _run_naturalbench_condition(adapter, calls, out_dir: Path, condition: dict[str, Any], seed: int, run_config: dict):
    condition_name = condition["condition"]
    out_path = out_dir / "naturalbench" / f"paligemma2_3b_attention_{condition_name}_seed{seed}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = []
    row_dicts = []
    with out_path.open("w", encoding="utf-8") as handle:
        for call in tqdm(calls, desc=f"NaturalBench {condition_name} seed{seed}"):
            maybe_pause()
            example = EvalExample(
                id=call.call_id,
                prompt=call.prompt,
                ground_truth=call.ground_truth,
                image=Image.open(call.image_path).convert("RGB"),
                image_path=call.image_path,
                topic="NaturalBench",
                sub_topic=call.question_type,
                metadata={
                    "group_id": call.group_id,
                    "question_id": call.question_id,
                    "image_id": call.image_id,
                    "source": call.source,
                },
            )
            raw_response = adapter.generate(example)
            maybe_pause()
            parsed_answer = extract_naturalbench_answer(raw_response, call.question_type)
            prediction = NaturalBenchPrediction(
                group_id=call.group_id,
                call_id=call.call_id,
                question_id=call.question_id,
                image_id=call.image_id,
                question_type=call.question_type,
                prompt=call.prompt,
                ground_truth=call.ground_truth,
                raw_response=raw_response,
                parsed_answer=parsed_answer,
                is_correct=normalize_naturalbench_answer(parsed_answer)
                == normalize_naturalbench_answer(call.ground_truth),
                source=call.source,
            )
            row = naturalbench_prediction_to_dict(prediction)
            generation_metadata = getattr(adapter, "last_generation_metadata", None)
            if generation_metadata:
                row["metadata"] = {"generation": generation_metadata}
            predictions.append(prediction)
            row_dicts.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    summary = summarize_naturalbench(predictions)
    summary.update(_attention_summary(row_dicts))
    summary["run_config"] = _condition_run_config(run_config, condition, seed, "naturalbench")
    _write_json(out_path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return _summary_row("naturalbench", condition, seed, summary, out_path)


def _attention_summary(rows: list[dict[str, Any]]) -> dict[str, float | str]:
    summary = {}
    for metric in ATTENTION_METRICS:
        values = [_attention_value(row, metric) for row in rows]
        summary[metric] = _mean_clean(values)
    return summary


def _attention_value(row: dict[str, Any], key: str) -> float | None:
    attention = (((row.get("metadata") or {}).get("generation") or {}).get("attention") or {})
    value = attention.get(key)
    return float(value) if value is not None else None


def _mean_clean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _condition_run_config(run_config: dict, condition: dict[str, Any], seed: int, benchmark: str) -> dict:
    return {
        **run_config,
        "benchmark": benchmark,
        "condition": condition["condition"],
        "seed": seed,
        "attention_alpha": condition["attention_alpha"],
        "layer_selection": condition["layer_selection"],
    }


def _summary_row(benchmark: str, condition: dict[str, Any], seed: int, summary: dict, out_path: Path) -> dict:
    row = {
        "benchmark": benchmark,
        "condition": condition["condition"],
        "seed": seed,
        "attention_alpha": condition["attention_alpha"],
        "layer_selection": condition["layer_selection"],
        "is_baseline": condition["is_baseline"],
        "out_path": str(out_path),
    }
    for key, value in summary.items():
        if isinstance(value, int | float) or value is None:
            row[key] = value
    return row


def _write_summary_tsv(rows: list[dict], out_path: Path, *, benchmark: str) -> None:
    metric_fields = list(VLMBIAS_METRICS if benchmark == "vlmbias" else NATURALBENCH_METRICS)
    fieldnames = [
        "benchmark",
        "condition",
        "seed",
        "attention_alpha",
        "layer_selection",
        "is_baseline",
        *metric_fields,
        *ATTENTION_METRICS,
        "out_path",
    ]
    _write_tsv(rows, out_path, fieldnames)
    print(out_path.read_text(encoding="utf-8"))
    print(f"Wrote {benchmark} per-seed summary to {out_path}")


def _write_aggregate_tsv(rows: list[dict], out_path: Path, metrics: tuple[str, ...]) -> None:
    grouped: dict[tuple[str, float, str], list[dict]] = {}
    for row in rows:
        key = (row["condition"], float(row["attention_alpha"]), row["layer_selection"])
        grouped.setdefault(key, []).append(row)

    aggregate_rows = []
    for (condition, attention_alpha, layer_selection), group_rows in sorted(grouped.items()):
        aggregate_row = {
            "condition": condition,
            "attention_alpha": attention_alpha,
            "layer_selection": layer_selection,
            "seeds": ",".join(str(row["seed"]) for row in sorted(group_rows, key=lambda item: item["seed"])),
            "runs": len(group_rows),
        }
        for metric in (*metrics, *ATTENTION_METRICS):
            values = [row.get(metric) for row in group_rows if row.get(metric) is not None]
            aggregate_row[f"{metric}_mean"] = mean(values) if values else None
            aggregate_row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else None
        aggregate_rows.append(aggregate_row)

    fieldnames = ["condition", "attention_alpha", "layer_selection", "seeds", "runs"]
    for metric in (*metrics, *ATTENTION_METRICS):
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
