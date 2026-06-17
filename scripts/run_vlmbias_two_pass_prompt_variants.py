from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from adapters.qwen25_vl_two_pass_variants import make_adapter
from vlm_eval.datasets import load_examples
from vlm_eval.metrics import prediction_to_dict, score_response, summarize
from vlm_eval.overheat import maybe_pause


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
    parser = argparse.ArgumentParser(description="Run fast VLMBias two-pass prompt variants without attention logging.")
    parser.add_argument("--dataset", default="data/vlmbias_400.jsonl")
    parser.add_argument("--out-dir", default="runs/vlmbias_two_pass_prompt_variants")
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--description-max-new-tokens", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", choices=[name for name, _, _ in CONDITIONS], default=None)
    args = parser.parse_args()

    selected = CONDITIONS
    if args.conditions:
        selected_names = set(args.conditions)
        selected = [condition for condition in CONDITIONS if condition[0] in selected_names]

    examples = load_examples(args.dataset, limit=args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = make_adapter(
        max_pixels=args.max_pixels,
        max_new_tokens=16,
        do_sample=True,
        temperature=args.temperature,
        seed=args.seed,
        device_map=args.device_map,
        description_max_new_tokens=args.description_max_new_tokens,
    )

    summaries = []
    for condition_name, variant, answer_include_image in selected:
        adapter.description_prompt_variant = variant
        adapter.answer_include_image = answer_include_image
        suffix = "image-again" if answer_include_image else "description-only"
        adapter.name = f"{adapter.model_id}-two-pass-fast-{variant}-{suffix}"
        out_path = out_dir / f"qwen25vl_3b_two_pass_{condition_name}.jsonl"
        predictions = _run_condition(adapter, examples, out_path, condition_name)
        summary = summarize(predictions)
        summary["run_config"] = {
            "dataset": args.dataset,
            "limit": args.limit,
            "condition": condition_name,
            "description_prompt_variant": variant,
            "answer_include_image": answer_include_image,
            "adapter_name": adapter.name,
            "adapter_config": getattr(adapter, "eval_config", None),
        }
        summary_path = out_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        summaries.append((condition_name, summary))
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote predictions to {out_path}")
        print(f"Wrote summary to {summary_path}")

    _write_report(summaries, out_dir / "summary.tsv")


def _run_condition(adapter, examples, out_path: Path, condition_name: str):
    predictions = []
    with out_path.open("w", encoding="utf-8") as handle:
        for example in tqdm(examples, desc=f"Evaluating {condition_name}"):
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
    return predictions


def _write_report(summaries: list[tuple[str, dict]], out_path: Path) -> None:
    header = [
        "condition",
        "n",
        "accuracy",
        "bias_aligned_fraction",
        "bias_aligned_error_rate",
        "error_rate",
    ]
    lines = ["\t".join(header)]
    for condition_name, summary in summaries:
        lines.append(
            "\t".join(
                [
                    condition_name,
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
    print(f"Wrote fast two-pass prompt-variant report to {out_path}")


if __name__ == "__main__":
    main()
