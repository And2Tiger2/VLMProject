from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from adapters.paligemma2 import DEFAULT_MODEL_ID, PaliGemma2Adapter
from vlm_eval.datasets import load_examples
from vlm_eval.metrics import prediction_to_dict, score_response, summarize
from vlm_eval.overheat import maybe_pause


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny PaliGemma2 VLMBias inference smoke test.")
    parser.add_argument("--dataset", default="data/vlmbias_400.jsonl")
    parser.add_argument("--split", default="main")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--prompt-mode", default="baseline")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-sample", action="store_true")
    parser.add_argument("--blank-image", action="store_true")
    parser.add_argument("--out", default="runs/paligemma2_smoke/vlmbias_smoke.jsonl")
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args()

    examples = load_examples(
        args.dataset,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    if not examples:
        raise ValueError(f"No examples loaded from {args.dataset}.")
    if not args.blank_image:
        missing_images = [example.id for example in examples if example.image is None and example.image_path is None]
        if missing_images:
            raise ValueError(
                "PaliGemma2 smoke tests require images. Missing image for examples: "
                + ", ".join(missing_images[:5])
            )

    adapter = PaliGemma2Adapter(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.no_sample,
        temperature=None if args.no_sample else args.temperature,
        include_image=not args.blank_image,
        seed=args.seed,
        device_map=args.device_map,
        prompt_mode=args.prompt_mode,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = []
    with out_path.open("w", encoding="utf-8") as handle:
        for example in tqdm(examples, desc=f"Smoke testing {adapter.name}"):
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

    summary = summarize(predictions)
    summary["run_config"] = {
        "dataset": args.dataset,
        "split": args.split,
        "limit": args.limit,
        "seed": args.seed,
        "shuffle": args.shuffle,
        "adapter_name": adapter.name,
        "adapter_config": adapter.eval_config,
    }
    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote predictions to {out_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
