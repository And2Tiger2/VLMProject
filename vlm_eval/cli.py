from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from vlm_eval.adapters import load_adapter
from vlm_eval.datasets import load_examples
from vlm_eval.metrics import prediction_to_dict, score_response, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a VLM on VLMBias-style examples.")
    parser.add_argument("--dataset", default="anvo25/vlms-are-biased")
    parser.add_argument("--split", default="main")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--adapter",
        default="adapters.dummy:make_adapter",
        help="Python adapter spec, e.g. `adapters.qwen25_vl:make_adapter,model_id=Qwen/Qwen2.5-VL-3B-Instruct`.",
    )
    parser.add_argument("--out", default="runs/vlmbias_eval.jsonl")
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args()

    examples = load_examples(
        args.dataset,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    adapter = load_adapter(args.adapter)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = []
    with out_path.open("w", encoding="utf-8") as handle:
        for example in tqdm(examples, desc=f"Evaluating {adapter.name}"):
            raw_response = adapter.generate(example)
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
        "adapter": args.adapter,
        "adapter_name": adapter.name,
        "adapter_config": getattr(adapter, "eval_config", None),
    }
    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote predictions to {out_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
