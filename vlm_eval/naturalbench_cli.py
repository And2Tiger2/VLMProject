from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from vlm_eval.adapters import load_adapter
from vlm_eval.naturalbench import (
    evaluate_naturalbench,
    load_naturalbench_calls,
    prediction_to_dict,
    summarize_naturalbench,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a VLM on a local NaturalBench slice.")
    parser.add_argument("--dataset", default="data/naturalbench_100_groups.jsonl")
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument(
        "--adapter",
        default="adapters.dummy:make_adapter,mode=truth",
        help="Python adapter spec, e.g. `adapters.qwen25_vl:make_adapter,model_id=Qwen/Qwen2.5-VL-3B-Instruct`.",
    )
    parser.add_argument("--out", default="runs/naturalbench_eval.jsonl")
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--resume", action="store_true", help="Append only missing calls from an existing JSONL output.")
    args = parser.parse_args()

    calls = load_naturalbench_calls(args.dataset, limit_groups=args.limit_groups)
    adapter = load_adapter(args.adapter)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions, completed = _load_existing_predictions(out_path) if args.resume else ([], set())
    remaining_calls = [
        call for call in calls if (call.group_id, call.call_id) not in completed
    ]
    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        for prediction in tqdm(
            evaluate_naturalbench(remaining_calls, adapter),
            total=len(remaining_calls),
            desc=f"Evaluating {adapter.name} on NaturalBench",
        ):
            predictions.append(prediction)
            handle.write(json.dumps(prediction_to_dict(prediction), ensure_ascii=False) + "\n")
            handle.flush()

    summary = summarize_naturalbench(predictions)
    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote predictions to {out_path}")
    print(f"Wrote summary to {summary_path}")


def _load_existing_predictions(out_path: Path):
    if not out_path.exists():
        return [], set()

    from vlm_eval.naturalbench import NaturalBenchPrediction

    predictions = []
    completed = set()
    with out_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            predictions.append(NaturalBenchPrediction(**row))
            completed.add((row["group_id"], row["call_id"]))
    return predictions, completed


if __name__ == "__main__":
    main()
