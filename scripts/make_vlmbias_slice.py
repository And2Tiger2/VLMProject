from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed local VLMBias eval slice.")
    parser.add_argument("--dataset", default="anvo25/vlms-are-biased")
    parser.add_argument("--split", default="main")
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/vlmbias_400.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    image_dir = out_path.parent / f"{out_path.stem}_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)
    rows_by_topic = defaultdict(list)
    for row in dataset:
        rows_by_topic[row.get("topic") or "unknown"].append(row)

    rng = random.Random(args.seed)
    for rows in rows_by_topic.values():
        rng.shuffle(rows)

    selected = []
    topics = sorted(rows_by_topic)
    while len(selected) < args.n and any(rows_by_topic.values()):
        for topic in topics:
            if rows_by_topic[topic]:
                selected.append(rows_by_topic[topic].pop())
                if len(selected) == args.n:
                    break

    rng.shuffle(selected)

    with out_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(selected):
            example_id = str(row.get("ID") or row.get("id") or idx)
            image = row["image"].convert("RGB")
            image_path = image_dir / f"{idx:04d}_{_safe_name(example_id)}.png"
            image.save(image_path)

            metadata = row.get("metadata")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {"raw_metadata": metadata}

            output = {
                "id": example_id,
                "prompt": row["prompt"],
                "ground_truth": row["ground_truth"],
                "expected_bias": row.get("expected_bias"),
                "topic": row.get("topic"),
                "sub_topic": row.get("sub_topic"),
                "image_path": str(image_path.relative_to(out_path.parent)),
                "metadata": metadata,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    manifest_path = out_path.with_suffix(".manifest.json")
    counts = defaultdict(int)
    for row in selected:
        counts[row.get("topic") or "unknown"] += 1
    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "n": len(selected),
        "seed": args.seed,
        "out": str(out_path),
        "image_dir": str(image_dir),
        "counts_by_topic": dict(sorted(counts.items())),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(json.dumps(manifest, indent=2, sort_keys=True))


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)[:80]


if __name__ == "__main__":
    main()

