from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed local NaturalBench eval slice.")
    parser.add_argument("--dataset", default="BaiqiL/NaturalBench")
    parser.add_argument("--split", default="train")
    parser.add_argument("--groups", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/naturalbench_100_groups.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    image_dir = out_path.parent / f"{out_path.stem}_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)
    rows_by_type = defaultdict(list)
    for row in dataset:
        rows_by_type[_question_type(row)].append(row)

    rng = random.Random(args.seed)
    for rows in rows_by_type.values():
        rng.shuffle(rows)

    selected = []
    question_types = sorted(rows_by_type)
    while len(selected) < args.groups and any(rows_by_type.values()):
        for question_type in question_types:
            if rows_by_type[question_type]:
                selected.append(rows_by_type[question_type].pop())
                if len(selected) == args.groups:
                    break

    rng.shuffle(selected)

    with out_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(selected):
            group_id = str(row.get("Index") or idx)
            image_0_path = image_dir / f"{idx:04d}_{_safe_name(group_id)}_image0.jpg"
            image_1_path = image_dir / f"{idx:04d}_{_safe_name(group_id)}_image1.jpg"
            row["Image_0"].convert("RGB").save(image_0_path)
            row["Image_1"].convert("RGB").save(image_1_path)

            output = {
                "id": group_id,
                "question_type": _question_type(row),
                "source": row.get("Source"),
                "question_0": row["Question_0"],
                "question_1": row["Question_1"],
                "image_0_path": str(image_0_path.relative_to(out_path.parent)),
                "image_1_path": str(image_1_path.relative_to(out_path.parent)),
                "answers": {
                    "q0_i0": row["Image_0_Question_0"],
                    "q0_i1": row["Image_1_Question_0"],
                    "q1_i0": row["Image_0_Question_1"],
                    "q1_i1": row["Image_1_Question_1"],
                },
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    manifest_path = out_path.with_suffix(".manifest.json")
    counts_by_type = defaultdict(int)
    counts_by_source = defaultdict(int)
    for row in selected:
        counts_by_type[_question_type(row)] += 1
        counts_by_source[row.get("Source") or "unknown"] += 1
    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "groups": len(selected),
        "model_calls": len(selected) * 4,
        "seed": args.seed,
        "out": str(out_path),
        "image_dir": str(image_dir),
        "counts_by_question_type": dict(sorted(counts_by_type.items())),
        "counts_by_source": dict(sorted(counts_by_source.items())),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(json.dumps(manifest, indent=2, sort_keys=True))


def _question_type(row: dict) -> str:
    return row.get("Question Type") or row.get("Question_Type")


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)[:80]


if __name__ == "__main__":
    main()

