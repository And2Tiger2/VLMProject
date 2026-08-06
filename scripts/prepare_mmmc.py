#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any

from vlm_eval.mechanistic_heads.config import (
    add_standard_run_arguments,
    effective_limit,
    load_json_config,
    prepare_output_directory,
)
from vlm_eval.mechanistic_heads.reproducibility import seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.schema import PairedExample, assert_no_group_leakage, write_paired_jsonl


DATASET_ID = "ustc-zhangzm/MMMC"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, pair, split, and audit MMMC.")
    add_standard_run_arguments(parser)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--skip-tokenization", action="store_true")
    args = parser.parse_args()
    config = load_json_config(args.config)
    prepare_output_directory(
        args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
        known_outputs=("audit.json", "object_pairs.jsonl"),
    )
    seed_everything(args.seed)
    audit, pairs = prepare_mmmc(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        config=config,
        seed=args.seed,
        smoke=args.smoke,
        limit=effective_limit(args),
        skip_tokenization=args.skip_tokenization,
    )
    pair_path = args.output_dir / "object_pairs.jsonl"
    write_paired_jsonl(pair_path, pairs)
    audit_path = args.output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    write_run_manifest(
        args.output_dir,
        config={
            **config,
            "dataset_id": DATASET_ID,
            "smoke": args.smoke,
            "limit": effective_limit(args),
            "skip_tokenization": args.skip_tokenization,
        },
        seeds={"split": args.seed},
        inputs=[args.config],
        outputs=[audit_path, pair_path],
        status="complete" if audit["valid"] else "failed",
        repo_root=Path.cwd(),
    )
    print(json.dumps(audit, indent=2))


def prepare_mmmc(
    *,
    output_dir: Path,
    cache_dir: Path | None,
    config: dict[str, Any],
    seed: int,
    smoke: bool,
    limit: int | None,
    skip_tokenization: bool,
) -> tuple[dict[str, Any], list[PairedExample]]:
    from datasets import Image as DatasetImage
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, cache_dir=str(cache_dir) if cache_dir else None)
    dataset_fingerprints = {
        split: str(rows._fingerprint) for split, rows in dataset.items()
    }
    # Pairing uses metadata only. Prevent PIL decoding all ~40k images during
    # the audit while still downloading and retaining the official dataset.
    dataset = {
        split: rows.cast_column("image", DatasetImage(decode=False))
        for split, rows in dataset.items()
    }
    split_sizes = {split: len(rows) for split, rows in dataset.items()}
    scanned: list[dict[str, Any]] = []
    per_split_limit = limit if smoke or limit is not None else None
    for split, rows in dataset.items():
        selected = rows if per_split_limit is None else rows.select(range(min(len(rows), per_split_limit)))
        for index, row in enumerate(selected):
            scanned.append({**row, "_split": split, "_index": index, "image": None})

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scanned:
        groups[str(row.get("image_id"))].append(row)
    clean_values = {str(value).casefold() for value in config.get("clean_conflict_values", ["", "null", "none", "clean", "no_conflict"])}
    def is_clean(row: dict[str, Any]) -> bool:
        value = row.get("conflict_type")
        return value is None or str(value).strip().casefold() in clean_values
    conflict_types = Counter(
        "clean" if is_clean(row) else str(row.get("conflict_type"))
        for row in scanned
    )
    pairs: list[PairedExample] = []
    exclusions = Counter()
    pairing_rows: list[dict[str, Any]] = []
    for image_id, rows in sorted(groups.items()):
        clean = [row for row in rows if is_clean(row)]
        conflicts = [row for row in rows if str(row.get("conflict_type")) == "object"]
        if not conflicts:
            continue
        if len(clean) != 1:
            exclusions["not_exactly_one_clean_row"] += len(conflicts)
            continue
        for conflict in conflicts:
            if clean[0]["_split"] != conflict["_split"]:
                exclusions["clean_conflict_cross_source_split"] += 1
                continue
            factual = str(conflict.get("answer") or "").strip()
            # Explicit documented candidate rule: the paired clean row's
            # provided answer is the hallucinated/prior candidate. It is not
            # generated or inferred from the conflict text.
            hallucinated = str(clean[0].get("answer") or "").strip()
            if not factual or not hallucinated:
                exclusions["empty_candidate"] += 1
                continue
            if factual.casefold() == hallucinated.casefold():
                exclusions["identical_candidates"] += 1
                continue
            source_split = str(conflict["_split"])
            pair_id = f"mmmc-{source_split}-{image_id}-{conflict['_index']}"
            pairing_rows.append(
                {
                    "pair_id": pair_id,
                    "image_id": image_id,
                    "source_split": source_split,
                    "clean_index": clean[0]["_index"],
                    "clean_source_split": clean[0]["_split"],
                    "conflict_index": conflict["_index"],
                    "factual_answer": factual,
                    "hallucinated_answer": hallucinated,
                    "conflict_question": conflict.get("question"),
                    "clean_question": clean[0].get("question"),
                    "key_component": conflict.get("key_component"),
                }
            )

    rng = random.Random(seed)
    group_ids = sorted({row["image_id"] for row in pairing_rows})
    rng.shuffle(group_ids)
    prototype_n = min(int(config.get("prototype_examples", 256)), len(group_ids))
    remaining = max(0, len(group_ids) - prototype_n)
    validation_n = min(int(config.get("validation_examples", 512)), max(0, remaining - 500))
    split_by_group = {
        group_id: (
            "prototype"
            if index < prototype_n
            else "validation"
            if index < prototype_n + validation_n
            else "locked_test"
        )
        for index, group_id in enumerate(group_ids)
    }
    for row in pairing_rows:
        split = split_by_group[row["image_id"]]
        pairs.append(
            PairedExample(
                pair_id=row["pair_id"],
                group_id=row["image_id"],
                donor_image=f"hf://{DATASET_ID}/{row['source_split']}/{row['clean_index']}",
                recipient_image=f"hf://{DATASET_ID}/{row['source_split']}/{row['conflict_index']}",
                donor_prompt=str(row["clean_question"]),
                recipient_prompt=str(row["conflict_question"]),
                donor_answer=row["hallucinated_answer"],
                recipient_answer=row["factual_answer"],
                correct_answer=row["factual_answer"],
                bias_answer=row["hallucinated_answer"],
                metadata={
                    "contrast": "clean_to_object_conflict",
                    "candidate_rule": (
                        "factual=conflict-row provided answer; hallucinated=paired-clean-row provided answer"
                    ),
                    "clean_index": row["clean_index"],
                    "conflict_index": row["conflict_index"],
                    "key_component": row["key_component"],
                },
                split=split,
                generator_seed=seed,
                source_id=row["image_id"],
            )
        )
    assert_no_group_leakage(pairs)

    tokenization = _tokenization_audit(pairs, config) if not skip_tokenization else {
        "skipped": True,
        "reason": "--skip-tokenization",
    }
    split_pair_counts = Counter(pair.split for pair in pairs)
    pairs_by_image = Counter(row["image_id"] for row in pairing_rows)
    pairs_per_image_histogram = Counter(pairs_by_image.values())
    split_group_counts = {
        split: len({pair.group_id for pair in pairs if pair.split == split})
        for split in split_pair_counts
    }
    locked_ok = smoke or split_group_counts.get("locked_test", 0) >= 500
    audit = {
        "valid": bool(pairs) and locked_ok,
        "label": "instrumentation smoke test" if smoke else "dataset preparation",
        "dataset_id": DATASET_ID,
        "dataset_fingerprints": dataset_fingerprints,
        "license": "CC BY-SA 3.0",
        "split_sizes": split_sizes,
        "n_rows_scanned": len(scanned),
        "conflict_types": dict(sorted(conflict_types.items())),
        "n_unique_image_ids": len(groups),
        "n_object_conflict_pairs": len(pairs),
        "object_pairs_per_image_id_histogram": {
            str(count): frequency
            for count, frequency in sorted(pairs_per_image_histogram.items())
        },
        "pairing_success_rate": (
            len(pairs) / max(1, conflict_types.get("object", 0))
        ),
        "candidate_rule": (
            "factual candidate is the object-conflict row's provided answer; "
            "hallucinated candidate is the paired clean row's provided answer"
        ),
        "exclusions": dict(sorted(exclusions.items())),
        "split_pair_counts": dict(split_pair_counts),
        "split_group_counts": split_group_counts,
        "tokenization": tokenization,
        "duplicate_pair_ids": len(pairs) - len({pair.pair_id for pair in pairs}),
        "locked_test_minimum_met": locked_ok,
        "errors": [] if bool(pairs) and locked_ok else [
            "no unambiguous object pairs" if not pairs else "locked test has fewer than 500 image groups"
        ],
    }
    return audit, pairs


def _tokenization_audit(
    pairs: list[PairedExample], config: dict[str, Any]
) -> dict[str, Any]:
    from transformers import AutoProcessor

    model_id = str(config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct"))
    processor = AutoProcessor.from_pretrained(model_id)
    rows = []
    for pair in pairs:
        factual = processor.tokenizer(pair.correct_answer, add_special_tokens=False).input_ids
        hallucinated = processor.tokenizer(pair.bias_answer, add_special_tokens=False).input_ids
        rows.append((len(factual), len(hallucinated)))
    return {
        "model_id": model_id,
        "n_pairs": len(rows),
        "factual_single_token": sum(left == 1 for left, _ in rows),
        "hallucinated_single_token": sum(right == 1 for _, right in rows),
        "both_single_token": sum(left == right == 1 for left, right in rows),
        "factual_length_histogram": dict(Counter(left for left, _ in rows)),
        "hallucinated_length_histogram": dict(Counter(right for _, right in rows)),
    }


if __name__ == "__main__":
    main()
