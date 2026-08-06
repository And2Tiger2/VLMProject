#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

from vlm_eval.mechanistic_heads.config import (
    add_standard_run_arguments,
    effective_limit,
    load_json_config,
    prepare_output_directory,
)
from vlm_eval.mechanistic_heads.reproducibility import referenced_image_paths, seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.schema import PairedExample, assert_no_group_leakage, write_paired_jsonl
from vlm_eval.mechanistic_heads.splits import group_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Build grouped VLMBias signed-head contrasts.")
    add_standard_run_arguments(parser)
    args = parser.parse_args()
    config = load_json_config(args.config)
    output = args.output_dir / "vlmbias_signed_contrasts.jsonl"
    prepare_output_directory(
        args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
        known_outputs=(output.name,),
    )
    seed_everything(args.seed)
    pairs, audit = prepare_contrasts(
        config=config,
        output_dir=args.output_dir,
        seed=args.seed,
        limit=effective_limit(args),
        smoke=args.smoke,
    )
    write_paired_jsonl(output, pairs)
    audit_path = args.output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    referenced_images = referenced_image_paths(pairs)
    output_root = args.output_dir.resolve()
    derived_outputs = [
        path for path in referenced_images if path.resolve().is_relative_to(output_root)
    ]
    source_inputs = [
        path for path in referenced_images if not path.resolve().is_relative_to(output_root)
    ]
    write_run_manifest(
        args.output_dir,
        config={**config, "smoke": args.smoke},
        seeds={"split": args.seed},
        inputs=[args.config, Path(__file__), Path(config["vlmbias_dataset"]), Path(config["accepted_masks"]), *source_inputs],
        outputs=[output, audit_path, *derived_outputs],
        status="complete",
        repo_root=Path.cwd(),
    )
    print(json.dumps(audit, indent=2))


def prepare_contrasts(
    *,
    config: dict[str, Any],
    output_dir: Path,
    seed: int,
    limit: int | None,
    smoke: bool,
) -> tuple[list[PairedExample], dict[str, Any]]:
    dataset_path = Path(config["vlmbias_dataset"])
    dataset_rows = {str(row["id"]): row for row in _read_jsonl(dataset_path)}
    accepted_path = Path(config["accepted_masks"])
    accepted = _read_jsonl(accepted_path)
    if smoke:
        # Each accepted row emits semantic/context and optionally detail
        # contrasts. Two source rows therefore exercise all paths while
        # keeping the smoke output below eight paired examples.
        accepted = accepted[: min(int(limit or 8), 2)]
    elif limit is not None:
        accepted = accepted[:limit]
    split_rows = group_split(
        accepted,
        group_key=lambda row: str(row["group_id"]),
        fractions={"prototype": 0.25, "validation": 0.25, "locked_test": 0.5},
        seed=seed,
    )
    split_by_group = {
        str(row["group_id"]): split
        for split, rows in split_rows.items()
        for row in rows
    }
    if smoke:
        # A tiny hashed split can otherwise contain no prototype row, causing
        # the signed-head smoke to emit an empty table. Smoke is an
        # instrumentation exercise, not a scientific split estimate, so make
        # its first source group a deterministic prototype while preserving
        # group integrity.
        smoke_splits = ("prototype", "validation", "locked_test")
        smoke_groups = list(dict.fromkeys(str(row["group_id"]) for row in accepted))
        split_by_group = {
            group_id: smoke_splits[index % len(smoke_splits)]
            for index, group_id in enumerate(smoke_groups)
        }
    pairs: list[PairedExample] = []
    exclusions: list[dict[str, str]] = []
    mask_root = accepted_path.parent
    candidates_root = Path(config["candidate_root"])
    for accepted_row in accepted:
        example_id = str(accepted_row["id"])
        row = dataset_rows.get(example_id)
        if row is None:
            exclusions.append({"id": example_id, "reason": "missing VLMBias row"})
            continue
        group_id = str(accepted_row["group_id"])
        split = split_by_group[group_id]
        source_image = Path(row["image_path"])
        if not source_image.is_absolute():
            source_image = dataset_path.parent / source_image
        correct = str(row["ground_truth"])
        bias = str(row["expected_bias"])
        loaded_prompt = str(row["prompt"])
        neutral_prompt = (
            "Treat this as an unfamiliar synthetic image with no standard real-world convention. "
            "Use only the visible pixels. " + loaded_prompt
        )
        pairs.append(
            PairedExample(
                pair_id=f"semantic-{example_id}",
                group_id=group_id,
                donor_image=str(source_image.resolve()),
                recipient_image=str(source_image.resolve()),
                donor_prompt=neutral_prompt,
                recipient_prompt=loaded_prompt,
                donor_answer=correct,
                recipient_answer=bias,
                correct_answer=correct,
                bias_answer=bias,
                metadata={"contrast": "semantic_prior", "prompt_rule": "neutral synthetic-image prefix"},
                split=split,
                generator_seed=seed,
                source_id=group_id,
            )
        )
        tight_mask = mask_root / str(accepted_row["artifacts"]["tight_mask"])
        if tight_mask.is_file():
            image = Image.open(source_image).convert("RGB")
            mask = Image.open(tight_mask).convert("L").resize(image.size)
            mode = str(config.get("context_removal", "whiten"))
            if mode == "whiten":
                background = Image.new("RGB", image.size, "white")
            elif mode == "blur":
                background = image.filter(ImageFilter.GaussianBlur(radius=16))
            else:
                raise ValueError(f"unknown context removal mode: {mode}")
            contextless = Image.composite(image, background, mask)
            context_path = output_dir / "derived_images" / "context" / f"{example_id}.png"
            context_path.parent.mkdir(parents=True, exist_ok=True)
            contextless.save(context_path)
            pairs.append(
                PairedExample(
                    pair_id=f"context-{example_id}",
                    group_id=group_id,
                    donor_image=str(context_path.resolve()),
                    recipient_image=str(source_image.resolve()),
                    donor_prompt=loaded_prompt,
                    recipient_prompt=loaded_prompt,
                    donor_answer=correct,
                    recipient_answer=bias,
                    correct_answer=correct,
                    bias_answer=bias,
                    donor_mask=str(tight_mask.resolve()),
                    recipient_mask=str(tight_mask.resolve()),
                    metadata={"contrast": "context", "context_removal": mode},
                    split=split,
                    generator_seed=seed,
                    source_id=group_id,
                )
            )
        topic_dir = "game_boards" if row.get("topic") == "Game Boards" else "optical_illusions"
        original = candidates_root / topic_dir / "originals" / f"{example_id}.png"
        if original.is_file():
            original_image = Image.open(original).convert("RGB")
            counterfactual = Image.open(source_image).convert("RGB")
            if original_image.size != counterfactual.size:
                exclusions.append({"id": example_id, "reason": "detail images have unequal pixel dimensions"})
            else:
                pairs.append(
                    PairedExample(
                        pair_id=f"detail-{example_id}",
                        group_id=group_id,
                        donor_image=str(original.resolve()),
                        recipient_image=str(source_image.resolve()),
                        donor_prompt=loaded_prompt,
                        recipient_prompt=loaded_prompt,
                        donor_answer=bias,
                        recipient_answer=correct,
                        correct_answer=correct,
                        bias_answer=bias,
                        metadata={"contrast": "detail", "alignment": "identical pixel dimensions required"},
                        split=split,
                        generator_seed=seed,
                        source_id=group_id,
                    )
                )
        else:
            exclusions.append(
                {
                    "id": example_id,
                    "reason": "detail contrast unavailable: original factual image is not present",
                }
            )
    assert_no_group_leakage(pairs)
    counts = {
        contrast: sum(pair.metadata["contrast"] == contrast for pair in pairs)
        for contrast in ("semantic_prior", "context", "detail")
    }
    return pairs, {
        "valid": bool(pairs),
        "label": "instrumentation smoke test" if smoke else "exploratory transfer dataset preparation",
        "n_pairs": len(pairs),
        "counts_by_contrast": counts,
        "counts_by_split": {
            split: sum(pair.split == split for pair in pairs)
            for split in ("prototype", "validation", "locked_test")
        },
        "exclusions": exclusions,
        "detail_status": (
            "available"
            if counts["detail"]
            else "computationally pending: no external original factual images were available"
        ),
        "errors": [] if pairs else ["no contrast pairs produced"],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
