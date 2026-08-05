#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

from PIL import Image, ImageDraw

from vlm_eval.mechanistic_heads.config import (
    add_standard_run_arguments,
    effective_limit,
    load_json_config,
    prepare_output_directory,
)
from vlm_eval.mechanistic_heads.reproducibility import seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.schema import PairedExample, write_paired_jsonl
from vlm_eval.mechanistic_heads.synthetic import (
    SYNDOT_PROMPT,
    fixed_eight_scene,
    render_syndot,
    render_syndot_mask,
    stable_seed,
    syndot_positions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic counting datasets.")
    add_standard_run_arguments(parser)
    args = parser.parse_args()
    config = load_json_config(args.config)
    prepare_output_directory(
        args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
        known_outputs=("dataset_manifest.json", "syndot.jsonl", "mechanistic_pairs.jsonl", "constant_complexity_pairs.jsonl"),
    )
    seed_everything(args.seed)
    result = generate_counting_datasets(
        args.output_dir,
        config=config,
        seed=args.seed,
        smoke=args.smoke,
        limit=effective_limit(args),
        resume=args.resume,
    )
    manifest_path = args.output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_run_manifest(
        args.output_dir,
        config={**config, "smoke": args.smoke, "limit": effective_limit(args)},
        seeds={"generator": args.seed},
        inputs=[args.config],
        outputs=[manifest_path, *(Path(path) for path in result["artifacts"])],
        status="complete",
        repo_root=Path.cwd(),
    )
    print(json.dumps(result, indent=2))


def generate_counting_datasets(
    output_dir: Path,
    *,
    config: dict[str, Any],
    seed: int,
    smoke: bool,
    limit: int | None,
    resume: bool,
) -> dict[str, Any]:
    train_n = int(config.get("syndot_train", 4000))
    test_n = int(config.get("syndot_test", 2000))
    pair_n = int(config.get("mechanistic_pairs", 100))
    constant_pairs = int(config.get("constant_complexity_pairs", 100))
    if smoke:
        train_n = min(train_n, 4)
        test_n = min(test_n, 4)
        pair_n = min(pair_n, 4)
        constant_pairs = min(constant_pairs, 4)
    if limit is not None:
        train_n = min(train_n, limit)
        test_n = min(test_n, limit)
        pair_n = min(pair_n, limit)
        constant_pairs = min(constant_pairs, limit)

    syndot_path = output_dir / "syndot.jsonl"
    syndot_rows: list[dict[str, Any]] = []
    for split, count in (("train", train_n), ("test", test_n)):
        for index in range(count):
            example_id = f"syndot-{split}-{index:05d}"
            answer = 1 + (stable_seed(seed, split, index) % 10)
            positions = syndot_positions(seed, example_id)
            image_path = output_dir / "images" / split / f"{example_id}.png"
            mask_path = output_dir / "masks" / split / f"{example_id}.png"
            if not resume or not image_path.exists():
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                render_syndot(answer, positions).save(image_path)
                render_syndot_mask(answer, positions).save(mask_path)
            syndot_rows.append(
                {
                    "id": example_id,
                    "group_id": example_id,
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "prompt": SYNDOT_PROMPT,
                    "ground_truth": str(answer),
                    "count": answer,
                    "split": split,
                    "generator_seed": seed,
                    "source_id": example_id,
                    "render_spec": {
                        "canvas": [336, 336],
                        "placement_grid": [28, 28],
                        "circle_radius": 4,
                    },
                }
            )
    _write_jsonl(syndot_path, syndot_rows)

    pairs: list[PairedExample] = []
    for index in range(pair_n):
        pair_id = f"syndot-pair-{index:04d}"
        rng = random.Random(stable_seed(seed, "mechanistic-pair", index))
        donor_count, recipient_count = rng.sample(range(1, 11), 2)
        positions = syndot_positions(seed, pair_id)
        pair_paths = {}
        for role, answer in (("donor", donor_count), ("recipient", recipient_count)):
            image_path = output_dir / "mechanistic" / "images" / f"{pair_id}-{role}.png"
            mask_path = output_dir / "mechanistic" / "masks" / f"{pair_id}-{role}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not resume or not image_path.exists():
                render_syndot(answer, positions).save(image_path)
                render_syndot_mask(answer, positions).save(mask_path)
            pair_paths[role] = (image_path, mask_path)
        pairs.append(
            PairedExample(
                pair_id=pair_id,
                group_id=pair_id,
                donor_image=str(pair_paths["donor"][0].resolve()),
                recipient_image=str(pair_paths["recipient"][0].resolve()),
                donor_prompt=SYNDOT_PROMPT,
                recipient_prompt=SYNDOT_PROMPT,
                donor_answer=str(donor_count),
                recipient_answer=str(recipient_count),
                donor_mask=str(pair_paths["donor"][1].resolve()),
                recipient_mask=str(pair_paths["recipient"][1].resolve()),
                metadata={"pair_type": "controlled-count", "positions_shared": True},
                split="mechanistic",
                generator_seed=seed,
                source_id=pair_id,
            )
        )
    pair_path = output_dir / "mechanistic_pairs.jsonl"
    write_paired_jsonl(pair_path, pairs)

    constant_rows: list[dict[str, Any]] = []
    for index in range(constant_pairs):
        group_id = f"constant-eight-{index:04d}"
        split = "train" if index % 5 else "locked_test"
        for variant in ("color", "shape"):
            for position_variant in ("standard", "target_relocation"):
                for role, red_count in (("recipient", 4), ("donor", 5)):
                    scene = fixed_eight_scene(
                        seed=seed,
                        scene_id=f"{group_id}-{variant}",
                        red_count=red_count,
                        variant=variant,
                        relocate=position_variant == "target_relocation",
                    )
                    stem = f"{group_id}-{variant}-{position_variant}-{role}"
                    image_path = output_dir / "constant_complexity" / "images" / f"{stem}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    if not resume or not image_path.exists():
                        scene.image.save(image_path)
                    sham_image_path = output_dir / "constant_complexity" / "sham_images" / f"{stem}.png"
                    sham_mask_path = output_dir / "constant_complexity" / "masks" / f"{stem}-sham.png"
                    if not resume or not sham_image_path.exists():
                        sham_image_path.parent.mkdir(parents=True, exist_ok=True)
                        sham_mask_path.parent.mkdir(parents=True, exist_ok=True)
                        sham = scene.image.copy()
                        ImageDraw.Draw(sham).rectangle((2, 2, 4, 4), fill=(245, 245, 245))
                        sham.save(sham_image_path)
                        sham_mask = Image.new("L", sham.size, 0)
                        ImageDraw.Draw(sham_mask).rectangle((2, 2, 4, 4), fill=255)
                        sham_mask.save(sham_mask_path)
                    mask_paths = {}
                    for name, mask in scene.masks.items():
                        mask_path = output_dir / "constant_complexity" / "masks" / f"{stem}-{name}.png"
                        mask_path.parent.mkdir(parents=True, exist_ok=True)
                        if not resume or not mask_path.exists():
                            mask.save(mask_path)
                        mask_paths[name] = str(mask_path.resolve())
                    constant_rows.append(
                        {
                        "id": stem,
                        "group_id": group_id,
                        "image_path": str(image_path.resolve()),
                        "prompt": (
                            "How many red objects are in the image? Answer with the number only."
                            if variant == "color"
                            else "How many circles are in the image? Answer with the number only."
                        ),
                        "ground_truth": str(red_count),
                        "total_objects": 8,
                        "variant": variant,
                        "position_variant": position_variant,
                        "role": role,
                        "answer_codes": {"4": "dax", "5": "wug"},
                        "answer_code_prompt": (
                            "Use the randomized codebook 4=dax and 5=wug. "
                            + ("How many red objects are in the image? Answer with the code only." if variant == "color" else "How many circles are in the image? Answer with the code only.")
                        ),
                        "sham_image_path": str(sham_image_path.resolve()),
                        "sham_mask_path": str(sham_mask_path.resolve()),
                        "masks": mask_paths,
                        "objects": scene.objects,
                        "split": split,
                        "generator_seed": seed,
                        "source_id": group_id,
                        }
                    )
    constant_path = output_dir / "constant_complexity.jsonl"
    _write_jsonl(constant_path, constant_rows)
    constant_pairs_out: list[PairedExample] = []
    by_condition: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in constant_rows:
        by_condition.setdefault((row["group_id"], row["variant"], row["position_variant"]), {})[row["role"]] = row
    for (group_id, variant, position_variant), roles in sorted(by_condition.items()):
        donor, recipient = roles["donor"], roles["recipient"]
        base = dict(group_id=group_id, donor_image=donor["image_path"], recipient_image=recipient["image_path"], donor_mask=donor["masks"]["changed_pixel"], recipient_mask=recipient["masks"]["changed_pixel"], split=donor["split"], generator_seed=seed, source_id=group_id)
        constant_pairs_out.append(PairedExample(pair_id=f"constant-{group_id}-{variant}-{position_variant}", donor_prompt=donor["prompt"], recipient_prompt=recipient["prompt"], donor_answer="5", recipient_answer="4", metadata={"pair_type": "constant-complexity", "variant": variant, "position_variant": position_variant, "total_objects": 8, "sham_images": [donor["sham_image_path"], recipient["sham_image_path"]]}, **base))
        constant_pairs_out.append(PairedExample(pair_id=f"constant-code-{group_id}-{variant}-{position_variant}", donor_prompt=donor["answer_code_prompt"], recipient_prompt=recipient["answer_code_prompt"], donor_answer="wug", recipient_answer="dax", metadata={"pair_type": "randomized-answer-code", "variant": variant, "position_variant": position_variant, "codebook": {"4": "dax", "5": "wug"}}, **base))
    constant_pair_path = output_dir / "constant_complexity_pairs.jsonl"
    write_paired_jsonl(constant_pair_path, constant_pairs_out)
    return {
        "valid": True,
        "label": "instrumentation smoke test" if smoke else "dataset preparation",
        "seed": seed,
        "counts": {
            "syndot_train": train_n,
            "syndot_test": test_n,
            "mechanistic_pairs": pair_n,
            "constant_complexity_rows": len(constant_rows),
        },
        "artifacts": [str(syndot_path), str(pair_path), str(constant_path), str(constant_pair_path)],
        "errors": [],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
