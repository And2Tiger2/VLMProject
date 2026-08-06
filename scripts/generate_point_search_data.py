#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

from PIL import ImageDraw

from vlm_eval.mechanistic_heads.config import (
    add_standard_run_arguments,
    effective_limit,
    load_json_config,
    prepare_output_directory,
)
from vlm_eval.mechanistic_heads.reproducibility import seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.schema import PairedExample, write_paired_jsonl
from vlm_eval.mechanistic_heads.synthetic import (
    SEARCH_COLORS,
    SEARCH_SHAPES,
    SEARCH_TEST_CONJUNCTIONS,
    SEARCH_TRAIN_CONJUNCTIONS,
    render_search_scene,
    render_waldo_like_scene,
    stable_seed,
    waldo_distractor_centers,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate color/shape search and synthetic Waldo-like datasets."
    )
    add_standard_run_arguments(parser)
    args = parser.parse_args()
    config = load_json_config(args.config)
    prepare_output_directory(
        args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
        known_outputs=("dataset_manifest.json", "point_search.jsonl", "waldo_like.jsonl"),
    )
    seed_everything(args.seed)
    result = generate_point_search_datasets(
        args.output_dir,
        config=config,
        seed=args.seed,
        smoke=args.smoke,
        limit=effective_limit(args),
        resume=args.resume,
    )
    manifest = args.output_dir / "dataset_manifest.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    generated_images = sorted(args.output_dir.rglob("*.png"))
    write_run_manifest(
        args.output_dir,
        config={**config, "smoke": args.smoke, "limit": effective_limit(args)},
        seeds={"generator": args.seed},
        # Reuse is safe only while both orchestration and renderer bytes match.
        inputs=[args.config, Path(__file__), Path(render_waldo_like_scene.__code__.co_filename)],
        outputs=[
            manifest,
            *(Path(path) for path in result["artifacts"]),
            *generated_images,
        ],
        status="complete",
        repo_root=Path.cwd(),
    )
    print(json.dumps(result, indent=2))


def generate_point_search_datasets(
    output_dir: Path,
    *,
    config: dict[str, Any],
    seed: int,
    smoke: bool,
    limit: int | None,
    resume: bool,
) -> dict[str, Any]:
    train_n = int(config.get("training_scenes", 2000))
    ood_per_condition = int(config.get("ood_scenes_per_condition", 50))
    waldo_n = int(config.get("waldo_like_scenes", 1000))
    if smoke:
        # Eight point-search rows total: two train plus one for each of the six
        # OOD target counts. Waldo and each causal-pair file remain <= 4 rows.
        train_n = min(train_n, 2)
        ood_per_condition = min(ood_per_condition, 1)
        waldo_n = min(waldo_n, 4)
    if limit is not None:
        train_n = min(train_n, limit)
        ood_per_condition = min(ood_per_condition, limit)
        waldo_n = min(waldo_n, limit)

    train_conjunctions = list(SEARCH_TRAIN_CONJUNCTIONS)
    test_conjunctions = list(SEARCH_TEST_CONJUNCTIONS)
    if set(train_conjunctions) & set(test_conjunctions):
        raise RuntimeError("target conjunction leakage")
    if set(SEARCH_COLORS) != {"red", "green", "blue", "purple", "gray", "black"}:
        raise RuntimeError("point-search colors drifted from the paper specification")
    if set(SEARCH_SHAPES) != {"L", "T", "H", "E", "F", "Γ"}:
        raise RuntimeError("point-search shapes drifted from the paper specification")

    point_rows: list[dict[str, Any]] = []
    for index in range(train_n):
        target_color, target_shape = train_conjunctions[index % len(train_conjunctions)]
        target_count = 1 if index % 2 == 0 else 0
        point_rows.append(
            _write_search_scene(
                output_dir,
                seed=seed,
                scene_id=f"search-train-{index:05d}",
                split="train",
                target_color=target_color,
                target_shape=target_shape,
                target_count=target_count,
                resume=resume,
            )
        )
    for target_count in (1, 2, 10, 30, 40, 50):
        for index in range(ood_per_condition):
            target_color, target_shape = test_conjunctions[
                (target_count + index) % len(test_conjunctions)
            ]
            point_rows.append(
                _write_search_scene(
                    output_dir,
                    seed=seed,
                    scene_id=f"search-ood-n{target_count}-{index:04d}",
                    split=f"ood_target_count_{target_count}",
                    target_color=target_color,
                    target_shape=target_shape,
                    target_count=target_count,
                    resume=resume,
                )
            )
    point_path = output_dir / "point_search.jsonl"
    _write_jsonl(point_path, point_rows)

    waldo_rows: list[dict[str, Any]] = []
    for index in range(waldo_n):
        scene_id = f"waldo-like-{index:05d}"
        template_index = index // 4
        target_present = index % 2 == 0
        scales = (0.75, 1.0, 1.25)
        backgrounds = ((225, 235, 220), (235, 225, 220), (218, 228, 240))
        zooms = (0.9, 1.0, 1.1)
        scale = scales[stable_seed(seed, scene_id, "scale") % len(scales)]
        background = backgrounds[
            stable_seed(seed, scene_id, "background") % len(backgrounds)
        ]
        zoom = zooms[stable_seed(seed, scene_id, "zoom") % len(zooms)]
        clutter = 24 + stable_seed(seed, scene_id, "clutter") % 12
        similarity = 1 + stable_seed(seed, scene_id, "similarity") % 3
        occluded = stable_seed(seed, scene_id, "occlusion") % 7 == 0
        scene = render_waldo_like_scene(
            seed=seed,
            scene_id=scene_id,
            target_present=target_present,
            clutter=clutter,
            similarity=similarity,
            target_cell=stable_seed(seed, scene_id) % 100,
            occluded=occluded,
            target_scale=scale,
            background=background,
            scene_zoom=zoom,
        )
        image_path = output_dir / "waldo_like" / "images" / f"{scene_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not resume or not image_path.exists():
            scene.image.save(image_path)
        visible_grid_path = output_dir / "waldo_like" / "visible_grid" / f"{scene_id}.png"
        if not resume or not visible_grid_path.exists():
            visible_grid_path.parent.mkdir(parents=True, exist_ok=True)
            gridded = scene.image.copy()
            draw = ImageDraw.Draw(gridded)
            for coordinate in range(0, 401, 40):
                draw.line((coordinate, 0, coordinate, 399), fill=(40, 40, 40), width=1)
                draw.line((0, coordinate, 399, coordinate), fill=(40, 40, 40), width=1)
            for cell in range(100):
                draw.text(((cell % 10) * 40 + 2, (cell // 10) * 40 + 2), f"{cell:02d}", fill=(20, 20, 20))
            gridded.save(visible_grid_path)
        mask_paths = {}
        for name, mask in scene.masks.items():
            mask_path = output_dir / "waldo_like" / "masks" / f"{scene_id}-{name}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not resume or not mask_path.exists():
                mask.save(mask_path)
            mask_paths[name] = str(mask_path.resolve())
        target_objects = [row for row in scene.objects if row["class"] == "target"]
        target_cell = target_objects[0]["cell"] if target_objects else None
        target_center = target_objects[0]["center"] if target_objects else None
        decoy_cells = [obj["cell"] for obj in scene.objects if obj["class"].startswith("distractor")][:3]
        candidate_cells = ([target_cell] if target_cell is not None else []) + decoy_cells
        candidate_rng = random.Random(stable_seed(seed, scene_id, "candidate-order"))
        candidate_rng.shuffle(candidate_cells)
        target_candidate = (
            candidate_cells.index(target_cell) if target_cell is not None else None
        )
        waldo_rows.append(
            {
                "id": scene_id,
                "group_id": f"waldo-template-{template_index:05d}",
                "image_path": str(image_path.resolve()),
                "target_present": target_present,
                "target_cell": target_cell,
                "target_center": target_center,
                "masks": mask_paths,
                "objects": scene.objects,
                "tasks": {
                    "grid_localization": "absent" if target_cell is None else f"cell={target_cell:02d}",
                    "normalized_point": (
                        "absent"
                        if target_center is None
                        else f"point=({target_center[0] / 399:.3f},{target_center[1] / 399:.3f})"
                    ),
                    "presence": "present" if target_present else "absent",
                    "visible_grid_ocr": "absent" if target_cell is None else f"cell={target_cell:02d}",
                    "four_candidate_selection": "absent" if target_candidate is None else f"candidate={target_candidate}",
                },
                "prompts": [
                    "Which 10x10 cell contains the four-feature target? Answer cell=NN.",
                    "Locate the character with stripes, round glasses, a pointed hat, and blue lower clothing. Answer cell=NN.",
                    "Is the four-feature target present? Answer present or absent.",
                ],
                "split": "locked_test" if template_index % 5 == 0 else "train",
                "generator_seed": seed,
                "source_id": f"waldo-template-{template_index:05d}",
                "metadata": {
                    "non_copyright_synthetic": True,
                    "feature_definition": [
                        "striped torso",
                        "round glasses",
                        "pointed hat",
                        "blue lower body",
                    ],
                    "visible_grid_image": str(visible_grid_path.resolve()),
                    "four_candidate_cells": candidate_cells,
                    "target_scale": scale,
                    "background_rgb": list(background),
                    "scene_zoom": zoom,
                    "zoom_condition": f"{zoom:.1f}x full-scene zoom",
                    "clutter": clutter,
                    "distractor_similarity": similarity,
                    "occluded": occluded,
                    "prompt_wording_variant": stable_seed(seed, scene_id, "prompt") % 3,
                },
            }
        )
    waldo_path = output_dir / "waldo_like.jsonl"
    owners: dict[str, str] = {}
    for row in waldo_rows:
        previous = owners.setdefault(str(row["group_id"]), str(row["split"]))
        if previous != row["split"]:
            raise RuntimeError(f"Waldo-like template leakage: {row['group_id']}")
    _write_jsonl(waldo_path, waldo_rows)
    search_pairs, verification_pairs, distractor_pairs = _write_waldo_pairs(
        output_dir, seed=seed, n_groups=max(1, waldo_n // 2), resume=resume
    )
    search_pair_path = output_dir / "search_pairs.jsonl"
    verification_pair_path = output_dir / "verification_pairs.jsonl"
    distractor_pair_path = output_dir / "distractor_pairs.jsonl"
    write_paired_jsonl(search_pair_path, search_pairs)
    write_paired_jsonl(verification_pair_path, verification_pairs)
    write_paired_jsonl(distractor_pair_path, distractor_pairs)
    return {
        "valid": True,
        "label": "instrumentation smoke test" if smoke else "dataset preparation",
        "seed": seed,
        "counts": {
            "point_search_train": train_n,
            "point_search_ood": len(point_rows) - train_n,
            "waldo_like": waldo_n,
        },
        "conjunction_split": {
            "train": [f"{color}-{shape}" for color, shape in train_conjunctions],
            "test": [f"{color}-{shape}" for color, shape in test_conjunctions],
        },
        "point_format": "points=[(037,064),...]; answer=N",
        "waldo_split_group_counts": {
            split: len({row["group_id"] for row in waldo_rows if row["split"] == split})
            for split in sorted({row["split"] for row in waldo_rows})
        },
        "waldo_pair_split_group_counts": {
            family: {
                split: len({row.group_id for row in pairs if row.split == split})
                for split in sorted({row.split for row in pairs})
            }
            for family, pairs in (
                ("search", search_pairs),
                ("verification", verification_pairs),
                ("distractor", distractor_pairs),
            )
        },
        "waldo_pair_masks_complete": all(
            pair.donor_mask and pair.recipient_mask
            for pairs in (search_pairs, verification_pairs, distractor_pairs)
            for pair in pairs
        ),
        "waldo_relocation_distractors_matched": True,
        "four_candidate_target_indices": sorted(
            {
                int(row["tasks"]["four_candidate_selection"].split("=")[1])
                for row in waldo_rows
                if row["target_present"]
            }
        ),
        "artifacts": [
            str(point_path),
            str(waldo_path),
            str(search_pair_path),
            str(verification_pair_path),
            str(distractor_pair_path),
        ],
        "errors": [],
    }


def _write_waldo_pairs(
    output_dir: Path, *, seed: int, n_groups: int, resume: bool
) -> tuple[list[PairedExample], list[PairedExample], list[PairedExample]]:
    search_pairs: list[PairedExample] = []
    verification_pairs: list[PairedExample] = []
    distractor_pairs: list[PairedExample] = []
    for index in range(n_groups):
        group_id = f"waldo-pair-{index:05d}"
        cell_a = stable_seed(seed, group_id, "a") % 100
        cell_b = stable_seed(seed, group_id, "b") % 100
        if cell_b == cell_a:
            cell_b = (cell_a + 37) % 100
        distractor_centers = waldo_distractor_centers(
            seed=seed,
            scene_id=group_id,
            clutter=24,
            forbidden_cells=[cell_a, cell_b],
        )
        scenes = {
            "location_a": render_waldo_like_scene(seed=seed, scene_id=group_id, target_present=True, target_cell=cell_a, similarity=2, distractor_centers=distractor_centers),
            "location_b": render_waldo_like_scene(seed=seed, scene_id=group_id, target_present=True, target_cell=cell_b, similarity=2, distractor_centers=distractor_centers),
            "true": render_waldo_like_scene(seed=seed, scene_id=group_id, target_present=True, target_cell=cell_a, similarity=3, distractor_centers=distractor_centers),
            "impostor": render_waldo_like_scene(seed=seed, scene_id=group_id, target_present=False, target_cell=cell_a, similarity=3, distractor_centers=distractor_centers),
            "low_decoy": render_waldo_like_scene(seed=seed, scene_id=group_id, target_present=True, target_cell=cell_a, similarity=1, distractor_centers=distractor_centers),
            # Matched high-decoy scene: only object 1 changes from a weak
            # distractor into an incorrect-binding four-feature impostor.
            "high_decoy": render_waldo_like_scene(seed=seed, scene_id=group_id, target_present=True, target_cell=cell_a, similarity=1, similarity_overrides={1: 3}, distractor_centers=distractor_centers),
        }
        paths: dict[str, Path] = {}
        mask_paths: dict[str, dict[str, Path]] = {}
        for name, scene in scenes.items():
            image_path = output_dir / "waldo_pairs" / "images" / f"{group_id}-{name}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not resume or not image_path.exists():
                scene.image.save(image_path)
            paths[name] = image_path.resolve()
            mask_paths[name] = {}
            for mask_name, mask in scene.masks.items():
                mask_path = output_dir / "waldo_pairs" / "masks" / f"{group_id}-{name}-{mask_name}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                if not resume or not mask_path.exists():
                    mask.save(mask_path)
                mask_paths[name][mask_name] = mask_path.resolve()
        split_bucket = index % 10
        split = (
            "prototype"
            if split_bucket < 6
            else "validation"
            if split_bucket < 8
            else "locked_test"
        )
        strong_decoy_cell = next(
            int(obj["cell"])
            for obj in scenes["high_decoy"].objects
            if obj["class"] == "distractor-incorrect-binding"
        )
        common = {
            "group_id": group_id,
            "metadata": {
                "template_id": group_id,
                "synthetic": True,
                "matched_distractor_centers": True,
            },
            "split": split,
            "generator_seed": seed,
            "source_id": group_id,
        }
        search_pairs.append(
            PairedExample(
                pair_id=f"search-{group_id}",
                donor_image=str(paths["location_b"]),
                recipient_image=str(paths["location_a"]),
                donor_prompt="Which 10x10 cell contains the four-feature target? Answer cell=NN.",
                recipient_prompt="Which 10x10 cell contains the four-feature target? Answer cell=NN.",
                donor_answer=f"cell={cell_b:02d}",
                recipient_answer=f"cell={cell_a:02d}",
                donor_mask=str(mask_paths["location_b"]["target"]),
                recipient_mask=str(mask_paths["location_a"]["target"]),
                **common,
            )
        )
        verification_pairs.append(
            PairedExample(
                pair_id=f"verification-{group_id}",
                donor_image=str(paths["true"]),
                recipient_image=str(paths["impostor"]),
                donor_prompt="Is the four-feature target present? Answer present or absent.",
                recipient_prompt="Is the four-feature target present? Answer present or absent.",
                donor_answer="present",
                recipient_answer="absent",
                donor_mask=str(mask_paths["true"]["target"]),
                recipient_mask=str(mask_paths["impostor"]["object_000"]),
                **common,
            )
        )
        distractor_pairs.append(
            PairedExample(
                pair_id=f"distractor-{group_id}",
                donor_image=str(paths["low_decoy"]),
                recipient_image=str(paths["high_decoy"]),
                donor_prompt="Which 10x10 cell contains the four-feature target? Answer cell=NN.",
                recipient_prompt="Which 10x10 cell contains the four-feature target? Answer cell=NN.",
                donor_answer=f"cell={cell_a:02d}",
                recipient_answer=f"cell={strong_decoy_cell:02d}",
                donor_mask=str(mask_paths["low_decoy"]["target"]),
                recipient_mask=str(mask_paths["high_decoy"]["object_001"]),
                metadata={
                    **common["metadata"],
                    "target_cell": cell_a,
                    "strong_decoy_cell": strong_decoy_cell,
                },
                group_id=common["group_id"],
                split=common["split"],
                generator_seed=common["generator_seed"],
                source_id=common["source_id"],
            )
        )
    return search_pairs, verification_pairs, distractor_pairs


def _write_search_scene(
    output_dir: Path,
    *,
    seed: int,
    scene_id: str,
    split: str,
    target_color: str,
    target_shape: str,
    target_count: int,
    resume: bool,
) -> dict[str, Any]:
    scene = render_search_scene(
        seed=seed,
        scene_id=scene_id,
        target_color=target_color,
        target_shape=target_shape,
        target_count=target_count,
    )
    image_path = output_dir / "point_search" / "images" / f"{scene_id}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume or not image_path.exists():
        scene.image.save(image_path)
    mask_paths = {}
    for name, mask in scene.masks.items():
        mask_path = output_dir / "point_search" / "masks" / f"{scene_id}-{name}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not resume or not mask_path.exists():
            mask.save(mask_path)
        mask_paths[name] = str(mask_path.resolve())
    targets = [obj for obj in scene.objects if obj["class"] == "target"]
    points = [tuple(obj["center"]) for obj in targets]
    point_answer = (
        "points=[]; answer=0"
        if not points
        else "points=["
        + ",".join(f"({x:03d},{y:03d})" for x, y in points)
        + f"]; answer={len(points)}"
    )
    # Shuffled Point-Answer must destroy image/coordinate binding even when a
    # training scene contains only one target; permuting a singleton is a no-op.
    # Use an equal-size deterministic sample of distractor centers instead.
    distractor_points = [tuple(obj["center"]) for obj in scene.objects if obj["class"] != "target"]
    shuffle_rng = random.Random(stable_seed(seed, "shuffle", scene_id))
    if len(distractor_points) >= len(points):
        shuffled = shuffle_rng.sample(distractor_points, len(points))
    else:
        shuffled = [((x + 37) % 224, (y + 53) % 224) for x, y in points]
    shuffled_answer = (
        "points=[]; answer=0"
        if not shuffled
        else "points=["
        + ",".join(f"({x:03d},{y:03d})" for x, y in shuffled)
        + f"]; answer={len(shuffled)}"
    )
    prompt = (
        f"Find every {target_color} {target_shape} in the image and report its center."
    )
    return {
        "id": scene_id,
        "group_id": scene_id,
        "image_path": str(image_path.resolve()),
        "prompt": prompt,
        "target": {"color": target_color, "shape": target_shape},
        "target_count": target_count,
        "masks": mask_paths,
        "objects": scene.objects,
        "answers": {
            "base": str(target_count),
            "direct": str(target_count),
            "direct_length_matched": str(target_count),
            "point": point_answer,
            "shuffled_point": shuffled_answer,
        },
        "split": split,
        "generator_seed": seed,
        "source_id": scene_id,
        "direct_length_matching": "materialized against the loaded Qwen tokenizer at train/evaluation time; non-spatial text only",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
