from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets
from PIL import Image, ImageDraw

from vlm_eval.vlmbias_roi import (
    SUPPORTED_PAIR_TOPICS,
    binary_mask_image,
    build_original_index,
    make_difference_mask,
    match_original_row,
    overlay_mask,
)


DEFAULT_SLICE = Path("segments/vlm_bias_attention/data/vlmbias_400.jsonl")
DEFAULT_OUTPUT = Path("segments/vlm_bias_attention/data/vlmbias_roi_masks_v1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pair a local VLMBias slice with cached originals and make binary ROI masks."
    )
    parser.add_argument("--slice", type=Path, default=DEFAULT_SLICE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--max-pairs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--max-mask-fraction", type=float, default=0.55)
    parser.add_argument("--max-median-difference", type=float, default=8.0)
    parser.add_argument("--contact-sheet-examples", type=int, default=48)
    parser.add_argument(
        "--include-car-logos",
        action="store_true",
        help="Include car-logo pairs despite known residual registration artifacts.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {args.output}; pass --overwrite to reuse it")
    args.output.mkdir(parents=True, exist_ok=True)
    for directory in ("originals", "differences", "masks", "overlays"):
        (args.output / directory).mkdir(parents=True, exist_ok=True)

    local_rows = _read_jsonl(args.slice)
    cache_dir = args.dataset_cache or _find_cached_dataset()
    main_rows = _load_cached_split(
        cache_dir, "main", columns=("ID", "image_path", "topic", "sub_topic")
    )
    original_rows = _load_cached_split(
        cache_dir, "original", columns=("image", "ID", "image_path", "topic")
    )
    main_by_id = {str(row["ID"]): row for row in main_rows}
    original_index = build_original_index(original_rows)
    local_ids_by_upstream_path: dict[str, list[str]] = defaultdict(list)
    for local_row in local_rows:
        main_row = main_by_id.get(str(local_row["id"]))
        if main_row is not None:
            local_ids_by_upstream_path[str(main_row.get("image_path") or "")].append(
                str(local_row["id"])
            )
    pilot_topics = tuple(
        topic
        for topic in SUPPORTED_PAIR_TOPICS
        if topic != "Logos" or args.include_car_logos
    )

    candidates_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    early_rejections: list[dict[str, Any]] = []
    for local_row in local_rows:
        edited_row = main_by_id.get(str(local_row["id"]))
        if edited_row is None:
            early_rejections.append(_rejection(local_row, "missing_from_cached_main_split"))
            continue
        topic = str(edited_row.get("topic") or "")
        if topic == "Logos" and not args.include_car_logos:
            early_rejections.append(
                _rejection(
                    local_row,
                    "car_logo_pairs_excluded_due_to_registration_artifacts",
                    edited_row,
                )
            )
            continue
        if topic not in pilot_topics:
            early_rejections.append(
                _rejection(local_row, f"topic_not_supported_for_subtraction:{topic}", edited_row)
            )
            continue
        candidates_by_topic[topic].append(
            {"local": local_row, "edited": edited_row}
        )

    rng = random.Random(args.seed)
    candidates: list[dict[str, Any]] = []
    for topic in pilot_topics:
        unique = _unique_candidates(candidates_by_topic.get(topic, []))
        rng.shuffle(unique)
        candidates.extend(unique)
    candidates = _round_robin_by_topic(candidates)

    accepted: list[dict[str, Any]] = []
    rejected = list(early_rejections)
    not_attempted: list[dict[str, Any]] = []
    accepted_visuals: list[tuple[str, list[Image.Image]]] = []
    rejected_visuals: list[tuple[str, list[Image.Image]]] = []
    for candidate in candidates:
        local_row = candidate["local"]
        edited_row = candidate["edited"]
        if len(accepted) >= args.max_pairs:
            not_attempted.append(
                _rejection(local_row, "pilot_limit_reached", edited_row)
            )
            continue
        original_row, pairing_error = match_original_row(edited_row, original_index)
        if pairing_error is not None or original_row is None:
            rejected.append(_rejection(local_row, pairing_error or "unknown_pairing_error", edited_row))
            continue

        edited_path = args.slice.parent / str(local_row["image_path"])
        edited_image = Image.open(edited_path).convert("RGB")
        original_image = original_row["image"].convert("RGB")
        original_resized = original_image.resize(edited_image.size, Image.Resampling.LANCZOS)
        mask = make_difference_mask(
            original_image,
            edited_image,
            threshold=args.threshold,
            max_mask_fraction=args.max_mask_fraction,
            max_median_difference=args.max_median_difference,
        )
        safe_id = _safe_name(str(local_row["id"]))
        visuals = [
            original_resized,
            edited_image,
            Image.fromarray(mask.difference).convert("RGB"),
            binary_mask_image(mask.clean_mask).convert("RGB"),
            overlay_mask(edited_image, mask.clean_mask),
        ]
        if not mask.accepted:
            rejected.append(
                {
                    **_base_manifest_row(local_row, edited_row, original_row),
                    "status": "rejected",
                    "rejection_reasons": list(mask.rejection_reasons),
                    "mask_stats": mask.stats,
                }
            )
            if len(rejected_visuals) < args.contact_sheet_examples:
                rejected_visuals.append((safe_id, visuals))
            continue
        artifact_paths = {
            "original_path": f"originals/{safe_id}.png",
            "difference_path": f"differences/{safe_id}.png",
            "mask_path": f"masks/{safe_id}.png",
            "overlay_path": f"overlays/{safe_id}.png",
        }
        original_resized.save(args.output / artifact_paths["original_path"])
        Image.fromarray(mask.difference).save(args.output / artifact_paths["difference_path"])
        binary_mask_image(mask.clean_mask).save(args.output / artifact_paths["mask_path"])
        visuals[-1].save(args.output / artifact_paths["overlay_path"])
        accepted.append(
            {
                **_base_manifest_row(local_row, edited_row, original_row),
                "status": "accepted_automatic",
                "rejection_reasons": [],
                "mask_stats": mask.stats,
                "artifacts": artifact_paths,
            }
        )
        if len(accepted_visuals) < args.contact_sheet_examples:
            accepted_visuals.append((safe_id, visuals))

    covered_local_ids: set[str] = set()
    for row in accepted:
        local_ids = local_ids_by_upstream_path[str(row["upstream_edited_path"])]
        row["covered_local_ids"] = local_ids
        row["n_covered_local_rows"] = len(local_ids)
        covered_local_ids.update(local_ids)

    _write_jsonl(args.output / "accepted.jsonl", accepted)
    _write_jsonl(args.output / "rejected.jsonl", rejected)
    _write_jsonl(args.output / "not_attempted.jsonl", not_attempted)
    _write_contact_sheets(args.output, "accepted_contact_sheet", accepted_visuals)
    _write_contact_sheets(args.output, "rejected_contact_sheet", rejected_visuals)
    summary = {
        "valid": bool(accepted),
        "source_slice": str(args.slice),
        "source_dataset": "anvo25/vlms-are-biased",
        "cached_dataset_directory": str(cache_dir),
        "max_pairs": args.max_pairs,
        "seed": args.seed,
        "pilot_topics": list(pilot_topics),
        "car_logos_included": args.include_car_logos,
        "mask_parameters": {
            "difference": "maximum absolute RGB-channel difference after resizing and Gaussian-smoothing both images",
            "threshold": args.threshold,
            "comparison_gaussian_blur_radius_pixels": 2.0,
            "morphological_opening_radius_pixels": 1,
            "dilation_fraction_of_short_side": 0.006,
            "minimum_mask_fraction": 0.0005,
            "maximum_mask_fraction": args.max_mask_fraction,
            "maximum_median_difference": args.max_median_difference,
        },
        "n_local_rows": len(local_rows),
        "n_supported_rows": sum(len(rows) for rows in candidates_by_topic.values()),
        "n_unique_supported_images": len(candidates),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "n_early_rejected_rows": len(early_rejections),
        "n_mask_rejected_unique_images": len(rejected) - len(early_rejections),
        "n_not_attempted_due_to_limit": len(not_attempted),
        "n_local_rows_covered_by_accepted_masks": len(covered_local_ids),
        "accepted_by_topic": dict(sorted(Counter(row["topic"] for row in accepted).items())),
        "rejected_by_reason": dict(
            sorted(Counter(reason for row in rejected for reason in row["rejection_reasons"]).items())
        ),
        "requires_manual_review": True,
        "manual_review_field": "Change status in accepted.jsonl from accepted_automatic to accepted_manual after inspecting the overlay.",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _find_cached_dataset() -> Path:
    candidates = sorted(
        (Path.home() / ".cache/huggingface/datasets/anvo25___vlms-are-biased").glob(
            "default/*/*"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if (path / "vlms-are-biased-original.arrow").exists() and list(
            path.glob("vlms-are-biased-main*.arrow")
        ):
            return path
    raise SystemExit(
        "Cached VLMBias main/original Arrow files were not found. Run "
        "`uv run python scripts/make_vlmbias_slice.py --n 1` once to populate the cache, "
        "or pass --dataset-cache."
    )


def _load_cached_split(
    cache_dir: Path,
    split: str,
    *,
    columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    files = sorted(cache_dir.glob(f"vlms-are-biased-{split}*.arrow"))
    if not files:
        raise SystemExit(f"No cached {split} Arrow files found under {cache_dir}")
    datasets = [Dataset.from_file(str(path)) for path in files]
    dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    dataset = dataset.select_columns(list(columns))
    return [dict(row) for row in dataset]


def _unique_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        path = str(candidate["edited"].get("image_path") or "")
        if path and path not in unique:
            unique[path] = candidate
    return list(unique.values())


def _round_robin_by_topic(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_topic[str(candidate["edited"].get("topic") or "")].append(candidate)
    ordered: list[dict[str, Any]] = []
    topics = sorted(by_topic)
    while any(by_topic.values()):
        for topic in topics:
            if by_topic[topic]:
                ordered.append(by_topic[topic].pop())
    return ordered


def _base_manifest_row(
    local_row: dict[str, Any],
    edited_row: dict[str, Any],
    original_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": local_row["id"],
        "topic": local_row.get("topic"),
        "sub_topic": local_row.get("sub_topic"),
        "prompt": local_row.get("prompt"),
        "ground_truth": local_row.get("ground_truth"),
        "expected_bias": local_row.get("expected_bias"),
        "edited_path": str(local_row.get("image_path")),
        "upstream_edited_path": str(edited_row.get("image_path")),
        "upstream_original_id": str(original_row.get("ID")),
        "upstream_original_path": str(original_row.get("image_path")),
        "metadata": local_row.get("metadata"),
    }


def _rejection(
    local_row: dict[str, Any],
    reason: str,
    edited_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": local_row.get("id"),
        "topic": local_row.get("topic"),
        "sub_topic": local_row.get("sub_topic"),
        "edited_path": local_row.get("image_path"),
        "upstream_edited_path": edited_row.get("image_path") if edited_row else None,
        "status": "rejected",
        "rejection_reasons": [reason],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_contact_sheets(
    output: Path,
    stem: str,
    examples: list[tuple[str, list[Image.Image]]],
    *,
    page_size: int = 12,
) -> None:
    if not examples:
        Image.new("RGB", (800, 120), "white").save(output / f"{stem}_001.png")
        return
    for page_index, offset in enumerate(range(0, len(examples), page_size), start=1):
        _write_contact_sheet(
            output / f"{stem}_{page_index:03d}.png",
            examples[offset : offset + page_size],
        )


def _write_contact_sheet(path: Path, examples: list[tuple[str, list[Image.Image]]]) -> None:
    columns = 5
    tile = 220
    label_height = 42
    sheet = Image.new("RGB", (columns * tile, len(examples) * (tile + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    headers = ("original", "edited", "difference", "binary mask", "overlay")
    for row_index, (example_id, images) in enumerate(examples):
        top = row_index * (tile + label_height)
        for column, image in enumerate(images):
            thumbnail = image.convert("RGB").copy()
            thumbnail.thumbnail((tile, tile))
            left = column * tile + (tile - thumbnail.width) // 2
            sheet.paste(thumbnail, (left, top + (tile - thumbnail.height) // 2))
            draw.text((column * tile + 4, top + 3), headers[column], fill="black")
        draw.text((4, top + tile + 4), example_id[:100], fill="black")
    sheet.save(path)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)[
        :120
    ]


if __name__ == "__main__":
    main()
