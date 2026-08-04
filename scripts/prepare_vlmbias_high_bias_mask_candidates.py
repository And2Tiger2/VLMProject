from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from vlm_eval.vlmbias_roi import binary_mask_image, dilate_binary_mask, overlay_mask


DEFAULT_SLICE = Path("segments/vlm_bias_attention/data/vlmbias_400.jsonl")
DEFAULT_OUTPUT = Path(
    "segments/vlm_bias_attention/data/vlmbias_high_bias_mask_candidates_v1"
)
TARGET_TOPICS = ("Game Boards", "Optical Illusion")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create manual-review mask candidates for VLMBias Game Boards and "
            "Optical Illusions. These outputs are candidates, not accepted masks."
        )
    )
    parser.add_argument("--slice", type=Path, default=DEFAULT_SLICE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--page-size", type=int, default=10)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {args.output}. Move it aside before regenerating."
        )
    for topic_dir in ("game_boards", "optical_illusions"):
        for artifact_dir in ("images", "masks_tight", "overlays_tight", "masks_broad", "overlays_broad"):
            (args.output / topic_dir / artifact_dir).mkdir(parents=True, exist_ok=True)

    rows = [row for row in _read_jsonl(args.slice) if row.get("topic") in TARGET_TOPICS]
    manifest: list[dict[str, Any]] = []
    sheets: dict[str, list[tuple[str, list[Image.Image]]]] = {
        "game_boards": [],
        "optical_illusions": [],
    }
    for row in rows:
        source = args.slice.parent / str(row["image_path"])
        image = Image.open(source).convert("RGB")
        safe_id = _safe_name(str(row["id"]))
        if row["topic"] == "Game Boards":
            topic_dir = "game_boards"
            tight, tight_details = _game_board_changed_band(image, row)
            broad = _structure_mask(image)
            method_names = ("metadata_changed_band", "grid_structure")
        else:
            topic_dir = "optical_illusions"
            tight = _foreground_mask(image)
            broad = dilate_binary_mask(tight, max(2, round(min(image.size) * 0.012)))
            tight_details = {
                "background_estimation": "median RGB over the outer image border",
                "foreground_threshold_rgb_distance": 22,
            }
            method_names = ("illusion_geometry_tight", "illusion_geometry_dilated")

        artifacts = {
            "image": f"{topic_dir}/images/{safe_id}.png",
            "tight_mask": f"{topic_dir}/masks_tight/{safe_id}.png",
            "tight_overlay": f"{topic_dir}/overlays_tight/{safe_id}.png",
            "broad_mask": f"{topic_dir}/masks_broad/{safe_id}.png",
            "broad_overlay": f"{topic_dir}/overlays_broad/{safe_id}.png",
        }
        image.save(args.output / artifacts["image"])
        binary_mask_image(tight).save(args.output / artifacts["tight_mask"])
        tight_overlay = overlay_mask(image, tight)
        tight_overlay.save(args.output / artifacts["tight_overlay"])
        binary_mask_image(broad).save(args.output / artifacts["broad_mask"])
        broad_overlay = overlay_mask(image, broad)
        broad_overlay.save(args.output / artifacts["broad_overlay"])

        manifest.append(
            {
                "id": row["id"],
                "topic": row["topic"],
                "sub_topic": row.get("sub_topic"),
                "prompt": row.get("prompt"),
                "ground_truth": row.get("ground_truth"),
                "expected_bias": row.get("expected_bias"),
                "source_image": str(source),
                "status": "candidate_requires_manual_review",
                "methods": {
                    "tight": method_names[0],
                    "broad": method_names[1],
                },
                "mask_fraction": {
                    "tight": float(tight.mean()),
                    "broad": float(broad.mean()),
                },
                "method_details": tight_details,
                "metadata": row.get("metadata", {}),
                "artifacts": artifacts,
            }
        )
        sheets[topic_dir].append(
            (
                safe_id,
                [
                    image,
                    binary_mask_image(tight).convert("RGB"),
                    tight_overlay,
                    binary_mask_image(broad).convert("RGB"),
                    broad_overlay,
                ],
            )
        )

    _write_jsonl(args.output / "manifest.jsonl", manifest)
    for topic_dir, examples in sheets.items():
        _write_contact_sheets(
            args.output,
            f"{topic_dir}_contact_sheet",
            examples,
            page_size=args.page_size,
        )

    by_topic: dict[str, dict[str, Any]] = {}
    for topic in TARGET_TOPICS:
        topic_rows = [row for row in manifest if row["topic"] == topic]
        by_topic[topic] = {
            "n": len(topic_rows),
            "tight_mask_fraction_range": _range(row["mask_fraction"]["tight"] for row in topic_rows),
            "broad_mask_fraction_range": _range(row["mask_fraction"]["broad"] for row in topic_rows),
        }
    summary = {
        "valid": len(manifest) == 114,
        "source_slice": str(args.slice),
        "output": str(args.output),
        "n_candidates": len(manifest),
        "by_topic": by_topic,
        "status": "candidate_requires_manual_review",
        "warning": (
            "These masks have not been accepted for experiments. Game-board changed-band masks "
            "are metadata-derived structural hypotheses, and optical-illusion masks cover the "
            "drawn illusion geometry rather than a subtraction-defined edit."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "README.md").write_text(_readme(), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _game_board_changed_band(
    image: Image.Image, row: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata = row.get("metadata", {})
    rows, columns = _parse_dimensions(str(metadata.get("new_dimensions") or ""))
    dimension = str(metadata.get("dimension_modified") or "")
    if dimension not in {"row", "col"} or rows <= 0 or columns <= 0:
        raise ValueError(f"Incomplete board metadata for {row['id']}")

    structure = _structure_mask(image)
    x0, y0, x1, y1 = _mask_bbox(structure, image.size)
    count = rows if dimension == "row" else columns
    action = str(metadata.get("action") or "")
    is_add = "add" in action
    position = str(metadata.get("position") or "")
    identifier = str(row["id"]).lower()
    index_key = "insert_index" if is_add else "remove_index"
    index = int(metadata.get(index_key) or 0)
    if position == "first" or "first_row" in identifier or "first_col" in identifier:
        index = 0
    elif position == "last" or "last_row" in identifier or "last_col" in identifier:
        index = count - 1
    index = max(0, min(count - 1, index))

    width = x1 - x0
    height = y1 - y0
    mask = np.zeros((image.height, image.width), dtype=bool)
    if dimension == "row":
        cell = height / rows
        center = y0 + (index + 0.5) * cell
        radius = 0.65 * cell
        start = max(0, int(round(center - radius)))
        stop = min(image.height, int(round(center + radius)))
        mask[start:stop, max(0, x0) : min(image.width, x1)] = True
    else:
        cell = width / columns
        center = x0 + (index + 0.5) * cell
        radius = 0.65 * cell
        start = max(0, int(round(center - radius)))
        stop = min(image.width, int(round(center + radius)))
        mask[max(0, y0) : min(image.height, y1), start:stop] = True
    return mask, {
        "dimension_modified": dimension,
        "action": action,
        "new_rows": rows,
        "new_columns": columns,
        "inferred_changed_index_zero_based": index,
        "content_bbox_xyxy": [x0, y0, x1, y1],
        "band_half_width_in_cells": 0.65,
    }


def _structure_mask(image: Image.Image) -> np.ndarray:
    gray = image.convert("L")
    radius = max(1, round(min(image.size) * 0.003))
    size = 2 * radius + 1
    local_max = np.asarray(gray.filter(ImageFilter.MaxFilter(size=size)), dtype=np.int16)
    local_min = np.asarray(gray.filter(ImageFilter.MinFilter(size=size)), dtype=np.int16)
    edges = local_max - local_min > 12
    edges = dilate_binary_mask(edges, max(1, round(min(image.size) * 0.002)))
    return edges


def _foreground_mask(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.int16)
    border = np.concatenate(
        [array[0, :, :], array[-1, :, :], array[:, 0, :], array[:, -1, :]], axis=0
    )
    background = np.median(border, axis=0)
    distance = np.max(np.abs(array - background), axis=2)
    mask = distance > 22
    mask = dilate_binary_mask(mask, max(1, round(min(image.size) * 0.003)))
    return mask


def _mask_bbox(mask: np.ndarray, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, image_size[0], image_size[1]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    if (x1 - x0) < width * 0.45 or (y1 - y0) < height * 0.45:
        return 0, 0, image_size[0], image_size[1]
    return x0, y0, x1, y1


def _parse_dimensions(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", value)
    if match is None:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def _write_contact_sheets(
    output: Path,
    stem: str,
    examples: list[tuple[str, list[Image.Image]]],
    *,
    page_size: int,
) -> None:
    for page_index, offset in enumerate(range(0, len(examples), page_size), start=1):
        _write_contact_sheet(
            output / f"{stem}_{page_index:03d}.png",
            examples[offset : offset + page_size],
        )


def _write_contact_sheet(path: Path, examples: list[tuple[str, list[Image.Image]]]) -> None:
    columns = 5
    tile = 230
    label_height = 46
    sheet = Image.new("RGB", (columns * tile, len(examples) * (tile + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    headers = ("image", "tight mask", "tight overlay", "broad mask", "broad overlay")
    for row_index, (example_id, images) in enumerate(examples):
        top = row_index * (tile + label_height)
        for column, image in enumerate(images):
            thumbnail = image.convert("RGB").copy()
            thumbnail.thumbnail((tile - 8, tile - 8))
            left = column * tile + (tile - thumbnail.width) // 2
            upper = top + (tile - thumbnail.height) // 2
            sheet.paste(thumbnail, (left, upper))
            draw.rectangle((column * tile, top, (column + 1) * tile - 1, top + tile - 1), outline="#d0d0d0")
            draw.rectangle((column * tile + 3, top + 3, column * tile + 112, top + 21), fill="white")
            draw.text((column * tile + 6, top + 5), headers[column], fill="black")
        draw.text((6, top + tile + 5), example_id[:110], fill="black")
    sheet.save(path)


def _range(values: Any) -> list[float]:
    values = list(values)
    return [float(min(values)), float(max(values))] if values else []


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    )[:160]


def _readme() -> str:
    return """# VLMBias high-bias-topic mask candidates

These files are **manual-review candidates**, not accepted experimental masks.

## Game Boards

- `masks_tight`: a metadata-derived band around the row or column that was added or removed.
- `masks_broad`: visible grid/board structure detected from local image edges.

For removed rows or columns, the tight mask necessarily marks the surviving boundary around the
removed location; deleted pixels do not exist in the edited input.

## Optical Illusions

- `masks_tight`: the drawn illusion geometry, separated from the border-estimated background.
- `masks_broad`: a dilated version that includes nearby context around the geometry.

Many optical-illusion examples have `diff=0`: this means the compared target elements are equal,
not that there is an image edit that can be recovered by subtraction. These candidates therefore
target the complete inducing geometry and should be reviewed before use.

Each contact-sheet row shows: source image, tight mask, tight overlay, broad mask, broad overlay.
"""


if __name__ == "__main__":
    main()
