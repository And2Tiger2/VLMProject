from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets
from PIL import Image, ImageDraw


DEFAULT_CANDIDATES = Path(
    "segments/vlm_bias_attention/data/vlmbias_high_bias_mask_candidates_v1"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make original-versus-biased VLMBias mask review sheets without overlays."
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--page-size", type=int, default=10)
    args = parser.parse_args()

    root = args.candidates
    candidate_rows = _read_jsonl(root / "manifest.jsonl")
    cache = args.dataset_cache or _find_cached_dataset()
    original_rows = _load_original_split(cache)
    illusion_index = {
        _original_illusion_key(row): row
        for row in original_rows
        if row.get("topic") == "Optical Illusion"
    }
    illusion_originals = [
        row for row in original_rows if row.get("topic") == "Optical Illusion"
    ]
    board_index: dict[str, dict[str, Any]] = {}
    for row in original_rows:
        if row.get("topic") == "Game Boards":
            board_index.setdefault(str(row.get("sub_topic")), row)

    sheets: dict[str, list[tuple[str, list[Image.Image]]]] = {
        "game_boards": [],
        "optical_illusions": [],
    }
    comparisons: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        topic_dir = "game_boards" if candidate["topic"] == "Game Boards" else "optical_illusions"
        if candidate["topic"] == "Game Boards":
            original_row = board_index.get(str(candidate.get("sub_topic")))
            pairing_method = "same game-board subtopic standard image"
        else:
            original_row, pairing_method = _match_illusion_original(
                candidate, illusion_index, illusion_originals
            )
        if original_row is None:
            raise RuntimeError(f"No upstream original match for {candidate['id']}")

        safe_id = _safe_name(str(candidate["id"]))
        originals_dir = root / topic_dir / "originals"
        originals_dir.mkdir(parents=True, exist_ok=True)
        original_path = originals_dir / f"{safe_id}.png"
        original_image = original_row["image"].convert("RGB")
        original_image.save(original_path)

        edited = Image.open(root / candidate["artifacts"]["image"]).convert("RGB")
        tight = Image.open(root / candidate["artifacts"]["tight_mask"]).convert("RGB")
        broad = Image.open(root / candidate["artifacts"]["broad_mask"]).convert("RGB")
        sheets[topic_dir].append((safe_id, [original_image, edited, tight, broad]))
        comparisons.append(
            {
                "id": candidate["id"],
                "topic": candidate["topic"],
                "sub_topic": candidate.get("sub_topic"),
                "pairing_method": pairing_method,
                "upstream_original_id": original_row.get("ID"),
                "upstream_original_path": original_row.get("image_path"),
                "artifacts": {
                    "original": str(original_path.relative_to(root)),
                    "biased_or_edited": candidate["artifacts"]["image"],
                    "tight_mask": candidate["artifacts"]["tight_mask"],
                    "broad_mask": candidate["artifacts"]["broad_mask"],
                },
            }
        )

    _write_jsonl(root / "comparison_manifest.jsonl", comparisons)
    for topic_dir, examples in sheets.items():
        _write_contact_sheets(
            root,
            f"{topic_dir}_original_biased_masks",
            examples,
            page_size=args.page_size,
        )
    summary = {
        "valid": len(comparisons) == len(candidate_rows),
        "n_pairs": len(comparisons),
        "n_game_boards": sum(row["topic"] == "Game Boards" for row in comparisons),
        "n_optical_illusions": sum(row["topic"] == "Optical Illusion" for row in comparisons),
        "columns": ["upstream original", "biased/edited slice image", "tight mask", "broad mask"],
        "dataset_cache": str(cache),
    }
    (root / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _find_cached_dataset() -> Path:
    candidates = sorted(
        (Path.home() / ".cache/huggingface/datasets/anvo25___vlms-are-biased").glob("default/*/*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if list(path.glob("vlms-are-biased-original*.arrow")):
            return path
    raise SystemExit("Cached anvo25/vlms-are-biased original split was not found")


def _load_original_split(cache: Path) -> list[dict[str, Any]]:
    paths = sorted(cache.glob("vlms-are-biased-original*.arrow"))
    parts = [Dataset.from_file(str(path)) for path in paths]
    dataset = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    return [
        {
            "image": row["image"],
            "ID": row["ID"],
            "image_path": row["image_path"],
            "topic": row["topic"],
            "sub_topic": row["sub_topic"],
        }
        for row in dataset
        if row.get("topic") in {"Game Boards", "Optical Illusion"}
    ]


def _candidate_illusion_key(row: dict[str, Any]) -> tuple[str, str, str]:
    metadata = row.get("metadata", {})
    family = str(metadata.get("illusion_type") or str(row["id"]).split("_", 1)[0])
    prompt_type = str(metadata.get("prompt_type") or _prompt_type(str(row["id"])))
    if family == "VerticalHorizontal":
        parameter = f"min:{float(metadata.get('size_min') or 0):.3f}"
    else:
        parameter = f"strength:{int(metadata.get('strength') or 0)}"
    return family, parameter, prompt_type


def _match_illusion_original(
    candidate: dict[str, Any],
    exact_index: dict[tuple[str, str, str], dict[str, Any]],
    originals: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    key = _candidate_illusion_key(candidate)
    exact = exact_index.get(key)
    if exact is not None:
        return exact, "exact same illusion family/strength/prompt with diff=0"

    family, parameter, prompt_type = key
    target_value = float(parameter.split(":", 1)[1])
    compatible: list[tuple[float, dict[str, Any]]] = []
    for row in originals:
        original_key = _original_illusion_key(row)
        if original_key[0] != family or original_key[2] != prompt_type:
            continue
        value = float(original_key[1].split(":", 1)[1])
        compatible.append((abs(value - target_value), row))
    if not compatible:
        return None, "no compatible diff=0 reference"
    distance, closest = min(compatible, key=lambda item: item[0])
    return closest, f"nearest available diff=0 reference; parameter_distance={distance:g}"


def _original_illusion_key(row: dict[str, Any]) -> tuple[str, str, str]:
    identifier = str(row["ID"])
    family = identifier.split("_", 1)[0]
    prompt_type = _prompt_type(identifier)
    path = str(row["image_path"])
    strength = re.search(r"_str(neg)?(\d+)_", path)
    minimum = re.search(r"_min(\d+)p(\d+)_", path)
    if minimum:
        parameter = f"min:{float(f'{minimum.group(1)}.{minimum.group(2)}'):.3f}"
    elif strength:
        value = int(strength.group(2)) * (-1 if strength.group(1) else 1)
        parameter = f"strength:{value}"
    else:
        raise ValueError(f"Could not parse original illusion parameter: {path}")
    return family, parameter, prompt_type


def _prompt_type(value: str) -> str:
    match = re.search(r"_(Q\d+)_", value)
    return match.group(1) if match else ""


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
    columns = 4
    tile = 270
    label_height = 46
    headers = ("upstream original", "biased / edited", "tight mask", "broad mask")
    sheet = Image.new("RGB", (columns * tile, len(examples) * (tile + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, (example_id, images) in enumerate(examples):
        top = row_index * (tile + label_height)
        for column, image in enumerate(images):
            thumbnail = image.convert("RGB").copy()
            thumbnail.thumbnail((tile - 8, tile - 8))
            left = column * tile + (tile - thumbnail.width) // 2
            upper = top + (tile - thumbnail.height) // 2
            sheet.paste(thumbnail, (left, upper))
            draw.rectangle(
                (column * tile, top, (column + 1) * tile - 1, top + tile - 1),
                outline="#d0d0d0",
            )
            draw.rectangle(
                (column * tile + 3, top + 3, column * tile + 145, top + 22), fill="white"
            )
            draw.text((column * tile + 6, top + 5), headers[column], fill="black")
        draw.text((6, top + tile + 5), example_id[:110], fill="black")
    sheet.save(path)


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


if __name__ == "__main__":
    main()
