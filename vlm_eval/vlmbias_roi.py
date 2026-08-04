from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


SUPPORTED_PAIR_TOPICS = ("Animals", "Chess Pieces", "Flags", "Logos")


@dataclass(frozen=True)
class DifferenceMask:
    difference: np.ndarray
    raw_mask: np.ndarray
    clean_mask: np.ndarray
    stats: dict[str, float]
    accepted: bool
    rejection_reasons: tuple[str, ...]


def unique_rows_by_image_path(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = str(row.get("image_path") or "")
        if path and path not in unique:
            unique[path] = row
    return list(unique.values())


def build_original_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in unique_rows_by_image_path(rows):
        topic = str(row.get("topic") or "")
        key = original_pair_key(row)
        if key is not None:
            index[(topic, key)] = row
    return index


def match_original_row(
    edited_row: dict[str, Any],
    original_index: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    topic = str(edited_row.get("topic") or "")
    if topic not in SUPPORTED_PAIR_TOPICS:
        return None, f"topic_not_supported_for_subtraction:{topic or 'missing'}"
    key = edited_pair_key(edited_row)
    if key is None:
        return None, "could_not_infer_pair_key"
    original = original_index.get((topic, key))
    if original is None:
        return None, f"no_original_match:{key}"
    return original, None


def original_pair_key(row: dict[str, Any]) -> str | None:
    topic = str(row.get("topic") or "")
    stem = Path(str(row.get("image_path") or "")).stem
    if not stem:
        return None
    if topic == "Animals":
        return _normalize(stem)
    if topic == "Flags":
        return _normalize(stem)
    if topic == "Logos":
        if stem in {"adidas", "nike", "mercedes"}:
            return None
        stem = re.sub(r"_0$", "", stem)
        return _normalize(stem)
    if topic == "Chess Pieces":
        if stem.startswith("xiangqi_pieces"):
            return "xiangqipieces"
        if stem.startswith("chess_pieces"):
            return "chesspieces"
    return None


def edited_pair_key(row: dict[str, Any]) -> str | None:
    topic = str(row.get("topic") or "")
    path = str(row.get("image_path") or "")
    stem = Path(path).stem
    if not stem:
        return None
    if topic == "Animals":
        stem = re.sub(r"_\d+_\d+_(384|768|1152)$", "", stem)
        return _normalize(stem)
    if topic == "Flags":
        stem = re.split(r"-(?:stars|stripes)=", stem, maxsplit=1)[0]
        return _normalize(stem)
    if topic == "Logos":
        if "/car_logos/" not in path.replace("\\", "/"):
            return None
        stem = re.sub(r"_(384|768|1152)$", "", stem)
        return _normalize(stem)
    if topic == "Chess Pieces":
        if "/xiangqi_pieces/" in path.replace("\\", "/"):
            return "xiangqipieces"
        if "/chess_pieces/" in path.replace("\\", "/"):
            return "chesspieces"
    return None


def make_difference_mask(
    original: Image.Image,
    edited: Image.Image,
    *,
    threshold: int = 30,
    comparison_blur_radius: float = 2.0,
    opening_radius: int = 1,
    dilation_fraction: float = 0.006,
    min_mask_fraction: float = 0.0005,
    max_mask_fraction: float = 0.55,
    max_median_difference: float = 8.0,
) -> DifferenceMask:
    edited_rgb = edited.convert("RGB")
    original_rgb = original.convert("RGB").resize(edited_rgb.size, Image.Resampling.LANCZOS)
    original_comparison = original_rgb.filter(
        ImageFilter.GaussianBlur(radius=comparison_blur_radius)
    )
    edited_comparison = edited_rgb.filter(
        ImageFilter.GaussianBlur(radius=comparison_blur_radius)
    )
    original_array = np.asarray(original_comparison, dtype=np.int16)
    edited_array = np.asarray(edited_comparison, dtype=np.int16)
    difference = np.max(np.abs(edited_array - original_array), axis=2).astype(np.uint8)

    raw_mask = difference > int(threshold)
    clean_mask = open_binary_mask(raw_mask, opening_radius)
    dilation_radius = max(1, int(round(min(edited_rgb.size) * dilation_fraction)))
    clean_mask = dilate_binary_mask(clean_mask, dilation_radius)

    stats = {
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "p90_difference": float(np.quantile(difference, 0.9)),
        "p99_difference": float(np.quantile(difference, 0.99)),
        "raw_mask_fraction": float(raw_mask.mean()),
        "clean_mask_fraction": float(clean_mask.mean()),
        "threshold": float(threshold),
        "comparison_blur_radius_pixels": float(comparison_blur_radius),
        "opening_radius_pixels": float(opening_radius),
        "dilation_radius_pixels": float(dilation_radius),
    }
    rejection_reasons: list[str] = []
    if stats["clean_mask_fraction"] < min_mask_fraction:
        rejection_reasons.append("mask_too_small")
    if stats["clean_mask_fraction"] > max_mask_fraction:
        rejection_reasons.append("mask_too_large")
    if stats["median_difference"] > max_median_difference:
        rejection_reasons.append("global_misalignment")
    return DifferenceMask(
        difference=difference,
        raw_mask=raw_mask,
        clean_mask=clean_mask,
        stats=stats,
        accepted=not rejection_reasons,
        rejection_reasons=tuple(rejection_reasons),
    )


def open_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Remove isolated speckles with a fast binary morphological opening."""
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    size = 2 * int(radius) + 1
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    opened = image.filter(ImageFilter.MinFilter(size=size)).filter(
        ImageFilter.MaxFilter(size=size)
    )
    return np.asarray(opened, dtype=np.uint8) > 0


def dilate_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    size = 2 * int(radius) + 1
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    return np.asarray(image.filter(ImageFilter.MaxFilter(size=size)), dtype=np.uint8) > 0


def binary_mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")


def overlay_mask(image: Image.Image, mask: np.ndarray, opacity: float = 0.45) -> Image.Image:
    base = image.convert("RGBA")
    alpha = np.asarray(mask, dtype=np.uint8) * int(round(255 * opacity))
    overlay_array = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay_array[..., 0] = 255
    overlay_array[..., 3] = alpha
    overlay = Image.fromarray(overlay_array, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return "".join(character for character in value.lower() if character.isalnum())
