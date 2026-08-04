from pathlib import Path

import numpy as np
from PIL import Image

from vlm_eval.vlmbias_roi import (
    build_original_index,
    make_difference_mask,
    match_original_row,
)


def _row(topic: str, path: str) -> dict:
    return {"topic": topic, "image_path": path, "ID": Path(path).stem, "image": None}


def test_pair_keys_match_supported_vlmbias_families() -> None:
    originals = [
        _row("Animals", "original_images/images/african wild dog.png"),
        _row("Flags", "original_images/images/Flag of Cuba.png"),
        _row("Logos", "original_images/images/mercedes-benz_black_sedan_0.png"),
        _row("Chess Pieces", "original_images/images/chess_pieces_notitle_px1152.png"),
    ]
    index = build_original_index(originals)
    edited = [
        _row(
            "Animals",
            "vlms-are-biased-notitle/animals_add_legs/images/african wild dog_2_3_768.png",
        ),
        _row(
            "Flags",
            "vlms-are-biased-notitle/flag_stripes/images/Flag of Cuba-stripes=6_1152.png",
        ),
        _row(
            "Logos",
            "vlms-are-biased-notitle/car_logos/images/mercedes_benz-black-sedan_768.png",
        ),
        _row(
            "Chess Pieces",
            "vlms-are-biased-notitle/chess_pieces/images/chess_pieces_004_remove_bishop.png",
        ),
    ]
    assert all(match_original_row(row, index)[0] is not None for row in edited)


def test_unaligned_or_unsupported_pairs_are_rejected() -> None:
    index = build_original_index([_row("Logos", "original_images/images/nike.png")])
    shoe = _row(
        "Logos",
        "vlms-are-biased-notitle/sportswear_logos/images/nike-red-running-0_768.png",
    )
    patterned = _row("Patterned Grid", "tally/images/tally_001.png")
    assert match_original_row(shoe, index)[1] == "could_not_infer_pair_key"
    assert match_original_row(patterned, index)[1].startswith("topic_not_supported")


def test_difference_mask_is_binary_local_and_dilated() -> None:
    original = Image.new("RGB", (64, 64), "white")
    edited_array = np.full((64, 64, 3), 255, dtype=np.uint8)
    edited_array[28:36, 28:36] = 0
    result = make_difference_mask(
        original,
        Image.fromarray(edited_array),
        threshold=20,
        comparison_blur_radius=0,
        dilation_fraction=0.02,
    )
    assert result.accepted
    assert result.clean_mask.dtype == np.bool_
    assert result.clean_mask[31, 31]
    assert result.clean_mask.sum() > 8 * 8
    assert result.stats["clean_mask_fraction"] < 0.1


def test_global_change_is_rejected_as_misalignment() -> None:
    original = Image.new("RGB", (32, 32), "white")
    edited = Image.new("RGB", (32, 32), "black")
    result = make_difference_mask(original, edited)
    assert not result.accepted
    assert "global_misalignment" in result.rejection_reasons
    assert "mask_too_large" in result.rejection_reasons
