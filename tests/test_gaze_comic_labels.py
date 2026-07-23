from __future__ import annotations

from pathlib import Path

from PIL import Image

from vlm_eval.gaze_comics import ComicStrip, add_panel_number_labels


def test_add_panel_number_labels_adds_banner_without_changing_strip_pixels() -> None:
    image = Image.new("RGB", (120, 20), (10, 20, 30))
    strip = ComicStrip(
        name="comic",
        strip=image,
        panels=[],
        panel_widths=[20] * 6,
        panel_paths=[Path(f"p{panel}.png") for panel in range(1, 7)],
        target_height=20,
    )

    labeled = add_panel_number_labels(strip, label_height=40)

    assert labeled.size == (120, 60)
    assert labeled.crop((0, 40, 120, 60)).tobytes() == image.tobytes()
    assert labeled.getpixel((10, 20)) != (255, 255, 255)
