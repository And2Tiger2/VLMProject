from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from vlm_eval.gaze_comics import build_strip, list_comic_dirs
from vlm_eval.gaze_regions import assign_panels_to_tokens, region_positions_from_ids


def test_build_strip_from_panel_folder(tmp_path: Path) -> None:
    comic_dir = tmp_path / "comic1"
    comic_dir.mkdir()
    for panel_idx in range(1, 7):
        Image.new("RGB", (40 + panel_idx, 20), (panel_idx, 0, 0)).save(comic_dir / f"p{panel_idx}.png")

    assert list_comic_dirs(tmp_path) == [comic_dir]
    strip = build_strip(comic_dir, target_height=64, gap=4)

    assert strip.name == "comic1"
    assert len(strip.panels) == 6
    assert len(strip.panel_widths) == 6
    assert strip.strip.size[1] == 64
    assert strip.strip.size[0] == sum(strip.panel_widths)


def test_assign_panels_to_tokens_and_positions() -> None:
    image_grid_thw = np.array([[1, 8, 24]])
    region_ids, grid_shape, ranges = assign_panels_to_tokens(
        image_grid_thw=image_grid_thw,
        panel_widths=[10, 10, 10, 10, 10, 10],
        spatial_merge=2,
    )

    assert grid_shape == (1, 4, 12)
    assert len(ranges) == 6
    assert region_ids.shape == (48,)
    assert set(region_ids.tolist()) == set(range(6))

    positions = region_positions_from_ids(img_start=5, region_ids=region_ids, n_regions=6)
    assert set(positions) == set(range(6))
    assert all(panel_positions for panel_positions in positions.values())
    assert min(min(panel_positions) for panel_positions in positions.values()) >= 5
