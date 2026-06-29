from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def get_merged_grid_shape(image_grid_thw, spatial_merge: int) -> tuple[int, int, int]:
    t, h, w = image_grid_thw[0].tolist()
    merge = max(1, int(spatial_merge))
    return int(t), int(h) // merge, int(w) // merge


def assign_panels_to_tokens(
    image_grid_thw,
    panel_widths: Sequence[int],
    spatial_merge: int,
) -> tuple[np.ndarray, tuple[int, int, int], list[tuple[int, int]]]:
    t, merged_h, merged_w = get_merged_grid_shape(image_grid_thw, spatial_merge)
    boundaries = _monotonic_boundaries(merged_w, panel_widths)
    panel_ranges = [(boundaries[idx], boundaries[idx + 1]) for idx in range(len(panel_widths))]

    column_to_panel = np.empty((merged_w,), dtype=int)
    for panel_idx, (col_start, col_end) in enumerate(panel_ranges):
        column_to_panel[col_start:col_end] = panel_idx

    panel_ids = np.tile(column_to_panel, t * merged_h)
    return panel_ids, (t, merged_h, merged_w), panel_ranges


def region_positions_from_ids(img_start: int, region_ids: np.ndarray, n_regions: int) -> dict[int, list[int]]:
    positions = {}
    for region_idx in range(n_regions):
        indices = np.where(region_ids == region_idx)[0]
        positions[region_idx] = (indices + img_start).astype(int).tolist()
    return positions


def _monotonic_boundaries(total_columns: int, widths: Sequence[int]) -> list[int]:
    n_regions = len(widths)
    if n_regions == 0:
        return [0, total_columns]

    total_width = max(1, int(sum(widths)))
    raw = [0.0]
    running = 0
    for width in widths:
        running += int(width)
        raw.append(total_columns * running / total_width)

    boundaries = [0]
    for region_idx in range(1, n_regions):
        proposed = int(round(raw[region_idx]))
        remaining = n_regions - region_idx
        min_allowed = boundaries[-1] + (1 if total_columns >= n_regions else 0)
        max_allowed = total_columns - remaining if total_columns >= n_regions else total_columns
        boundaries.append(min(max(proposed, min_allowed), max_allowed))
    boundaries.append(total_columns)
    return boundaries
