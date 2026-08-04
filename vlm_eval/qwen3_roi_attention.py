from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


EXPERIMENT_VERSION = "qwen3_vlmbias_roi_attention_v1"
DEFAULT_EXPERIMENT_ROOT = Path(
    "segments/vlm_bias_attention/experiments/qwen3_roi_attention_v1"
)
DEFAULT_RUN_ROOT = Path("segments/vlm_bias_attention/runs/qwen3_roi_attention_v1")
DEFAULT_REPORT_ROOT = Path("segments/vlm_bias_attention/reports/qwen3_roi_attention_v1")
DEFAULT_ROI_ROOT = Path("segments/vlm_bias_attention/data/vlmbias_roi_masks_v1")
DEFAULT_VLMBIAS = Path("segments/vlm_bias_attention/data/vlmbias_400.jsonl")
DEFAULT_GAZE_RANKING = Path(
    "segments/gaze_heads_qwen3_8b/runs/gaze_discovery_seed42_merged/"
    "gaze_head_ranking.json"
)


def pixel_mask_to_visual_tokens(
    mask: Image.Image,
    image_grid_thw: Iterable[int],
    *,
    spatial_merge_size: int,
    min_token_coverage: float = 0.05,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map a pixel ROI to Qwen's row-major, spatially merged image tokens."""
    grid_t, grid_h, grid_w = (int(value) for value in image_grid_thw)
    if min(grid_t, grid_h, grid_w) <= 0:
        raise ValueError(f"invalid image_grid_thw: {(grid_t, grid_h, grid_w)}")
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")
    if grid_h % spatial_merge_size or grid_w % spatial_merge_size:
        raise ValueError(
            "Qwen image grid must be divisible by spatial_merge_size: "
            f"grid={(grid_t, grid_h, grid_w)}, merge={spatial_merge_size}"
        )
    if not 0.0 <= min_token_coverage <= 1.0:
        raise ValueError("min_token_coverage must be within [0, 1]")

    merged_h = grid_h // spatial_merge_size
    merged_w = grid_w // spatial_merge_size
    binary = mask.convert("L").point(lambda value: 255 if value > 0 else 0)
    occupancy = (
        np.asarray(
            binary.resize((merged_w, merged_h), Image.Resampling.BOX),
            dtype=np.float32,
        )
        / 255.0
    )
    spatial = occupancy >= float(min_token_coverage)
    if bool((occupancy > 0).any()) and not bool(spatial.any()):
        spatial.flat[int(np.argmax(occupancy))] = True
    tokens = np.tile(spatial.reshape(-1), grid_t).astype(bool, copy=False)
    return tokens, {
        "image_grid_thw": [grid_t, grid_h, grid_w],
        "spatial_merge_size": int(spatial_merge_size),
        "merged_grid_hw": [merged_h, merged_w],
        "n_image_tokens": int(tokens.size),
        "n_target_tokens": int(tokens.sum()),
        "target_token_indices": np.flatnonzero(tokens).astype(int).tolist(),
        "target_token_fraction": float(tokens.mean()),
        "min_token_coverage": float(min_token_coverage),
        "mean_selected_token_coverage": (
            float(np.tile(occupancy.reshape(-1), grid_t)[tokens].mean())
            if bool(tokens.any())
            else 0.0
        ),
    }


def spatial_control_mask(mask: Image.Image, example_id: str, seed: int) -> Image.Image:
    """Deterministically roll a mask while preserving its exact pixel area."""
    array = np.asarray(mask.convert("L"), dtype=np.uint8)
    digest = hashlib.sha256(f"{seed}:{example_id}".encode()).digest()
    height, width = array.shape
    shift_y = max(
        1, height // 3 + int.from_bytes(digest[:2], "big") % max(1, height // 3)
    )
    shift_x = max(
        1, width // 3 + int.from_bytes(digest[2:4], "big") % max(1, width // 3)
    )
    shifted = np.roll(array, shift=(shift_y, shift_x), axis=(0, 1))
    return Image.fromarray(shifted, mode="L")


def base_condition(
    name: str,
    *,
    region: str,
    alpha: float,
    heads: int,
    selection: str,
    head_seed: int = 0,
) -> dict[str, Any]:
    return {
        "name": name,
        "region": region,
        "alpha": float(alpha),
        "head_count": int(heads),
        "head_selection": selection,
        "head_seed": int(head_seed),
        "seed": 0,
        "do_sample": False,
        "temperature": None,
    }


def smoke_conditions() -> list[dict[str, Any]]:
    return [
        base_condition(
            "baseline", region="full_image", alpha=0, heads=0, selection="gaze_global"
        ),
        base_condition(
            "roi_gaze50_alpha2",
            region="roi",
            alpha=2,
            heads=50,
            selection="gaze_global",
        ),
        base_condition(
            "full_gaze50_alpha2",
            region="full_image",
            alpha=2,
            heads=50,
            selection="gaze_global",
        ),
        base_condition(
            "roi_random50_alpha2",
            region="roi",
            alpha=2,
            heads=50,
            selection="layer_matched_random",
            head_seed=55,
        ),
    ]


def tune_conditions() -> list[dict[str, Any]]:
    rows = [
        base_condition(
            "baseline", region="full_image", alpha=0, heads=0, selection="gaze_global"
        )
    ]
    rows.extend(
        base_condition(
            f"roi_gaze50_alpha{slug(alpha)}",
            region="roi",
            alpha=alpha,
            heads=50,
            selection="gaze_global",
        )
        for alpha in (0.5, 1.0, 2.0, 5.0)
    )
    rows.extend(
        [
            base_condition(
                "full_gaze50_alpha2",
                region="full_image",
                alpha=2,
                heads=50,
                selection="gaze_global",
            ),
            base_condition(
                "shifted_gaze50_alpha2",
                region="shifted_roi",
                alpha=2,
                heads=50,
                selection="gaze_global",
                head_seed=0,
            ),
        ]
    )
    return rows


def head_conditions(alpha: float) -> list[dict[str, Any]]:
    definitions = [
        ("gaze_global10", "gaze_global", 10, 0),
        ("gaze_global50", "gaze_global", 50, 0),
        ("gaze_global100", "gaze_global", 100, 0),
        ("gaze_early50", "gaze_early", 50, 0),
        ("gaze_middle50", "gaze_middle", 50, 0),
        ("gaze_late50", "gaze_late", 50, 0),
        ("layer_random50_seed55", "layer_matched_random", 50, 55),
        ("layer_random50_seed56", "layer_matched_random", 50, 56),
        ("layer_random100_seed55", "layer_matched_random", 100, 55),
        ("layer_low50", "layer_matched_low", 50, 0),
        ("paper_random50_seed55", "paper_random", 50, 55),
    ]
    rows = [
        base_condition(
            "baseline", region="full_image", alpha=0, heads=0, selection="gaze_global"
        )
    ]
    rows.extend(
        base_condition(
            f"roi_{name}_alpha{slug(alpha)}",
            region="roi",
            alpha=alpha,
            heads=count,
            selection=selection,
            head_seed=seed,
        )
        for name, selection, count, seed in definitions
    )
    return rows


def confirm_conditions(
    alpha_spec: dict[str, Any], head_spec: dict[str, Any]
) -> list[dict[str, Any]]:
    alpha = float(alpha_spec["alpha"])
    count = int(head_spec["head_count"])
    selection = str(head_spec["head_selection"])
    seed = int(head_spec["head_seed"])
    locked = f"{selection}{count}_alpha{slug(alpha)}"
    return [
        base_condition(
            "baseline", region="full_image", alpha=0, heads=0, selection="gaze_global"
        ),
        base_condition(
            f"roi_locked_{locked}",
            region="roi",
            alpha=alpha,
            heads=count,
            selection=selection,
            head_seed=seed,
        ),
        base_condition(
            f"full_locked_{locked}",
            region="full_image",
            alpha=alpha,
            heads=count,
            selection=selection,
            head_seed=seed,
        ),
        base_condition(
            f"shifted_locked_{locked}",
            region="shifted_roi",
            alpha=alpha,
            heads=count,
            selection=selection,
            head_seed=seed,
        ),
        base_condition(
            f"roi_layer_random{count}_seed57_alpha{slug(alpha)}",
            region="roi",
            alpha=alpha,
            heads=count,
            selection="layer_matched_random",
            head_seed=57,
        ),
    ]


def prepare_roi_splits(
    *,
    vlmbias_path: Path,
    roi_root: Path,
    out_dir: Path,
    dev_groups: int = 50,
    smoke_groups: int = 8,
    seed: int = 2026,
) -> dict[str, Any]:
    source_rows = read_jsonl(vlmbias_path)
    accepted = read_jsonl(roi_root / "accepted.jsonl")
    source_by_id = {str(row["id"]): row for row in source_rows}
    groups: list[dict[str, Any]] = []
    for mask_row in accepted:
        rows = []
        for example_id in mask_row.get("covered_local_ids") or [mask_row["id"]]:
            source = source_by_id.get(str(example_id))
            if source is None:
                raise ValueError(
                    f"ROI manifest references missing VLMBias ID: {example_id}"
                )
            image_path = _resolve(vlmbias_path.parent, source["image_path"])
            mask_path = _resolve(roi_root, mask_row["artifacts"]["mask_path"])
            metadata = dict(source.get("metadata") or {})
            metadata.update(
                {
                    "roi_mask_path": str(mask_path),
                    "roi_group_id": str(mask_row["id"]),
                    "roi_mask_fraction": mask_row["mask_stats"]["clean_mask_fraction"],
                }
            )
            rows.append({**source, "image_path": str(image_path), "metadata": metadata})
        groups.append(
            {"id": str(mask_row["id"]), "topic": str(mask_row["topic"]), "rows": rows}
        )

    dev_ids = _stratified_group_ids(groups, dev_groups, seed)
    dev_groups_rows = [group for group in groups if group["id"] in dev_ids]
    confirm_groups_rows = [group for group in groups if group["id"] not in dev_ids]
    smoke_ids = _stratified_group_ids(dev_groups_rows, smoke_groups, seed + 1)
    splits = {
        "smoke": _flatten(
            group for group in dev_groups_rows if group["id"] in smoke_ids
        ),
        "dev": _flatten(dev_groups_rows),
        "confirm": _flatten(confirm_groups_rows),
        "all": _flatten(groups),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, rows in splits.items():
        path = out_dir / f"{name}_vlmbias_roi.jsonl"
        write_jsonl(path, rows)
        paths[name] = str(path)
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "seed": seed,
        "source_vlmbias": str(vlmbias_path),
        "source_roi_manifest": str(roi_root / "accepted.jsonl"),
        "n_unique_roi_groups": len(groups),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "group_counts": {
            "smoke": len(smoke_ids),
            "dev": len(dev_ids),
            "confirm": len(groups) - len(dev_ids),
            "all": len(groups),
        },
        "stratification": "topic, grouped by shared edited image/mask",
        "paths": paths,
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def write_stage_manifest(
    *,
    stage: str,
    conditions: list[dict[str, Any]],
    split: str,
    split_manifest: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    rows = [
        {**condition, "split": split, "dataset": split_manifest["paths"][split]}
        for condition in conditions
    ]
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": stage,
        "split": split,
        "split_manifest": str(path.parent / "split_manifest.json"),
        "conditions": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def methodology() -> dict[str, Any]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "intervention": "Add fixed alpha before softmax only to ROI visual-token keys in selected language-attention heads, for every prefill and decoding query.",
        "token_mapping": "Resize the reviewed binary pixel mask to image_grid_thw / spatial_merge_size with BOX coverage; select tokens with at least 5% ROI coverage.",
        "stages": {
            "smoke": "8 unique mask groups; baseline, ROI gaze-50, full-image gaze-50, and ROI random-50.",
            "tune": "50 development mask groups; gaze top-50 alpha in {0.5,1,2,5}, plus full-image and shifted-mask controls.",
            "heads": "Same development split at locked alpha; gaze top-k/layer bands and random/low-score controls.",
            "confirm": "Disjoint held-out mask groups; baseline, locked ROI, full-image, shifted-mask, and layer-matched-random-head control.",
        },
        "selection": "Among gaze-eligible conditions, maximize VLMBias accuracy, then minimize bias-aligned fraction and invalid rate; random controls cannot be selected.",
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _stratified_group_ids(
    groups: list[dict[str, Any]], n_select: int, seed: int
) -> set[str]:
    if not 0 < n_select <= len(groups):
        raise ValueError(f"requested {n_select} groups from {len(groups)}")
    by_topic: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        by_topic[group["topic"]].append(group["id"])
    rng = random.Random(seed)
    for values in by_topic.values():
        rng.shuffle(values)
    selected: set[str] = set()
    topics = sorted(by_topic)
    while len(selected) < n_select:
        for topic in topics:
            if by_topic[topic] and len(selected) < n_select:
                selected.add(by_topic[topic].pop())
    return selected


def _flatten(groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for group in groups for row in group["rows"]]
