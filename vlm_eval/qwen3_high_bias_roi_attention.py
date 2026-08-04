from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from vlm_eval.qwen3_roi_attention import (
    base_condition,
    read_jsonl,
    slug,
    write_jsonl,
)


EXPERIMENT_VERSION = "qwen3_vlmbias_high_bias_roi_attention_v1"
DEFAULT_EXPERIMENT_ROOT = Path(
    "segments/vlm_bias_attention/experiments/qwen3_high_bias_roi_attention_v1"
)
DEFAULT_RUN_ROOT = Path(
    "segments/vlm_bias_attention/runs/qwen3_high_bias_roi_attention_v1"
)
DEFAULT_REPORT_ROOT = Path(
    "segments/vlm_bias_attention/reports/qwen3_high_bias_roi_attention_v1"
)
DEFAULT_ROI_ROOT = Path(
    "segments/vlm_bias_attention/data/vlmbias_high_bias_roi_masks_v1"
)
DEFAULT_VLMBIAS = DEFAULT_ROI_ROOT / "dataset/vlmbias_high_bias_114.jsonl"
DEFAULT_RUNTIME_BUNDLE = Path(
    "segments/vlm_bias_attention/assets/vlmbias_high_bias_roi_masks_v1_runtime.tar.gz"
)
TARGET_TOPICS = ("Game Boards", "Optical Illusion")
MASK_VARIANTS = ("tight", "broad")


def condition(
    name: str,
    *,
    region: str,
    mask_variant: str,
    alpha: float,
    heads: int,
    selection: str,
    head_seed: int = 0,
) -> dict[str, Any]:
    if mask_variant not in MASK_VARIANTS:
        raise ValueError(f"unknown mask variant: {mask_variant}")
    return {
        **base_condition(
            name,
            region=region,
            alpha=alpha,
            heads=heads,
            selection=selection,
            head_seed=head_seed,
        ),
        "mask_variant": mask_variant,
    }


def smoke_conditions() -> list[dict[str, Any]]:
    return [
        condition(
            "baseline",
            region="full_image",
            mask_variant="tight",
            alpha=0,
            heads=0,
            selection="gaze_global",
        ),
        condition(
            "roi_tight_gaze50_alpha2",
            region="roi",
            mask_variant="tight",
            alpha=2,
            heads=50,
            selection="gaze_global",
        ),
        condition(
            "roi_broad_gaze50_alpha2",
            region="roi",
            mask_variant="broad",
            alpha=2,
            heads=50,
            selection="gaze_global",
        ),
        condition(
            "full_gaze50_alpha2",
            region="full_image",
            mask_variant="tight",
            alpha=2,
            heads=50,
            selection="gaze_global",
        ),
        condition(
            "roi_tight_random50_alpha2",
            region="roi",
            mask_variant="tight",
            alpha=2,
            heads=50,
            selection="layer_matched_random",
            head_seed=55,
        ),
        condition(
            "roi_broad_random50_alpha2",
            region="roi",
            mask_variant="broad",
            alpha=2,
            heads=50,
            selection="layer_matched_random",
            head_seed=55,
        ),
    ]


def tune_conditions() -> list[dict[str, Any]]:
    rows = [
        condition(
            "baseline",
            region="full_image",
            mask_variant="tight",
            alpha=0,
            heads=0,
            selection="gaze_global",
        )
    ]
    for mask_variant in MASK_VARIANTS:
        for alpha in (0.5, 1.0, 2.0, 5.0):
            rows.append(
                condition(
                    f"roi_{mask_variant}_gaze50_alpha{slug(alpha)}",
                    region="roi",
                    mask_variant=mask_variant,
                    alpha=alpha,
                    heads=50,
                    selection="gaze_global",
                )
            )
    rows.extend(
        [
            condition(
                "full_gaze50_alpha2",
                region="full_image",
                mask_variant="tight",
                alpha=2,
                heads=50,
                selection="gaze_global",
            ),
            condition(
                "shifted_tight_gaze50_alpha2",
                region="shifted_roi",
                mask_variant="tight",
                alpha=2,
                heads=50,
                selection="gaze_global",
            ),
            condition(
                "shifted_broad_gaze50_alpha2",
                region="shifted_roi",
                mask_variant="broad",
                alpha=2,
                heads=50,
                selection="gaze_global",
            ),
        ]
    )
    return rows


def head_conditions(alpha: float, mask_variant: str) -> list[dict[str, Any]]:
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
        condition(
            "baseline",
            region="full_image",
            mask_variant=mask_variant,
            alpha=0,
            heads=0,
            selection="gaze_global",
        )
    ]
    rows.extend(
        condition(
            f"roi_{mask_variant}_{name}_alpha{slug(alpha)}",
            region="roi",
            mask_variant=mask_variant,
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
    selected_variant = str(alpha_spec["mask_variant"])
    alternate_variant = next(value for value in MASK_VARIANTS if value != selected_variant)
    count = int(head_spec["head_count"])
    selection = str(head_spec["head_selection"])
    seed = int(head_spec["head_seed"])
    locked = f"{selected_variant}_{selection}{count}_alpha{slug(alpha)}"
    return [
        condition(
            "baseline",
            region="full_image",
            mask_variant=selected_variant,
            alpha=0,
            heads=0,
            selection="gaze_global",
        ),
        condition(
            f"roi_locked_{locked}",
            region="roi",
            mask_variant=selected_variant,
            alpha=alpha,
            heads=count,
            selection=selection,
            head_seed=seed,
        ),
        condition(
            f"roi_alternate_{alternate_variant}_{selection}{count}_alpha{slug(alpha)}",
            region="roi",
            mask_variant=alternate_variant,
            alpha=alpha,
            heads=count,
            selection=selection,
            head_seed=seed,
        ),
        condition(
            f"full_locked_{locked}",
            region="full_image",
            mask_variant=selected_variant,
            alpha=alpha,
            heads=count,
            selection=selection,
            head_seed=seed,
        ),
        condition(
            f"shifted_locked_{locked}",
            region="shifted_roi",
            mask_variant=selected_variant,
            alpha=alpha,
            heads=count,
            selection=selection,
            head_seed=seed,
        ),
        condition(
            f"roi_random{count}_seed57_{selected_variant}_alpha{slug(alpha)}",
            region="roi",
            mask_variant=selected_variant,
            alpha=alpha,
            heads=count,
            selection="layer_matched_random",
            head_seed=57,
        ),
    ]


def canonical_group_id(example_id: str) -> str:
    value = re.sub(r"_px\d+", "", example_id)
    value = re.sub(r"_Q\d+", "", value)
    value = re.sub(r"_prompt\d+", "", value)
    return re.sub(r"_+", "_", value).strip("_")


def prepare_splits(
    *,
    vlmbias_path: Path,
    roi_root: Path,
    out_dir: Path,
    dev_groups: int = 40,
    smoke_groups: int = 8,
    seed: int = 20260804,
) -> dict[str, Any]:
    source_rows = read_jsonl(vlmbias_path)
    source_by_id = {str(row["id"]): row for row in source_rows}
    accepted = read_jsonl(roi_root / "accepted.jsonl")
    grouped: dict[str, dict[str, Any]] = {}
    for mask_row in accepted:
        example_id = str(mask_row["id"])
        source = source_by_id.get(example_id)
        if source is None:
            raise ValueError(f"mask manifest references missing VLMBias ID: {example_id}")
        topic = str(source.get("topic") or "")
        if topic not in TARGET_TOPICS:
            raise ValueError(f"unexpected topic in high-bias mask manifest: {topic}")
        group_id = str(mask_row.get("group_id") or canonical_group_id(example_id))
        mask_paths = {
            variant: str(_resolve(roi_root, mask_row["artifacts"][f"{variant}_mask"]))
            for variant in MASK_VARIANTS
        }
        for path in mask_paths.values():
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        image_path = str(_resolve(vlmbias_path.parent, str(source["image_path"])))
        metadata = dict(source.get("metadata") or {})
        metadata.update(
            {
                "roi_group_id": group_id,
                "roi_mask_paths": mask_paths,
                "roi_mask_fractions": dict(mask_row["mask_fractions"]),
                "roi_review_status": mask_row["status"],
            }
        )
        row = {**source, "image_path": image_path, "metadata": metadata}
        if group_id not in grouped:
            grouped[group_id] = {"id": group_id, "topic": topic, "rows": []}
        if grouped[group_id]["topic"] != topic:
            raise ValueError(f"group spans multiple topics: {group_id}")
        grouped[group_id]["rows"].append(row)

    groups = list(grouped.values())
    if len(accepted) != 114 or len(groups) != 79:
        raise ValueError(
            f"expected 114 reviewed rows in 79 visual groups; got {len(accepted)} rows in {len(groups)} groups"
        )
    dev_ids = _stratified_group_ids(groups, dev_groups, seed)
    dev = [group for group in groups if group["id"] in dev_ids]
    confirm = [group for group in groups if group["id"] not in dev_ids]
    smoke_ids = _stratified_group_ids(dev, smoke_groups, seed + 1)
    splits = {
        "smoke": _flatten(group for group in dev if group["id"] in smoke_ids),
        "dev": _flatten(dev),
        "confirm": _flatten(confirm),
        "all": _flatten(groups),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, rows in splits.items():
        path = out_dir / f"{name}_vlmbias_roi.jsonl"
        write_jsonl(path, rows)
        paths[name] = str(path)
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "seed": seed,
        "source_vlmbias": str(vlmbias_path),
        "source_roi_manifest": str(roi_root / "accepted.jsonl"),
        "target_topics": list(TARGET_TOPICS),
        "mask_variants": list(MASK_VARIANTS),
        "n_rows": len(accepted),
        "n_unique_visual_groups": len(groups),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "group_counts": {
            "smoke": len(smoke_ids),
            "dev": len(dev_ids),
            "confirm": len(groups) - len(dev_ids),
            "all": len(groups),
        },
        "row_topics": {
            name: dict(sorted(Counter(row["topic"] for row in rows).items()))
            for name, rows in splits.items()
        },
        "group_topics": {
            "dev": dict(sorted(Counter(group["topic"] for group in dev).items())),
            "confirm": dict(sorted(Counter(group["topic"] for group in confirm).items())),
        },
        "grouping": "canonical manipulation/illusion ID after removing prompt and resolution suffixes",
        "stratification": "topic at canonical visual-group level",
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
        {**row, "split": split, "dataset": split_manifest["paths"][split]}
        for row in conditions
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
        "topics": list(TARGET_TOPICS),
        "rows_and_groups": "114 rows grouped into 79 canonical visual/manipulation groups before splitting.",
        "intervention": "Add fixed alpha before softmax to tight or broad ROI visual-token keys in selected language-attention heads for every prefill and decoding query.",
        "tight_masks": "Game Boards: inferred changed row/column band. Optical Illusions: drawn illusion geometry.",
        "broad_masks": "Game Boards: complete visible board/grid structure. Optical Illusions: dilated illusion geometry.",
        "token_mapping": "Resize each binary mask to image_grid_thw / spatial_merge_size using BOX coverage; select tokens with at least 5% ROI coverage.",
        "stages": {
            "smoke": "8 development visual groups; baseline plus tight/broad gaze-50, full-image, and tight/broad random-50 mechanics controls.",
            "tune": "40 development visual groups; tight and broad gaze top-50 with alpha in {0.5,1,2,5}, plus full-image and shifted-mask controls.",
            "heads": "Same development split with the locked mask/alpha; gaze top-k/layer bands and random/low-score controls.",
            "confirm": "39 disjoint held-out visual groups; baseline, locked ROI, alternate mask, full-image, shifted-mask, and fresh layer-matched-random control.",
        },
        "metrics": "Accuracy, bias-aligned fraction/count, conditional bias-aligned error rate, invalid rate, target-token fraction, and attention telemetry overall and by topic.",
        "selection": "Among gaze-eligible ROI conditions passing accuracy/invalid guardrails, maximize accuracy, then minimize bias-aligned fraction and invalid rate. Random controls cannot be selected.",
    }


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
    total = len(groups)
    exact = {
        topic: n_select * len(values) / total for topic, values in by_topic.items()
    }
    quotas = {topic: int(value) for topic, value in exact.items()}
    remaining = n_select - sum(quotas.values())
    for topic in sorted(by_topic, key=lambda key: (-(exact[key] - quotas[key]), key)):
        if remaining <= 0:
            break
        quotas[topic] += 1
        remaining -= 1
    selected = {
        group_id
        for topic, values in by_topic.items()
        for group_id in values[: quotas[topic]]
    }
    if len(selected) != n_select:
        raise AssertionError(f"stratified selection produced {len(selected)} != {n_select}")
    return selected


def _flatten(groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for group in groups for row in group["rows"]]
