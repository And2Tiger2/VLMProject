from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from vlm_eval.qwen3_roi_attention import pixel_mask_to_visual_tokens


CUE_MODES = ("text", "target_exemplar", "impostor_exemplar")
SCENE_OFFSET = (160, 20)
COMPOSITE_SIZE = (580, 440)


@dataclass(frozen=True)
class SearchProbe:
    image: Image.Image
    prompt: str
    answer: str
    candidate_answers: tuple[str, ...]
    masks: dict[str, Image.Image]
    target_candidate: int


def assert_base_only(config: dict[str, Any]) -> None:
    forbidden = [key for key in ("adapter_path", "checkpoint", "lora") if config.get(key)]
    if forbidden:
        raise ValueError(f"base-search experiments forbid trained checkpoints: {forbidden}")


def assert_unmodified_runtime(runtime: Any) -> None:
    if getattr(runtime, "adapter_path", None) is not None or bool(getattr(runtime, "adapter_merged", False)):
        raise RuntimeError("base-search runtime unexpectedly loaded or merged an adapter")


def configured_cues(config: dict[str, Any]) -> tuple[str, ...]:
    cues = tuple(str(value) for value in config.get("cue_modes", CUE_MODES))
    unknown = set(cues) - set(CUE_MODES)
    if unknown:
        raise ValueError(f"unknown cue modes: {sorted(unknown)}")
    return cues


def crop_masked_object(image_path: str | Path, mask_path: str | Path, *, padding: int = 8) -> tuple[Image.Image, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError(f"empty exemplar mask: {mask_path}")
    left, top, right, bottom = bbox
    bbox = (
        max(0, left - padding),
        max(0, top - padding),
        min(image.width, right + padding),
        min(image.height, bottom + padding),
    )
    return image.crop(bbox), mask.crop(bbox)


def exemplar_source_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    present = next(
        row for row in rows if row.get("split") == "train" and row.get("target_present")
    )
    impostor_row = next(
        (
            row
            for row in rows
            if row.get("split") == "train"
            and any(obj.get("class") == "distractor-incorrect-binding" for obj in row["objects"])
        ),
        present,
    )
    return present, impostor_row


def find_exemplars(rows: list[dict[str, Any]]) -> dict[str, tuple[Image.Image, Image.Image]]:
    present, impostor_row = exemplar_source_rows(rows)
    target = crop_masked_object(present["image_path"], present["masks"]["target"])
    impostor = next(
        (obj for obj in impostor_row["objects"] if obj.get("class") == "distractor-incorrect-binding"),
        next(obj for obj in impostor_row["objects"] if str(obj.get("class", "")).startswith("distractor")),
    )
    impostor_mask = impostor_row["masks"][f"object_{int(impostor['index']):03d}"]
    return {
        "target_exemplar": target,
        "impostor_exemplar": crop_masked_object(impostor_row["image_path"], impostor_mask),
    }


def build_search_probe(
    row: dict[str, Any],
    *,
    cue_mode: str,
    exemplars: dict[str, tuple[Image.Image, Image.Image]],
) -> SearchProbe:
    if cue_mode not in CUE_MODES:
        raise ValueError(f"unknown cue mode: {cue_mode}")
    if not row.get("target_present"):
        raise ValueError("candidate localization probes require a present target")
    candidates = [int(value) for value in row["metadata"]["four_candidate_cells"]]
    target_cell = int(row["target_cell"])
    if target_cell not in candidates:
        raise ValueError(f"target cell {target_cell} is absent from candidates")
    target_candidate = candidates.index(target_cell)

    scene = Image.open(row["image_path"]).convert("RGB")
    if scene.size != (400, 400):
        scene = scene.resize((400, 400), Image.Resampling.BICUBIC)
    composite = Image.new("RGB", COMPOSITE_SIZE, (245, 245, 242))
    composite.paste(scene, SCENE_OFFSET)
    draw = ImageDraw.Draw(composite)
    draw.rectangle((150, 10, 570, 430), outline=(35, 35, 35), width=2)
    draw.text((164, 2), "SEARCH SCENE", fill=(20, 20, 20))
    draw.text((18, 15), "SEARCH CUE", fill=(20, 20, 20))

    masks = {
        name: Image.new("L", COMPOSITE_SIZE, 0)
        for name in ("reference", "scene", "target_object", "target_candidate", "distractor_candidates")
    }
    masks.update({f"candidate_{index}": Image.new("L", COMPOSITE_SIZE, 0) for index in range(4)})
    masks["scene"].paste(Image.new("L", (400, 400), 255), SCENE_OFFSET)
    target_object = Image.open(row["masks"]["target"]).convert("L")
    masks["target_object"].paste(target_object, SCENE_OFFSET)

    if cue_mode == "text":
        lines = ("striped", "torso", "+ round", "glasses", "+ pointed", "hat", "+ blue", "lower body")
        for index, line in enumerate(lines):
            draw.text((20, 75 + 24 * index), line, fill=(30, 30, 30))
    else:
        exemplar, exemplar_mask = exemplars[cue_mode]
        exemplar.thumbnail((105, 170), Image.Resampling.LANCZOS)
        exemplar_mask = exemplar_mask.resize(exemplar.size, Image.Resampling.NEAREST)
        paste_xy = ((140 - exemplar.width) // 2, 95)
        composite.paste(exemplar, paste_xy)
        masks["reference"].paste(exemplar_mask, paste_xy)
        label = "MATCH THIS" if cue_mode == "target_exemplar" else "CONTROL CUE"
        draw.text((25, 280), label, fill=(20, 20, 20))

    palette = ((0, 90, 180), (180, 60, 0), (0, 130, 65), (135, 45, 150))
    for candidate_index, cell in enumerate(candidates):
        column, row_index = cell % 10, cell // 10
        x0 = SCENE_OFFSET[0] + column * 40
        y0 = SCENE_OFFSET[1] + row_index * 40
        x1, y1 = x0 + 39, y0 + 39
        color = palette[candidate_index]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        draw.rectangle((x0 + 1, y0 + 1, x0 + 15, y0 + 14), fill=(250, 250, 250))
        draw.text((x0 + 4, y0 + 1), str(candidate_index), fill=color)
        cell_mask = Image.new("L", COMPOSITE_SIZE, 0)
        ImageDraw.Draw(cell_mask).rectangle((x0, y0, x1, y1), fill=255)
        masks[f"candidate_{candidate_index}"] = cell_mask
        destination = "target_candidate" if cell == target_cell else "distractor_candidates"
        masks[destination] = Image.fromarray(
            np.maximum(np.asarray(masks[destination]), np.asarray(cell_mask)).astype(np.uint8),
            mode="L",
        )

    if cue_mode == "text":
        cue = "Find the character with a striped torso, round glasses, a pointed hat, and blue lower clothing."
    elif cue_mode == "target_exemplar":
        cue = "Use the reference character in the left panel as the visual search target."
    else:
        cue = "The left panel is a deliberately incorrect visual cue. Find the four-feature target described here: striped torso, round glasses, pointed hat, and blue lower clothing."
    prompt = (
        f"{cue} Search only the right scene. Candidate boxes are labeled 0, 1, 2, and 3. "
        "Which candidate contains the target? Answer exactly candidate=N."
    )
    answers = tuple(f"candidate={index}" for index in range(4))
    return SearchProbe(
        image=composite,
        prompt=prompt,
        answer=answers[target_candidate],
        candidate_answers=answers,
        masks=masks,
        target_candidate=target_candidate,
    )


def build_presence_probe(
    row: dict[str, Any],
    *,
    cue_mode: str,
    exemplars: dict[str, tuple[Image.Image, Image.Image]],
) -> SearchProbe:
    if cue_mode not in CUE_MODES:
        raise ValueError(f"unknown cue mode: {cue_mode}")
    scene = Image.open(row["image_path"]).convert("RGB")
    if scene.size != (400, 400):
        scene = scene.resize((400, 400), Image.Resampling.BICUBIC)
    composite = Image.new("RGB", COMPOSITE_SIZE, (245, 245, 242))
    composite.paste(scene, SCENE_OFFSET)
    draw = ImageDraw.Draw(composite)
    draw.rectangle((150, 10, 570, 430), outline=(35, 35, 35), width=2)
    draw.text((164, 2), "SEARCH SCENE", fill=(20, 20, 20))
    draw.text((18, 15), "SEARCH CUE", fill=(20, 20, 20))
    masks = {
        name: Image.new("L", COMPOSITE_SIZE, 0)
        for name in ("reference", "scene", "target_object", "target_candidate", "distractor_candidates")
    }
    masks["scene"].paste(Image.new("L", (400, 400), 255), SCENE_OFFSET)
    target_mask_path = row["masks"].get("target")
    if target_mask_path:
        masks["target_object"].paste(Image.open(target_mask_path).convert("L"), SCENE_OFFSET)
    if cue_mode == "text":
        for index, line in enumerate(("striped torso", "round glasses", "pointed hat", "blue lower body")):
            draw.text((15, 85 + 32 * index), line, fill=(30, 30, 30))
        cue = "Look for the character with a striped torso, round glasses, a pointed hat, and blue lower clothing."
    else:
        exemplar, exemplar_mask = exemplars[cue_mode]
        exemplar.thumbnail((105, 170), Image.Resampling.LANCZOS)
        exemplar_mask = exemplar_mask.resize(exemplar.size, Image.Resampling.NEAREST)
        paste_xy = ((140 - exemplar.width) // 2, 95)
        composite.paste(exemplar, paste_xy)
        masks["reference"].paste(exemplar_mask, paste_xy)
        if cue_mode == "target_exemplar":
            cue = "Use the reference character in the left panel as the visual search target."
        else:
            cue = "The left panel is an incorrect control cue. Look instead for the striped-torso, round-glasses, pointed-hat character with blue lower clothing."
    prompt = f"{cue} Search only the right scene. Is the target present? Answer exactly present or absent."
    answer = "present" if row.get("target_present") else "absent"
    return SearchProbe(
        image=composite,
        prompt=prompt,
        answer=answer,
        candidate_answers=("present", "absent"),
        masks=masks,
        target_candidate=-1,
    )


def visual_region_indices(inputs: Any, runtime: Any, image_positions: list[int], masks: dict[str, Image.Image]) -> dict[str, list[int]]:
    grid = inputs["image_grid_thw"][0].detach().cpu().tolist()
    merge = int(runtime.model.config.vision_config.spatial_merge_size)
    result: dict[str, list[int]] = {}
    for name, mask in masks.items():
        local, _ = pixel_mask_to_visual_tokens(mask, grid, spatial_merge_size=merge)
        if len(local) != len(image_positions):
            raise RuntimeError(f"{name} token mask has {len(local)} entries; expected {len(image_positions)}")
        result[name] = [image_positions[index] for index in np.flatnonzero(local)]
    if not result["target_candidate"] or not result["target_object"]:
        raise RuntimeError("target ROI mapped to no visual tokens")
    return result


def attention_density(attention: Any, positions: list[int], *, total_image_tokens: int) -> float:
    if not positions:
        return 0.0
    mass = float(attention[..., positions].sum(dim=-1).mean().detach().cpu())
    return mass / (len(positions) / total_image_tokens)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
