from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
from typing import Any, Iterable

from PIL import Image, ImageDraw


SYNDOT_PROMPT = "What is the number of the black dots in the image? Answer with the number only."
SEARCH_COLORS = {
    "red": (220, 45, 45),
    "green": (45, 165, 80),
    "blue": (45, 100, 220),
    "purple": (145, 70, 190),
    "gray": (125, 125, 125),
    "black": (25, 25, 25),
}
SEARCH_SHAPES = ("L", "T", "H", "E", "F", "Γ")
SEARCH_TRAIN_CONJUNCTIONS = (
    ("red", "L"),
    ("green", "T"),
    ("blue", "H"),
    ("purple", "E"),
    ("gray", "F"),
    ("black", "Γ"),
)
SEARCH_TEST_CONJUNCTIONS = (
    ("gray", "L"),
    ("green", "E"),
    ("green", "L"),
    ("red", "E"),
    ("red", "F"),
    ("gray", "T"),
    ("blue", "F"),
    ("black", "H"),
    ("blue", "E"),
    ("red", "H"),
)


def stable_seed(seed: int, *parts: Any) -> int:
    payload = ":".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def syndot_positions(seed: int, example_id: str, *, maximum: int = 10) -> list[tuple[int, int]]:
    rng = random.Random(stable_seed(seed, "syndot", example_id))
    grid = [(14 + 12 * x, 14 + 12 * y) for y in range(28) for x in range(28)]
    return rng.sample(grid, maximum)


def render_syndot(count: int, positions: list[tuple[int, int]]) -> Image.Image:
    if not 1 <= count <= 10:
        raise ValueError("SynDot count must be in 1..10")
    image = Image.new("RGB", (336, 336), "white")
    draw = ImageDraw.Draw(image)
    for x, y in positions[:count]:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="black")
    return image


def render_syndot_mask(count: int, positions: list[tuple[int, int]]) -> Image.Image:
    mask = Image.new("L", (336, 336), 0)
    draw = ImageDraw.Draw(mask)
    for x, y in positions[:count]:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=255)
    return mask


@dataclass(frozen=True)
class RenderedScene:
    image: Image.Image
    masks: dict[str, Image.Image]
    objects: list[dict[str, Any]]


def fixed_eight_scene(
    *,
    seed: int,
    scene_id: str,
    red_count: int,
    variant: str = "color",
    relocate: bool = False,
) -> RenderedScene:
    """Render constant-total scenes with an exact one-object category change."""

    if red_count not in {4, 5}:
        raise ValueError("constant-complexity red_count must be 4 or 5")
    rng = random.Random(stable_seed(seed, "constant-eight", scene_id))
    image = Image.new("RGB", (336, 336), "white")
    positions = rng.sample(
        [(42 + 42 * x, 42 + 42 * y) for y in range(7) for x in range(7)], 8
    )
    if relocate:
        # Index four is the only object whose category differs between the
        # 4-target and 5-target scenes. Relocate that causal edit—not an
        # unrelated already-target object—so the position-transfer control
        # actually moves the count-changing evidence.
        positions[4] = next(
            point
            for point in reversed([(42 + 42 * x, 42 + 42 * y) for y in range(7) for x in range(7)])
            if point not in positions
        )
    masks = {
        name: Image.new("L", image.size, 0)
        for name in ("target", "distractor", "changed_pixel")
    }
    objects: list[dict[str, Any]] = []
    for index, (x, y) in enumerate(positions):
        is_target = index < red_count
        if variant == "color":
            color = SEARCH_COLORS["red" if is_target else "blue"]
            shape = "circle"
        elif variant == "shape":
            color = SEARCH_COLORS["green"]
            shape = "circle" if is_target else "square"
        else:
            raise ValueError(f"unknown fixed-eight variant: {variant}")
        _draw_shape(ImageDraw.Draw(image), shape, x, y, 12, fill=color)
        mask_name = "target" if is_target else "distractor"
        _draw_shape(ImageDraw.Draw(masks[mask_name]), shape, x, y, 12, fill=255)
        if index == 4:
            _draw_shape(ImageDraw.Draw(masks["changed_pixel"]), shape, x, y, 12, fill=255)
        objects.append(
            {
                "index": index,
                "center": [x, y],
                "box": [x - 12, y - 12, x + 12, y + 12],
                "class": f"{('target' if is_target else 'distractor')}-{shape}",
            }
        )
    return RenderedScene(image=image, masks=masks, objects=objects)


def render_search_scene(
    *,
    seed: int,
    scene_id: str,
    target_color: str,
    target_shape: str,
    target_count: int,
    n_objects: int = 50,
    size: int = 336,
) -> RenderedScene:
    if target_color not in SEARCH_COLORS or target_shape not in SEARCH_SHAPES:
        raise ValueError("unknown search target conjunction")
    if not 0 <= target_count <= n_objects:
        raise ValueError("target_count must be within scene size")
    rng = random.Random(stable_seed(seed, "point-search", scene_id))
    image = Image.new("RGB", (size, size), (245, 245, 242))
    target_mask = Image.new("L", image.size, 0)
    distractor_mask = Image.new("L", image.size, 0)
    centers = _sample_centers(rng, n_objects, size=size, margin=14, min_distance=26)
    objects: list[dict[str, Any]] = []
    other_colors = [name for name in SEARCH_COLORS if name != target_color]
    other_shapes = [name for name in SEARCH_SHAPES if name != target_shape]
    for index, (x, y) in enumerate(centers):
        if index < target_count:
            color_name, shape = target_color, target_shape
            kind = "target"
        else:
            distractor_type = index % 3
            if distractor_type == 0:
                color_name, shape = target_color, rng.choice(other_shapes)
                kind = "same-color"
            elif distractor_type == 1:
                color_name, shape = rng.choice(other_colors), target_shape
                kind = "same-shape"
            else:
                color_name, shape = rng.choice(other_colors), rng.choice(other_shapes)
                kind = "neither"
        _draw_shape(ImageDraw.Draw(image), shape, x, y, 8, fill=SEARCH_COLORS[color_name])
        target = target_mask if kind == "target" else distractor_mask
        _draw_shape(ImageDraw.Draw(target), shape, x, y, 9, fill=255)
        objects.append(
            {
                "index": index,
                "center": [x, y],
                "normalized_center": [round(x / (size - 1), 6), round(y / (size - 1), 6)],
                "box": [x - 9, y - 9, x + 9, y + 9],
                "color": color_name,
                "shape": shape,
                "class": kind,
            }
        )
    return RenderedScene(
        image=image,
        masks={"target": target_mask, "distractor": distractor_mask},
        objects=objects,
    )


WALDO_FEATURES = ("striped_torso", "round_glasses", "pointed_hat", "blue_lower")


def length_matched_nonspatial_answer(tokenizer: Any, *, direct_answer: str, point_answer: str) -> str:
    """Return non-spatial text with exactly the point answer's token count."""

    target = len(tokenizer(point_answer, add_special_tokens=False).input_ids)
    if target < 1:
        raise ValueError("point answer tokenized to an empty sequence")
    bases = [f"answer={direct_answer}", str(direct_answer)]
    neutral_words = (" evidence", " seen", " object", " result", " neutral")
    for base in bases:
        for word in neutral_words:
            for repeats in range(target + 1):
                candidate = base + word * repeats
                ids = tokenizer(candidate, add_special_tokens=False).input_ids
                if len(ids) == target and "(" not in candidate and ")" not in candidate:
                    return candidate
                if len(ids) > target:
                    break
    raise RuntimeError(
        f"could not build a non-spatial {target}-token answer for {direct_answer!r}"
    )


def point_condition_prompt(row: dict[str, Any], condition: str) -> str:
    """Select the declared output-format prompt for a matched search scene."""

    prompts = row.get("prompts")
    if not isinstance(prompts, dict):
        return str(row["prompt"])
    key = {
        "base": "direct",
        "direct_answer": "direct",
        "direct_length_matched": "direct_length_matched",
        "point_answer": "point",
        "shuffled_point_answer": "point",
    }.get(condition)
    if key is None or key not in prompts:
        raise ValueError(f"unknown or undeclared point-search condition: {condition}")
    return str(prompts[key])


def render_waldo_like_scene(
    *,
    seed: int,
    scene_id: str,
    target_present: bool,
    clutter: int = 24,
    similarity: int = 2,
    target_cell: int | None = None,
    occluded: bool = False,
    size: int = 400,
    target_scale: float = 1.0,
    background: tuple[int, int, int] = (225, 235, 220),
    distractor_centers: list[tuple[int, int]] | None = None,
    similarity_overrides: dict[int, int] | None = None,
    scene_zoom: float = 1.0,
) -> RenderedScene:
    """Render an original four-feature conjunction-search character scene."""

    if similarity not in {1, 2, 3}:
        raise ValueError("similarity must be one, two, or three shared features")
    rng = random.Random(stable_seed(seed, "waldo-like", scene_id))
    image = Image.new("RGB", (size, size), background)
    masks = {"target": Image.new("L", image.size, 0)}
    objects: list[dict[str, Any]] = []
    if target_cell is None:
        target_cell = rng.randrange(100)
    target_center = (
        (target_cell % 10) * (size // 10) + size // 20,
        (target_cell // 10) * (size // 10) + size // 20,
    )
    if distractor_centers is None:
        distractor_centers = _sample_centers(
            rng, clutter, size=size, margin=18, min_distance=28, forbidden=[target_center]
        )
    elif len(distractor_centers) != clutter:
        raise ValueError(
            f"expected {clutter} distractor centers, received {len(distractor_centers)}"
        )
    centers = [target_center, *distractor_centers]
    for index, center in enumerate(centers):
        is_target = index == 0 and target_present
        incorrect_binding = False
        if is_target:
            features = set(WALDO_FEATURES)
            class_name = "target"
        else:
            object_similarity = int((similarity_overrides or {}).get(index, similarity))
            if object_similarity not in {1, 2, 3}:
                raise ValueError("per-object similarity must be one, two, or three")
            # Feature choice is keyed by object index, so changing whether the
            # target is present cannot silently change every later distractor.
            feature_rng = random.Random(
                stable_seed(seed, "waldo-like", scene_id, "features", index)
            )
            feature_order = list(WALDO_FEATURES)
            feature_rng.shuffle(feature_order)
            features = set(feature_order[:object_similarity])
            class_name = f"distractor-shared-{object_similarity}"
        if not is_target and index == 1 and object_similarity == 3:
            features = set(WALDO_FEATURES)
            class_name = "distractor-incorrect-binding"
            incorrect_binding = True
        mask = masks.setdefault(
            f"object_{index:03d}", Image.new("L", image.size, 0)
        )
        scale = target_scale if is_target else 1.0
        _draw_character(image, mask, center, features, occluded=is_target and occluded, scale=scale, binding_correct=not incorrect_binding)
        if is_target:
            masks["target"] = mask.copy()
        x, y = center
        objects.append(
            {
                "index": index,
                "center": [x, y],
                "box": [x - int(round(12 * scale)), y - int(round(29 * scale)), x + int(round(12 * scale)), y + int(round(18 * scale))],
                "cell": int((y // (size // 10)) * 10 + x // (size // 10)),
                "class": class_name,
                "features": sorted(features),
                "binding_correct": not incorrect_binding,
            }
        )
    if scene_zoom <= 0:
        raise ValueError("scene zoom must be positive")
    if scene_zoom != 1.0:
        image, masks, objects = _zoom_scene(
            image,
            masks,
            objects,
            zoom=float(scene_zoom),
            background=background,
        )
    return RenderedScene(image=image, masks=masks, objects=objects)


def _zoom_scene(
    image: Image.Image,
    masks: dict[str, Image.Image],
    objects: list[dict[str, Any]],
    *,
    zoom: float,
    background: tuple[int, int, int],
) -> tuple[Image.Image, dict[str, Image.Image], list[dict[str, Any]]]:
    width, height = image.size
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2
    inverse = (
        1.0 / zoom,
        0.0,
        center_x - center_x / zoom,
        0.0,
        1.0 / zoom,
        center_y - center_y / zoom,
    )
    zoomed_image = image.transform(
        image.size,
        Image.Transform.AFFINE,
        inverse,
        resample=Image.Resampling.BICUBIC,
        fillcolor=background,
    )
    zoomed_masks = {
        name: mask.transform(
            mask.size,
            Image.Transform.AFFINE,
            inverse,
            resample=Image.Resampling.NEAREST,
            fillcolor=0,
        )
        for name, mask in masks.items()
    }

    def coordinate(value: float, center: float, maximum: int) -> int:
        return max(0, min(maximum, int(round(center + (value - center) * zoom))))

    zoomed_objects = []
    for value in objects:
        row = dict(value)
        x, y = value["center"]
        new_x = coordinate(x, center_x, width - 1)
        new_y = coordinate(y, center_y, height - 1)
        x0, y0, x1, y1 = value["box"]
        row["center"] = [new_x, new_y]
        row["box"] = [
            coordinate(x0, center_x, width - 1),
            coordinate(y0, center_y, height - 1),
            coordinate(x1, center_x, width - 1),
            coordinate(y1, center_y, height - 1),
        ]
        row["cell"] = (new_y // 40) * 10 + new_x // 40
        row["scene_zoom"] = zoom
        zoomed_objects.append(row)
    return zoomed_image, zoomed_masks, zoomed_objects


def waldo_distractor_centers(
    *,
    seed: int,
    scene_id: str,
    clutter: int,
    forbidden_cells: list[int],
    size: int = 400,
) -> list[tuple[int, int]]:
    """Return deterministic distractor locations shared by matched scenes."""
    rng = random.Random(stable_seed(seed, "waldo-like", scene_id, "matched-centers"))
    forbidden = set(int(cell) for cell in forbidden_cells)
    if any(cell < 0 or cell >= 100 for cell in forbidden):
        raise ValueError("forbidden Waldo cells must lie in 0..99")
    cell_size = size // 10
    centers: list[tuple[int, int]] = []
    attempts = 0
    while len(centers) < clutter and attempts < clutter * 1000:
        attempts += 1
        point = (rng.randint(18, size - 18), rng.randint(18, size - 18))
        cell = (point[1] // cell_size) * 10 + point[0] // cell_size
        if cell in forbidden:
            continue
        if all(math.dist(point, previous) >= 28 for previous in centers):
            centers.append(point)
    if len(centers) != clutter:
        raise RuntimeError(f"could place only {len(centers)}/{clutter} Waldo distractors")
    return centers


def _sample_centers(
    rng: random.Random,
    count: int,
    *,
    size: int,
    margin: int,
    min_distance: int,
    forbidden: Iterable[tuple[int, int]] = (),
) -> list[tuple[int, int]]:
    centers = list(forbidden)
    added: list[tuple[int, int]] = []
    attempts = 0
    while len(added) < count and attempts < count * 500:
        attempts += 1
        point = (rng.randint(margin, size - margin), rng.randint(margin, size - margin))
        if all(math.dist(point, previous) >= min_distance for previous in centers):
            centers.append(point)
            added.append(point)
    if len(added) != count:
        raise RuntimeError(f"could place only {len(added)}/{count} objects")
    return added


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    shape: str,
    x: int,
    y: int,
    radius: int,
    *,
    fill: Any,
) -> None:
    if shape == "circle":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    elif shape == "square":
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=fill)
    elif shape == "triangle":
        draw.polygon([(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)], fill=fill)
    elif shape == "diamond":
        draw.polygon([(x, y - radius), (x - radius, y), (x, y + radius), (x + radius, y)], fill=fill)
    elif shape == "cross":
        width = max(2, radius // 3)
        draw.rectangle((x - width, y - radius, x + width, y + radius), fill=fill)
        draw.rectangle((x - radius, y - width, x + radius, y + width), fill=fill)
    elif shape == "pentagon":
        points = [
            (
                x + radius * math.cos(-math.pi / 2 + 2 * math.pi * index / 5),
                y + radius * math.sin(-math.pi / 2 + 2 * math.pi * index / 5),
            )
            for index in range(5)
        ]
        draw.polygon(points, fill=fill)
    elif shape in SEARCH_SHAPES:
        # Paper-style visual-search glyphs. Drawing these from rectangles keeps
        # the renderer deterministic and avoids font/version dependencies.
        stroke = max(2, radius // 3)

        def vertical(x_center: int, y0: int, y1: int) -> None:
            draw.rectangle(
                (x_center - stroke, y0, x_center + stroke, y1), fill=fill
            )

        def horizontal(y_center: int, x0: int, x1: int) -> None:
            draw.rectangle(
                (x0, y_center - stroke, x1, y_center + stroke), fill=fill
            )

        left, right = x - radius, x + radius
        top, bottom = y - radius, y + radius
        if shape == "L":
            vertical(left + stroke, top, bottom)
            horizontal(bottom - stroke, left, right)
        elif shape == "T":
            horizontal(top + stroke, left, right)
            vertical(x, top, bottom)
        elif shape == "H":
            vertical(left + stroke, top, bottom)
            vertical(right - stroke, top, bottom)
            horizontal(y, left, right)
        elif shape == "E":
            vertical(left + stroke, top, bottom)
            horizontal(top + stroke, left, right)
            horizontal(y, left, right - stroke)
            horizontal(bottom - stroke, left, right)
        elif shape == "F":
            vertical(left + stroke, top, bottom)
            horizontal(top + stroke, left, right)
            horizontal(y, left, right - stroke)
        elif shape == "Γ":
            horizontal(top + stroke, left, right)
            vertical(left + stroke, top, bottom)
    else:
        raise ValueError(f"unknown shape: {shape}")


def _draw_character(
    image: Image.Image,
    mask: Image.Image,
    center: tuple[int, int],
    features: set[str],
    *,
    occluded: bool,
    scale: float = 1.0,
    binding_correct: bool = True,
) -> None:
    x, y = center
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)
    def s(value: int) -> int: return int(round(value * scale))
    skin = (225, 175, 135)
    torso = (215, 45, 45) if "striped_torso" in features else (45, 165, 80)
    lower = (45, 80, 190) if "blue_lower" in features else (110, 75, 50)
    if not binding_correct and "blue_lower" in features:
        torso, lower = lower, torso
    draw.ellipse((x - s(6), y - s(17), x + s(6), y - s(5)), fill=skin, outline="black")
    draw.rectangle((x - s(8), y - s(5), x + s(8), y + s(8)), fill=torso, outline="black")
    if "striped_torso" in features:
        for offset in (-3, 2, 7):
            draw.line((x - s(7), y + s(offset), x + s(7), y + s(offset)), fill="white", width=max(1, s(2)))
    draw.rectangle((x - s(7), y + s(8), x + s(7), y + s(17)), fill=lower, outline="black")
    if "round_glasses" in features:
        draw.ellipse((x - s(6), y - s(14), x - s(1), y - s(9)), outline="black", width=1)
        draw.ellipse((x + s(1), y - s(14), x + s(6), y - s(9)), outline="black", width=1)
    if "pointed_hat" in features:
        draw.polygon([(x, y - s(29)), (x - s(9), y - s(17)), (x + s(9), y - s(17))], fill=(235, 190, 35), outline="black")
    mask_draw.rectangle((x - s(10), y - s(29), x + s(10), y + s(18)), fill=255)
    if occluded:
        draw.rectangle((x - s(12), y - s(2), x + s(12), y + s(7)), fill=(120, 105, 80))
