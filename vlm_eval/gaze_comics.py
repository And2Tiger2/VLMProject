from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
DEFAULT_N_PANELS = 6
DEFAULT_TARGET_HEIGHT = 256
DEFAULT_GAP = 6


@dataclass(frozen=True)
class ComicStrip:
    name: str
    strip: Image.Image
    panels: list[Image.Image]
    panel_widths: list[int]
    panel_paths: list[Path]
    target_height: int


def round_to_multiple(value: int, multiple: int = 32) -> int:
    return max(multiple, multiple * round(float(value) / float(multiple)))


def list_comic_dirs(root: Path, n_panels: int = DEFAULT_N_PANELS) -> list[Path]:
    if not root.exists():
        return []
    valid = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and all(_find_panel_file(path, idx) is not None for idx in range(1, n_panels + 1)):
            valid.append(path)
    return valid


def build_strip(
    panel_dir: Path,
    n_panels: int = DEFAULT_N_PANELS,
    target_height: int = DEFAULT_TARGET_HEIGHT,
    gap: int = DEFAULT_GAP,
) -> ComicStrip:
    panel_items = []
    for panel_index in range(1, n_panels + 1):
        panel_path = _find_panel_file(panel_dir, panel_index)
        if panel_path is None:
            raise FileNotFoundError(f"Missing p{panel_index} image under {panel_dir}")
        panel_items.append((panel_path, _open_rgb(panel_path)))

    resized, panel_widths = _resize_panels([image for _, image in panel_items], target_height)
    strip = _assemble_strip(resized, panel_widths, gap)
    return ComicStrip(
        name=panel_dir.name,
        strip=strip,
        panels=resized,
        panel_widths=panel_widths,
        panel_paths=[path for path, _ in panel_items],
        target_height=int(strip.size[1]),
    )


def build_strip_from_paths(
    panel_paths: list[Path],
    *,
    name: str,
    target_height: int = DEFAULT_TARGET_HEIGHT,
    gap: int = DEFAULT_GAP,
) -> ComicStrip:
    if not panel_paths:
        raise ValueError("A comic strip needs at least one panel path.")
    images = [_open_rgb(path) for path in panel_paths]
    resized, panel_widths = _resize_panels(images, target_height)
    strip = _assemble_strip(resized, panel_widths, gap)
    return ComicStrip(
        name=name,
        strip=strip,
        panels=resized,
        panel_widths=panel_widths,
        panel_paths=list(panel_paths),
        target_height=int(strip.size[1]),
    )


def add_panel_number_labels(
    strip: ComicStrip, *, label_height: int = 56
) -> Image.Image:
    """Add a judge-only numbered banner without modifying generation images."""
    if label_height <= 0:
        raise ValueError("label_height must be positive")
    canvas = Image.new(
        "RGB",
        (strip.strip.width, strip.strip.height + label_height),
        (255, 255, 255),
    )
    canvas.paste(strip.strip, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=max(28, int(label_height * 0.65)))
    except TypeError:  # Pillow before load_default(size=...) support.
        font = ImageFont.load_default()
    x_offset = 0
    for panel, width in enumerate(strip.panel_widths, start=1):
        center_x = x_offset + int(width) // 2
        box_half_width = max(24, int(label_height * 0.55))
        draw.rounded_rectangle(
            [
                center_x - box_half_width,
                4,
                center_x + box_half_width,
                label_height - 4,
            ],
            radius=8,
            fill=(0, 0, 0),
        )
        draw.text(
            (center_x, label_height // 2),
            str(panel),
            fill=(255, 255, 255),
            font=font,
            anchor="mm",
        )
        x_offset += int(width)
    return canvas


def sample_raw_comics_windows(
    root: Path,
    *,
    n_panels: int = DEFAULT_N_PANELS,
    n_samples: int = 500,
    seed: int = 42,
) -> list[tuple[str, list[Path]]]:
    """Reproduce the official raw-COMICS sampling protocol.

    Each sample chooses a random comic, then a random page containing at least
    ``n_panels`` panels, then a consecutive within-page window. For comics with
    more than ten pages, the first and last five pages are excluded when
    possible. Sampling is with replacement, matching the reference code.
    """
    samples: list[tuple[str, list[Path]]] = []
    if not root.exists():
        return samples
    comic_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not comic_dirs:
        return samples

    import numpy as np

    rng = np.random.RandomState(seed)
    requested = max(0, int(n_samples))
    for _ in range(requested):
        for _attempt in range(200):
            comic_dir = comic_dirs[int(rng.randint(len(comic_dirs)))]
            by_page = _group_raw_panels_by_page(comic_dir)
            pages = sorted(by_page)
            interior_pages = pages[5:-5] if len(pages) > 10 else pages
            eligible = [page for page in interior_pages if len(by_page[page]) >= n_panels]
            if not eligible:
                eligible = [page for page in pages if len(by_page[page]) >= n_panels]
            if not eligible:
                continue
            page = int(eligible[int(rng.randint(len(eligible)))])
            panel_paths = by_page[page]
            max_start = len(panel_paths) - n_panels
            start = int(rng.randint(max_start + 1)) if max_start > 0 else 0
            window = panel_paths[start : start + n_panels]
            samples.append((f"{comic_dir.name}_page{page}_start{start}", window))
            break
        else:
            raise RuntimeError(f"Could not sample a valid {n_panels}-panel strip from {root}.")
    return samples


def _find_panel_file(panel_dir: Path, panel_index: int) -> Path | None:
    stem = f"p{panel_index}"
    for suffix in IMAGE_SUFFIXES:
        candidate = panel_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _raw_panel_sort_key(path: Path) -> tuple[int, int]:
    numbers = [int(value) for value in re.findall(r"\d+", path.stem)]
    if len(numbers) < 2:
        raise ValueError(f"Raw COMICS panel must be named <page>_<panel>: {path}")
    return numbers[-2], numbers[-1]


def _group_raw_panels_by_page(comic_dir: Path) -> dict[int, list[Path]]:
    grouped: dict[int, list[tuple[int, Path]]] = {}
    for path in sorted(comic_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            page, panel = _raw_panel_sort_key(path)
        except ValueError:
            continue
        grouped.setdefault(page, []).append((panel, path))
    return {
        page: [path for _, path in sorted(items)]
        for page, items in sorted(grouped.items())
    }


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _resize_panels(images: list[Image.Image], target_height: int) -> tuple[list[Image.Image], list[int]]:
    resized = []
    widths = []
    rounded_height = round_to_multiple(target_height)
    for image in images:
        width, height = image.size
        new_width = round_to_multiple(int(width * rounded_height / max(height, 1)))
        resized_image = image.resize((new_width, rounded_height), Image.LANCZOS)
        resized.append(resized_image)
        widths.append(new_width)
    return resized, widths


def _assemble_strip(images: list[Image.Image], panel_widths: list[int], gap: int) -> Image.Image:
    total_width = int(sum(panel_widths))
    target_height = int(images[0].size[1])
    strip = Image.new("RGB", (total_width, target_height), (255, 255, 255))
    x_offset = 0
    for image in images:
        strip.paste(image, (x_offset, 0))
        x_offset += int(image.size[0])

    if gap > 0 and len(panel_widths) > 1:
        draw = ImageDraw.Draw(strip)
        x_offset = 0
        for width in panel_widths[:-1]:
            x_offset += width
            draw.rectangle(
                [x_offset - gap // 2, 0, x_offset + (gap - gap // 2) - 1, target_height - 1],
                fill="white",
            )
    return strip
