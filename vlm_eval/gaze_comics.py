from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


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


def _find_panel_file(panel_dir: Path, panel_index: int) -> Path | None:
    stem = f"p{panel_index}"
    for suffix in IMAGE_SUFFIXES:
        candidate = panel_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


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
