from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REFERENCE_ROWS = [
    {
        "condition": "Direct image",
        "group": "Reference",
        "n": "400",
        "accuracy": "0.1425",
        "bias_aligned_fraction": "0.395",
        "bias_aligned_error_rate": "0.4606413994169096",
        "error_rate": "0.8575",
    },
    {
        "condition": "Direct no-image",
        "group": "Reference",
        "n": "400",
        "accuracy": "0.12",
        "bias_aligned_fraction": "0.295",
        "bias_aligned_error_rate": "0.3352272727272727",
        "error_rate": "0.88",
    },
    {
        "condition": "Basic 2-pass image",
        "group": "Reference",
        "n": "400",
        "accuracy": "0.1725",
        "bias_aligned_fraction": "0.38",
        "bias_aligned_error_rate": "0.459214501510574",
        "error_rate": "0.8275",
    },
    {
        "condition": "Basic 2-pass desc-only",
        "group": "Reference",
        "n": "400",
        "accuracy": "0.1875",
        "bias_aligned_fraction": "0.15",
        "bias_aligned_error_rate": "0.18461538461538463",
        "error_rate": "0.8125",
    },
]

METRICS = [
    ("accuracy", "Accuracy", (48, 126, 190)),
    ("bias_aligned_fraction", "Bias-aligned fraction", (216, 91, 92)),
    ("bias_aligned_error_rate", "Bias-aligned error rate", (105, 158, 77)),
]

DISPLAY_NAMES = {
    "neutral_image_again": "neutral image again",
    "neutral_description_only": "neutral description only",
    "counterfactual_image_again": "counterfactual image again",
    "counterfactual_description_only": "counterfactual description only",
    "structured_image_again": "structured image again",
    "structured_description_only": "structured description only",
    "verification_image_again": "verification image again",
    "verification_description_only": "verification description only",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VLMBias two-pass prompt variant comparison artifacts.")
    parser.add_argument(
        "--variant-summary",
        default="reports/vlmbias_two_pass_prompt_variants_fast_summary.tsv",
        help="TSV with the 8 prompt-variant rows.",
    )
    parser.add_argument(
        "--out-tsv",
        default="reports/vlmbias_two_pass_prompt_variants_comparison.tsv",
        help="Combined comparison TSV to write.",
    )
    parser.add_argument(
        "--out-png",
        default="reports/vlmbias_two_pass_prompt_variants_comparison.png",
        help="Comparison chart PNG to write.",
    )
    args = parser.parse_args()

    rows = build_rows(Path(args.variant_summary))
    write_tsv(rows, Path(args.out_tsv))
    draw_chart(rows, Path(args.out_png))


def build_rows(summary_path: Path) -> list[dict[str, str]]:
    rows = [dict(row) for row in REFERENCE_ROWS]
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            row = dict(row)
            row["condition"] = DISPLAY_NAMES.get(row["condition"], row["condition"].replace("_", " "))
            row["group"] = "Prompt variants"
            rows.append(row)
    return rows


def write_tsv(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["condition", "group", "n", "accuracy", "bias_aligned_fraction", "bias_aligned_error_rate", "error_rate"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def draw_chart(rows: list[dict[str, str]], out_path: Path) -> None:
    width, height = 1800, 940
    margin_left, margin_right = 110, 36
    margin_top, margin_bottom = 160, 170
    plot_left, plot_right = margin_left, width - margin_right
    plot_top, plot_bottom = margin_top, height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    max_value = 0.55

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font(17)
    small = _font(13)
    title_font = _font(28, bold=True)
    subtitle_font = _font(15)

    draw.text(
        (width / 2, 28),
        "Qwen2.5-VL-3B on VLMBias: Two-Pass Prompt Variant Comparison",
        fill=(36, 36, 36),
        font=title_font,
        anchor="ma",
    )
    draw.text((370, 72), "Original/reference runs", fill=(70, 70, 70), font=subtitle_font, anchor="mm")
    draw.text((1175, 72), "New fast 8-condition prompt variants", fill=(70, 70, 70), font=subtitle_font, anchor="mm")

    legend_x = margin_left
    for metric_index, (_, label, color) in enumerate(METRICS):
        x = legend_x + metric_index * 300
        draw.rectangle((x, 100, x + 15, 115), fill=color)
        draw.text((x + 22, 97), label, fill=(30, 30, 30), font=font)

    for tick in range(0, 6):
        value = tick / 10
        y = plot_bottom - (value / max_value) * plot_height
        draw.line((plot_left, y, plot_right, y), fill=(224, 224, 224), width=1)
        draw.text((plot_left - 18, y), f"{value:.1f}", fill=(70, 70, 70), font=small, anchor="rm")

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(20, 20, 20), width=3)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(160, 160, 160), width=1)
    draw.text((plot_left, plot_top - 28), "Metric value", fill=(30, 30, 30), font=font, anchor="lm")

    split_x = plot_left + plot_width * (4 / len(rows))
    draw.line((split_x, plot_top, split_x, plot_bottom), fill=(95, 95, 95), width=2)

    group_width = plot_width / len(rows)
    bar_width = min(28, group_width * 0.22)
    for row_index, row in enumerate(rows):
        center_x = plot_left + group_width * (row_index + 0.5)
        for metric_index, (key, _, color) in enumerate(METRICS):
            value = float(row[key])
            x0 = center_x + (metric_index - 1) * bar_width * 1.25 - bar_width / 2
            x1 = x0 + bar_width
            y0 = plot_bottom - (value / max_value) * plot_height
            draw.rectangle((x0, y0, x1, plot_bottom), fill=color)
            draw.text((x0 + bar_width / 2, y0 - 18), f"{value:.3f}", fill=(25, 25, 25), font=small, anchor="mm")
        draw.text(
            (center_x, plot_bottom + 58),
            _short_label(row["condition"]),
            fill=(45, 45, 45),
            font=small,
            anchor="mm",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _short_label(label: str) -> str:
    return label.replace("counterfactual", "counterf.").replace("description only", "desc only")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
