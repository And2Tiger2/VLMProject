from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


COLORS = {
    "blue": (48, 126, 190),
    "red": (216, 91, 92),
    "green": (105, 158, 77),
    "purple": (132, 94, 194),
    "orange": (230, 159, 75),
    "gray": (100, 100, 100),
    "light_grid": (224, 224, 224),
    "text": (34, 34, 34),
}


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "two_pass_unified").mkdir(parents=True, exist_ok=True)
    (REPORTS / "attention_alpha_sweep").mkdir(parents=True, exist_ok=True)
    two_pass = _read_tsv(ROOT / "runs/vlmbias_two_pass_unified/summary_aggregate.tsv")
    vanilla_image = _read_json(ROOT / "runs/qwen25vl_3b_vlmbias_400_temp07_image.summary.json")
    vanilla_no_image = _read_json(ROOT / "runs/qwen25vl_3b_vlmbias_400_temp07_no_image.summary.json")
    vlmbias_attention = _read_tsv(ROOT / "runs/attention_alpha_sweep/vlmbias_summary_aggregate.tsv")
    naturalbench_attention = _read_tsv(ROOT / "runs/attention_alpha_sweep/naturalbench_summary_aggregate.tsv")
    vlmbias_attention_ci = _attention_ci_from_runs(ROOT / "runs/attention_alpha_sweep/vlmbias_summary_by_seed.tsv")
    naturalbench_attention_ci = _attention_ci_from_runs(ROOT / "runs/attention_alpha_sweep/naturalbench_summary_by_seed.tsv")

    draw_two_pass(two_pass, vanilla_image, vanilla_no_image, REPORTS / "two_pass_unified/results.png")
    draw_attention_vlmbias(vlmbias_attention, vlmbias_attention_ci, REPORTS / "attention_alpha_sweep/vlmbias.png")
    draw_attention_naturalbench(
        naturalbench_attention,
        naturalbench_attention_ci,
        REPORTS / "attention_alpha_sweep/naturalbench.png",
    )


def draw_two_pass(
    rows: list[dict[str, str]],
    vanilla_image: dict,
    vanilla_no_image: dict,
    out_path: Path,
) -> None:
    width, height = 1700, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = _font(30, bold=True)
    font = _font(17)
    small = _font(14)
    tiny = _font(12)

    draw.text((width / 2, 30), "VLMBias Unified Two-Pass Prompt Experiment", font=title, fill=COLORS["text"], anchor="ma")
    draw.text(
        (width / 2, 70),
        "Vanilla references* plus five prompt conditions x image-again / description-only",
        font=font,
        fill=(70, 70, 70),
        anchor="ma",
    )

    rows_by_condition = {}
    for row in rows:
        rows_by_condition.setdefault(row["description_prompt_variant"], {})[row["answer_mode"]] = row
    rows_by_condition["vanilla*"] = {
        "image_again": _reference_row(vanilla_image),
        "description_only": _reference_row(vanilla_no_image),
    }
    condition_order = ["vanilla*", "direct", "original", "counterfactual", "structured", "verification"]

    _draw_metric_panel(
        draw,
        rows_by_condition,
        condition_order,
        metric="accuracy_mean",
        ci_metric="accuracy_std",
        title="Accuracy (higher is better)",
        box=(90, 135, 800, 855),
        max_value=0.50,
        colors={"image_again": COLORS["blue"], "description_only": COLORS["green"]},
        font=font,
        small=small,
        tiny=tiny,
    )
    _draw_metric_panel(
        draw,
        rows_by_condition,
        condition_order,
        metric="bias_aligned_fraction_mean",
        ci_metric="bias_aligned_fraction_std",
        title="Bias-aligned fraction (lower is better)",
        box=(890, 135, 1600, 855),
        max_value=0.50,
        colors={"image_again": COLORS["red"], "description_only": COLORS["purple"]},
        font=font,
        small=small,
        tiny=tiny,
    )
    _legend(
        draw,
        [
            ("image / image-again", COLORS["blue"]),
            ("no-image* / description-only", COLORS["green"]),
            ("image bias", COLORS["red"]),
            ("no-image* / desc-only bias", COLORS["purple"]),
        ],
        x=90,
        y=880,
        font=font,
    )
    draw.text(
        (90, 930),
        "* Vanilla references are earlier temp=0.7 seed-0 runs and have no seed CI. Two-pass error bars are 95% CI over seeds 0-4.",
        font=small,
        fill=(70, 70, 70),
    )
    image.save(out_path)


def draw_attention_vlmbias(
    rows: list[dict[str, str]],
    ci_by_condition: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    width, height = 1700, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = _font(30, bold=True)
    font = _font(17)
    small = _font(14)
    tiny = _font(12)
    draw.text((width / 2, 30), "VLMBias Image-Attention Alpha Sweep", font=title, fill=COLORS["text"], anchor="ma")
    draw.text(
        (width / 2, 70),
        "Attention logit boost added to image-token keys before softmax; seed 0",
        font=font,
        fill=(70, 70, 70),
        anchor="ma",
    )
    _draw_attention_panel(
        draw,
        rows,
        ci_by_condition,
        metric="bias_aligned_fraction_mean",
        ci_metric="bias_aligned_fraction",
        title="Bias-aligned fraction (lower is better)",
        box=(90, 135, 800, 850),
        y_max=0.32,
        font=font,
        small=small,
        tiny=tiny,
    )
    _draw_attention_panel(
        draw,
        rows,
        ci_by_condition,
        metric="mean_image_attention_mass_mean",
        ci_metric="mean_image_attention_mass",
        title="Mean image attention mass",
        box=(890, 135, 1600, 850),
        y_max=0.60,
        font=font,
        small=small,
        tiny=tiny,
    )
    _legend(
        draw,
        [("last25", COLORS["blue"]), ("last50", COLORS["green"]), ("all", COLORS["red"]), ("baseline", COLORS["gray"])],
        x=90,
        y=890,
        font=font,
    )
    draw.text(
        (90, 930),
        "Error bars are within-run 95% intervals over examples; attention sweep currently has seed 0 only.",
        font=small,
        fill=(70, 70, 70),
    )
    image.save(out_path)


def draw_attention_naturalbench(
    rows: list[dict[str, str]],
    ci_by_condition: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    width, height = 1700, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = _font(30, bold=True)
    font = _font(17)
    small = _font(14)
    tiny = _font(12)
    draw.text((width / 2, 30), "NaturalBench Sanity Check: Attention Alpha Sweep", font=title, fill=COLORS["text"], anchor="ma")
    draw.text(
        (width / 2, 70),
        "Checks whether image-attention boosting damages or improves general VLM behavior; seed 0",
        font=font,
        fill=(70, 70, 70),
        anchor="ma",
    )
    _draw_attention_panel(
        draw,
        rows,
        ci_by_condition,
        metric="Acc_mean",
        ci_metric="Acc",
        title="NaturalBench call accuracy",
        box=(90, 135, 800, 850),
        y_max=0.62,
        font=font,
        small=small,
        tiny=tiny,
    )
    _draw_attention_panel(
        draw,
        rows,
        ci_by_condition,
        metric="G_Acc_mean",
        ci_metric="G_Acc",
        title="NaturalBench group accuracy",
        box=(890, 135, 1600, 850),
        y_max=0.10,
        font=font,
        small=small,
        tiny=tiny,
    )
    _legend(
        draw,
        [("last25", COLORS["blue"]), ("last50", COLORS["green"]), ("all", COLORS["red"]), ("baseline", COLORS["gray"])],
        x=90,
        y=890,
        font=font,
    )
    draw.text(
        (90, 930),
        "Error bars are within-run 95% intervals over calls/groups; attention sweep currently has seed 0 only.",
        font=small,
        fill=(70, 70, 70),
    )
    image.save(out_path)


def _draw_metric_panel(
    draw: ImageDraw.ImageDraw,
    rows_by_condition: dict[str, dict[str, dict[str, str]]],
    condition_order: list[str],
    *,
    metric: str,
    ci_metric: str,
    title: str,
    box: tuple[int, int, int, int],
    max_value: float,
    colors: dict[str, tuple[int, int, int]],
    font,
    small,
    tiny,
) -> None:
    x0, y0, x1, y1 = box
    draw.text((x0, y0 - 30), title, font=font, fill=COLORS["text"], anchor="la")
    plot_top, plot_bottom = y0 + 25, y1 - 95
    plot_left, plot_right = x0 + 65, x1 - 20
    _axes(draw, plot_left, plot_top, plot_right, plot_bottom, max_value, small)
    group_width = (plot_right - plot_left) / len(condition_order)
    bar_width = min(32, group_width * 0.26)
    for idx, condition in enumerate(condition_order):
        center = plot_left + group_width * (idx + 0.5)
        for offset, mode in [(-0.58, "image_again"), (0.58, "description_only")]:
            value = float(rows_by_condition[condition][mode][metric])
            ci = _seed_ci(rows_by_condition[condition][mode], ci_metric)
            bx0 = center + offset * bar_width - bar_width / 2
            bx1 = bx0 + bar_width
            by0 = plot_bottom - (value / max_value) * (plot_bottom - plot_top)
            draw.rectangle((bx0, by0, bx1, plot_bottom), fill=colors[mode])
            _draw_error_bar(draw, bx0 + bar_width / 2, value, ci, max_value, plot_top, plot_bottom)
            draw.text((bx0 + bar_width / 2, by0 - 14), f"{value:.3f}", font=tiny, fill=COLORS["text"], anchor="mm")
        draw.text((center, plot_bottom + 42), condition, font=tiny, fill=COLORS["text"], anchor="mm")


def _draw_attention_panel(
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, str]],
    ci_by_condition: dict[str, dict[str, float]],
    *,
    metric: str,
    ci_metric: str,
    title: str,
    box: tuple[int, int, int, int],
    y_max: float,
    font,
    small,
    tiny,
) -> None:
    x0, y0, x1, y1 = box
    draw.text((x0, y0 - 30), title, font=font, fill=COLORS["text"], anchor="la")
    plot_top, plot_bottom = y0 + 25, y1 - 80
    plot_left, plot_right = x0 + 65, x1 - 20
    _axes(draw, plot_left, plot_top, plot_right, plot_bottom, y_max, small)

    baseline = next(row for row in rows if row["layer_selection"] == "baseline")
    baseline_value = float(baseline[metric]) if baseline.get(metric) else 0.0
    by_layer = {"last25": [], "last50": [], "all": []}
    for row in rows:
        layer = row["layer_selection"]
        if layer in by_layer:
            by_layer[layer].append((float(row["attention_alpha"]), float(row[metric])))
    alphas = sorted({alpha for values in by_layer.values() for alpha, _ in values})
    min_log, max_log = math.log10(min(alphas)), math.log10(max(alphas))

    def x_pos(alpha: float) -> float:
        return plot_left + ((math.log10(alpha) - min_log) / (max_log - min_log)) * (plot_right - plot_left)

    def y_pos(value: float) -> float:
        return plot_bottom - (value / y_max) * (plot_bottom - plot_top)

    draw.line((plot_left, y_pos(baseline_value), plot_right, y_pos(baseline_value)), fill=COLORS["gray"], width=2)
    draw.text((plot_right - 5, y_pos(baseline_value) - 12), f"baseline {baseline_value:.3f}", font=tiny, fill=COLORS["gray"], anchor="ra")

    line_colors = {"last25": COLORS["blue"], "last50": COLORS["green"], "all": COLORS["red"]}
    for layer, values in by_layer.items():
        values = sorted(values)
        points = [(x_pos(alpha), y_pos(value)) for alpha, value in values]
        if len(points) >= 2:
            draw.line(points, fill=line_colors[layer], width=3)
        for (alpha, value), point in zip(values, points):
            x, y = point
            condition = f"alpha{_alpha_label(alpha)}_{layer}"
            ci = ci_by_condition.get(condition, {}).get(ci_metric, 0.0)
            _draw_error_bar(draw, x, value, ci, y_max, plot_top, plot_bottom)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=line_colors[layer])
            if alpha in {0.25, 1.0, 3.0}:
                draw.text((x, y - 17), f"{value:.3f}", font=tiny, fill=COLORS["text"], anchor="mm")

    for alpha in alphas:
        x = x_pos(alpha)
        draw.text((x, plot_bottom + 28), f"{alpha:g}", font=tiny, fill=COLORS["text"], anchor="mm")
    draw.text(((plot_left + plot_right) / 2, plot_bottom + 58), "alpha (log scale)", font=small, fill=COLORS["text"], anchor="mm")


def _axes(draw, x0: int, y0: int, x1: int, y1: int, y_max: float, font) -> None:
    draw.line((x0, y1, x1, y1), fill=(20, 20, 20), width=2)
    draw.line((x0, y0, x0, y1), fill=(150, 150, 150), width=1)
    for tick in range(6):
        value = y_max * tick / 5
        y = y1 - (value / y_max) * (y1 - y0)
        draw.line((x0, y, x1, y), fill=COLORS["light_grid"], width=1)
        draw.text((x0 - 12, y), f"{value:.2f}", font=font, fill=(70, 70, 70), anchor="rm")


def _draw_error_bar(
    draw: ImageDraw.ImageDraw,
    x: float,
    value: float,
    ci: float,
    y_max: float,
    plot_top: float,
    plot_bottom: float,
) -> None:
    if ci <= 0:
        return
    y_low = plot_bottom - (max(0.0, value - ci) / y_max) * (plot_bottom - plot_top)
    y_high = plot_bottom - (min(y_max, value + ci) / y_max) * (plot_bottom - plot_top)
    draw.line((x, y_high, x, y_low), fill=(30, 30, 30), width=2)
    draw.line((x - 6, y_high, x + 6, y_high), fill=(30, 30, 30), width=2)
    draw.line((x - 6, y_low, x + 6, y_low), fill=(30, 30, 30), width=2)


def _legend(draw, items: list[tuple[str, tuple[int, int, int]]], *, x: int, y: int, font) -> None:
    cursor = x
    for label, color in items:
        draw.rectangle((cursor, y, cursor + 18, y + 18), fill=color)
        draw.text((cursor + 26, y - 1), label, font=font, fill=COLORS["text"])
        cursor += 250


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _attention_ci_from_runs(summary_path: Path) -> dict[str, dict[str, float]]:
    ci_by_condition: dict[str, dict[str, float]] = {}
    for row in _read_tsv(summary_path):
        condition = row["condition"]
        raw_rows = _read_jsonl(Path(row["out_path"]))
        if not raw_rows:
            continue
        if row["benchmark"] == "vlmbias":
            ci_by_condition[condition] = {
                "bias_aligned_fraction": _proportion_ci(
                    sum(bool(raw_row["is_bias_aligned_error"]) for raw_row in raw_rows),
                    len(raw_rows),
                ),
                "mean_image_attention_mass": _mean_ci(
                    [_attention_metric(raw_row, "mean_image_attention_mass") for raw_row in raw_rows]
                ),
            }
        elif row["benchmark"] == "naturalbench":
            groups: dict[str, list[dict]] = {}
            for raw_row in raw_rows:
                groups.setdefault(raw_row["group_id"], []).append(raw_row)
            group_correct = [
                int(len(group_rows) == 4 and all(bool(group_row["is_correct"]) for group_row in group_rows))
                for group_rows in groups.values()
            ]
            ci_by_condition[condition] = {
                "Acc": _proportion_ci(sum(bool(raw_row["is_correct"]) for raw_row in raw_rows), len(raw_rows)),
                "G_Acc": _proportion_ci(sum(group_correct), len(group_correct)),
            }
    return ci_by_condition


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _attention_metric(row: dict, key: str) -> float | None:
    value = (((row.get("metadata") or {}).get("generation") or {}).get("attention") or {}).get(key)
    return float(value) if value is not None else None


def _proportion_ci(successes: int, total: int) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    return 1.96 * math.sqrt((p * (1 - p)) / total)


def _mean_ci(values: list[float | None]) -> float:
    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        return 0.0
    avg = sum(clean) / len(clean)
    variance = sum((value - avg) ** 2 for value in clean) / (len(clean) - 1)
    return 1.96 * math.sqrt(variance / len(clean))


def _seed_ci(row: dict[str, str], std_key: str) -> float:
    if not row.get(std_key) or not row.get("runs"):
        return 0.0
    runs = int(row["runs"])
    if runs <= 1:
        return 0.0
    return 2.776 * float(row[std_key]) / math.sqrt(runs)


def _reference_row(summary: dict) -> dict[str, str]:
    return {
        "accuracy_mean": str(summary["accuracy"]),
        "bias_aligned_fraction_mean": str(summary["bias_aligned_fraction"]),
        "bias_aligned_error_rate_mean": str(summary["bias_aligned_error_rate"]),
        "error_rate_mean": str(summary["error_rate"]),
    }


def _alpha_label(alpha: float) -> str:
    return f"{alpha:g}".replace("-", "m").replace(".", "p")


def _font(size: int, *, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
