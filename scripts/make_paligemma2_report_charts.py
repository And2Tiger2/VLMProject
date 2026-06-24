from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.make_completed_experiment_charts import (
    COLORS,
    _axes,
    _draw_attention_panel,
    _draw_error_bar,
    _draw_metric_panel,
    _font,
    _legend,
    _read_tsv,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> None:
    two_pass_dir = REPORTS / "paligemma2_two_pass_unified"
    attention_dir = REPORTS / "paligemma2_attention_alpha_sweep"
    two_pass_dir.mkdir(parents=True, exist_ok=True)
    attention_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(ROOT / "runs/paligemma2_two_pass_unified/summary_aggregate.tsv", two_pass_dir / "summary_aggregate.tsv")
    shutil.copyfile(ROOT / "runs/paligemma2_two_pass_unified/summary_by_seed.tsv", two_pass_dir / "summary_by_seed.tsv")
    shutil.copyfile(
        ROOT / "runs/paligemma2_attention_alpha_sweep/vlmbias_summary_aggregate.tsv",
        attention_dir / "vlmbias_summary_aggregate.tsv",
    )
    shutil.copyfile(
        ROOT / "runs/paligemma2_attention_alpha_sweep/vlmbias_summary_by_seed.tsv",
        attention_dir / "vlmbias_summary_by_seed.tsv",
    )
    shutil.copyfile(
        ROOT / "runs/paligemma2_attention_alpha_sweep/naturalbench_summary_aggregate.tsv",
        attention_dir / "naturalbench_summary_aggregate.tsv",
    )
    shutil.copyfile(
        ROOT / "runs/paligemma2_attention_alpha_sweep/naturalbench_summary_by_seed.tsv",
        attention_dir / "naturalbench_summary_by_seed.tsv",
    )

    two_pass = _read_tsv(ROOT / "runs/paligemma2_two_pass_unified/summary_aggregate.tsv")
    vlmbias_attention = _read_tsv(ROOT / "runs/paligemma2_attention_alpha_sweep/vlmbias_summary_aggregate.tsv")
    naturalbench_attention = _read_tsv(ROOT / "runs/paligemma2_attention_alpha_sweep/naturalbench_summary_aggregate.tsv")

    draw_paligemma_two_pass(two_pass, two_pass_dir / "results.png")
    draw_paligemma_attention_vlmbias(vlmbias_attention, attention_dir / "vlmbias.png")
    draw_paligemma_attention_naturalbench(naturalbench_attention, attention_dir / "naturalbench.png")
    write_notes(two_pass, vlmbias_attention, naturalbench_attention, two_pass_dir / "notes.md", attention_dir / "notes.md")


def draw_paligemma_two_pass(rows: list[dict[str, str]], out_path: Path) -> None:
    width, height = 1700, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = _font(30, bold=True)
    font = _font(17)
    small = _font(14)
    tiny = _font(12)

    draw.text((width / 2, 30), "PaliGemma2 VLMBias Unified Two-Pass Experiment", font=title, fill=COLORS["text"], anchor="ma")
    draw.text(
        (width / 2, 70),
        "Five first-pass prompt policies x image-again / blank-image description-only; 95% CI over seeds 0-4",
        font=font,
        fill=(70, 70, 70),
        anchor="ma",
    )

    rows_by_condition: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        rows_by_condition.setdefault(row["description_prompt_variant"], {})[row["answer_mode"]] = row
    condition_order = ["direct", "original", "counterfactual", "structured", "verification"]

    _draw_metric_panel(
        draw,
        rows_by_condition,
        condition_order,
        metric="accuracy_mean",
        ci_metric="accuracy_std",
        title="Accuracy (higher is better)",
        box=(90, 135, 800, 855),
        max_value=0.36,
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
        max_value=0.40,
        colors={"image_again": COLORS["red"], "description_only": COLORS["purple"]},
        font=font,
        small=small,
        tiny=tiny,
    )
    _legend(
        draw,
        [
            ("image-again accuracy", COLORS["blue"]),
            ("blank-image desc-only accuracy", COLORS["green"]),
            ("image-again bias", COLORS["red"]),
            ("blank-image desc-only bias", COLORS["purple"]),
        ],
        x=90,
        y=880,
        font=font,
    )
    draw.text(
        (90, 930),
        "PaliGemma requires an image input; description-only uses a blank white image in pass 2. Error bars are 95% CI over 5 seeds.",
        font=small,
        fill=(70, 70, 70),
    )
    image.save(out_path)


def draw_paligemma_attention_vlmbias(rows: list[dict[str, str]], out_path: Path) -> None:
    width, height = 1700, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = _font(30, bold=True)
    font = _font(17)
    small = _font(14)
    tiny = _font(12)
    draw.text((width / 2, 30), "PaliGemma2 VLMBias Image-Attention Alpha Sweep", font=title, fill=COLORS["text"], anchor="ma")
    draw.text(
        (width / 2, 70),
        "Attention logit boost added to image-token keys before softmax; 10 seeds",
        font=font,
        fill=(70, 70, 70),
        anchor="ma",
    )
    ci = _seed_ci_by_condition(rows, ("bias_aligned_fraction", "mean_image_attention_mass"))
    _draw_attention_panel(
        draw,
        rows,
        ci,
        metric="bias_aligned_fraction_mean",
        ci_metric="bias_aligned_fraction",
        title="Bias-aligned fraction (lower is better)",
        box=(90, 135, 800, 850),
        y_max=0.36,
        font=font,
        small=small,
        tiny=tiny,
    )
    _draw_attention_panel(
        draw,
        rows,
        ci,
        metric="mean_image_attention_mass_mean",
        ci_metric="mean_image_attention_mass",
        title="Mean image attention mass",
        box=(890, 135, 1600, 850),
        y_max=1.00,
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
    draw.text((90, 930), "Error bars are 95% CI over seeds 0-9.", font=small, fill=(70, 70, 70))
    image.save(out_path)


def draw_paligemma_attention_naturalbench(rows: list[dict[str, str]], out_path: Path) -> None:
    width, height = 1700, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = _font(30, bold=True)
    font = _font(17)
    small = _font(14)
    tiny = _font(12)
    draw.text((width / 2, 30), "PaliGemma2 NaturalBench Sanity Check", font=title, fill=COLORS["text"], anchor="ma")
    draw.text(
        (width / 2, 70),
        "Checks whether image-attention boosting damages general VLM behavior; 10 seeds",
        font=font,
        fill=(70, 70, 70),
        anchor="ma",
    )
    ci = _seed_ci_by_condition(rows, ("Acc", "G_Acc"))
    _draw_attention_panel(
        draw,
        rows,
        ci,
        metric="Acc_mean",
        ci_metric="Acc",
        title="NaturalBench call accuracy",
        box=(90, 135, 800, 850),
        y_max=0.80,
        font=font,
        small=small,
        tiny=tiny,
    )
    _draw_attention_panel(
        draw,
        rows,
        ci,
        metric="G_Acc_mean",
        ci_metric="G_Acc",
        title="NaturalBench group accuracy",
        box=(890, 135, 1600, 850),
        y_max=0.30,
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
    draw.text((90, 930), "Error bars are 95% CI over seeds 0-9.", font=small, fill=(70, 70, 70))
    image.save(out_path)


def write_notes(
    two_pass: list[dict[str, str]],
    vlmbias_attention: list[dict[str, str]],
    naturalbench_attention: list[dict[str, str]],
    two_pass_notes: Path,
    attention_notes: Path,
) -> None:
    direct_image = _find_two_pass(two_pass, "direct", "image_again")
    best_desc_acc = max((row for row in two_pass if row["answer_mode"] == "description_only"), key=lambda row: float(row["accuracy_mean"]))
    best_desc_bias = min((row for row in two_pass if row["answer_mode"] == "description_only"), key=lambda row: float(row["bias_aligned_fraction_mean"]))
    best_image_bias = min((row for row in two_pass if row["answer_mode"] == "image_again"), key=lambda row: float(row["bias_aligned_fraction_mean"]))

    two_pass_notes.write_text(
        f"""# PaliGemma2 Unified Two-Pass Experiment

Model: `google/paligemma2-3b-mix-448`

Dataset: `data/vlmbias_400.jsonl`

Seeds: `0,1,2,3,4`

PaliGemma requires an image input, so the `description_only` answer mode uses a blank white image in pass 2.

## Main Readout

- Direct image-again baseline-like condition: accuracy `{float(direct_image['accuracy_mean']):.3f}`, bias-aligned fraction `{float(direct_image['bias_aligned_fraction_mean']):.3f}`.
- Best description-only accuracy: `{best_desc_acc['description_prompt_variant']}` with accuracy `{float(best_desc_acc['accuracy_mean']):.3f}`.
- Lowest description-only bias: `{best_desc_bias['description_prompt_variant']}` with bias-aligned fraction `{float(best_desc_bias['bias_aligned_fraction_mean']):.3f}`.
- Lowest image-again bias: `{best_image_bias['description_prompt_variant']}` with bias-aligned fraction `{float(best_image_bias['bias_aligned_fraction_mean']):.3f}`.

## Interpretation

For PaliGemma2, image-again conditions preserve more accuracy but remain substantially more bias-aligned. Blank-image description-only conditions sharply reduce bias-aligned answers, but accuracy drops close to zero. This is qualitatively similar to the Qwen two-pass direction on bias reduction, but PaliGemma pays a much larger accuracy cost in the blank-image answer pass.
""",
        encoding="utf-8",
    )

    baseline_v = next(row for row in vlmbias_attention if row["condition"] == "baseline")
    min_bias = min(vlmbias_attention, key=lambda row: float(row["bias_aligned_fraction_mean"]))
    max_acc_nb = max(naturalbench_attention, key=lambda row: float(row["Acc_mean"]))
    bad_nb = min(naturalbench_attention, key=lambda row: float(row["Acc_mean"]))

    attention_notes.write_text(
        f"""# PaliGemma2 Attention Alpha Sweep

Model: `google/paligemma2-3b-mix-448`

Benchmarks:

- VLMBias: `data/vlmbias_400.jsonl`
- NaturalBench: `data/naturalbench_100_groups.jsonl`

Seeds: `0,1,2,3,4,5,6,7,8,9`

Alpha values: `0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0`

Layer selections: `last25`, `last50`, `all`, plus baseline.

## Main Readout

- VLMBias baseline: accuracy `{float(baseline_v['accuracy_mean']):.3f}`, bias-aligned fraction `{float(baseline_v['bias_aligned_fraction_mean']):.3f}`.
- Lowest VLMBias bias-aligned fraction: `{min_bias['condition']}` at `{float(min_bias['bias_aligned_fraction_mean']):.3f}` with accuracy `{float(min_bias['accuracy_mean']):.3f}`.
- Best NaturalBench call accuracy: `{max_acc_nb['condition']}` at `{float(max_acc_nb['Acc_mean']):.3f}`.
- Worst NaturalBench call accuracy: `{bad_nb['condition']}` at `{float(bad_nb['Acc_mean']):.3f}`.

## Interpretation

The attention hook is active: image attention mass rises monotonically with alpha, and extreme `all`-layer boosts push image attention near saturation. Moderate boosts do not reliably reduce VLMBias bias and mostly leave NaturalBench stable. Extreme all-layer boosts, especially `alpha10_all`, severely damage NaturalBench and VLMBias accuracy, so the intervention can overpower useful language-model behavior rather than selectively correcting bias.
""",
        encoding="utf-8",
    )


def _seed_ci_by_condition(rows: list[dict[str, str]], metrics: tuple[str, ...]) -> dict[str, dict[str, float]]:
    ci: dict[str, dict[str, float]] = {}
    for row in rows:
        runs = int(row["runs"])
        t_value = 2.262 if runs == 10 else 1.96
        ci[row["condition"]] = {}
        for metric in metrics:
            std_key = f"{metric}_std"
            ci[row["condition"]][metric] = t_value * float(row.get(std_key, 0.0)) / math.sqrt(runs)
    return ci


def _find_two_pass(rows: list[dict[str, str]], description_prompt_variant: str, answer_mode: str) -> dict[str, str]:
    return next(
        row
        for row in rows
        if row["description_prompt_variant"] == description_prompt_variant and row["answer_mode"] == answer_mode
    )


if __name__ == "__main__":
    main()
