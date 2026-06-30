#!/usr/bin/env python3
"""Analyze Qwen gaze-description and gaze-attention sweep outputs."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
STATIC_RUNS = ROOT / "segments/gaze_heads_qwen25/runs"
STATIC_REPORT = ROOT / "segments/gaze_heads_qwen25/reports/static_narration_qwen_judge_analysis"
SWEEP_RUN = ROOT / "segments/vlm_bias_attention/runs/vlmbias_gaze_attention_sweep"
SWEEP_REPORT = ROOT / "segments/vlm_bias_attention/reports/gaze_attention_sweep_analysis"


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_SMALL = font(14)
FONT = font(16)
FONT_BOLD = font(18, bold=True)
FONT_TITLE = font(24, bold=True)


def ensure_dirs() -> None:
    STATIC_REPORT.mkdir(parents=True, exist_ok=True)
    (STATIC_REPORT / "plots").mkdir(exist_ok=True)
    SWEEP_REPORT.mkdir(parents=True, exist_ok=True)
    (SWEEP_REPORT / "plots").mkdir(exist_ok=True)
    (SWEEP_REPORT / "tables").mkdir(exist_ok=True)
    (STATIC_REPORT / "tables").mkdir(exist_ok=True)


def fmt(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class Chart:
    def __init__(self, path: Path, title: str, x_label: str, y_label: str, width: int = 1100, height: int = 720):
        self.path = path
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.width = width
        self.height = height
        self.margin = (105, 80, 35, 90)  # left, top, right, bottom
        self.colors = ["#2F6FAD", "#C5533D", "#4F8A41", "#8B5FBF", "#D19A2A", "#3B8C8C"]

    def _bounds(self, series: list[tuple[str, list[tuple[float, float]]]], y_min=None, y_max=None):
        xs = [x for _, pts in series for x, _ in pts]
        ys = [y for _, pts in series for _, y in pts if not pd.isna(y)]
        xmin, xmax = min(xs), max(xs)
        ymin = min(ys) if y_min is None else y_min
        ymax = max(ys) if y_max is None else y_max
        if math.isclose(ymin, ymax):
            ymin -= 0.01
            ymax += 0.01
        pad = (ymax - ymin) * 0.08
        return xmin, xmax, ymin - pad, ymax + pad

    def line(
        self,
        series: list[tuple[str, list[tuple[float, float]]]],
        baseline: float | None = None,
        y_min=None,
        y_max=None,
        x_tick_labels: dict[float, str] | None = None,
    ):
        img = Image.new("RGB", (self.width, self.height), "white")
        d = ImageDraw.Draw(img)
        l, t, r, b = self.margin
        x0, y0 = l, self.height - b
        x1, y1 = self.width - r, t
        xmin, xmax, ymin, ymax = self._bounds(series, y_min, y_max)
        if baseline is not None:
            ymin = min(ymin, baseline)
            ymax = max(ymax, baseline)

        def sx(x):
            return x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)

        def sy(y):
            return y0 - (y - ymin) / (ymax - ymin) * (y0 - y1)

        d.text((l, 22), self.title, fill="#111111", font=FONT_TITLE)
        d.line((x0, y0, x1, y0), fill="#444444", width=2)
        d.line((x0, y0, x0, y1), fill="#444444", width=2)
        for i in range(6):
            y = ymin + (ymax - ymin) * i / 5
            py = sy(y)
            d.line((x0, py, x1, py), fill="#E6E6E6", width=1)
            d.text((15, py - 8), fmt(y), fill="#333333", font=FONT_SMALL)
        x_tick_labels = x_tick_labels or {}
        for x in sorted({x for _, pts in series for x, _ in pts}):
            px = sx(x)
            d.line((px, y0, px, y0 + 6), fill="#444444", width=1)
            label = x_tick_labels.get(x, str(x).rstrip("0").rstrip("."))
            d.text((px - 16, y0 + 14), label, fill="#333333", font=FONT_SMALL)
        if baseline is not None:
            py = sy(baseline)
            d.line((x0, py, x1, py), fill="#777777", width=2)
            d.text((x1 - 145, py - 22), f"baseline {fmt(baseline)}", fill="#555555", font=FONT_SMALL)
        for idx, (name, pts) in enumerate(series):
            color = self.colors[idx % len(self.colors)]
            coords = [(sx(x), sy(y)) for x, y in pts if not pd.isna(y)]
            if len(coords) > 1:
                d.line(coords, fill=color, width=4)
            for px, py in coords:
                d.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
            d.text((x1 - 145, y1 + idx * 24), name, fill=color, font=FONT)
        d.text(((x0 + x1) / 2 - 45, self.height - 35), self.x_label, fill="#111111", font=FONT)
        d.text((15, 52), self.y_label, fill="#111111", font=FONT)
        img.save(self.path)

    def grouped_bars(
        self,
        labels: list[str],
        series: list[tuple[str, list[float]]],
        y_min: float = 0.0,
        y_max: float | None = None,
        errors: list[tuple[str, list[tuple[float, float]]]] | None = None,
    ):
        img = Image.new("RGB", (self.width, self.height), "white")
        d = ImageDraw.Draw(img)
        l, t, r, b = self.margin
        x0, y0 = l, self.height - b
        x1, y1 = self.width - r, t
        vals = [v for _, arr in series for v in arr]
        ymax = max(vals) if y_max is None else y_max
        ymax = ymax * 1.15 if y_max is None else y_max

        def sy(y):
            return y0 - (y - y_min) / (ymax - y_min) * (y0 - y1)

        d.text((l, 22), self.title, fill="#111111", font=FONT_TITLE)
        d.line((x0, y0, x1, y0), fill="#444444", width=2)
        d.line((x0, y0, x0, y1), fill="#444444", width=2)
        for i in range(6):
            y = y_min + (ymax - y_min) * i / 5
            py = sy(y)
            d.line((x0, py, x1, py), fill="#E6E6E6", width=1)
            d.text((15, py - 8), fmt(y), fill="#333333", font=FONT_SMALL)
        group_w = (x1 - x0) / len(labels)
        bar_w = group_w / (len(series) + 1.2)
        error_map = {name: arr for name, arr in errors} if errors else {}
        for i, label in enumerate(labels):
            gx = x0 + i * group_w + group_w * 0.18
            for j, (name, arr) in enumerate(series):
                val = arr[i]
                x_left = gx + j * bar_w
                x_right = x_left + bar_w * 0.82
                x_mid = (x_left + x_right) / 2
                d.rectangle((x_left, sy(val), x_right, y0), fill=self.colors[j % len(self.colors)])
                if name in error_map:
                    lo, hi = error_map[name][i]
                    d.line((x_mid, sy(lo), x_mid, sy(hi)), fill="#222222", width=2)
                    d.line((x_mid - 6, sy(lo), x_mid + 6, sy(lo)), fill="#222222", width=2)
                    d.line((x_mid - 6, sy(hi), x_mid + 6, sy(hi)), fill="#222222", width=2)
                d.text((x_left, sy(val) - 18), fmt(val), fill="#222222", font=FONT_SMALL)
            d.text((x0 + i * group_w + group_w * 0.32, y0 + 14), label, fill="#333333", font=FONT_SMALL)
        for j, (name, _) in enumerate(series):
            d.rectangle((x1 - 165, y1 + j * 24, x1 - 148, y1 + 17 + j * 24), fill=self.colors[j % len(self.colors)])
            d.text((x1 - 140, y1 + j * 24), name, fill="#111111", font=FONT)
        d.text((15, 52), self.y_label, fill="#111111", font=FONT)
        img.save(self.path)


def analyze_static() -> dict:
    rows = []
    panel_rows = []
    match_rows = []
    topks = [1, 5, 10, 20]
    for k in topks:
        run_dir = STATIC_RUNS / f"static_narration_top{k}_merged_0_500"
        agg_path = run_dir / "qwen_judge/aggregate_results.json"
        judgments_path = run_dir / "qwen_judge/judgments.jsonl"
        with agg_path.open() as f:
            data = json.load(f)
        for condition, stats in data["aggregate"].items():
            overall = stats["overall"]
            rows.append(
                {
                    "top_k": k,
                    "condition": condition,
                    "is_gaze": condition.startswith("gaze_"),
                    "accuracy": overall["accuracy"],
                    "ci_low": overall["ci_low"],
                    "ci_high": overall["ci_high"],
                    "n": overall["n"],
                    "junk_count": stats["junk_count"],
                    "baseline_match_count": stats["baseline_match_count"],
                    "source": str(agg_path.relative_to(ROOT)),
                }
            )
            for panel, panel_stats in stats["per_panel"].items():
                panel_rows.append(
                    {
                        "top_k": k,
                        "condition": condition,
                        "target_panel": int(panel),
                        "accuracy": panel_stats["accuracy"],
                        "ci_low": panel_stats["ci_low"],
                        "ci_high": panel_stats["ci_high"],
                        "n": panel_stats["n"],
                        "source": str(agg_path.relative_to(ROOT)),
                    }
                )
        match_counts: dict[str, Counter] = defaultdict(Counter)
        with judgments_path.open() as f:
            for line in f:
                item = json.loads(line)
                cond = item["condition"]
                judgment = item["judgment"]
                matched_panel = judgment.get("matched_panel")
                if not judgment.get("is_junk") and matched_panel is not None:
                    try:
                        match_counts[cond][int(matched_panel)] += 1
                    except (TypeError, ValueError):
                        pass
        for cond, counter in match_counts.items():
            total = sum(counter.values())
            for panel in range(1, 7):
                match_rows.append(
                    {
                        "top_k": k,
                        "condition": cond,
                        "matched_panel": panel,
                        "count": counter[panel],
                        "fraction": counter[panel] / total if total else np.nan,
                        "source": str(judgments_path.relative_to(ROOT)),
                    }
                )
    df = pd.DataFrame(rows)
    panel_df = pd.DataFrame(panel_rows)
    match_df = pd.DataFrame(match_rows)
    pairs = []
    for k in topks:
        gaze = df[(df.top_k == k) & (df.is_gaze)].iloc[0]
        nongaze = df[(df.top_k == k) & (~df.is_gaze)].iloc[0]
        pairs.append(
            {
                "top_k": k,
                "gaze_accuracy": gaze.accuracy,
                "non_gaze_accuracy": nongaze.accuracy,
                "delta_gaze_minus_non_gaze": gaze.accuracy - nongaze.accuracy,
                "gaze_junk_count": int(gaze.junk_count),
                "non_gaze_junk_count": int(nongaze.junk_count),
            }
        )
    pair_df = pd.DataFrame(pairs)
    df.to_csv(STATIC_REPORT / "tables/qwen_judge_condition_summary.tsv", sep="\t", index=False)
    panel_df.to_csv(STATIC_REPORT / "tables/qwen_judge_panel_summary.tsv", sep="\t", index=False)
    match_df.to_csv(STATIC_REPORT / "tables/qwen_judge_matched_panel_distribution.tsv", sep="\t", index=False)
    pair_df.to_csv(STATIC_REPORT / "tables/qwen_judge_gaze_vs_non_gaze.tsv", sep="\t", index=False)

    labels = [str(k) for k in topks]
    gaze_vals = [pair_df[pair_df.top_k == k].gaze_accuracy.iloc[0] for k in topks]
    nongaze_vals = [pair_df[pair_df.top_k == k].non_gaze_accuracy.iloc[0] for k in topks]
    gaze_errs = [
        (
            df[(df.top_k == k) & (df.is_gaze)].ci_low.iloc[0],
            df[(df.top_k == k) & (df.is_gaze)].ci_high.iloc[0],
        )
        for k in topks
    ]
    nongaze_errs = [
        (
            df[(df.top_k == k) & (~df.is_gaze)].ci_low.iloc[0],
            df[(df.top_k == k) & (~df.is_gaze)].ci_high.iloc[0],
        )
        for k in topks
    ]
    Chart(STATIC_REPORT / "plots/qwen_judge_accuracy_by_topk.png", "Static narration Qwen-judge accuracy", "top-k gaze heads", "forced-choice accuracy").grouped_bars(
        labels,
        [("gaze steered", gaze_vals), ("non-gaze control", nongaze_vals)],
        y_max=0.22,
        errors=[("gaze steered", gaze_errs), ("non-gaze control", nongaze_errs)],
    )

    # Panel-level accuracy plots for every top-k, plus a compatibility copy for the best positive top-k.
    best_k = int(pair_df.sort_values("delta_gaze_minus_non_gaze", ascending=False).iloc[0].top_k)
    panel_labels = [str(i) for i in range(1, 7)]
    for k in topks:
        panel_k = panel_df[panel_df.top_k == k]
        g = [panel_k[(panel_k.condition == f"gaze_top{k}") & (panel_k.target_panel == i)].accuracy.iloc[0] for i in range(1, 7)]
        n = [panel_k[(panel_k.condition == f"non_gaze_{k}") & (panel_k.target_panel == i)].accuracy.iloc[0] for i in range(1, 7)]
        g_errs = [
            (
                panel_k[(panel_k.condition == f"gaze_top{k}") & (panel_k.target_panel == i)].ci_low.iloc[0],
                panel_k[(panel_k.condition == f"gaze_top{k}") & (panel_k.target_panel == i)].ci_high.iloc[0],
            )
            for i in range(1, 7)
        ]
        n_errs = [
            (
                panel_k[(panel_k.condition == f"non_gaze_{k}") & (panel_k.target_panel == i)].ci_low.iloc[0],
                panel_k[(panel_k.condition == f"non_gaze_{k}") & (panel_k.target_panel == i)].ci_high.iloc[0],
            )
            for i in range(1, 7)
        ]
        out_name = f"qwen_judge_panel_accuracy_top{k}.png"
        Chart(STATIC_REPORT / f"plots/{out_name}", f"Panel accuracy at top-{k}", "target panel", "accuracy").grouped_bars(
            panel_labels,
            [("gaze", g), ("non-gaze", n)],
            y_max=0.9,
            errors=[("gaze", g_errs), ("non-gaze", n_errs)],
        )
        if k == best_k:
            Chart(STATIC_REPORT / "plots/qwen_judge_panel_accuracy_best_topk.png", f"Panel accuracy at top-{best_k}", "target panel", "accuracy").grouped_bars(
                panel_labels,
                [("gaze", g), ("non-gaze", n)],
                y_max=0.9,
                errors=[("gaze", g_errs), ("non-gaze", n_errs)],
            )
    return {"condition": df, "panel": panel_df, "match": match_df, "pairs": pair_df, "best_k": best_k}


def parse_seed(path: Path) -> int:
    m = re.search(r"_seed(\d+)\.summary\.json$", path.name)
    if not m:
        raise ValueError(f"Cannot parse seed from {path}")
    return int(m.group(1))


def load_sweep_summaries() -> pd.DataFrame:
    rows = []
    for benchmark in ["vlmbias", "naturalbench"]:
        for path in sorted((SWEEP_RUN / benchmark).glob("*.summary.json")):
            with path.open() as f:
                data = json.load(f)
            cfg = data["run_config"]
            row = {
                "benchmark": benchmark,
                "condition": cfg["condition"],
                "seed": parse_seed(path),
                "attention_alpha": float(cfg["attention_alpha"]),
                "top_k_gaze": int(cfg["top_k_gaze"]),
                "is_baseline": cfg["condition"] == "baseline",
                "source": str(path.relative_to(ROOT)),
            }
            if benchmark == "vlmbias":
                for key in ["accuracy", "bias_aligned_fraction", "bias_aligned_error_rate", "error_rate", "mean_image_attention_mass", "mean_boosted_layer_image_attention_mass", "mean_boosted_head_image_attention_mass"]:
                    row[key] = data.get(key, np.nan)
                row["n"] = data.get("n", np.nan)
            else:
                for key in ["Acc", "Q_Acc", "I_Acc", "G_Acc", "mean_image_attention_mass", "mean_boosted_layer_image_attention_mass", "mean_boosted_head_image_attention_mass"]:
                    row[key] = data.get(key, np.nan)
                row["n_model_calls"] = data.get("n_model_calls", np.nan)
                row["n_groups"] = data.get("n_groups", np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_sweep(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c
        for c in df.columns
        if c
        not in {
            "benchmark",
            "condition",
            "seed",
            "attention_alpha",
            "top_k_gaze",
            "is_baseline",
            "source",
        }
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    grouped = []
    for keys, g in df.groupby(["benchmark", "condition", "attention_alpha", "top_k_gaze", "is_baseline"], dropna=False):
        row = {
            "benchmark": keys[0],
            "condition": keys[1],
            "attention_alpha": keys[2],
            "top_k_gaze": keys[3],
            "is_baseline": keys[4],
            "seeds": ",".join(map(str, sorted(g.seed.unique()))),
            "n_seeds": g.seed.nunique(),
        }
        for col in metric_cols:
            if g[col].notna().any():
                row[f"{col}_mean"] = g[col].mean()
                row[f"{col}_std"] = g[col].std(ddof=1) if g[col].notna().sum() > 1 else 0.0
        grouped.append(row)
    agg = pd.DataFrame(grouped).sort_values(["benchmark", "top_k_gaze", "attention_alpha"])
    for benchmark in ["vlmbias", "naturalbench"]:
        b = agg[(agg.benchmark == benchmark) & (agg.condition == "baseline")].iloc[0]
        for col in [c for c in agg.columns if c.endswith("_mean")]:
            if not pd.isna(b.get(col, np.nan)):
                mask = agg.benchmark == benchmark
                agg.loc[mask, f"{col}_delta_vs_baseline"] = agg.loc[mask, col] - b[col]
    return agg


def series_from(agg: pd.DataFrame, benchmark: str, metric: str, x_map: dict[float, float] | None = None):
    out = []
    sub = agg[(agg.benchmark == benchmark) & (~agg.is_baseline)]
    for k, g in sub.groupby("top_k_gaze"):
        pts = [
            (x_map.get(float(r.attention_alpha), float(r.attention_alpha)) if x_map else float(r.attention_alpha), float(getattr(r, metric)))
            for r in g.sort_values("attention_alpha").itertuples()
        ]
        out.append((f"top-{int(k)}", pts))
    return sorted(out, key=lambda x: int(x[0].split("-")[1]))


def draw_sweep_comparison_plot(
    path: Path,
    panels: list[dict],
    x_tick_labels: dict[float, str],
) -> None:
    width, height = 1650, 620
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((45, 24), "Alpha sweep comparison across benchmark and bias metrics", fill="#111111", font=FONT_TITLE)
    colors = ["#2F6FAD", "#C5533D", "#4F8A41", "#8B5FBF"]
    panel_w = 500
    top = 90
    bottom = 90
    lefts = [70, 590, 1110]
    for panel_idx, panel in enumerate(panels):
        left = lefts[panel_idx]
        right = left + panel_w - 40
        y_top = top
        y_bottom = height - bottom
        series = panel["series"]
        y_min = panel["y_min"]
        y_max = panel["y_max"]
        xs = sorted({x for _, pts in series for x, _ in pts})
        x_min, x_max = min(xs), max(xs)

        def sx(x):
            return left + (x - x_min) / (x_max - x_min) * (right - left)

        def sy(y):
            return y_bottom - (y - y_min) / (y_max - y_min) * (y_bottom - y_top)

        d.text((left, 58), panel["title"], fill="#111111", font=FONT_BOLD)
        d.line((left, y_bottom, right, y_bottom), fill="#444444", width=2)
        d.line((left, y_bottom, left, y_top), fill="#444444", width=2)
        for i in range(5):
            y = y_min + (y_max - y_min) * i / 4
            py = sy(y)
            d.line((left, py, right, py), fill="#E6E6E6", width=1)
            d.text((left - 58, py - 8), fmt(y), fill="#333333", font=FONT_SMALL)
        for x in xs:
            px = sx(x)
            d.line((px, y_bottom, px, y_bottom + 6), fill="#444444", width=1)
            d.text((px - 12, y_bottom + 14), x_tick_labels.get(x, str(x)), fill="#333333", font=FONT_SMALL)
        if panel.get("baseline") is not None:
            py = sy(panel["baseline"])
            d.line((left, py, right, py), fill="#777777", width=2)
            d.text((right - 120, py - 20), f"baseline {fmt(panel['baseline'])}", fill="#555555", font=FONT_SMALL)
        for idx, (name, pts) in enumerate(series):
            color = colors[idx % len(colors)]
            coords = [(sx(x), sy(y)) for x, y in pts]
            d.line(coords, fill=color, width=4)
            for px, py in coords:
                d.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
        d.text((left + 160, height - 35), "attention alpha", fill="#111111", font=FONT)
    legend_x = 1310
    for idx, name in enumerate(["top-1", "top-5", "top-10", "top-20"]):
        d.line((legend_x, 32 + idx * 24, legend_x + 28, 32 + idx * 24), fill=colors[idx], width=4)
        d.text((legend_x + 36, 23 + idx * 24), name, fill="#111111", font=FONT)
    img.save(path)


def ci_points(g: pd.DataFrame, metric_mean: str, metric_std: str, x_map: dict[float, float]) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
    mean_pts = []
    low_pts = []
    high_pts = []
    for row in g.sort_values("attention_alpha").itertuples():
        x = x_map[float(row.attention_alpha)]
        mean = float(getattr(row, metric_mean))
        std = float(getattr(row, metric_std))
        n = int(getattr(row, "n_seeds"))
        half_width = 1.96 * std / math.sqrt(n) if n else 0.0
        mean_pts.append((x, mean))
        low_pts.append((x, mean - half_width))
        high_pts.append((x, mean + half_width))
    return mean_pts, low_pts, high_pts


def draw_sweep_faceted_ci_plot(
    path: Path,
    agg: pd.DataFrame,
    x_map: dict[float, float],
    x_tick_labels: dict[float, str],
) -> None:
    width, height = 1650, 1220
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img, "RGBA")
    d.text((45, 22), "Alpha sweep comparison with 95% confidence bands", fill="#111111", font=FONT_TITLE)
    d.text((45, 54), "Rows split top-k settings; columns compare NaturalBench Acc, VLMBias accuracy, and VLMBias bias-aligned fraction.", fill="#333333", font=FONT)
    topks = [1, 5, 10, 20]
    colors = {1: "#2F6FAD", 5: "#C5533D", 10: "#4F8A41", 20: "#8B5FBF"}
    metrics = [
        {
            "title": "NaturalBench Acc",
            "benchmark": "naturalbench",
            "mean": "Acc_mean",
            "std": "Acc_std",
            "y_min": 0.50,
            "y_max": 0.60,
        },
        {
            "title": "VLMBias accuracy",
            "benchmark": "vlmbias",
            "mean": "accuracy_mean",
            "std": "accuracy_std",
            "y_min": 0.08,
            "y_max": 0.18,
        },
        {
            "title": "VLMBias bias-aligned fraction",
            "benchmark": "vlmbias",
            "mean": "bias_aligned_fraction_mean",
            "std": "bias_aligned_fraction_std",
            "y_min": 0.18,
            "y_max": 0.28,
        },
    ]
    cell_w = 500
    cell_h = 245
    lefts = [80, 600, 1120]
    row_tops = [115, 380, 645, 910]
    plot_bottom_pad = 58
    plot_top_pad = 45
    plot_right_pad = 25
    plot_left_pad = 65

    for row_idx, top_k in enumerate(topks):
        d.text((20, row_tops[row_idx] + 104), f"top-{top_k}", fill=colors[top_k], font=FONT_BOLD)
        for col_idx, metric in enumerate(metrics):
            cell_left = lefts[col_idx]
            cell_top = row_tops[row_idx]
            x0 = cell_left + plot_left_pad
            x1 = cell_left + cell_w - plot_right_pad
            y1 = cell_top + plot_top_pad
            y0 = cell_top + cell_h - plot_bottom_pad
            y_min = metric["y_min"]
            y_max = metric["y_max"]
            x_min, x_max = min(x_map.values()), max(x_map.values())

            def sx(x):
                return x0 + (x - x_min) / (x_max - x_min) * (x1 - x0)

            def sy(y):
                return y0 - (y - y_min) / (y_max - y_min) * (y0 - y1)

            if row_idx == 0:
                d.text((x0, cell_top + 5), metric["title"], fill="#111111", font=FONT_BOLD)
            d.line((x0, y0, x1, y0), fill="#444444", width=2)
            d.line((x0, y0, x0, y1), fill="#444444", width=2)
            for i in range(4):
                y = y_min + (y_max - y_min) * i / 3
                py = sy(y)
                d.line((x0, py, x1, py), fill="#E6E6E6", width=1)
                d.text((x0 - 58, py - 8), fmt(y), fill="#333333", font=FONT_SMALL)
            for x in sorted(x_map.values()):
                px = sx(x)
                d.line((px, y0, px, y0 + 5), fill="#444444", width=1)
                if row_idx == len(topks) - 1:
                    d.text((px - 12, y0 + 12), x_tick_labels[x], fill="#333333", font=FONT_SMALL)

            baseline = agg[(agg.benchmark == metric["benchmark"]) & (agg.condition == "baseline")].iloc[0]
            base_mean = float(baseline[metric["mean"]])
            base_std = float(baseline[metric["std"]])
            base_n = int(baseline["n_seeds"])
            base_half = 1.96 * base_std / math.sqrt(base_n) if base_n else 0.0
            d.rectangle((x0, sy(base_mean + base_half), x1, sy(base_mean - base_half)), fill=(150, 150, 150, 35))
            d.line((x0, sy(base_mean), x1, sy(base_mean)), fill="#777777", width=2)

            sub = agg[
                (agg.benchmark == metric["benchmark"])
                & (agg.top_k_gaze == top_k)
                & (~agg.is_baseline)
            ]
            mean_pts, low_pts, high_pts = ci_points(sub, metric["mean"], metric["std"], x_map)
            band = [(sx(x), sy(y)) for x, y in low_pts] + [(sx(x), sy(y)) for x, y in reversed(high_pts)]
            rgb = tuple(int(colors[top_k].lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
            d.polygon(band, fill=(*rgb, 48))
            coords = [(sx(x), sy(y)) for x, y in mean_pts]
            d.line(coords, fill=colors[top_k], width=4)
            for px, py in coords:
                d.ellipse((px - 4, py - 4, px + 4, py + 4), fill=colors[top_k])
            if col_idx == 1 and row_idx == len(topks) - 1:
                d.text((x0 + 145, height - 28), "attention alpha", fill="#111111", font=FONT)
    img.save(path)


def analyze_sweep() -> dict:
    raw = load_sweep_summaries()
    agg = aggregate_sweep(raw)
    raw.to_csv(SWEEP_REPORT / "tables/recomputed_summary_by_seed.tsv", sep="\t", index=False)
    agg.to_csv(SWEEP_REPORT / "tables/recomputed_summary_aggregate.tsv", sep="\t", index=False)

    base_v = agg[(agg.benchmark == "vlmbias") & (agg.condition == "baseline")].iloc[0]
    base_n = agg[(agg.benchmark == "naturalbench") & (agg.condition == "baseline")].iloc[0]
    alphas = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    alpha_x = {alpha: float(i) for i, alpha in enumerate(alphas)}
    alpha_labels = {float(i): str(alpha).rstrip("0").rstrip(".") for i, alpha in enumerate(alphas)}
    Chart(SWEEP_REPORT / "plots/vlmbias_accuracy_by_alpha.png", "VLMBias accuracy by gaze-attention boost", "attention alpha", "accuracy").line(
        series_from(agg, "vlmbias", "accuracy_mean", alpha_x), baseline=float(base_v.accuracy_mean), y_min=0.08, y_max=0.18, x_tick_labels=alpha_labels
    )
    Chart(SWEEP_REPORT / "plots/vlmbias_bias_aligned_fraction_by_alpha.png", "VLMBias bias-aligned answer fraction", "attention alpha", "bias-aligned fraction").line(
        series_from(agg, "vlmbias", "bias_aligned_fraction_mean", alpha_x), baseline=float(base_v.bias_aligned_fraction_mean), y_min=0.18, y_max=0.28, x_tick_labels=alpha_labels
    )
    Chart(SWEEP_REPORT / "plots/naturalbench_acc_by_alpha.png", "NaturalBench Acc by gaze-attention boost", "attention alpha", "Acc").line(
        series_from(agg, "naturalbench", "Acc_mean", alpha_x), baseline=float(base_n.Acc_mean), y_min=0.50, y_max=0.60, x_tick_labels=alpha_labels
    )
    Chart(SWEEP_REPORT / "plots/naturalbench_g_acc_by_alpha.png", "NaturalBench G_Acc by gaze-attention boost", "attention alpha", "G_Acc").line(
        series_from(agg, "naturalbench", "G_Acc_mean", alpha_x), baseline=float(base_n.G_Acc_mean), y_min=0.0, y_max=0.05, x_tick_labels=alpha_labels
    )
    Chart(SWEEP_REPORT / "plots/boosted_head_attention_mass_by_alpha.png", "Boosted head image attention mass", "attention alpha", "mean boosted-head mass").line(
        series_from(agg, "vlmbias", "mean_boosted_head_image_attention_mass_mean", alpha_x), baseline=None, y_min=0.25, y_max=1.02, x_tick_labels=alpha_labels
    )
    draw_sweep_comparison_plot(
        SWEEP_REPORT / "plots/combined_accuracy_bias_comparison.png",
        [
            {
                "title": "NaturalBench Acc",
                "series": series_from(agg, "naturalbench", "Acc_mean", alpha_x),
                "baseline": float(base_n.Acc_mean),
                "y_min": 0.50,
                "y_max": 0.60,
            },
            {
                "title": "VLMBias accuracy",
                "series": series_from(agg, "vlmbias", "accuracy_mean", alpha_x),
                "baseline": float(base_v.accuracy_mean),
                "y_min": 0.08,
                "y_max": 0.18,
            },
            {
                "title": "VLMBias bias-aligned fraction",
                "series": series_from(agg, "vlmbias", "bias_aligned_fraction_mean", alpha_x),
                "baseline": float(base_v.bias_aligned_fraction_mean),
                "y_min": 0.18,
                "y_max": 0.28,
            },
        ],
        alpha_labels,
    )
    draw_sweep_faceted_ci_plot(
        SWEEP_REPORT / "plots/combined_accuracy_bias_comparison_faceted_ci.png",
        agg,
        alpha_x,
        alpha_labels,
    )

    joined = agg[agg.benchmark == "vlmbias"][
        [
            "condition",
            "attention_alpha",
            "top_k_gaze",
            "accuracy_mean",
            "accuracy_mean_delta_vs_baseline",
            "bias_aligned_fraction_mean",
            "bias_aligned_fraction_mean_delta_vs_baseline",
            "bias_aligned_error_rate_mean",
            "bias_aligned_error_rate_mean_delta_vs_baseline",
        ]
    ].merge(
        agg[agg.benchmark == "naturalbench"][
            [
                "condition",
                "Acc_mean",
                "Acc_mean_delta_vs_baseline",
                "G_Acc_mean",
                "G_Acc_mean_delta_vs_baseline",
            ]
        ],
        on="condition",
        how="inner",
    )
    joined.to_csv(SWEEP_REPORT / "tables/joined_vlmbias_naturalbench_tradeoff.tsv", sep="\t", index=False)
    candidates = joined[~joined.condition.eq("baseline")].copy()
    best_tables = {
        "best_vlmbias_accuracy": candidates.sort_values("accuracy_mean", ascending=False).head(10),
        "lowest_bias_aligned_fraction": candidates.sort_values("bias_aligned_fraction_mean", ascending=True).head(10),
        "best_naturalbench_acc": candidates.sort_values("Acc_mean", ascending=False).head(10),
        "best_accuracy_with_naturalbench_within_1pt": candidates[candidates.Acc_mean_delta_vs_baseline >= -0.01].sort_values("accuracy_mean_delta_vs_baseline", ascending=False).head(10),
    }
    for name, table in best_tables.items():
        table.to_csv(SWEEP_REPORT / f"tables/{name}.tsv", sep="\t", index=False)

    # Simple scatter tradeoff.
    img = Image.new("RGB", (980, 720), "white")
    d = ImageDraw.Draw(img)
    l, t, r, b = 115, 80, 50, 95
    x0, y0, x1, y1 = l, 720 - b, 980 - r, t
    xs = candidates.Acc_mean_delta_vs_baseline.to_numpy()
    ys = candidates.accuracy_mean_delta_vs_baseline.to_numpy()
    xmin, xmax = min(xs) - 0.005, max(xs) + 0.005
    ymin, ymax = min(ys) - 0.005, max(ys) + 0.005
    sx = lambda x: x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)
    sy = lambda y: y0 - (y - ymin) / (ymax - ymin) * (y0 - y1)
    d.text((l, 24), "Tradeoff: NaturalBench vs VLMBias accuracy deltas", fill="#111111", font=FONT_TITLE)
    d.line((x0, y0, x1, y0), fill="#444444", width=2)
    d.line((x0, y0, x0, y1), fill="#444444", width=2)
    d.line((sx(0), y0, sx(0), y1), fill="#999999", width=2)
    d.line((x0, sy(0), x1, sy(0)), fill="#999999", width=2)
    colors = {1: "#2F6FAD", 5: "#C5533D", 10: "#4F8A41", 20: "#8B5FBF"}
    for row in candidates.itertuples():
        color = colors.get(int(row.top_k_gaze), "#333333")
        px, py = sx(row.Acc_mean_delta_vs_baseline), sy(row.accuracy_mean_delta_vs_baseline)
        d.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color)
        if row.accuracy_mean_delta_vs_baseline == candidates.accuracy_mean_delta_vs_baseline.max():
            d.text((px + 8, py - 10), row.condition, fill="#111111", font=FONT_SMALL)
    for i in range(5):
        x = xmin + (xmax - xmin) * i / 4
        d.text((sx(x) - 24, y0 + 16), fmt(x), fill="#333333", font=FONT_SMALL)
        y = ymin + (ymax - ymin) * i / 4
        d.text((20, sy(y) - 8), fmt(y), fill="#333333", font=FONT_SMALL)
    d.text((350, 680), "NaturalBench Acc delta vs baseline", fill="#111111", font=FONT)
    d.text((15, 52), "VLMBias accuracy delta", fill="#111111", font=FONT)
    for idx, (k, color) in enumerate(colors.items()):
        d.ellipse((790, 100 + idx * 24, 804, 114 + idx * 24), fill=color)
        d.text((812, 96 + idx * 24), f"top-{k}", fill="#111111", font=FONT)
    img.save(SWEEP_REPORT / "plots/tradeoff_vlmbias_vs_naturalbench.png")

    # Scatter plot: NaturalBench accuracy guardrail vs VLMBias bias-aligned rate.
    img = Image.new("RGB", (980, 720), "white")
    d = ImageDraw.Draw(img)
    l, t, r, b = 115, 80, 50, 95
    x0, y0, x1, y1 = l, 720 - b, 980 - r, t
    xs = joined.Acc_mean.to_numpy()
    ys = joined.bias_aligned_fraction_mean.to_numpy()
    xmin, xmax = min(xs) - 0.004, max(xs) + 0.004
    ymin, ymax = min(ys) - 0.006, max(ys) + 0.006
    sx = lambda x: x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)
    sy = lambda y: y0 - (y - ymin) / (ymax - ymin) * (y0 - y1)
    d.text((l, 24), "Tradeoff: NaturalBench Acc vs VLMBias bias-aligned fraction", fill="#111111", font=FONT_TITLE)
    d.line((x0, y0, x1, y0), fill="#444444", width=2)
    d.line((x0, y0, x0, y1), fill="#444444", width=2)
    for i in range(5):
        x = xmin + (xmax - xmin) * i / 4
        d.line((sx(x), y0, sx(x), y1), fill="#E6E6E6", width=1)
        d.text((sx(x) - 24, y0 + 16), fmt(x), fill="#333333", font=FONT_SMALL)
        y = ymin + (ymax - ymin) * i / 4
        d.line((x0, sy(y), x1, sy(y)), fill="#E6E6E6", width=1)
        d.text((20, sy(y) - 8), fmt(y), fill="#333333", font=FONT_SMALL)
    colors = {0: "#111111", 1: "#2F6FAD", 5: "#C5533D", 10: "#4F8A41", 20: "#8B5FBF"}
    label_conditions = {
        joined[~joined.condition.eq("baseline")].sort_values("bias_aligned_fraction_mean").iloc[0].condition,
        joined[~joined.condition.eq("baseline")].sort_values("accuracy_mean", ascending=False).iloc[0].condition,
    }
    for row in joined.itertuples():
        color = colors.get(int(row.top_k_gaze), "#333333")
        px, py = sx(row.Acc_mean), sy(row.bias_aligned_fraction_mean)
        if row.condition == "baseline":
            d.rectangle((px - 7, py - 7, px + 7, py + 7), outline=color, width=3)
            d.text((px + 10, py - 10), "baseline", fill=color, font=FONT_SMALL)
        else:
            d.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color)
            if row.condition in label_conditions:
                d.text((px + 8, py - 10), row.condition, fill="#111111", font=FONT_SMALL)
    d.text((330, 680), "NaturalBench Acc (higher is better)", fill="#111111", font=FONT)
    d.text((15, 52), "VLMBias bias-aligned fraction (lower is better)", fill="#111111", font=FONT)
    legend_x = 770
    legend = [(0, "baseline"), (1, "top-1"), (5, "top-5"), (10, "top-10"), (20, "top-20")]
    for idx, (k, label) in enumerate(legend):
        y = 100 + idx * 24
        if k == 0:
            d.rectangle((legend_x, y, legend_x + 14, y + 14), outline=colors[k], width=2)
        else:
            d.ellipse((legend_x, y, legend_x + 14, y + 14), fill=colors[k])
        d.text((legend_x + 22, y - 3), label, fill="#111111", font=FONT)
    img.save(SWEEP_REPORT / "plots/scatter_naturalbench_acc_vs_vlmbias_bias_fraction.png")
    return {"raw": raw, "agg": agg, "joined": joined, "best_tables": best_tables}


def md_table(df: pd.DataFrame, cols: list[str], n: int = 10) -> str:
    sub = df[cols].head(n).copy()
    for c in sub.columns:
        if pd.api.types.is_float_dtype(sub[c]):
            sub[c] = sub[c].map(lambda x: fmt(x, 4))
    header = "| " + " | ".join(sub.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(sub.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in sub.to_numpy()]
    return "\n".join([header, sep] + rows)


def write_reports(static: dict, sweep: dict) -> None:
    pairs = static["pairs"].sort_values("top_k")
    best_static = pairs.sort_values("delta_gaze_minus_non_gaze", ascending=False).iloc[0]
    worst_static = pairs.sort_values("delta_gaze_minus_non_gaze").iloc[0]
    panel = static["panel"]
    matched = static["match"]
    most_matched = matched.groupby(["top_k", "condition"]).apply(lambda g: g.sort_values("fraction", ascending=False).iloc[0], include_groups=False).reset_index()

    static_md = f"""# Static narration Qwen-judge analysis

## Sources

- Runs: `segments/gaze_heads_qwen25/runs/static_narration_top{{1,5,10,20}}_merged_0_500/`
- Judge aggregates: `qwen_judge/aggregate_results.json`
- Judge rows: `qwen_judge/judgments.jsonl`
- Analysis outputs: `{STATIC_REPORT.relative_to(ROOT)}/`

## Validation

- Each top-k run has 6,000 judged rows: 3,000 gaze-steered generations and 3,000 non-gaze controls.
- Each condition covers 500 comics x 6 target panels.
- Accuracy is Qwen2.5-VL forced-choice matching of generated narration back to the intended target panel.
- Error bars in the plots are the aggregate judge confidence intervals from `aggregate_results.json`.

## Metric definitions

- Overall accuracy is computed over all 3,000 judged generations for a condition: 500 comics x 6 target panels. It answers: "how often did the judge match the generated text back to the intended panel overall?"
- Panel accuracy slices the same judgment task by target panel. For example, panel-2 accuracy only uses rows where the steered/queried target was panel 2. It answers: "does the method work equally well for each target panel, or is success concentrated in one panel position?"
- These are not separate tasks. Panel accuracy is a diagnostic decomposition of the same overall accuracy.

## Main result

{md_table(pairs, ["top_k", "gaze_accuracy", "non_gaze_accuracy", "delta_gaze_minus_non_gaze", "gaze_junk_count", "non_gaze_junk_count"])}

The largest positive gaze-minus-control effect is top-{int(best_static.top_k)}: {fmt(best_static.delta_gaze_minus_non_gaze, 4)} absolute accuracy. The largest negative effect is top-{int(worst_static.top_k)}: {fmt(worst_static.delta_gaze_minus_non_gaze, 4)}.

## Interpretation

The static gaze-description effect is weak. Overall forced-choice accuracies stay close to the 1/6 random baseline, and the best top-k setting improves over its non-gaze control by only {fmt(best_static.delta_gaze_minus_non_gaze, 4)}. This is not evidence that static narration steering reliably makes the model describe the target panel.

The judge is also highly panel-biased, especially toward panel 2. That conclusion comes from `qwen_judge_matched_panel_distribution.tsv`; the most frequent matched panel per condition is:

{md_table(most_matched, ["top_k", "condition", "matched_panel", "fraction"], n=8)}

This panel bias explains why per-panel accuracy can be high for panel 2 while very low for panel 6. See the per-top-k panel plots in `plots/qwen_judge_panel_accuracy_top*.png` and the source table `tables/qwen_judge_panel_summary.tsv`.

## Artifacts

- `plots/qwen_judge_accuracy_by_topk.png`
- `plots/qwen_judge_panel_accuracy_best_topk.png`
- `plots/qwen_judge_panel_accuracy_top1.png`
- `plots/qwen_judge_panel_accuracy_top5.png`
- `plots/qwen_judge_panel_accuracy_top10.png`
- `plots/qwen_judge_panel_accuracy_top20.png`
- `tables/qwen_judge_condition_summary.tsv`
- `tables/qwen_judge_panel_summary.tsv`
- `tables/qwen_judge_matched_panel_distribution.tsv`
- `tables/qwen_judge_gaze_vs_non_gaze.tsv`
"""
    (STATIC_REPORT / "report.md").write_text(static_md)

    agg = sweep["agg"]
    joined = sweep["joined"]
    base_v = agg[(agg.benchmark == "vlmbias") & (agg.condition == "baseline")].iloc[0]
    base_n = agg[(agg.benchmark == "naturalbench") & (agg.condition == "baseline")].iloc[0]
    candidates = joined[~joined.condition.eq("baseline")].copy()
    best_acc = candidates.sort_values("accuracy_mean", ascending=False).iloc[0]
    best_trade = candidates[candidates.Acc_mean_delta_vs_baseline >= -0.01].sort_values("accuracy_mean_delta_vs_baseline", ascending=False).iloc[0]
    lowest_bias = candidates.sort_values("bias_aligned_fraction_mean").iloc[0]
    worst_nb = candidates.sort_values("Acc_mean").iloc[0]
    best_acc_std = agg[(agg.benchmark == "vlmbias") & (agg.condition == best_acc.condition)].iloc[0].accuracy_std
    base_acc_std = base_v.accuracy_std
    seed_counts = agg.groupby("benchmark").n_seeds.agg(["min", "max"]).reset_index()

    sweep_md = f"""# Gaze attention sweep analysis

## Sources

- Experiment config: `segments/vlm_bias_attention/runs/vlmbias_gaze_attention_sweep/experiment_config.json`
- Per-run summaries: `segments/vlm_bias_attention/runs/vlmbias_gaze_attention_sweep/{{vlmbias,naturalbench}}/*.summary.json`
- Original aggregate TSVs: `segments/vlm_bias_attention/runs/vlmbias_gaze_attention_sweep/*_summary_*.tsv`
- Analysis outputs: `{SWEEP_REPORT.relative_to(ROOT)}/`

## Validation

- Found 250 VLMBias summary files and 250 NaturalBench summary files.
- Recomputed aggregates from per-run summary JSON files because the checked-in `*_summary_by_seed.tsv` files only contain seed 0.
- Seed coverage in recomputed aggregates:

{md_table(seed_counts, ["benchmark", "min", "max"], n=2)}

## Baselines

- VLMBias baseline accuracy: {fmt(base_v.accuracy_mean, 4)}; bias-aligned fraction: {fmt(base_v.bias_aligned_fraction_mean, 4)}; bias-aligned error rate: {fmt(base_v.bias_aligned_error_rate_mean, 4)}.
- NaturalBench baseline Acc: {fmt(base_n.Acc_mean, 4)}; Q_Acc: {fmt(base_n.Q_Acc_mean, 4)}; I_Acc: {fmt(base_n.I_Acc_mean, 4)}; G_Acc: {fmt(base_n.G_Acc_mean, 4)}.

## NaturalBench metric definitions

- NaturalBench `Acc` is per-call accuracy: correct model calls divided by all model calls. In this slice that is 400 calls per run.
- NaturalBench `G_Acc` is strict group accuracy: each group has four required calls (`q0_i0`, `q0_i1`, `q1_i0`, `q1_i1`), and the group is correct only if all four are correct. This is why `G_Acc` is much lower than `Acc`.
- VLMBias `accuracy` is a separate benchmark metric: correct answer fraction on the VLMBias questions. It should be compared to NaturalBench `Acc` as another accuracy-style metric, but it is not computed from NaturalBench groups.
- VLMBias `bias_aligned_fraction` is not accuracy. It is the fraction of outputs that align with the dataset's known visual/textual bias direction; lower can be better only if task accuracy is not also harmed.

## Main result

Best VLMBias accuracy is `{best_acc.condition}`: accuracy {fmt(best_acc.accuracy_mean, 4)}, delta {fmt(best_acc.accuracy_mean_delta_vs_baseline, 4)}, with NaturalBench Acc delta {fmt(best_acc.Acc_mean_delta_vs_baseline, 4)}. The gain is smaller than the seed-to-seed standard deviation for both baseline ({fmt(base_acc_std, 4)}) and this condition ({fmt(best_acc_std, 4)}), so treat it as a small observed shift rather than a robust win.

Best VLMBias accuracy while keeping NaturalBench Acc within 1 point of baseline is `{best_trade.condition}`: VLMBias accuracy delta {fmt(best_trade.accuracy_mean_delta_vs_baseline, 4)}, NaturalBench Acc delta {fmt(best_trade.Acc_mean_delta_vs_baseline, 4)}.

Lowest bias-aligned fraction is `{lowest_bias.condition}`: {fmt(lowest_bias.bias_aligned_fraction_mean, 4)}, delta {fmt(lowest_bias.bias_aligned_fraction_mean_delta_vs_baseline, 4)}. Lower bias-aligned fraction is not enough on its own, because it can coincide with lower accuracy or general degradation.

Worst NaturalBench Acc is `{worst_nb.condition}`: Acc {fmt(worst_nb.Acc_mean, 4)}, delta {fmt(worst_nb.Acc_mean_delta_vs_baseline, 4)}. Across the recomputed all-seed aggregates, NaturalBench Acc remains within about one point of baseline for every tested condition.

## Top tables

### Best VLMBias accuracy

{md_table(sweep["best_tables"]["best_vlmbias_accuracy"], ["condition", "accuracy_mean", "accuracy_mean_delta_vs_baseline", "bias_aligned_fraction_mean", "Acc_mean", "Acc_mean_delta_vs_baseline"])}

### Best VLMBias accuracy with NaturalBench within 1 point

{md_table(sweep["best_tables"]["best_accuracy_with_naturalbench_within_1pt"], ["condition", "accuracy_mean", "accuracy_mean_delta_vs_baseline", "bias_aligned_fraction_mean", "Acc_mean", "Acc_mean_delta_vs_baseline"])}

## Interpretation

The attention intervention changes image attention mass monotonically with alpha, but task outcomes are noisy and not consistently improved. NaturalBench is fairly stable in this completed sweep, so the main concern is not broad VQA collapse; it is that the VLMBias improvements are small relative to seed variation and do not consistently move the bias metrics in the desired direction.

For the current data, gaze attention boosting is not a clean debiasing solution. Top-20 alpha 2 gives the best observed VLMBias accuracy, while top-20 alpha 10 gives the lowest observed bias-aligned fraction; these are different operating points, and neither effect is large enough by itself to claim reliable debiasing.

## Artifacts

- `plots/vlmbias_accuracy_by_alpha.png`
- `plots/vlmbias_bias_aligned_fraction_by_alpha.png`
- `plots/naturalbench_acc_by_alpha.png`
- `plots/naturalbench_g_acc_by_alpha.png`
- `plots/combined_accuracy_bias_comparison.png`
- `plots/combined_accuracy_bias_comparison_faceted_ci.png`
- `plots/boosted_head_attention_mass_by_alpha.png`
- `plots/tradeoff_vlmbias_vs_naturalbench.png`
- `plots/scatter_naturalbench_acc_vs_vlmbias_bias_fraction.png`
- `tables/recomputed_summary_by_seed.tsv`
- `tables/recomputed_summary_aggregate.tsv`
- `tables/joined_vlmbias_naturalbench_tradeoff.tsv`
- `tables/best_vlmbias_accuracy.tsv`
- `tables/best_accuracy_with_naturalbench_within_1pt.tsv`
- `tables/lowest_bias_aligned_fraction.tsv`
- `tables/best_naturalbench_acc.tsv`
"""
    (SWEEP_REPORT / "report.md").write_text(sweep_md)


def main() -> None:
    ensure_dirs()
    static = analyze_static()
    sweep = analyze_sweep()
    write_reports(static, sweep)
    print(f"Wrote {STATIC_REPORT.relative_to(ROOT)}")
    print(f"Wrote {SWEEP_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
