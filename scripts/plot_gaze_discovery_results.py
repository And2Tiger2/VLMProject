from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SVG plots for merged Qwen2.5-VL gaze discovery results.")
    parser.add_argument("--run-dir", default="segments/gaze_heads_qwen25/runs/gaze_discovery_merged_0_500")
    parser.add_argument("--out-dir", default="segments/gaze_heads_qwen25/reports/discovery_0_500_plots")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranking = json.loads((run_dir / "gaze_head_ranking.json").read_text(encoding="utf-8"))
    scores = np.load(run_dir / "gaze_scores.npy")
    stability = _read_tsv(run_dir / "top20_stability.tsv")

    _write_bar_chart(out_dir / "top20_gaze_scores.svg", ranking[:20], title="Top 20 gaze-head scores")
    _write_bar_chart(out_dir / "top100_gaze_scores.svg", ranking[:100], title="Top 100 gaze-head scores", label_every=5)
    _write_score_heatmap(out_dir / "layer_head_score_heatmap.svg", scores)
    _write_stability_heatmap(out_dir / "top20_shard_rank_stability.svg", stability)

    print(f"Wrote plots to {out_dir}")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _head_label(row: dict[str, Any]) -> str:
    return f"L{int(row['layer'])}H{int(row['head'])}"


def _svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<style>\n'
        'text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937} '
        '.small{font-size:12px}.tiny{font-size:10px}.title{font-size:22px;font-weight:700} '
        '.axis{stroke:#475569;stroke-width:1}.grid{stroke:#e2e8f0;stroke-width:1} '
        '</style>\n'
        f"{body}\n</svg>\n"
    )


def _write_bar_chart(path: Path, ranking: list[dict[str, Any]], *, title: str, label_every: int = 1) -> None:
    width = 1320
    height = 560
    margin_left = 76
    margin_right = 34
    margin_top = 68
    margin_bottom = 110
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_score = max(float(row["score"]) for row in ranking)
    bar_gap = 2
    bar_w = max(2, (plot_w - bar_gap * (len(ranking) - 1)) / len(ranking))

    parts = [
        '<rect x="0" y="0" width="1320" height="560" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="36" class="title">{title}</text>',
        f'<text x="{margin_left}" y="56" class="small">Merged 500-comic discovery run. Higher score means stronger diagonal panel attention.</text>',
    ]

    for tick in range(5):
        value = max_score * tick / 4
        y = margin_top + plot_h - (value / max_score) * plot_h
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" class="small">{value:.3f}</text>')

    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" class="axis"/>')

    for idx, row in enumerate(ranking):
        score = float(row["score"])
        x = margin_left + idx * (bar_w + bar_gap)
        h = (score / max_score) * plot_h
        y = margin_top + plot_h - h
        color = "#2563eb" if idx == 0 else "#64748b"
        if idx < 20 and int(row["layer"]) in {32, 33}:
            color = "#0f766e"
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}"/>')
        if idx % label_every == 0:
            label = _head_label(row)
            parts.append(
                f'<text x="{x + bar_w / 2:.2f}" y="{margin_top + plot_h + 18}" text-anchor="end" '
                f'transform="rotate(-55 {x + bar_w / 2:.2f},{margin_top + plot_h + 18})" class="tiny">{label}</text>'
            )

    path.write_text(_svg(width, height, "\n".join(parts)), encoding="utf-8")


def _write_score_heatmap(path: Path, scores: np.ndarray) -> None:
    n_layers, n_heads = scores.shape
    cell = 20
    margin_left = 72
    margin_top = 76
    width = margin_left + n_heads * cell + 180
    height = margin_top + n_layers * cell + 54
    max_score = float(scores.max())
    min_score = float(scores.min())
    parts = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="36" class="title">Layer-head gaze score heatmap</text>',
        f'<text x="{margin_left}" y="56" class="small">Rows are layers, columns are heads. Darker cells have stronger gaze scores.</text>',
    ]
    for head in range(n_heads):
        x = margin_left + head * cell + cell / 2
        parts.append(f'<text x="{x:.1f}" y="{margin_top - 10}" text-anchor="middle" class="tiny">H{head}</text>')
    for layer in range(n_layers):
        y = margin_top + layer * cell + cell / 2 + 4
        if layer % 2 == 0 or layer in {5, 32, 33}:
            parts.append(f'<text x="{margin_left - 10}" y="{y:.1f}" text-anchor="end" class="tiny">L{layer}</text>')
        for head in range(n_heads):
            x0 = margin_left + head * cell
            y0 = margin_top + layer * cell
            parts.append(
                f'<rect x="{x0}" y="{y0}" width="{cell - 1}" height="{cell - 1}" '
                f'fill="{_blue_scale(float(scores[layer, head]), min_score, max_score)}"/>'
            )
    _legend(parts, margin_left + n_heads * cell + 34, margin_top, max_score, "score")
    path.write_text(_svg(width, height, "\n".join(parts)), encoding="utf-8")


def _write_stability_heatmap(path: Path, stability: list[dict[str, str]]) -> None:
    width = 900
    row_h = 24
    margin_left = 90
    margin_top = 76
    cell = 44
    height = margin_top + row_h * len(stability) + 60
    parts = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="36" class="title">Top-20 stability across shards</text>',
        f'<text x="{margin_left}" y="56" class="small">Cells show how many of 10 shards placed each head in top-k buckets.</text>',
    ]
    cols = [("top5_batches", "top5"), ("top10_batches", "top10"), ("top20_batches", "top20"), ("top50_batches", "top50")]
    for idx, (_, label) in enumerate(cols):
        x = margin_left + idx * cell + cell / 2
        parts.append(f'<text x="{x:.1f}" y="{margin_top - 12}" text-anchor="middle" class="small">{label}</text>')
    for row_idx, row in enumerate(stability):
        y = margin_top + row_idx * row_h
        parts.append(f'<text x="{margin_left - 12}" y="{y + 16}" text-anchor="end" class="small">{row["head"]}</text>')
        for col_idx, (field, _) in enumerate(cols):
            value = int(row[field])
            x = margin_left + col_idx * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{row_h - 2}" '
                f'fill="{_green_scale(value / 10)}"/>'
            )
            parts.append(f'<text x="{x + cell / 2:.1f}" y="{y + 16}" text-anchor="middle" class="tiny">{value}</text>')
    path.write_text(_svg(width, height, "\n".join(parts)), encoding="utf-8")


def _legend(parts: list[str], x: int, y: int, max_value: float, label: str) -> None:
    h = 180
    for idx in range(h):
        value = max_value * (1 - idx / max(1, h - 1))
        parts.append(f'<rect x="{x}" y="{y + idx}" width="18" height="1" fill="{_blue_scale(value, 0.0, max_value)}"/>')
    parts.append(f'<text x="{x + 28}" y="{y + 5}" class="tiny">{max_value:.3f}</text>')
    parts.append(f'<text x="{x + 28}" y="{y + h}" class="tiny">0.000</text>')
    parts.append(f'<text x="{x}" y="{y + h + 24}" class="small">{label}</text>')


def _blue_scale(value: float, min_value: float, max_value: float) -> str:
    if max_value <= min_value:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (value - min_value) / (max_value - min_value)))
    low = np.array([239, 246, 255])
    high = np.array([30, 64, 175])
    rgb = (low + (high - low) * t).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _green_scale(t: float) -> str:
    t = max(0.0, min(1.0, t))
    low = np.array([240, 253, 244])
    high = np.array([21, 128, 61])
    rgb = (low + (high - low) * t).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


if __name__ == "__main__":
    main()
