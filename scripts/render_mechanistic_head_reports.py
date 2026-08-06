#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from vlm_eval.mechanistic_heads.atlas import ATLAS_COLUMNS, initialize_head_atlas, write_head_atlas
from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, load_json_config, prepare_output_directory
from vlm_eval.mechanistic_heads.reproducibility import write_run_manifest


SCORE_SOURCES = {
    "count_causal_score": ("count_scores", "symmetric_causal_score"),
    "search_causal_score": ("search_scores", "search_causal_score"),
    "verification_causal_score": ("verification_scores", "verification_causal_score"),
    "distractor_suppression_score": ("distractor_scores", "distractor_suppression_score"),
    "mmmc_signed_score": ("mmmc_scores", "mean_signed_intervention_score"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the unified mechanistic head atlas and reports.")
    add_standard_run_arguments(parser)
    args = parser.parse_args()
    config = load_json_config(args.config)
    atlas_path = args.output_dir / "unified_head_atlas.tsv"
    prepare_output_directory(args.output_dir, resume=args.resume, overwrite=args.overwrite, known_outputs=(atlas_path.name,))
    n_layers, n_heads, architecture_source = resolve_architecture(config)
    rows = initialize_head_atlas(n_layers, n_heads)
    by_head = {(row["layer"], row["head"]): row for row in rows}
    inputs = [args.config]
    gaze_path = Path(config["gaze_ranking"])
    if gaze_path.is_file():
        inputs.append(gaze_path)
        for gaze in json.loads(gaze_path.read_text(encoding="utf-8")):
            head = by_head.get((int(gaze["layer"]), int(gaze["head"])))
            if head is not None:
                head["comic_gaze_score"] = gaze.get("score", gaze.get("gaze_score"))
    for atlas_column, (config_key, score_column) in SCORE_SOURCES.items():
        path_value = config.get(config_key)
        if not path_value:
            continue
        path = Path(path_value)
        if not path.is_file():
            continue
        inputs.append(path)
        aggregates = _aggregate_tsv(path, score_column)
        for key, value in aggregates.items():
            if key in by_head:
                by_head[key][atlas_column] = value
    count_value = config.get("count_scores")
    if count_value and Path(count_value).is_file():
        count_path = Path(count_value)
        for atlas_column, source_column in (("image_attention", "image_attention_ratio"), ("projected_output_norm", "projected_output_norm")):
            aggregates = _aggregate_tsv(count_path, source_column)
            for key, value in aggregates.items():
                if key in by_head: by_head[key][atlas_column] = value
    vlmbias_value = config.get("vlmbias_scores")
    if vlmbias_value and Path(vlmbias_value).is_file():
        path = Path(vlmbias_value); inputs.append(path)
        with path.open("r", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle, delimiter="\t"))
        for contrast, column in {
            "semantic_prior": "vlmbias_semantic_prior_score",
            "context": "vlmbias_context_score",
            "detail": "vlmbias_detail_score",
        }.items():
            grouped = _aggregate_rows([row for row in source_rows if row.get("contrast") == contrast], "mean_signed_score")
            for key, value in grouped.items():
                if key in by_head:
                    by_head[key][column] = value
        attribution = _aggregate_rows(source_rows, "direct_correct_vs_bias_logit_attribution")
        for key, value in attribution.items():
            if key in by_head: by_head[key]["correct_vs_bias_logit_attribution"] = value
    write_head_atlas(atlas_path, rows)
    figures = render_figures(rows, args.output_dir)
    overlap_path = args.output_dir / "top_k_overlaps.tsv"
    _write_overlaps(rows, overlap_path, k=int(config.get("top_k", 50)))
    correlation_path = args.output_dir / "rank_correlations.tsv"
    _write_correlations(rows, correlation_path)
    double_path = args.output_dir / "task_by_head_set_double_dissociation.tsv"
    _write_double_dissociation(rows, double_path, k=int(config.get("top_k", 50)))
    validation_statuses = load_validation_statuses(config, inputs)
    status = args.output_dir / "STATUS.md"
    status.write_text(_status_markdown(rows, figures, inputs, validation_statuses), encoding="utf-8")
    write_run_manifest(args.output_dir, config={**config, "verified_architecture": {"n_layers": n_layers, "n_heads": n_heads, "source": architecture_source}}, seeds={"render": args.seed}, inputs=inputs, outputs=[atlas_path, overlap_path, correlation_path, double_path, status, *figures], status="complete", repo_root=Path.cwd())
    print(json.dumps({"valid": True, "atlas": str(atlas_path), "figures": [str(path) for path in figures]}, indent=2))


def resolve_architecture(config: dict[str, Any]) -> tuple[int, int, str]:
    """Resolve dimensions from a loaded model config or completed run manifest."""
    for key, _ in SCORE_SOURCES.values():
        value = config.get(key)
        if not value:
            continue
        manifest = Path(value).parent / "run_manifest.json"
        if manifest.is_file():
            arch = json.loads(manifest.read_text(encoding="utf-8")).get("config", {}).get("architecture")
            if arch:
                dimensions = (int(arch["n_layers"]), int(arch["n_heads"]))
                if dimensions != (36, 32):
                    raise RuntimeError(f"unexpected Qwen3 architecture in {manifest}: {dimensions}")
                return *dimensions, str(manifest)
    instrumentation = config.get("status_sources", {}).get("instrumentation")
    if instrumentation:
        manifest = Path(instrumentation).parent / "run_manifest.json"
        if manifest.is_file():
            arch = json.loads(manifest.read_text(encoding="utf-8")).get("config", {}).get("architecture")
            if arch:
                dimensions = (int(arch["n_layers"]), int(arch["n_heads"]))
                if dimensions != (36, 32):
                    raise RuntimeError(f"unexpected Qwen3 architecture in {manifest}: {dimensions}")
                return *dimensions, str(manifest)
    try:
        from transformers import AutoConfig
        loaded = AutoConfig.from_pretrained(str(config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")), local_files_only=True)
        text_config = loaded.text_config
        dimensions = (int(text_config.num_hidden_layers), int(text_config.num_attention_heads))
    except Exception as exc:
        raise RuntimeError(
            "cannot verify atlas dimensions: supply a completed score run_manifest.json or cache the Qwen3 model config"
        ) from exc
    if dimensions != (36, 32):
        raise RuntimeError(f"unexpected Qwen3 architecture: {dimensions}")
    return *dimensions, "AutoConfig.text_config"


def _aggregate_tsv(path: Path, score_column: str) -> dict[tuple[int, int], float]:
    with path.open("r", encoding="utf-8") as handle:
        return _aggregate_rows(list(csv.DictReader(handle, delimiter="\t")), score_column)


def _aggregate_rows(rows: list[dict[str, str]], score_column: str) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        if row.get("layer") in (None, "") or row.get("head") in (None, "") or row.get(score_column) in (None, ""):
            continue
        values.setdefault((int(row["layer"]), int(row["head"])), []).append(float(row[score_column]))
    return {key: sum(group) / len(group) for key, group in values.items()}


def render_figures(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Report rendering requires `uv sync --extra mechanistic`.") from exc
    score_columns = [column for column in ATLAS_COLUMNS[2:] if any(row.get(column) is not None for row in rows)]
    paths = []
    if score_columns:
        figure, axes = plt.subplots(len(score_columns), 1, figsize=(12, max(3.2, 2.5 * len(score_columns))), squeeze=False, constrained_layout=True)
        n_layers = max(row["layer"] for row in rows) + 1
        n_heads = max(row["head"] for row in rows) + 1
        for axis, column in zip(axes[:, 0], score_columns):
            matrix = np.full((n_layers, n_heads), np.nan)
            for row in rows:
                if row.get(column) is not None:
                    matrix[row["layer"], row["head"]] = float(row[column])
            scale = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1
            image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-scale, vmax=scale, interpolation="nearest")
            axis.set_title(column.replace("_", " ").title(), loc="left", fontweight="bold")
            axis.set_xlabel("Head"); axis.set_ylabel("Layer")
            figure.colorbar(image, ax=axis, fraction=0.02, pad=0.02)
        path = output_dir / "functional_head_heatmaps.png"
        figure.savefig(path, dpi=220, facecolor="white"); plt.close(figure); paths.append(path)
        for column in score_columns:
            matrix = np.full((n_layers, n_heads), np.nan)
            for row in rows:
                if row.get(column) is not None: matrix[row["layer"], row["head"]] = float(row[column])
            scale = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1
            figure, axis = plt.subplots(figsize=(10, 6.2), constrained_layout=True)
            image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-scale, vmax=scale, interpolation="nearest")
            axis.set_title(column.replace("_", " ").title(), loc="left", fontweight="bold"); axis.set_xlabel("Head"); axis.set_ylabel("Language layer"); figure.colorbar(image, ax=axis, label="Causal score")
            single = output_dir / f"{column}_heatmap.png"; figure.savefig(single, dpi=220, facecolor="white"); plt.close(figure); paths.append(single)
    numeric = [column for column in score_columns if column != "comic_gaze_score"]
    if numeric and any(row.get("comic_gaze_score") is not None for row in rows):
        figure, axes = plt.subplots(1, len(numeric), figsize=(4.3 * len(numeric), 4), squeeze=False, constrained_layout=True)
        for axis, column in zip(axes[0], numeric):
            pairs = [(row.get("comic_gaze_score"), row.get(column)) for row in rows if row.get("comic_gaze_score") is not None and row.get(column) is not None]
            if pairs:
                axis.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=13, alpha=0.55, color="#31688e")
            axis.axhline(0, color="#777777", linewidth=0.8); axis.set_xlabel("Comic gaze score"); axis.set_ylabel(column.replace("_", " "))
        path = output_dir / "gaze_vs_mechanistic_scores.png"
        figure.savefig(path, dpi=220, facecolor="white"); plt.close(figure); paths.append(path)
    if score_columns:
        figure, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
        for column in score_columns:
            layer_values = []
            for layer in range(max(row["layer"] for row in rows) + 1):
                values = [abs(float(row[column])) for row in rows if row["layer"] == layer and row.get(column) is not None]
                layer_values.append(float(np.mean(values)) if values else np.nan)
            axis.plot(layer_values, linewidth=1.8, label=column.replace("_", " "))
        axis.set_xlabel("Language layer"); axis.set_ylabel("Mean |score| across heads")
        axis.set_title("Where each mechanistic function is concentrated", loc="left", fontweight="bold")
        axis.legend(fontsize=7, ncol=2, frameon=False)
        path = output_dir / "functional_layer_distributions.png"
        figure.savefig(path, dpi=220, facecolor="white"); plt.close(figure); paths.append(path)
    complete_columns = [column for column in score_columns if sum(row.get(column) is not None for row in rows) >= 3]
    if len(complete_columns) >= 2:
        matrix = np.full((len(complete_columns), len(complete_columns)), np.nan)
        for i, left in enumerate(complete_columns):
            for j, right in enumerate(complete_columns):
                pairs = [(float(row[left]), float(row[right])) for row in rows if row.get(left) is not None and row.get(right) is not None]
                if len(pairs) >= 3:
                    matrix[i, j] = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
        figure, axis = plt.subplots(figsize=(max(6, .65 * len(complete_columns)), max(5, .58 * len(complete_columns))), constrained_layout=True)
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        labels = [column.replace("_score", "").replace("_", " ") for column in complete_columns]
        axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(range(len(labels)), labels=labels, fontsize=8)
        axis.set_title("Head-function rank correlations", loc="left", fontweight="bold")
        figure.colorbar(image, ax=axis, label="Spearman rho", fraction=.04)
        path = output_dir / "head_function_rank_correlations.png"
        figure.savefig(path, dpi=220, facecolor="white"); plt.close(figure); paths.append(path)
        feature_rows = [row for row in rows if all(row.get(column) is not None for column in complete_columns)]
        if len(feature_rows) >= 3:
            features = np.asarray([[float(row[column]) for column in complete_columns] for row in feature_rows], dtype=float)
            features = (features - features.mean(0)) / np.where(features.std(0) == 0, 1, features.std(0))
            _, _, vh = np.linalg.svd(features, full_matrices=False)
            order = np.argsort(features @ vh[0])
            figure, axis = plt.subplots(figsize=(max(7, .7 * len(complete_columns)), max(5, len(feature_rows) / 45)), constrained_layout=True)
            image = axis.imshow(features[order], aspect="auto", cmap="coolwarm", interpolation="nearest")
            axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right", fontsize=8)
            axis.set_ylabel("Heads ordered by functional profile")
            axis.set_title("Clustered functional head atlas", loc="left", fontweight="bold")
            figure.colorbar(image, ax=axis, label="Standardized score", fraction=.025)
            path = output_dir / "clustered_functional_atlas.png"
            figure.savefig(path, dpi=220, facecolor="white"); plt.close(figure); paths.append(path)
    return paths


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]: end += 1
        rank = (start + end - 1) / 2
        for idx in order[start:end]: ranks[idx] = rank
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    import numpy as np
    if len(left) < 2: return float("nan")
    return float(np.corrcoef(_rank(left), _rank(right))[0, 1])


def _write_correlations(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [column for column in ATLAS_COLUMNS[2:] if any(row.get(column) is not None for row in rows)]
    output = []
    for left_idx, left in enumerate(columns):
        for right in columns[left_idx + 1:]:
            pairs = [(float(row[left]), float(row[right])) for row in rows if row.get(left) is not None and row.get(right) is not None]
            output.append({"left": left, "right": right, "n": len(pairs), "spearman_rho": _spearman([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) >= 3 else ""})
    if not output:
        output.append({"left": "pending", "right": "pending", "n": 0, "spearman_rho": ""})
    _write_table(path, output, ["left", "right", "n", "spearman_rho"])


def _write_double_dissociation(rows: list[dict[str, Any]], path: Path, *, k: int) -> None:
    columns = [column for column in ATLAS_COLUMNS[2:] if any(row.get(column) is not None for row in rows)]
    sets = {column: sorted([row for row in rows if row.get(column) is not None], key=lambda row: abs(float(row[column])), reverse=True)[:k] for column in columns}
    output = []
    for head_set, selected in sets.items():
        for measured in columns:
            values = [float(row[measured]) for row in selected if row.get(measured) is not None]
            output.append({"head_set": f"top{k}_{head_set}", "measured_task": measured, "n": len(values), "mean_signed_score": sum(values) / len(values) if values else "", "mean_absolute_score": sum(abs(value) for value in values) / len(values) if values else ""})
    if not output:
        output.append({"head_set": "pending", "measured_task": "pending", "n": 0, "mean_signed_score": "", "mean_absolute_score": ""})
    _write_table(path, output, ["head_set", "measured_task", "n", "mean_signed_score", "mean_absolute_score"])


def _write_table(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t"); writer.writeheader(); writer.writerows(rows)


def _write_overlaps(rows: list[dict[str, Any]], path: Path, *, k: int) -> None:
    columns = [column for column in ATLAS_COLUMNS[2:] if any(row.get(column) is not None for row in rows)]
    sets = {column: {(row["layer"], row["head"]) for row in sorted([item for item in rows if item.get(column) is not None], key=lambda item: abs(float(item[column])), reverse=True)[:k]} for column in columns}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["left", "right", "k", "intersection", "jaccard"], delimiter="\t"); writer.writeheader()
        wrote = False
        for left_idx, left in enumerate(columns):
            for right in columns[left_idx + 1:]:
                intersection = len(sets[left] & sets[right]); union = len(sets[left] | sets[right])
                writer.writerow({"left": left, "right": right, "k": k, "intersection": intersection, "jaccard": intersection / union if union else 0})
                wrote = True
        if not wrote:
            writer.writerow({"left": "pending", "right": "pending", "k": k, "intersection": 0, "jaccard": ""})


def load_validation_statuses(config: dict[str, Any], inputs: list[Path]) -> dict[str, dict[str, Any]]:
    statuses = {}
    for name, value in config.get("status_sources", {}).items():
        path = Path(value)
        if not path.is_file():
            statuses[name] = {
                "valid": None,
                "label": "computationally pending",
                "missing_path": str(path),
            }
        else:
            statuses[name] = json.loads(path.read_text(encoding="utf-8"))
            inputs.append(path)
    return statuses


def _status_markdown(rows: list[dict[str, Any]], figures: list[Path], inputs: list[Path], validation_statuses: dict[str, dict[str, Any]]) -> str:
    available = [column for column in ATLAS_COLUMNS[2:] if any(row.get(column) is not None for row in rows)]
    passed=[];failed=[];pending=[]
    for name, summary in validation_statuses.items():
        label=str(summary.get("label","unlabeled"))
        calibration=str(summary.get("calibration_result",""))
        item=f"{name}: {label}"+(f" ({calibration})" if calibration else "")
        if "pending" in label:
            pending.append(item)
        elif summary.get("valid") is not True or "failed" in label or "failed" in calibration:
            failed.append(item)
        else:
            passed.append(item)
    absent=[column for column in ATLAS_COLUMNS[2:] if column not in available]
    if absent:pending.append("Missing atlas score families: "+", ".join(absent))
    return "\n".join(["# Mechanistic Heads Status", "", "## Implemented", "", "- Unified runtime-verified 36x32 head atlas and report renderer.", "- Available score families: " + (", ".join(available) if available else "none"), "", "## Run", "", *([f"- {name}" for name in validation_statuses] or ["- No validation summaries were supplied."]), "", "## Passed", "", *([f"- {item}" for item in passed] or ["- None."]), f"- Joined {len(rows)} architecture slots from {len(inputs)} hashed input artifacts.", "", "## Failed", "", *([f"- {item}" for item in failed] or ["- None reported by supplied summaries."]), "", "## Computationally pending", "", *([f"- {item}" for item in pending] or ["- No atlas score family is missing."]), "- VLMBias detail contrast remains pending if original factual images were unavailable during preparation.", "", "## Figures", "", *[f"- `{path.name}`" for path in figures], "", "## Deviations", "", "- See `../IMPLEMENTATION_PLAN.md`; causal labels remain provisional unless their stability and locked-control gates pass.", ""])


if __name__ == "__main__":
    main()
