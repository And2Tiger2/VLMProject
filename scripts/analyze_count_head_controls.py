#!/usr/bin/env python3
"""Aggregate count heads, stability, gaze overlap, and matched controls."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, load_json_config, prepare_output_directory
from vlm_eval.mechanistic_heads.controls import layer_matched_control_draws, multivariate_matched_control_draws
from vlm_eval.mechanistic_heads.reproducibility import write_run_manifest
from vlm_eval.mechanistic_heads.io import write_tsv
from vlm_eval.mechanistic_heads.preflight import require_current_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Build held-out count-head control distributions.")
    add_standard_run_arguments(parser)
    args = parser.parse_args()
    config = load_json_config(args.config)
    outputs = [args.output_dir / name for name in ("count_head_ranking.tsv", "count_head_controls.tsv", "count_gaze_overlap.tsv", "count_head_stability.tsv")]
    prepare_output_directory(args.output_dir, resume=args.resume, overwrite=args.overwrite, known_outputs=tuple(path.name for path in outputs))
    source = Path(config["count_scores"])
    source_manifest = require_current_artifact(source)
    rows = read_tsv(source)
    architecture = source_manifest["config"]["architecture"]
    n_layers, n_heads = int(architecture["n_layers"]), int(architecture["n_heads"])
    if (n_layers, n_heads) != (36, 32): raise RuntimeError(f"unexpected Qwen3 architecture: {(n_layers, n_heads)}")
    gaze = {(int(row["layer"]), int(row["head"])): float(row.get("score", row.get("gaze_score", 0))) for row in json.loads(Path(config["gaze_ranking"]).read_text(encoding="utf-8"))}
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows: grouped[(int(row["layer"]), int(row["head"]))].append(row)
    ranking = []
    for head, group in grouped.items():
        ranking.append({"layer": head[0], "head": head[1], "n": len(group), "count_causal_score": mean(group, "symmetric_causal_score"), "image_attention": mean(group, "image_attention_ratio"), "projected_output_norm": mean(group, "projected_output_norm"), "attention_entropy": mean(group, "attention_entropy"), "gaze_score": gaze.get(head, 0.0)})
    ranking.sort(key=lambda row: abs(row["count_causal_score"]), reverse=True)
    write_tsv(outputs[0], ranking)
    features = {(row["layer"], row["head"]): {key: float(row[key]) for key in ("image_attention", "projected_output_norm", "attention_entropy", "gaze_score")} for row in ranking}
    general_path = config.get("general_causal_importance")
    if general_path and not Path(general_path).is_file():
        raise RuntimeError(f"required general causal importance artifact is missing: {general_path}")
    if general_path and Path(general_path).is_file():
        require_current_artifact(Path(general_path))
        for row in read_tsv(Path(general_path)):
            features[(int(row["layer"]), int(row["head"]))]["general_causal_importance"] = float(row["general_causal_importance"])
    control_rows = []
    overlap_rows = []
    gaze_ranked = sorted(gaze, key=lambda head: abs(gaze[head]), reverse=True)
    control_draws = max(20, int(config.get("control_draws", 20)))
    for k in (10, 25, 50):
        selected = [(row["layer"], row["head"]) for row in ranking[:k]]
        gaze_set = set(gaze_ranked[:k]); selected_set = set(selected)
        overlap_rows.append({"k": k, "intersection": len(selected_set & gaze_set), "union": len(selected_set | gaze_set), "jaccard": len(selected_set & gaze_set) / len(selected_set | gaze_set)})
        families = {
            "layer_matched": layer_matched_control_draws(selected, n_layers=n_layers, n_heads=n_heads, n_draws=control_draws, seed=args.seed),
            "image_attention_matched": multivariate_matched_control_draws(selected, features, feature_names=("image_attention",), n_draws=control_draws, seed=args.seed + 1),
            "output_norm_matched": multivariate_matched_control_draws(selected, features, feature_names=("projected_output_norm",), n_draws=control_draws, seed=args.seed + 2),
            "attention_entropy_matched": multivariate_matched_control_draws(selected, features, feature_names=("attention_entropy",), n_draws=control_draws, seed=args.seed + 3),
            "gaze_score_matched": multivariate_matched_control_draws(selected, features, feature_names=("gaze_score",), n_draws=control_draws, seed=args.seed + 4),
        }
        if general_path and Path(general_path).is_file():
            families["general_importance_matched"] = multivariate_matched_control_draws(selected, features, feature_names=("general_causal_importance",), n_draws=control_draws, seed=args.seed + 5)
            families["fully_matched"] = multivariate_matched_control_draws(selected, features, n_draws=control_draws, seed=args.seed + 6)
        low = [(row["layer"], row["head"]) for row in sorted(ranking, key=lambda row: abs(row["count_causal_score"]))[:k]]
        families["low_count_score"] = [low]
        for family, draws in families.items():
            for draw, heads in enumerate(draws):
                for layer, head in heads: control_rows.append({"selected_k": k, "control_family": family, "draw": draw, "layer": layer, "head": head})
    write_tsv(outputs[1], control_rows); write_tsv(outputs[2], overlap_rows)
    stability = split_half_stability(rows)
    configured_repeats = [Path(value) for value in config.get("count_score_repeats", [])]
    missing_repeats = [path for path in configured_repeats if not path.is_file()]
    if missing_repeats:
        raise RuntimeError(f"configured count-score repeats are missing: {missing_repeats}")
    for path in configured_repeats: require_current_artifact(path)
    repeat_paths = configured_repeats
    repeat_rankings = [aggregate_head_scores(read_tsv(path)) for path in repeat_paths]
    for left in range(len(repeat_rankings)):
        for right in range(left + 1, len(repeat_rankings)):
            common = sorted(set(repeat_rankings[left]) & set(repeat_rankings[right]))
            stability.append({"comparison": f"cross_seed_{left}_{right}", "n_pairs_half_1": "", "n_pairs_half_2": "", "n_heads": len(common), "spearman_rho": correlation(ranks([repeat_rankings[left][head] for head in common]), ranks([repeat_rankings[right][head] for head in common])) if common else None})
    write_tsv(outputs[3], stability)
    figures = render(ranking, stability, args.output_dir, n_layers=n_layers, n_heads=n_heads)
    status = {"valid": True, "label": "methods-based reproduction", "n_heads": len(ranking), "control_draws": control_draws, "cross_seed_repeats": len(repeat_rankings), "cross_seed_status": "computed" if len(repeat_rankings) >= 2 else "computationally pending", "general_causal_importance": "provided" if general_path and Path(general_path).is_file() else "unavailable; fully matched controls intentionally withheld", "figures": [str(path) for path in figures]}
    status_path = args.output_dir / "summary.json"; status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    report_path=args.output_dir/"report.md";report_path.write_text("\n".join(["# Counting-head discovery report","","- Label: methods-based reproduction","- Heads are ranked by symmetric bidirectional candidate-margin shift, not attention mass.",f"- Runtime-verified architecture: {n_layers} layers × {n_heads} heads.",f"- Matched-control draws per family: {control_draws}.",f"- Cross-seed rank status: {status['cross_seed_status']}.","- Do not declare count heads until locked necessity/sufficiency, constant-complexity, answer-code, sham, relocation, renderer-transfer, and matched-control gates pass.","","## PNG files","",*[f"- `{path.name}`" for path in figures]]),encoding="utf-8")
    manifest_inputs=[args.config, source, Path(config["gaze_ranking"])]
    if general_path and Path(general_path).is_file(): manifest_inputs.append(Path(general_path))
    write_run_manifest(args.output_dir, config=config, seeds={"controls": args.seed}, inputs=manifest_inputs, outputs=[*outputs, *figures, status_path,report_path], status="complete", repo_root=Path.cwd())
    print(json.dumps(status, indent=2))


def split_half_stability(rows: list[dict]) -> list[dict]:
    pairs = sorted({row["pair_id"] for row in rows}); midpoint = len(pairs) // 2; halves = [set(pairs[:midpoint]), set(pairs[midpoint:])]
    scores = []
    for half in halves:
        grouped = defaultdict(list)
        for row in rows:
            if row["pair_id"] in half: grouped[(int(row["layer"]), int(row["head"]))].append(float(row["symmetric_causal_score"]))
        scores.append({head: sum(values) / len(values) for head, values in grouped.items()})
    common = sorted(set(scores[0]) & set(scores[1]))
    rank0 = ranks([scores[0][head] for head in common]); rank1 = ranks([scores[1][head] for head in common])
    rho = correlation(rank0, rank1) if common else None
    return [{"comparison": "split_half", "n_pairs_half_1": len(halves[0]), "n_pairs_half_2": len(halves[1]), "n_heads": len(common), "spearman_rho": rho}]


def aggregate_head_scores(rows: list[dict]) -> dict[tuple[int, int], float]:
    grouped = defaultdict(list)
    for row in rows: grouped[(int(row["layer"]), int(row["head"]))].append(float(row["symmetric_causal_score"]))
    return {head: sum(values) / len(values) for head, values in grouped.items()}


def render(ranking: list[dict], stability: list[dict], output: Path, *, n_layers: int, n_heads: int) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError: return []
    matrix = np.full((n_layers, n_heads), np.nan)
    for row in ranking: matrix[row["layer"], row["head"]] = row["count_causal_score"]
    scale = np.nanmax(np.abs(matrix)) or 1
    figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True); image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-scale, vmax=scale); axis.set_xlabel("Head"); axis.set_ylabel("Layer"); axis.set_title("Bidirectional count-head causal scores", loc="left", fontweight="bold"); figure.colorbar(image, ax=axis, label="Symmetric margin shift")
    heatmap = output / "count_head_heatmap.png"; figure.savefig(heatmap, dpi=220, facecolor="white"); plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True); axis.scatter([row["gaze_score"] for row in ranking], [row["count_causal_score"] for row in ranking], s=14, alpha=.55); axis.axhline(0, color="gray", linewidth=.8); axis.set_xlabel("Comic gaze score"); axis.set_ylabel("Count causal score"); axis.set_title("Gaze is not assumed to imply counting", loc="left", fontweight="bold")
    scatter = output / "gaze_vs_count_scatter.png"; figure.savefig(scatter, dpi=220, facecolor="white"); plt.close(figure)
    finite=[row for row in stability if row.get("spearman_rho") not in (None,"")]
    figure,axis=plt.subplots(figsize=(7.5,4.6),constrained_layout=True);axis.bar([row["comparison"] for row in finite],[float(row["spearman_rho"]) for row in finite],color="#35b779");axis.axhline(.5,color="#666666",linestyle="--",linewidth=1);axis.set_ylim(-1,1);axis.set_ylabel("Spearman rank correlation");axis.set_title("Count-head rank stability",loc="left",fontweight="bold");axis.tick_params(axis="x",rotation=25)
    stability_path=output/"count_head_stability.png";figure.savefig(stability_path,dpi=220,facecolor="white");plt.close(figure)
    return [heatmap, scatter,stability_path]


def ranks(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: values[index]); result = [0] * len(values)
    for rank, index in enumerate(order): result[index] = rank
    return result


def correlation(left: list[float], right: list[float]) -> float:
    import math
    lm, rm = sum(left)/len(left), sum(right)/len(right); numerator = sum((a-lm)*(b-rm) for a,b in zip(left,right)); denominator = math.sqrt(sum((a-lm)**2 for a in left)*sum((b-rm)**2 for b in right)); return numerator / denominator if denominator else 0.0


def mean(rows: list[dict], key: str) -> float: return sum(float(row[key]) for row in rows) / len(rows)
def read_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle: return list(csv.DictReader(handle, delimiter="\t"))
if __name__ == "__main__": main()
