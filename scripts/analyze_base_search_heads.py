#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, load_json_config, prepare_output_directory
from vlm_eval.mechanistic_heads.reproducibility import write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank frozen-base gaze-style search heads and select causal controls.")
    add_standard_run_arguments(parser)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    config = load_json_config(args.config)
    source = args.source or Path(config["merged_head_scores"])
    rows = read_tsv(source)
    ranking_cues = tuple(str(value) for value in config.get("ranking_cues", ["text", "target_exemplar"]))
    ranking = rank_heads(rows, ranking_cues=ranking_cues)
    top_k = int(config.get("top_k", 16))
    controls = select_controls(ranking, top_k=top_k, seed=args.seed)
    ranking_path = args.output_dir / "base_search_head_ranking.json"
    summary_path = args.output_dir / "summary.json"
    table_path = args.output_dir / "base_search_head_ranking.tsv"
    prepare_output_directory(
        args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
        known_outputs=(ranking_path.name, summary_path.name, table_path.name),
    )
    ranking_path.write_text(json.dumps(ranking, indent=2), encoding="utf-8")
    write_tsv(table_path, ranking)
    summary = {
        "valid": True,
        "base_model_only": True,
        "trained_checkpoint": None,
        "n_heads": len(ranking),
        "ranking_cues": list(ranking_cues),
        "score_definition": "mean correct-candidate attention-density advantage over three matched decoys",
        "top_k": top_k,
        "top_heads": controls["search_heads"],
        "controls": controls,
        "top_head": ranking[0] if ranking else None,
        "interpretation": "This is gaze-style passive discovery; causal claims require the locked validation stage.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_manifest(
        args.output_dir,
        config={**config, "ranking_cues": ranking_cues, "base_model_only": True},
        seeds={"control_selection": args.seed},
        inputs=[args.config, source],
        outputs=[ranking_path, table_path, summary_path],
        status="complete",
        repo_root=Path.cwd(),
    )
    print(json.dumps(summary, indent=2))


def rank_heads(rows: list[dict[str, str]], *, ranking_cues: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["cue_mode"] in ranking_cues:
            grouped[(int(row["layer"]), int(row["head"]))].append(row)
    ranking: list[dict[str, Any]] = []
    for (layer, head), group in grouped.items():
        cue_scores = {
            cue: mean(float(row["target_selectivity"]) for row in group if row["cue_mode"] == cue)
            for cue in ranking_cues
        }
        halves = []
        for parity in (0, 1):
            half = [row for row in group if stable_parity(row["id"]) == parity]
            halves.append(mean(float(row["target_selectivity"]) for row in half))
        ranking.append(
            {
                "layer": layer,
                "head": head,
                "score": mean(cue_scores.values()),
                "minimum_cue_score": min(cue_scores.values()),
                "text_score": cue_scores.get("text"),
                "target_exemplar_score": cue_scores.get("target_exemplar"),
                "routing_accuracy": mean(float(row["routing_correct"]) for row in group),
                "target_object_density": mean(float(row["target_object_density"]) for row in group),
                "image_attention": mean(float(row["image_attention"]) for row in group),
                "projected_output_norm": mean(float(row["projected_output_norm"]) for row in group),
                "split_half_0_score": halves[0],
                "split_half_1_score": halves[1],
                "split_half_sign_agreement": int(halves[0] * halves[1] > 0),
                "n_rows": len(group),
            }
        )
    ranking.sort(key=lambda row: (row["minimum_cue_score"], row["routing_accuracy"], row["score"]), reverse=True)
    for index, row in enumerate(ranking, 1):
        row["rank"] = index
    return ranking


def select_controls(ranking: list[dict[str, Any]], *, top_k: int, seed: int) -> dict[str, list[dict[str, int]]]:
    if top_k <= 0 or top_k > len(ranking):
        raise ValueError(f"top_k must lie in 1..{len(ranking)}")
    search = ranking[:top_k]
    selected = {(row["layer"], row["head"]) for row in search}
    remaining = [row for row in ranking if (row["layer"], row["head"]) not in selected]
    image = sorted(remaining, key=lambda row: row["image_attention"], reverse=True)[:top_k]
    image_selected = {(row["layer"], row["head"]) for row in image}
    random_pool = [row for row in remaining if (row["layer"], row["head"]) not in image_selected]
    random_pool.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['layer']}:{row['head']}".encode()).hexdigest())
    random_rows = random_pool[:top_k]
    def encode(values: list[dict[str, Any]]) -> list[dict[str, int]]:
        return [{"layer": int(row["layer"]), "head": int(row["head"])} for row in values]

    return {
        "search_heads": encode(search),
        "high_image_attention_control": encode(image),
        "random_control": encode(random_rows),
    }


def stable_parity(value: str) -> int:
    return hashlib.sha256(value.encode()).digest()[0] % 2


def mean(values: Any) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["layer"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
