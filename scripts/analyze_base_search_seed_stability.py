#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

from vlm_eval.mechanistic_heads.config import (
    add_standard_run_arguments,
    load_json_config,
    prepare_output_directory,
)
from vlm_eval.mechanistic_heads.preflight import require_completed_manifest
from vlm_eval.mechanistic_heads.reproducibility import write_run_manifest


SEGMENT = Path("segments/mechanistic_heads_qwen3_8b")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate frozen-base visual-search results across replication seeds."
    )
    add_standard_run_arguments(parser)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    config = load_json_config(args.config)
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seed stability requires at least two unique seeds")

    summary_path = args.output_dir / "summary.json"
    table_path = args.output_dir / "seed_metrics.tsv"
    prepare_output_directory(
        args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
        known_outputs=(summary_path.name, table_path.name),
    )

    rankings: dict[int, list[dict[str, Any]]] = {}
    ranking_summaries: dict[int, dict[str, Any]] = {}
    validation_summaries: dict[int, dict[str, Any]] = {}
    inputs: list[Path] = [args.config]
    for seed in args.seeds:
        namespace = f"seed-{seed}"
        ranking_root = SEGMENT / f"reports/base_search_heads/{args.mode}/{namespace}"
        validation_root = SEGMENT / f"runs/base_search_validation/{args.mode}/{namespace}"
        ranking_path = ranking_root / "base_search_head_ranking.json"
        ranking_summary_path = ranking_root / "summary.json"
        validation_summary_path = validation_root / "summary.json"
        require_completed_manifest(
            ranking_root,
            expected_outputs=(ranking_path, ranking_summary_path),
            require_current_git=True,
        )
        require_completed_manifest(
            validation_root,
            expected_outputs=(validation_summary_path,),
            require_current_git=True,
        )
        rankings[seed] = json.loads(ranking_path.read_text(encoding="utf-8"))
        ranking_summaries[seed] = json.loads(
            ranking_summary_path.read_text(encoding="utf-8")
        )
        validation_summaries[seed] = json.loads(
            validation_summary_path.read_text(encoding="utf-8")
        )
        inputs.extend(
            [
                ranking_path,
                ranking_summary_path,
                ranking_root / "run_manifest.json",
                validation_summary_path,
                validation_root / "run_manifest.json",
            ]
        )

    behavior_root = SEGMENT / f"runs/base_search_behavior/{args.mode}"
    behavior_path = behavior_root / "summary.json"
    require_completed_manifest(
        behavior_root, expected_outputs=(behavior_path,), require_current_git=True
    )
    behavior = json.loads(behavior_path.read_text(encoding="utf-8"))
    inputs.extend([behavior_path, behavior_root / "run_manifest.json"])

    result, seed_rows = summarize_seed_stability(
        args.seeds,
        rankings=rankings,
        ranking_summaries=ranking_summaries,
        validation_summaries=validation_summaries,
        behavior=behavior,
    )
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_tsv(table_path, seed_rows)
    write_run_manifest(
        args.output_dir,
        config={**config, "mode": args.mode, "replication_seeds": args.seeds},
        seeds={
            f"replication_{index}": seed
            for index, seed in enumerate(args.seeds, start=1)
        },
        inputs=inputs,
        outputs=[summary_path, table_path],
        status="complete",
        repo_root=Path.cwd(),
    )
    print(json.dumps(result, indent=2))


def summarize_seed_stability(
    seeds: list[int],
    *,
    rankings: dict[int, list[dict[str, Any]]],
    ranking_summaries: dict[int, dict[str, Any]],
    validation_summaries: dict[int, dict[str, Any]],
    behavior: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    top_sets: dict[int, set[tuple[int, int]]] = {}
    counts: dict[tuple[int, int], int] = {}
    for seed in seeds:
        top = ranking_summaries[seed]["top_heads"]
        top_set = {(int(row["layer"]), int(row["head"])) for row in top}
        top_sets[seed] = top_set
        for head in top_set:
            counts[head] = counts.get(head, 0) + 1
        validation = validation_summaries[seed]
        intact_accuracy = weighted_condition_metric(
            validation, condition="intact", metric="accuracy"
        )
        search_harm = weighted_condition_metric(
            validation, condition="search_heads", metric="mean_ablation_harm"
        )
        contrasts = validation["contrasts"]
        row = {
            "seed": seed,
            "top_head_layer": int(top[0]["layer"]),
            "top_head": int(top[0]["head"]),
            "intact_accuracy": intact_accuracy,
            "mean_search_head_ablation_harm": search_harm,
            "search_minus_high_image_attention_harm": float(
                contrasts["search_minus_high_image_attention_harm"]
            ),
            "search_minus_random_harm": float(
                contrasts["search_minus_random_harm"]
            ),
        }
        row["passes_seed_claim_gate"] = bool(
            intact_accuracy > 0.25
            and search_harm > 0
            and row["search_minus_high_image_attention_harm"] > 0
            and row["search_minus_random_harm"] > 0
        )
        seed_rows.append(row)

    pairwise = []
    for left, right in combinations(seeds, 2):
        overlap = len(top_sets[left] & top_sets[right])
        union = len(top_sets[left] | top_sets[right])
        pairwise.append(
            {
                "left_seed": left,
                "right_seed": right,
                "top_k_overlap": overlap,
                "top_k_jaccard": overlap / union if union else 0.0,
                "rank_spearman": rank_spearman(rankings[left], rankings[right]),
            }
        )

    consensus = [
        {"layer": layer, "head": head, "seed_count": count}
        for (layer, head), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )
    ]
    return (
        {
            "valid": True,
            "label": "multi-seed frozen-base visual-search head stability",
            "base_model_only": True,
            "trained_checkpoint": None,
            "seeds": seeds,
            "shared_behavior": behavior,
            "per_seed": seed_rows,
            "pairwise_stability": pairwise,
            "mean_top_k_overlap": mean(row["top_k_overlap"] for row in pairwise),
            "mean_rank_spearman": mean(row["rank_spearman"] for row in pairwise),
            "consensus_top_heads": consensus,
            "all_seeds_pass_claim_gate": all(
                row["passes_seed_claim_gate"] for row in seed_rows
            ),
            "claim_gate": (
                "Each seed must have above-four-way-chance intact accuracy, positive "
                "absolute search-head ablation harm, and harm exceeding both controls."
            ),
        },
        seed_rows,
    )


def weighted_condition_metric(
    summary: dict[str, Any], *, condition: str, metric: str
) -> float:
    groups = [
        row
        for name, row in summary["by_condition"].items()
        if name.endswith(f":{condition}")
    ]
    total = sum(int(row["n"]) for row in groups)
    if not total:
        raise ValueError(f"validation summary has no {condition!r} rows")
    return sum(float(row[metric]) * int(row["n"]) for row in groups) / total


def rank_spearman(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> float:
    left_ranks = {
        (int(row["layer"]), int(row["head"])): index
        for index, row in enumerate(left, 1)
    }
    right_ranks = {
        (int(row["layer"]), int(row["head"])): index
        for index, row in enumerate(right, 1)
    }
    keys = sorted(left_ranks.keys() & right_ranks.keys())
    if len(keys) < 2:
        return 0.0
    x = [left_ranks[key] for key in keys]
    y = [right_ranks[key] for key in keys]
    x_mean, y_mean = mean(x), mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = (
        sum((a - x_mean) ** 2 for a in x)
        * sum((b - y_mean) ** 2 for b in y)
    ) ** 0.5
    return numerator / denominator if denominator else 0.0


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
