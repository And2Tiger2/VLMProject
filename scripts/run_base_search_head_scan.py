#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from vlm_eval.mechanistic_heads.base_search import (
    attention_density,
    assert_base_only,
    assert_unmodified_runtime,
    build_search_probe,
    configured_cues,
    exemplar_source_rows,
    find_exemplars,
    read_jsonl,
    visual_region_indices,
)
from vlm_eval.mechanistic_heads.causal import capture_teacher_forced
from vlm_eval.mechanistic_heads.checkpoint import JsonlCheckpoint
from vlm_eval.mechanistic_heads.config import (
    add_standard_run_arguments,
    effective_limit,
    enforce_smoke_layer_limit,
    load_json_config,
    parse_layer_spec,
    prepare_output_directory,
)
from vlm_eval.mechanistic_heads.qwen3_runtime import Qwen3MechanisticRuntime
from vlm_eval.mechanistic_heads.reproducibility import hash_paths, referenced_image_paths, seed_everything, write_run_manifest


def materialize_projected_query_heads(projected_heads: Any, queries: list[int]) -> Any:
    """Materialize only answer-query contributions from a lazy head capture."""

    selected = projected_heads[0, queries, :, :]
    materialize = getattr(selected, "materialize", None)
    if callable(materialize):
        selected = materialize()
    if selected.ndim != 3:
        raise ValueError(
            "projected query heads must have [query, head, model_width] dimensions"
        )
    return selected.float()


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover frozen-base visual-search heads using gaze-style ROI routing.")
    add_standard_run_arguments(parser)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--layers")
    args = parser.parse_args()
    config = load_json_config(args.config)
    assert_base_only(config)
    output = args.output_dir / "base_search_head_scores.tsv"
    checkpoint_path = args.output_dir / "base_search_head_scores.checkpoint.jsonl"
    prepare_output_directory(
        args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
        known_outputs=(output.name, checkpoint_path.name),
    )
    seed_everything(args.seed)
    all_rows = read_jsonl(Path(config["dataset"]))
    exemplars = find_exemplars(all_rows)
    rows = [
        row for row in all_rows
        if row["split"] == str(config.get("discovery_split", "train")) and row.get("target_present")
    ]
    limit = effective_limit(args, smoke_max=4)
    if limit is None:
        limit = int(config.get("discovery_examples", 64))
    rows = rows[:limit]
    runtime = Qwen3MechanisticRuntime(
        model_id=str(config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")),
        device_map=args.device_map,
    )
    assert_unmodified_runtime(runtime)
    layers = list(range(runtime.architecture.n_layers))
    if args.smoke:
        layers = [0, runtime.architecture.n_layers - 1]
    cli_layers = parse_layer_spec(args.layers, n_layers=runtime.architecture.n_layers)
    if cli_layers is not None:
        layers = cli_layers
    layers = enforce_smoke_layer_limit(args, layers)
    manifest_inputs = [args.config, Path(config["dataset"]), *referenced_image_paths([*rows, *exemplar_source_rows(all_rows)])]
    context = {
        "config": config,
        "seed": args.seed,
        "smoke": args.smoke,
        "layers": layers,
        "input_sha256": hash_paths(manifest_inputs),
    }
    checkpoint = JsonlCheckpoint(
        checkpoint_path,
        key=lambda row: (row["id"], row["cue_mode"], row["layer"], row["head"]),
        resume=args.resume,
        context=context,
    )
    records = list(checkpoint.rows)
    for row in rows:
        for cue_mode in configured_cues(config):
            expected = [
                (row["id"], cue_mode, layer, head)
                for layer in layers
                for head in range(runtime.architecture.n_heads)
            ]
            if all(not checkpoint.missing(key) for key in expected):
                continue
            probe = build_search_probe(row, cue_mode=cue_mode, exemplars=exemplars)
            capture = capture_teacher_forced(
                runtime,
                image_path=probe.image,
                prompt=probe.prompt,
                answer=probe.answer,
                layers=layers,
            )
            regions = visual_region_indices(capture.inputs, runtime, capture.image_positions, probe.masks)
            queries = list(range(capture.prompt_length - 1, capture.prompt_length + capture.answer_length - 1))
            chunk: list[dict[str, Any]] = []
            for layer in layers:
                probabilities = capture.store.attention_probabilities[layer][0, :, queries, :].float()
                projected = materialize_projected_query_heads(
                    capture.store.projected_heads[layer], queries
                )
                for head in range(runtime.architecture.n_heads):
                    key = (row["id"], cue_mode, layer, head)
                    if not checkpoint.missing(key):
                        continue
                    attention = probabilities[head]
                    densities = {
                        f"candidate_{index}": attention_density(
                            attention,
                            regions[f"candidate_{index}"],
                            total_image_tokens=len(capture.image_positions),
                        )
                        for index in range(4)
                    }
                    target_density = densities[f"candidate_{probe.target_candidate}"]
                    decoy_density = sum(
                        value for name, value in densities.items()
                        if name != f"candidate_{probe.target_candidate}"
                    ) / 3.0
                    image_attention = float(
                        attention[..., capture.image_positions].sum(dim=-1).mean().detach().cpu()
                    )
                    safe = attention.clamp_min(1e-12)
                    chunk.append(
                        {
                            "id": row["id"],
                            "group_id": row["group_id"],
                            "cue_mode": cue_mode,
                            "layer": layer,
                            "head": head,
                            "target_candidate": probe.target_candidate,
                            "target_candidate_density": target_density,
                            "mean_decoy_candidate_density": decoy_density,
                            "target_selectivity": target_density - decoy_density,
                            "routing_correct": int(target_density == max(densities.values())),
                            "target_object_density": attention_density(attention, regions["target_object"], total_image_tokens=len(capture.image_positions)),
                            "reference_density": attention_density(attention, regions["reference"], total_image_tokens=len(capture.image_positions)),
                            "scene_density": attention_density(attention, regions["scene"], total_image_tokens=len(capture.image_positions)),
                            "image_attention": image_attention,
                            "attention_entropy": float((-(safe * safe.log()).sum(dim=-1)).mean().detach().cpu()),
                            "projected_output_norm": float(projected[:, head, :].norm(dim=-1).mean().detach().cpu()),
                            "n_query_tokens": len(queries),
                            "score_definition": "correct-candidate attention density minus mean matched-decoy density",
                        }
                    )
            records.extend(chunk)
            checkpoint.append(chunk)
    write_tsv(output, records)
    write_run_manifest(
        args.output_dir,
        config={**config, "smoke": args.smoke, "layers": layers, "architecture": vars(runtime.architecture), "base_model_only": True},
        seeds={"global": args.seed},
        inputs=manifest_inputs,
        outputs=[output, checkpoint_path, checkpoint.meta_path],
        status="complete",
        repo_root=Path.cwd(),
    )
    print(json.dumps({"valid": True, "base_model_only": True, "rows": len(records), "output": str(output)}, indent=2))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}) or ["id"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
