#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

from vlm_eval.mechanistic_heads.causal import (
    batched_candidate_margin,
    batched_projected_head_patch,
    bounded_head_microbatch,
    candidate_margin,
    capture_prefill,
    repeat_model_inputs,
)
from vlm_eval.mechanistic_heads.config import (
    add_standard_run_arguments,
    effective_limit,
    load_json_config,
    parse_layer_spec,
    prepare_output_directory,
)
from vlm_eval.mechanistic_heads.qwen3_runtime import Qwen3MechanisticRuntime
from vlm_eval.mechanistic_heads.preflight import require_scientific_validation, validation_path_from_config
from vlm_eval.mechanistic_heads.mmmc import MMMCImages
from vlm_eval.mechanistic_heads.reproducibility import seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.schema import read_paired_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run signed clean-to-conflict MACI head path patching.")
    add_standard_run_arguments(parser)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--scope", choices=["last_prefill", "all_aligned_prefill"], default="last_prefill"
    )
    parser.add_argument("--head-microbatch", type=int, default=32)
    parser.add_argument("--layers")
    args = parser.parse_args()
    config = load_json_config(args.config)
    if not args.smoke:
        require_scientific_validation(validation_path_from_config(config))
    per_example = args.output_dir / "maci_head_scores_per_example.tsv"
    aggregate = args.output_dir / "maci_head_scores.tsv"
    prepare_output_directory(
        args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
        known_outputs=(per_example.name, aggregate.name),
    )
    seed_everything(args.seed)
    runtime = Qwen3MechanisticRuntime(
        model_id=str(config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")),
        device_map=args.device_map,
    )
    layers = list(range(runtime.architecture.n_layers))
    if args.smoke:
        layers = [0, runtime.architecture.n_layers - 1]
    if config.get("layers") is not None:
        layers = [int(value) for value in config["layers"]]
    cli_layers = parse_layer_spec(args.layers, n_layers=runtime.architecture.n_layers)
    if cli_layers is not None: layers = cli_layers
    pairs = [
        pair
        for pair in read_paired_jsonl(Path(config["paired_dataset"]))
        if pair.split == str(config.get("split", "prototype"))
    ]
    limit = effective_limit(args)
    if limit is not None:
        pairs = pairs[:limit]
    images = MMMCImages(cache_dir=args.cache_dir)
    rows = run_maci_scan(runtime, pairs=pairs, layers=layers, scope=args.scope, images=images, head_microbatch=args.head_microbatch)
    _write_tsv(per_example, rows)
    aggregates = aggregate_scores(rows)
    _write_tsv(aggregate, aggregates)
    write_run_manifest(
        args.output_dir,
        config={**config, "scope": args.scope, "layers": layers, "smoke": args.smoke, "head_microbatch": args.head_microbatch, "architecture": vars(runtime.architecture)},
        seeds={"global": args.seed},
        inputs=[args.config, Path(config["paired_dataset"])],
        outputs=[per_example, aggregate],
        status="complete",
        repo_root=Path.cwd(),
    )
    print(json.dumps({"valid": True, "rows": len(rows), "aggregate": str(aggregate)}, indent=2))


def run_maci_scan(
    runtime: Any,
    *,
    pairs: list[Any],
    layers: list[int],
    scope: str,
    images: MMMCImages,
    head_microbatch: int = 32,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        clean = capture_prefill(
            runtime,
            image_path=images.resolve(pair.donor_image),
            prompt=pair.donor_prompt,
            layers=layers,
        )
        conflict = capture_prefill(
            runtime,
            image_path=images.resolve(pair.recipient_image),
            prompt=pair.recipient_prompt,
            layers=layers,
        )
        baseline_l, _ = candidate_margin(
            runtime,
            conflict.inputs,
            positive_answer=pair.bias_answer,
            negative_answer=pair.correct_answer,
        )
        if scope == "last_prefill":
            recipient_positions = [conflict.prompt_length - 1]
            donor_positions = [clean.prompt_length - 1]
        else:
            if conflict.prompt_length != clean.prompt_length:
                rows.append(
                    {
                        "pair_id": pair.pair_id,
                        "layer": None,
                        "head": None,
                        "scope": scope,
                        "excluded": True,
                        "exclusion_reason": "unequal prefill lengths",
                    }
                )
                continue
            recipient_positions = list(range(conflict.prompt_length))
            donor_positions = list(range(clean.prompt_length))
        effective_head_microbatch = bounded_head_microbatch(
            head_microbatch,
            max(clean.prompt_length, conflict.prompt_length),
        )
        for layer_idx in layers:
            for start in range(0, runtime.architecture.n_heads, effective_head_microbatch):
                head_indices = list(range(start, min(start + effective_head_microbatch, runtime.architecture.n_heads)))
                repeated = repeat_model_inputs(conflict.inputs, len(head_indices))
                with batched_projected_head_patch(
                    runtime.model,
                    layer_idx=layer_idx,
                    head_indices=head_indices,
                    donor_projected=clean.store.projected_heads[layer_idx],
                    recipient_projected=conflict.store.projected_heads[layer_idx],
                    positions=recipient_positions,
                    donor_positions=donor_positions,
                ):
                    patched_values, _ = batched_candidate_margin(
                        runtime,
                        repeated,
                        positive_answer=pair.bias_answer,
                        negative_answer=pair.correct_answer,
                    )
                for batch_idx, head_idx in enumerate(head_indices):
                    patched_l = float(patched_values[batch_idx].detach().cpu())
                    attention = conflict.store.attention_probabilities[layer_idx][0, head_idx, recipient_positions, :].float().clamp_min(1e-12)
                    projected = conflict.store.projected_heads[layer_idx][0, recipient_positions, head_idx, :]
                    image_index = runtime.torch.as_tensor(conflict.image_positions, dtype=runtime.torch.long, device=attention.device)
                    rows.append({
                        "pair_id": pair.pair_id,
                        "group_id": pair.group_id,
                        "split": pair.split,
                        "layer": layer_idx,
                        "head": head_idx,
                        "scope": scope,
                        "baseline_hallucination_advantage": baseline_l,
                        "patched_hallucination_advantage": patched_l,
                        "signed_intervention_score": baseline_l - patched_l,
                        "sign_interpretation": "driving" if baseline_l - patched_l > 0 else "resisting",
                        "image_attention": float(attention.index_select(-1, image_index).sum(-1).mean().detach().cpu()),
                        "projected_output_norm": float(projected.float().norm(dim=-1).mean().detach().cpu()),
                        "attention_entropy": float((-(attention * attention.log()).sum(-1)).mean().detach().cpu()),
                        "excluded": False,
                        "exclusion_reason": None,
                    })
    return rows


def aggregate_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        if row.get("excluded"):
            continue
        grouped[(int(row["layer"]), int(row["head"]), str(row["scope"]))].append(row)
    output = []
    for (layer, head, scope), group in sorted(grouped.items()):
        values = [float(row["signed_intervention_score"]) for row in group]
        mean = sum(values) / len(values)
        output.append(
            {
                "layer": layer,
                "head": head,
                "scope": scope,
                "n": len(values),
                "mean_signed_intervention_score": mean,
                "role": "driving" if mean > 0 else "resisting",
                "image_attention": sum(float(row["image_attention"]) for row in group) / len(group),
                "projected_output_norm": sum(float(row["projected_output_norm"]) for row in group) / len(group),
                "attention_entropy": sum(float(row["attention_entropy"]) for row in group) / len(group),
            }
        )
    return output


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["pair_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
