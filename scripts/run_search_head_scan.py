#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from vlm_eval.mechanistic_heads.causal import (
    batched_candidate_margin,
    batched_visual_attention_map_patch_many,
    bounded_head_microbatch,
    candidate_margin,
    capture_teacher_forced,
    repeat_model_inputs,
)
from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, effective_limit, load_json_config, parse_layer_spec, prepare_output_directory
from vlm_eval.mechanistic_heads.checkpoint import JsonlCheckpoint
from vlm_eval.mechanistic_heads.qwen3_runtime import checkpoint_manifest_inputs, runtime_from_config
from vlm_eval.mechanistic_heads.preflight import require_calibration_report, require_scientific_validation, validation_path_from_config
from vlm_eval.mechanistic_heads.reproducibility import hash_paths, referenced_image_paths, seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.scan import symmetric_bidirectional_score
from vlm_eval.mechanistic_heads.schema import read_paired_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan visual attention-map transplantation search heads.")
    add_standard_run_arguments(parser)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--head-microbatch", type=int, default=32)
    parser.add_argument("--layers")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    config = load_json_config(args.config)
    if not args.smoke:
        require_scientific_validation(validation_path_from_config(config));require_calibration_report(Path(config["point_calibration"]),boolean_key="calibration_passed");require_calibration_report(Path(config["waldo_calibration"]),boolean_key="calibration_passed")
    output = args.output_dir / "search_head_scores.tsv"
    checkpoint_path=args.output_dir/"search_head_scores.checkpoint.jsonl"
    prepare_output_directory(args.output_dir, resume=args.resume, overwrite=args.overwrite, known_outputs=(output.name,checkpoint_path.name))
    seed_everything(args.seed)
    runtime = runtime_from_config(
        config, device_map=args.device_map, checkpoint_override=args.checkpoint
    )
    layers = list(range(runtime.architecture.n_layers))
    if args.smoke:
        layers = [0, runtime.architecture.n_layers - 1]
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
    manifest_inputs=[args.config,Path(config["paired_dataset"]),*referenced_image_paths(pairs),*checkpoint_manifest_inputs(config,checkpoint_override=args.checkpoint)]
    checkpoint=JsonlCheckpoint(checkpoint_path,key=lambda row:(row["pair_id"],row["layer"],row["head"]),resume=args.resume,context={"config":config,"seed":args.seed,"smoke":args.smoke,"layers":layers,"input_sha256":hash_paths(manifest_inputs)})
    rows = scan(runtime, pairs=pairs, layers=layers, head_microbatch=args.head_microbatch,checkpoint=checkpoint)
    _write_tsv(output, rows)
    if not args.smoke:manifest_inputs.extend([Path(config["point_calibration"]),Path(config["waldo_calibration"])])
    write_run_manifest(args.output_dir, config={**config, "layers": layers, "smoke": args.smoke, "head_microbatch": args.head_microbatch, "architecture": vars(runtime.architecture), "normalization_rule": "donor visual pattern, recipient visual mass, unchanged nonvisual keys"}, seeds={"global": args.seed}, inputs=manifest_inputs, outputs=[output,checkpoint_path,checkpoint.meta_path], status="complete", repo_root=Path.cwd())
    print(json.dumps({"valid": True, "rows": len(rows), "output": str(output)}, indent=2))


def scan(runtime: Any, *, pairs: list[Any], layers: list[int], head_microbatch: int = 32,checkpoint:JsonlCheckpoint|None=None) -> list[dict[str, Any]]:
    rows = list(checkpoint.rows) if checkpoint else []
    for pair in pairs:
        expected=[(pair.pair_id,layer,head) for layer in layers for head in range(runtime.architecture.n_heads)]
        if checkpoint and all(not checkpoint.missing(key) for key in expected):continue
        donor = capture_teacher_forced(runtime, image_path=pair.donor_image, prompt=pair.donor_prompt, answer=pair.donor_answer, layers=layers)
        recipient = capture_teacher_forced(runtime, image_path=pair.recipient_image, prompt=pair.recipient_prompt, answer=pair.recipient_answer, layers=layers)
        if len(donor.image_positions) != len(recipient.image_positions):
            raise RuntimeError(f"unaligned visual token counts for {pair.pair_id}")
        if donor.answer_length != recipient.answer_length:
            raise RuntimeError(f"unaligned coordinate-sequence token counts for {pair.pair_id}")
        donor_queries = list(range(donor.prompt_length - 1, donor.prompt_length + donor.answer_length - 1))
        recipient_queries = list(range(recipient.prompt_length - 1, recipient.prompt_length + recipient.answer_length - 1))
        effective_head_microbatch = bounded_head_microbatch(
            head_microbatch,
            max(
                donor.prompt_length + donor.answer_length,
                recipient.prompt_length + recipient.answer_length,
            ),
        )
        base_forward, _ = candidate_margin(runtime, recipient.inputs, positive_answer=pair.donor_answer, negative_answer=pair.recipient_answer)
        base_reverse, _ = candidate_margin(runtime, donor.inputs, positive_answer=pair.recipient_answer, negative_answer=pair.donor_answer)
        for layer in layers:
            for start in range(0, runtime.architecture.n_heads, effective_head_microbatch):
                heads = list(range(start, min(start + effective_head_microbatch, runtime.architecture.n_heads)))
                if checkpoint:heads=[head for head in heads if checkpoint.missing((pair.pair_id,layer,head))]
                if not heads:continue
                recipient_batch = repeat_model_inputs(recipient.inputs, len(heads))
                with batched_visual_attention_map_patch_many(runtime.model, layer_idx=layer, head_indices=heads, donor_probabilities=donor.store.attention_probabilities[layer], donor_query_positions=donor_queries, recipient_query_positions=recipient_queries, donor_visual_positions=donor.image_positions, recipient_visual_positions=recipient.image_positions):
                    patched_forward, _ = batched_candidate_margin(runtime, recipient_batch, positive_answer=pair.donor_answer, negative_answer=pair.recipient_answer)
                donor_batch = repeat_model_inputs(donor.inputs, len(heads))
                with batched_visual_attention_map_patch_many(runtime.model, layer_idx=layer, head_indices=heads, donor_probabilities=recipient.store.attention_probabilities[layer], donor_query_positions=recipient_queries, recipient_query_positions=donor_queries, donor_visual_positions=recipient.image_positions, recipient_visual_positions=donor.image_positions):
                    patched_reverse, _ = batched_candidate_margin(runtime, donor_batch, positive_answer=pair.recipient_answer, negative_answer=pair.donor_answer)
                chunk_rows=[]
                for batch_idx, head in enumerate(heads):
                    forward = float(patched_forward[batch_idx].detach().cpu()) - base_forward
                    reverse = float(patched_reverse[batch_idx].detach().cpu()) - base_reverse
                    attention = recipient.store.attention_probabilities[layer][0, head, recipient_queries, :].float().clamp_min(1e-12)
                    image_index = runtime.torch.as_tensor(recipient.image_positions, dtype=runtime.torch.long, device=attention.device)
                    projected = recipient.store.projected_heads[layer][0, recipient_queries, head, :]
                    chunk_rows.append({"pair_id": pair.pair_id, "layer": layer, "head": head, "forward_margin_shift": forward, "reverse_margin_shift": reverse, "search_causal_score": symmetric_bidirectional_score(forward, reverse), "image_attention": float(attention.index_select(-1,image_index).sum(-1).mean().detach().cpu()), "projected_output_norm": float(projected.float().norm(dim=-1).mean().detach().cpu()), "attention_entropy": float((-(attention*attention.log()).sum(-1)).mean().detach().cpu()), "n_coordinate_sequence_queries": donor.answer_length, "normalization_rule": "transplant every aligned answer-query visual pattern while preserving recipient visual mass"})
                rows.extend(chunk_rows)
                if checkpoint:checkpoint.append(chunk_rows)
    return rows


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}) or ["pair_id"], delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
