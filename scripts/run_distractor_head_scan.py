#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from vlm_eval.mechanistic_heads.causal import batched_candidate_margin, batched_head_scaling, bounded_head_microbatch, candidate_margin, capture_prefill, repeat_model_inputs
from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, effective_limit, load_json_config, parse_layer_spec, prepare_output_directory
from vlm_eval.mechanistic_heads.preflight import require_scientific_validation, validation_path_from_config
from vlm_eval.mechanistic_heads.qwen3_runtime import checkpoint_manifest_inputs, runtime_from_config
from vlm_eval.mechanistic_heads.reproducibility import seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.schema import read_paired_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank heads whose ablation increases strong-decoy selection.")
    add_standard_run_arguments(parser)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--head-microbatch", type=int, default=32)
    parser.add_argument("--layers")
    args = parser.parse_args()
    config = load_json_config(args.config)
    if not args.smoke:
        require_scientific_validation(validation_path_from_config(config))
    output = args.output_dir / "distractor_head_scores.tsv"
    prepare_output_directory(args.output_dir, resume=args.resume, overwrite=args.overwrite, known_outputs=(output.name,))
    seed_everything(args.seed)
    runtime = runtime_from_config(config, device_map=args.device_map)
    layers = list(range(runtime.architecture.n_layers))
    if args.smoke:
        layers = [0, runtime.architecture.n_layers - 1]
    cli_layers = parse_layer_spec(args.layers, n_layers=runtime.architecture.n_layers)
    if cli_layers is not None: layers = cli_layers
    pairs = read_paired_jsonl(Path(config["paired_dataset"])); limit = effective_limit(args)
    if limit is not None: pairs = pairs[:limit]
    rows = []
    for pair in pairs:
        image = Image.open(pair.recipient_image).convert("RGB")
        capture = capture_prefill(runtime, image_path=image, prompt=pair.recipient_prompt, layers=layers)
        inputs = capture.inputs
        effective_head_microbatch = bounded_head_microbatch(
            args.head_microbatch, capture.prompt_length
        )
        baseline, _ = candidate_margin(runtime, inputs, positive_answer=pair.donor_answer, negative_answer=pair.recipient_answer)
        for layer in layers:
            for start in range(0, runtime.architecture.n_heads, effective_head_microbatch):
                heads = list(range(start, min(start + effective_head_microbatch, runtime.architecture.n_heads)))
                repeated = repeat_model_inputs(inputs, len(heads))
                with batched_head_scaling(runtime.model, layer_idx=layer, head_indices=heads, scale=0.0):
                    ablated_values, _ = batched_candidate_margin(runtime, repeated, positive_answer=pair.donor_answer, negative_answer=pair.recipient_answer)
                for batch_idx, head in enumerate(heads):
                    ablated = float(ablated_values[batch_idx].detach().cpu())
                    query=[capture.prompt_length-1];attention=capture.store.attention_probabilities[layer][0,head,query,:].float().clamp_min(1e-12);image_index=runtime.torch.as_tensor(capture.image_positions,dtype=runtime.torch.long,device=attention.device);projected=capture.store.projected_heads[layer][0,query,head,:]
                    rows.append({"pair_id": pair.pair_id, "layer": layer, "head": head, "baseline_correct_vs_decoy_margin": baseline, "ablated_correct_vs_decoy_margin": ablated, "distractor_suppression_score": baseline - ablated, "image_attention":float(attention.index_select(-1,image_index).sum(-1).mean().detach().cpu()),"projected_output_norm":float(projected.float().norm(dim=-1).mean().detach().cpu()),"attention_entropy":float((-(attention*attention.log()).sum(-1)).mean().detach().cpu())})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["pair_id"], delimiter="\t"); writer.writeheader(); writer.writerows(rows)
    write_run_manifest(args.output_dir, config={**config, "layers": layers, "smoke": args.smoke, "architecture": vars(runtime.architecture)}, seeds={"global": args.seed}, inputs=[args.config, Path(config["paired_dataset"]), *checkpoint_manifest_inputs(config)], outputs=[output], status="complete", repo_root=Path.cwd())
    print(json.dumps({"valid": True, "rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__": main()
