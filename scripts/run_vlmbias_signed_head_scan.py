#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

from vlm_eval.mechanistic_heads.causal import batched_candidate_margin, batched_projected_head_patch, candidate_margin, capture_prefill, repeat_model_inputs
from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, effective_limit, load_json_config, parse_layer_spec, prepare_output_directory
from vlm_eval.mechanistic_heads.qwen3_runtime import Qwen3MechanisticRuntime
from vlm_eval.mechanistic_heads.preflight import require_scientific_validation, validation_path_from_config
from vlm_eval.mechanistic_heads.reproducibility import seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.schema import read_paired_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan contrast-specific signed VLMBias heads.")
    add_standard_run_arguments(parser)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--scope", choices=["last_prefill", "all_aligned_prefill"], default="last_prefill")
    parser.add_argument("--head-microbatch", type=int, default=32)
    parser.add_argument("--layers")
    args = parser.parse_args()
    config = load_json_config(args.config)
    if not args.smoke:
        require_scientific_validation(validation_path_from_config(config))
    per_example = args.output_dir / "vlmbias_signed_scores_per_example.tsv"
    aggregate = args.output_dir / "vlmbias_signed_head_scores.tsv"
    prepare_output_directory(args.output_dir, resume=args.resume, overwrite=args.overwrite, known_outputs=(per_example.name, aggregate.name))
    seed_everything(args.seed)
    runtime = Qwen3MechanisticRuntime(model_id=str(config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")), device_map=args.device_map)
    layers = list(range(runtime.architecture.n_layers))
    if args.smoke:
        layers = [0, runtime.architecture.n_layers - 1]
    cli_layers = parse_layer_spec(args.layers, n_layers=runtime.architecture.n_layers)
    if cli_layers is not None: layers = cli_layers
    pairs = [pair for pair in read_paired_jsonl(Path(config["paired_dataset"])) if pair.split == str(config.get("split", "prototype"))]
    limit = effective_limit(args)
    if limit is not None:
        pairs = pairs[:limit]
    rows = scan(runtime, pairs=pairs, layers=layers, scope=args.scope, head_microbatch=args.head_microbatch)
    _write_tsv(per_example, rows)
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    attributions: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    diagnostics: dict[tuple[str, int, int], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        if not row.get("excluded"):
            grouped[(row["contrast"], row["layer"], row["head"])].append(row["signed_score"])
            if row.get("direct_correct_vs_bias_logit_attribution") is not None:
                attributions[(row["contrast"], row["layer"], row["head"])].append(float(row["direct_correct_vs_bias_logit_attribution"]))
            diagnostics[(row["contrast"], row["layer"], row["head"])].append(row)
    aggregate_rows = [
        {"contrast": key[0], "layer": key[1], "head": key[2], "n": len(values), "mean_signed_score": sum(values) / len(values), "image_attention": sum(float(row["image_attention"]) for row in diagnostics[key]) / len(diagnostics[key]), "projected_output_norm": sum(float(row["projected_output_norm"]) for row in diagnostics[key]) / len(diagnostics[key]), "attention_entropy": sum(float(row["attention_entropy"]) for row in diagnostics[key]) / len(diagnostics[key]), "direct_correct_vs_bias_logit_attribution": (sum(attributions[key]) / len(attributions[key]) if attributions[key] else None)}
        for key, values in sorted(grouped.items())
    ]
    _write_tsv(aggregate, aggregate_rows)
    write_run_manifest(args.output_dir, config={**config, "scope": args.scope, "layers": layers, "smoke": args.smoke, "head_microbatch": args.head_microbatch, "architecture": vars(runtime.architecture)}, seeds={"global": args.seed}, inputs=[args.config, Path(config["paired_dataset"])], outputs=[per_example, aggregate], status="complete", repo_root=Path.cwd())
    print(json.dumps({"valid": True, "rows": len(rows), "aggregate": str(aggregate)}, indent=2))


def scan(runtime: Any, *, pairs: list[Any], layers: list[int], scope: str, head_microbatch: int = 32) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        donor = capture_prefill(runtime, image_path=pair.donor_image, prompt=pair.donor_prompt, layers=layers)
        recipient = capture_prefill(runtime, image_path=pair.recipient_image, prompt=pair.recipient_prompt, layers=layers)
        baseline, _ = candidate_margin(runtime, recipient.inputs, positive_answer=pair.bias_answer, negative_answer=pair.correct_answer)
        correct_ids = runtime.answer_token_ids(pair.correct_answer)
        bias_ids = runtime.answer_token_ids(pair.bias_answer)
        single_token_candidates = correct_ids.shape[1] == bias_ids.shape[1] == 1
        unembedding_delta = None
        if single_token_candidates:
            unembedding_delta = runtime.model.lm_head.weight[int(correct_ids[0, 0])] - runtime.model.lm_head.weight[int(bias_ids[0, 0])]
        if scope == "all_aligned_prefill" and donor.prompt_length != recipient.prompt_length:
            rows.append({"pair_id": pair.pair_id, "contrast": pair.metadata["contrast"], "excluded": True, "reason": "unequal prefill lengths"})
            continue
        donor_positions = [donor.prompt_length - 1] if scope == "last_prefill" else list(range(donor.prompt_length))
        recipient_positions = [recipient.prompt_length - 1] if scope == "last_prefill" else list(range(recipient.prompt_length))
        # Detail contrasts require exact processed visual-token alignment.
        if pair.metadata["contrast"] == "detail" and donor.image_positions != recipient.image_positions:
            rows.append({"pair_id": pair.pair_id, "contrast": "detail", "excluded": True, "reason": "processed visual-token spans differ"})
            continue
        for layer in layers:
            for start in range(0, runtime.architecture.n_heads, head_microbatch):
                heads = list(range(start, min(start + head_microbatch, runtime.architecture.n_heads)))
                repeated = repeat_model_inputs(recipient.inputs, len(heads))
                with batched_projected_head_patch(runtime.model, layer_idx=layer, head_indices=heads, donor_projected=donor.store.projected_heads[layer], recipient_projected=recipient.store.projected_heads[layer], positions=recipient_positions, donor_positions=donor_positions):
                    patched_values, _ = batched_candidate_margin(runtime, repeated, positive_answer=pair.bias_answer, negative_answer=pair.correct_answer)
                for batch_idx, head in enumerate(heads):
                    patched = float(patched_values[batch_idx].detach().cpu())
                    attention = recipient.store.attention_probabilities[layer][0, head, recipient_positions, :].float().clamp_min(1e-12)
                    projected = recipient.store.projected_heads[layer][0, recipient_positions, head, :]
                    image_index = runtime.torch.as_tensor(recipient.image_positions, dtype=runtime.torch.long, device=attention.device)
                    direct = None
                    if unembedding_delta is not None:
                        contribution = recipient.store.projected_heads[layer][0, recipient.prompt_length - 1, head, :]
                        direct = float((contribution.float() * unembedding_delta.float()).sum().detach().cpu())
                    rows.append({"pair_id": pair.pair_id, "group_id": pair.group_id, "split": pair.split, "contrast": pair.metadata["contrast"], "layer": layer, "head": head, "baseline_bias_margin": baseline, "patched_bias_margin": patched, "signed_score": baseline - patched, "image_attention": float(attention.index_select(-1, image_index).sum(-1).mean().detach().cpu()), "projected_output_norm": float(projected.float().norm(dim=-1).mean().detach().cpu()), "attention_entropy": float((-(attention * attention.log()).sum(-1)).mean().detach().cpu()), "direct_correct_vs_bias_logit_attribution": direct, "attribution_definition": "pre-final-norm projected head dot (correct-first-token minus bias-first-token unembedding); single-token candidates only", "excluded": False})
    return rows


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["pair_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
