#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from vlm_eval.mechanistic_heads.causal import (
    batched_candidate_margin,
    batched_projected_head_patch,
    candidate_margin,
    capture_prefill,
    repeat_model_inputs,
    scope_positions,
)
from vlm_eval.mechanistic_heads.config import (
    add_standard_run_arguments,
    effective_limit,
    load_json_config,
    parse_layer_spec,
    prepare_output_directory,
)
from vlm_eval.mechanistic_heads.checkpoint import JsonlCheckpoint
from vlm_eval.mechanistic_heads.qwen3_runtime import Qwen3MechanisticRuntime
from vlm_eval.mechanistic_heads.preflight import require_scientific_validation, validation_path_from_config
from vlm_eval.mechanistic_heads.reproducibility import seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.scan import symmetric_bidirectional_score
from vlm_eval.mechanistic_heads.schema import read_paired_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan exact post-W_O counting-head contributions.")
    add_standard_run_arguments(parser)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--head-microbatch", type=int, default=32)
    parser.add_argument("--layers")
    args = parser.parse_args()
    config = load_json_config(args.config)
    if not args.smoke:
        require_scientific_validation(validation_path_from_config(config))
    output = args.output_dir / "count_head_scores.tsv"
    checkpoint_path = args.output_dir / "count_head_scores.checkpoint.jsonl"
    prepare_output_directory(
        args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
        known_outputs=(output.name,),
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
    pairs = read_paired_jsonl(Path(config["paired_dataset"]))
    limit = effective_limit(args)
    if limit is not None:
        pairs = pairs[:limit]
    checkpoint = JsonlCheckpoint(
        checkpoint_path,
        key=lambda row: (row["pair_id"], int(row["layer"]), int(row["head"]), row["scope"]),
        resume=args.resume,
    )
    rows = run_head_scan(
        runtime,
        pairs=pairs,
        layers=layers,
        scope=str(config.get("patch_scope", "final_prompt_token")),
        head_microbatch=args.head_microbatch,
        checkpoint=checkpoint,
    )
    _write_tsv(output, rows)
    write_run_manifest(
        args.output_dir,
        config={
            **config,
            "layers": layers,
            "smoke": args.smoke,
            "head_microbatch": args.head_microbatch,
            "execution_note": "32 head candidates are scored as layer-local model microbatches",
            "architecture": vars(runtime.architecture),
        },
        seeds={"global": args.seed},
        inputs=[args.config, Path(config["paired_dataset"])],
        outputs=[output, checkpoint_path],
        status="complete",
        repo_root=Path.cwd(),
    )
    print(json.dumps({"valid": True, "rows": len(rows), "output": str(output)}, indent=2))


def run_head_scan(
    runtime: Any,
    *,
    pairs: list[Any],
    layers: list[int],
    scope: str,
    head_microbatch: int = 32,
    checkpoint: JsonlCheckpoint | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = list(checkpoint.rows) if checkpoint else []
    for pair in pairs:
        expected = [(pair.pair_id, layer, head, scope) for layer in layers for head in range(runtime.architecture.n_heads)]
        if checkpoint and all(not checkpoint.missing(key) for key in expected):
            continue
        donor = capture_prefill(runtime, image_path=pair.donor_image, prompt=pair.donor_prompt, layers=layers)
        recipient = capture_prefill(
            runtime, image_path=pair.recipient_image, prompt=pair.recipient_prompt, layers=layers
        )
        if donor.prompt_length != recipient.prompt_length:
            raise RuntimeError(f"unaligned sequences for {pair.pair_id}")
        positions = scope_positions(recipient, scope)
        recipient_margin, _ = candidate_margin(
            runtime,
            recipient.inputs,
            positive_answer=pair.donor_answer,
            negative_answer=pair.recipient_answer,
        )
        donor_margin, _ = candidate_margin(
            runtime,
            donor.inputs,
            positive_answer=pair.recipient_answer,
            negative_answer=pair.donor_answer,
        )
        for layer_idx in layers:
            for start in range(0, runtime.architecture.n_heads, head_microbatch):
                head_indices = list(
                    range(
                        start,
                        min(start + head_microbatch, runtime.architecture.n_heads),
                    )
                )
                if checkpoint:
                    head_indices = [head for head in head_indices if checkpoint.missing((pair.pair_id, layer_idx, head, scope))]
                if not head_indices:
                    continue
                forward_inputs = repeat_model_inputs(recipient.inputs, len(head_indices))
                with batched_projected_head_patch(
                    runtime.model,
                    layer_idx=layer_idx,
                    head_indices=head_indices,
                    donor_projected=donor.store.projected_heads[layer_idx],
                    recipient_projected=recipient.store.projected_heads[layer_idx],
                    positions=positions,
                ):
                    forward_margins, _ = batched_candidate_margin(
                        runtime,
                        forward_inputs,
                        positive_answer=pair.donor_answer,
                        negative_answer=pair.recipient_answer,
                    )
                reverse_positions = scope_positions(donor, scope)
                reverse_inputs = repeat_model_inputs(donor.inputs, len(head_indices))
                with batched_projected_head_patch(
                    runtime.model,
                    layer_idx=layer_idx,
                    head_indices=head_indices,
                    donor_projected=recipient.store.projected_heads[layer_idx],
                    recipient_projected=donor.store.projected_heads[layer_idx],
                    positions=reverse_positions,
                ):
                    reverse_margins, _ = batched_candidate_margin(
                        runtime,
                        reverse_inputs,
                        positive_answer=pair.recipient_answer,
                        negative_answer=pair.donor_answer,
                    )
                chunk_rows = []
                for batch_idx, head_idx in enumerate(head_indices):
                    forward_shift = float(forward_margins[batch_idx].cpu()) - recipient_margin
                    reverse_shift = float(reverse_margins[batch_idx].cpu()) - donor_margin
                    raw = recipient.store.raw_heads[layer_idx][0, positions, head_idx, :]
                    projected = recipient.store.projected_heads[layer_idx][0, positions, head_idx, :]
                    attention = recipient.store.attention_probabilities[layer_idx][
                        0, head_idx, positions, :
                    ]
                    positive_attention = attention.float().clamp_min(1e-12)
                    image_index = runtime.torch.as_tensor(
                        recipient.image_positions,
                        dtype=runtime.torch.long,
                        device=attention.device,
                    )
                    chunk_rows.append(
                        {
                            "pair_id": pair.pair_id,
                            "layer": layer_idx,
                            "head": head_idx,
                            "scope": scope,
                            "forward_margin_shift": forward_shift,
                            "reverse_margin_shift": reverse_shift,
                            "symmetric_causal_score": symmetric_bidirectional_score(
                                forward_shift, reverse_shift
                            ),
                            "image_attention_ratio": float(
                                attention.index_select(-1, image_index)
                                .sum(-1)
                                .mean()
                                .detach()
                                .cpu()
                            ),
                            "raw_output_norm": float(
                                raw.float().norm(dim=-1).mean().detach().cpu()
                            ),
                            "projected_output_norm": float(
                                projected.float().norm(dim=-1).mean().detach().cpu()
                            ),
                            "attention_entropy": float(
                                (-(positive_attention * positive_attention.log()).sum(-1)).mean().detach().cpu()
                            ),
                        }
                    )
                if checkpoint:
                    checkpoint.append(chunk_rows)
                rows.extend(chunk_rows)
    return rows


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["pair_id"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
