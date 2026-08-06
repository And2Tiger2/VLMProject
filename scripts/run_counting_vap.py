#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from vlm_eval.mechanistic_heads.causal import (
    candidate_margin,
    capture_prefill,
    module_activation_patch,
    scope_positions,
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
from vlm_eval.mechanistic_heads.reproducibility import seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.schema import read_paired_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run layerwise counting visual activation patching.")
    add_standard_run_arguments(parser)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--layers")
    args = parser.parse_args()
    config = load_json_config(args.config)
    if not args.smoke:
        require_scientific_validation(validation_path_from_config(config))
    output = args.output_dir / "layerwise_vap.tsv"
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
    configured_layers = config.get("layers")
    if configured_layers is not None:
        layers = [int(value) for value in configured_layers]
    cli_layers = parse_layer_spec(args.layers, n_layers=runtime.architecture.n_layers)
    if cli_layers is not None: layers = cli_layers
    pairs = read_paired_jsonl(Path(config["paired_dataset"]))
    limit = effective_limit(args)
    if limit is not None:
        pairs = pairs[:limit]
    rows = run_vap(runtime, pairs=pairs, layers=layers)
    _write_tsv(output, rows)
    write_run_manifest(
        args.output_dir,
        config={**config, "layers": layers, "smoke": args.smoke, "architecture": vars(runtime.architecture)},
        seeds={"global": args.seed},
        inputs=[args.config, Path(config["paired_dataset"])],
        outputs=[output],
        status="complete",
        repo_root=Path.cwd(),
    )
    print(json.dumps({"valid": True, "rows": len(rows), "output": str(output)}, indent=2))


def run_vap(runtime: Any, *, pairs: list[Any], layers: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes = (
        "all_image_tokens",
        "last_image_token",
        "system_tokens",
        "user_prompt",
        "final_prompt_token",
        "all_prefill_positions",
    )
    modules = ("attention_output", "mlp_output", "full_residual")
    for pair in pairs:
        donor = capture_prefill(
            runtime,
            image_path=pair.donor_image,
            prompt=pair.donor_prompt,
            layers=layers,
        )
        recipient = capture_prefill(
            runtime,
            image_path=pair.recipient_image,
            prompt=pair.recipient_prompt,
            layers=layers,
        )
        if donor.prompt_length != recipient.prompt_length:
            raise RuntimeError(f"unaligned prefill lengths for {pair.pair_id}")
        directions = (
            ("donor_to_recipient", donor, recipient, pair.donor_answer, pair.recipient_answer),
            ("recipient_to_donor", recipient, donor, pair.recipient_answer, pair.donor_answer),
        )
        for direction, source, target, source_answer, target_answer in directions:
            baseline_margin, _ = candidate_margin(
                runtime, target.inputs, positive_answer=source_answer, negative_answer=target_answer
            )
            for layer_idx in layers:
                source_by_module = {
                    "attention_output": source.store.attention_outputs[layer_idx],
                    "mlp_output": source.store.mlp_outputs[layer_idx],
                    "full_residual": source.store.layer_outputs[layer_idx],
                }
                for scope in scopes:
                    target_positions = scope_positions(target, scope)
                    source_positions = scope_positions(source, scope)
                    if not target_positions and not source_positions:
                        rows.append({"pair_id": pair.pair_id, "direction": direction, "layer": layer_idx, "scope": scope, "module": "not_applicable", "baseline_source_minus_target_margin": baseline_margin, "patched_source_minus_target_margin": baseline_margin, "margin_shift": 0.0, "answer_flip": 0, "note": "repository template contains no system message"})
                        continue
                    if target_positions != source_positions:
                        raise RuntimeError(
                            f"unaligned {scope} positions for {pair.pair_id}; refusing silent truncation"
                        )
                    for module_kind in modules:
                        with module_activation_patch(
                            runtime.model,
                            layer_idx=layer_idx,
                            module_kind=module_kind,
                            donor_activation=source_by_module[module_kind],
                            positions=target_positions,
                        ):
                            patched_margin, _ = candidate_margin(
                                runtime,
                                target.inputs,
                                positive_answer=source_answer,
                                negative_answer=target_answer,
                            )
                        rows.append(
                            {
                                "pair_id": pair.pair_id,
                                "direction": direction,
                                "layer": layer_idx,
                                "scope": scope,
                                "module": module_kind,
                                "baseline_source_minus_target_margin": baseline_margin,
                                "patched_source_minus_target_margin": patched_margin,
                                "margin_shift": patched_margin - baseline_margin,
                                "answer_flip": int(baseline_margin <= 0 < patched_margin),
                            }
                        )
    return rows


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in rows for key in row}) or ["pair_id"]
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
