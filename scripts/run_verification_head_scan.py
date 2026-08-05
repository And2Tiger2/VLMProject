#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_counting_head_scan import run_head_scan
from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, effective_limit, load_json_config, parse_layer_spec, prepare_output_directory
from vlm_eval.mechanistic_heads.qwen3_runtime import checkpoint_manifest_inputs, runtime_from_config
from vlm_eval.mechanistic_heads.preflight import require_scientific_validation, validation_path_from_config
from vlm_eval.mechanistic_heads.reproducibility import seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.schema import read_paired_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan post-W_O fine-detail verification heads.")
    add_standard_run_arguments(parser)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--layers")
    args = parser.parse_args()
    config = load_json_config(args.config)
    if not args.smoke:
        require_scientific_validation(validation_path_from_config(config))
    output = args.output_dir / "verification_head_scores.tsv"
    prepare_output_directory(args.output_dir, resume=args.resume, overwrite=args.overwrite, known_outputs=(output.name,))
    seed_everything(args.seed)
    runtime = runtime_from_config(config, device_map=args.device_map)
    layers = list(range(runtime.architecture.n_layers))
    if args.smoke:
        layers = [0, runtime.architecture.n_layers - 1]
    cli_layers = parse_layer_spec(args.layers, n_layers=runtime.architecture.n_layers)
    if cli_layers is not None: layers = cli_layers
    pairs = read_paired_jsonl(Path(config["paired_dataset"]))
    limit = effective_limit(args)
    if limit is not None:
        pairs = pairs[:limit]
    rows = run_head_scan(
        runtime,
        pairs=pairs,
        layers=layers,
        scope="final_prompt_token",
        head_microbatch=int(config.get("head_microbatch", 32)),
    )
    for row in rows:
        row["verification_causal_score"] = row.pop("symmetric_causal_score")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["pair_id"], delimiter="\t"); writer.writeheader(); writer.writerows(rows)
    write_run_manifest(args.output_dir, config={**config, "layers": layers, "smoke": args.smoke, "architecture": vars(runtime.architecture)}, seeds={"global": args.seed}, inputs=[args.config, Path(config["paired_dataset"]), *checkpoint_manifest_inputs(config)], outputs=[output], status="complete", repo_root=Path.cwd())
    print(json.dumps({"valid": True, "rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
