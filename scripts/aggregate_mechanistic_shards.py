#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, load_json_config, prepare_output_directory
from vlm_eval.mechanistic_heads.reproducibility import write_run_manifest


ARTIFACTS = {
    "counting-vap": ["layerwise_vap.tsv"],
    "counting-heads": ["count_head_scores.tsv"],
    "point-centroids": ["centroid_rmse_by_layer.tsv", "attention_centroids_per_point.tsv"],
    "search-heads": ["search_head_scores.tsv"],
    "verification-heads": ["verification_head_scores.tsv"],
    "distractor-heads": ["distractor_head_scores.tsv"],
    "maci-heads": ["maci_head_scores_per_example.tsv", "maci_head_scores.tsv"],
    "maci-heads-aligned": ["maci_head_scores_per_example.tsv", "maci_head_scores.tsv"],
    "vlmbias-heads": ["vlmbias_signed_scores_per_example.tsv", "vlmbias_signed_head_scores.tsv"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate verified per-layer mechanistic scans.")
    add_standard_run_arguments(parser); parser.add_argument("--task", choices=sorted(ARTIFACTS), required=True); parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args(); config = load_json_config(args.config)
    shards = sorted(args.input_root.glob("layer-*"), key=lambda path: int(path.name.split("-")[-1]))
    if not shards:
        raise RuntimeError(f"no layer shards found under {args.input_root}")
    manifests = []
    runtime_architecture = None
    for index, shard in enumerate(shards):
        manifest_path = shard / "run_manifest.json"
        if not manifest_path.is_file(): raise RuntimeError(f"missing shard manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")); arch = manifest["config"]["architecture"]
        observed = (int(arch["n_layers"]), int(arch["n_heads"]))
        if runtime_architecture is None:
            runtime_architecture = observed
        elif observed != runtime_architecture:
            raise RuntimeError(f"architecture mismatch in {manifest_path}: {observed} != {runtime_architecture}")
        if manifest["config"]["layers"] != [index]: raise RuntimeError(f"shard {shard} does not contain layer {index}")
        manifests.append(manifest_path)
    assert runtime_architecture is not None
    n_layers, n_heads = runtime_architecture
    if len(shards) != n_layers:
        raise RuntimeError(f"runtime config reports {n_layers} layers, but found {len(shards)} shards")
    configured_layers = config.get("expected_layers")
    configured_heads = config.get("expected_heads")
    if configured_layers is not None and int(configured_layers) != n_layers:
        raise RuntimeError(f"configured layer expectation {configured_layers} != loaded runtime {n_layers}")
    if configured_heads is not None and int(configured_heads) != n_heads:
        raise RuntimeError(f"configured head expectation {configured_heads} != loaded runtime {n_heads}")
    outputs = [args.output_dir / name for name in ARTIFACTS[args.task]]
    prepare_output_directory(args.output_dir, resume=args.resume, overwrite=args.overwrite, known_outputs=tuple(path.name for path in outputs))
    inputs = list(manifests)
    for name, output in zip(ARTIFACTS[args.task], outputs):
        sources = [shard / name for shard in shards]
        if not all(path.is_file() for path in sources): raise RuntimeError(f"one or more shards are missing {name}")
        merge_tsv(sources, output); inputs.extend(sources)
    write_run_manifest(args.output_dir, config={**config, "task": args.task, "architecture": {"n_layers": n_layers, "n_heads": n_heads}, "layers": list(range(n_layers))}, seeds={"aggregate": args.seed}, inputs=inputs, outputs=outputs, status="complete", repo_root=Path.cwd())
    print(json.dumps({"valid": True, "task": args.task, "shards": len(shards), "outputs": [str(path) for path in outputs]}, indent=2))


def merge_tsv(paths: list[Path], output: Path) -> None:
    fieldnames = None; rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if fieldnames is None: fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames: raise RuntimeError(f"TSV schema mismatch: {path}")
            rows.extend(reader)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"], delimiter="\t"); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
