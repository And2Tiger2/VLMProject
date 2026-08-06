#!/usr/bin/env python3
"""Measure coordinate-token attention centroids for Point-Answer models."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from PIL import Image

from vlm_eval.mechanistic_heads.capture import Qwen3CaptureHooks
from vlm_eval.mechanistic_heads.checkpoint import JsonlCheckpoint
from vlm_eval.mechanistic_heads.likelihood import append_answer_tokens
from vlm_eval.mechanistic_heads.io import write_tsv
from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, effective_limit, enforce_smoke_layer_limit, load_json_config, parse_layer_spec, prepare_output_directory
from vlm_eval.mechanistic_heads.preflight import require_calibration_report, require_scientific_validation, validation_path_from_config
from vlm_eval.mechanistic_heads.qwen3_runtime import checkpoint_manifest_inputs, runtime_from_config
from vlm_eval.mechanistic_heads.reproducibility import hash_paths, referenced_image_paths, seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.synthetic import point_condition_prompt
from vlm_eval.mechanistic_heads.token_spans import locate_subsequence


COORDINATE = re.compile(r"\((\d{3}),(\d{3})\)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Point-token attention centroid calibration.")
    add_standard_run_arguments(parser); parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--checkpoint")
    parser.add_argument("--layers")
    args = parser.parse_args(); config = load_json_config(args.config)
    if not args.smoke:
        require_scientific_validation(validation_path_from_config(config));require_calibration_report(Path(config["point_calibration"]),boolean_key="calibration_passed")
    output = args.output_dir / "centroid_rmse_by_layer.tsv"
    per_point = args.output_dir / "attention_centroids_per_point.tsv"
    checkpoint_path=args.output_dir/"attention_centroids.checkpoint.jsonl"
    prepare_output_directory(args.output_dir, resume=args.resume, overwrite=args.overwrite, known_outputs=(output.name, per_point.name,checkpoint_path.name))
    seed_everything(args.seed)
    rows = [row for row in read_jsonl(Path(config["dataset"])) if row.get("split") == str(config.get("split", "ood_target_count_1"))]
    limit = effective_limit(args)
    if limit is not None: rows = rows[:limit]
    runtime = runtime_from_config(
        config, device_map=args.device_map, checkpoint_override=args.checkpoint
    )
    layers = list(range(runtime.architecture.n_layers));
    if args.smoke: layers = [0, runtime.architecture.n_layers - 1]
    cli_layers = parse_layer_spec(args.layers, n_layers=runtime.architecture.n_layers)
    if cli_layers is not None: layers = cli_layers
    layers = enforce_smoke_layer_limit(args, layers)
    manifest_inputs=[args.config, Path(config["dataset"]), *referenced_image_paths(rows), *checkpoint_manifest_inputs(config, checkpoint_override=args.checkpoint)]
    checkpoint=JsonlCheckpoint(checkpoint_path,key=lambda row:(row["id"],row["point_index"],row["layer"]),resume=args.resume,context={"config":config,"seed":args.seed,"smoke":args.smoke,"layers":layers,"input_sha256":hash_paths(manifest_inputs)})
    records = list(checkpoint.rows)
    for row in rows:
        n_points=len(COORDINATE.findall(str(row["answers"]["point"])))
        missing_layers=[layer for layer in layers if any(checkpoint.missing((row["id"],point,layer)) for point in range(n_points))]
        if not missing_layers:continue
        fresh=trace_example(runtime,row,missing_layers);checkpoint.append(fresh);records.extend(fresh)
    write_tsv(per_point, records)
    summaries = []
    for layer in layers:
        group = [row for row in records if row["layer"] == layer]
        summaries.append({"layer": layer, "n_points": len(group), "centroid_rmse_pixels": math.sqrt(sum(row["squared_error"] for row in group) / len(group)) if group else None, "normalization": "positive excess above uniform visual attention; raw visual attention fallback"})
    write_tsv(output, summaries)
    summary = {"valid": bool(records), "label": "instrumentation smoke test" if args.smoke else "modified replication", "n_examples": len(rows), "n_point_layer_rows": len(records), "architecture": vars(runtime.architecture), "deviation": "deterministic text coordinate tokens replace HTML point boxes"}
    summary_path = args.output_dir / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.smoke:manifest_inputs.append(Path(config["point_calibration"]))
    write_run_manifest(args.output_dir, config={**config, "layers": layers, "smoke": args.smoke, "architecture": vars(runtime.architecture)}, seeds={"global": args.seed}, inputs=manifest_inputs, outputs=[output, per_point, summary_path,checkpoint_path,checkpoint.meta_path], status="complete" if records else "failed", repo_root=Path.cwd())
    print(json.dumps(summary, indent=2))


def trace_example(runtime, row: dict, layers: list[int]) -> list[dict]:
    image = Image.open(row["image_path"]).convert("RGB")
    inputs = runtime.prepare(
        image, point_condition_prompt(row, "point_answer"), prompt_mode="raw"
    )
    answer = str(row["answers"]["point"]); answer_ids = runtime.answer_token_ids(answer)
    kwargs, answer_ids, prompt_length = append_answer_tokens(inputs, answer_ids)
    with runtime.torch.no_grad(), Qwen3CaptureHooks(runtime.model, layers=layers, to_cpu=False) as store:
        runtime.model(**kwargs, use_cache=False, return_dict=True)
    image_positions = inputs.input_ids[0].eq(int(runtime.model.config.image_token_id)).nonzero().flatten().tolist()
    visual_xy = visual_token_centers(inputs, runtime, image.size, len(image_positions))
    tokenized_answer = runtime.processor.tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    offsets = tokenized_answer["offset_mapping"]
    records = []
    for point_index, match in enumerate(COORDINATE.finditer(answer)):
        covered = [index for index, (start_char, end_char) in enumerate(offsets) if end_char > match.start() and start_char < match.end()]
        if not covered: raise RuntimeError(f"cannot align coordinate tokens for {row['id']}: {match.group(0)}")
        start, end = covered[0], covered[-1] + 1
        # A causal LM predicts answer token t from query position t-1.
        truth_x, truth_y = int(match.group(1)), int(match.group(2)); query_positions = list(range(prompt_length - 1 + start, prompt_length - 1 + end))
        for layer in layers:
            probabilities = store.attention_probabilities[layer][0, :, query_positions, :][:, :, image_positions].float().mean((0, 1))
            denoised = (probabilities - probabilities.mean()).clamp_min(0)
            weights = denoised if float(denoised.sum()) > 0 else probabilities
            weights = weights / weights.sum()
            cx = sum(float(weight) * xy[0] for weight, xy in zip(weights.detach().cpu(), visual_xy)); cy = sum(float(weight) * xy[1] for weight, xy in zip(weights.detach().cpu(), visual_xy))
            records.append({"id": row["id"], "point_index": point_index, "layer": layer, "truth_x": truth_x, "truth_y": truth_y, "centroid_x": cx, "centroid_y": cy, "squared_error": (cx-truth_x)**2 + (cy-truth_y)**2, "query_token_start": query_positions[0], "query_token_end": query_positions[-1] + 1})
    return records


def visual_token_centers(inputs, runtime, image_size: tuple[int, int], n_tokens: int) -> list[tuple[float, float]]:
    grid = inputs["image_grid_thw"][0].tolist(); merge = int(runtime.model.config.vision_config.spatial_merge_size)
    temporal, height, width = (int(value) for value in grid); height //= merge; width //= merge
    if temporal * height * width != n_tokens: raise RuntimeError(f"visual-grid token count {temporal*height*width} != observed {n_tokens}")
    image_width, image_height = image_size
    return [((column + .5) / width * image_width, (row + .5) / height * image_height) for _ in range(temporal) for row in range(height) for column in range(width)]


def read_jsonl(path: Path) -> list[dict]: return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if __name__ == "__main__": main()
