#!/usr/bin/env python3
"""Behavioral calibration for base/direct/point-search checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from PIL import Image

from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, effective_limit, load_json_config, prepare_output_directory
from vlm_eval.mechanistic_heads.preflight import require_scientific_validation, validation_path_from_config
from vlm_eval.mechanistic_heads.qwen3_runtime import checkpoint_manifest_inputs, runtime_from_config
from vlm_eval.mechanistic_heads.reproducibility import referenced_image_paths, seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.synthetic import length_matched_nonspatial_answer, point_condition_prompt


CONDITION_KEYS = {"base": "base", "direct_answer": "direct", "direct_length_matched": "direct_length_matched", "point_answer": "point", "shuffled_point_answer": "shuffled_point"}
POINT_RE = re.compile(r"\((\d{1,3}),(\d{1,3})\)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate point-search behavior and OOD target counts.")
    add_standard_run_arguments(parser)
    parser.add_argument("--condition", choices=sorted(CONDITION_KEYS), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device-map", default="cuda")
    args = parser.parse_args()
    config = load_json_config(args.config)
    if not args.smoke:
        require_scientific_validation(validation_path_from_config(config))
    output = args.output_dir / "behavior.tsv"
    prepare_output_directory(args.output_dir, resume=args.resume, overwrite=args.overwrite, known_outputs=(output.name,))
    seed_everything(args.seed)
    rows = read_jsonl(Path(config["dataset"]))
    rows = [row for row in rows if row.get("split") != "train"]
    limit = effective_limit(args)
    if limit is not None: rows = rows[:limit]
    runtime = runtime_from_config(config, device_map=args.device_map, checkpoint_override=args.checkpoint)
    answer_key = CONDITION_KEYS[args.condition]
    records = []
    for row in rows:
        inputs = runtime.prepare(Image.open(row["image_path"]).convert("RGB"), point_condition_prompt(row, args.condition), prompt_mode="raw")
        with runtime.torch.no_grad():
            generated = runtime.model.generate(**inputs, do_sample=False, max_new_tokens=int(config.get("max_new_tokens", 256)))
        text = runtime.processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        expected = str(row["answers"][answer_key])
        if args.condition == "direct_length_matched":
            expected = length_matched_nonspatial_answer(runtime.processor.tokenizer, direct_answer=str(row["target_count"]), point_answer=str(row["answers"]["point"]))
        expected_points = [(int(x), int(y)) for x, y in POINT_RE.findall(expected)]
        predicted_points = [(int(x), int(y)) for x, y in POINT_RE.findall(text)]
        expected_count = int(row["target_count"])
        count_match = re.search(r"(?:answer|count)\s*=\s*(\d+)", text)
        if count_match is None and re.fullmatch(r"\s*\d+\s*", text): count_match = re.match(r"\s*(\d+)", text)
        predicted_count = int(count_match.group(1)) if count_match else None
        rmse = point_rmse(predicted_points, expected_points)
        records.append({"id": row["id"], "split": row["split"], "condition": args.condition, "target_count": expected_count, "expected": expected, "output": text, "parsed_count": "" if predicted_count is None else predicted_count, "count_correct": int(predicted_count == expected_count), "sequence_exact": int(text == expected), "point_rmse": "" if rmse is None else rmse})
    write_tsv(output, records)
    by_split = {}
    for split in sorted({row["split"] for row in records}):
        group = [row for row in records if row["split"] == split]
        rmses = [float(row["point_rmse"]) for row in group if row["point_rmse"] != ""]
        positive = [row for row in group if int(row["target_count"]) > 0]
        by_split[split] = {"n": len(group), "count_accuracy": sum(row["count_correct"] for row in group) / len(group), "sequence_exact": sum(row["sequence_exact"] for row in group) / len(group), "point_parse_rate": sum(row["point_rmse"] != "" for row in positive) / len(positive) if positive else None, "point_rmse": sum(rmses) / len(rmses) if rmses else None}
    calibration_split=str(config.get("calibration_split","ood_target_count_1"));calibration=by_split.get(calibration_split,{})
    calibration_passed=(args.condition!="point_answer") or (float(calibration.get("count_accuracy",0))>=float(config.get("minimum_calibration_count_accuracy",0.5)) and float(calibration.get("point_parse_rate",0))>=float(config.get("minimum_calibration_point_parse_rate",0.5)))
    summary = {"valid": True, "label": "instrumentation smoke test" if args.smoke else ("modified replication" if calibration_passed else "failed calibration"), "condition": args.condition, "n": len(records), "by_split": by_split, "calibration_split":calibration_split,"calibration_passed":calibration_passed,"architecture": vars(runtime.architecture), "deviation": "deterministic textual coordinates replace paper HTML point boxes"}
    summary_path = args.output_dir / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_manifest(args.output_dir, config={**config, "condition": args.condition, "smoke": args.smoke, "architecture": vars(runtime.architecture)}, seeds={"global": args.seed}, inputs=[args.config, Path(config["dataset"]), *referenced_image_paths(rows), *checkpoint_manifest_inputs(config, checkpoint_override=args.checkpoint)], outputs=[output, summary_path], status="complete", repo_root=Path.cwd())
    print(json.dumps(summary, indent=2))
    if not args.smoke and not calibration_passed:
        raise SystemExit("Point-Answer behavioral calibration failed; causal scans are blocked")


def point_rmse(predicted: list[tuple[int, int]], expected: list[tuple[int, int]]) -> float | None:
    if not expected or len(predicted) != len(expected): return None
    predicted_array = np.asarray(predicted, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    squared_distances = (
        (expected_array[:, None, :] - predicted_array[None, :, :]) ** 2
    ).sum(axis=-1)
    expected_indices, predicted_indices = linear_sum_assignment(squared_distances)
    return float(
        np.sqrt(squared_distances[expected_indices, predicted_indices].mean())
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}) or ["id"], delimiter="\t"); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
