#!/usr/bin/env python3
"""Evaluate SynDot/constant-complexity counting before any causal scan."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from PIL import Image

from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, effective_limit, load_json_config, prepare_output_directory
from vlm_eval.mechanistic_heads.preflight import require_scientific_validation, validation_path_from_config
from vlm_eval.mechanistic_heads.qwen3_runtime import Qwen3MechanisticRuntime
from vlm_eval.mechanistic_heads.reproducibility import referenced_image_paths, seed_everything, write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy Qwen3 counting calibration.")
    add_standard_run_arguments(parser)
    parser.add_argument("--device-map", default="cuda")
    args = parser.parse_args()
    config = load_json_config(args.config)
    if not args.smoke:
        require_scientific_validation(validation_path_from_config(config))
    output = args.output_dir / "counting_behavior.tsv"
    prepare_output_directory(args.output_dir, resume=args.resume, overwrite=args.overwrite, known_outputs=(output.name,))
    seed_everything(args.seed)
    rows = read_jsonl(Path(config["dataset"]))
    rows = [row for row in rows if row.get("split") in set(config.get("splits", ["test", "locked_test"]))]
    limit = effective_limit(args)
    if limit is not None:
        rows = rows[:limit]
    runtime = Qwen3MechanisticRuntime(model_id=str(config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")), device_map=args.device_map)
    records = []
    for row in rows:
        inputs = runtime.prepare(Image.open(row["image_path"]).convert("RGB"), str(row["prompt"]), prompt_mode="raw")
        with runtime.torch.no_grad():
            generated = runtime.model.generate(**inputs, do_sample=False, max_new_tokens=8)
        text = runtime.processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        match = re.fullmatch(r"\s*(\d+)\s*", text)
        prediction = int(match.group(1)) if match else None
        truth = int(row["ground_truth"])
        error = None if prediction is None else prediction - truth
        records.append({"id": row["id"], "split": row["split"], "truth": truth, "raw_output": text, "prediction": "" if prediction is None else prediction, "valid": int(prediction is not None), "correct": int(prediction == truth), "absolute_error": "" if error is None else abs(error), "squared_error": "" if error is None else error * error, "off_by_one": int(error is not None and abs(error) <= 1)})
    write_tsv(output, records)
    valid = [row for row in records if row["valid"]]
    summary = {
        "valid": True,
        "label": "instrumentation smoke test" if args.smoke else "methods-based reproduction",
        "n": len(records), "n_valid": len(valid), "invalid_rate": 1 - len(valid) / len(records) if records else 0,
        "accuracy": sum(row["correct"] for row in records) / len(records) if records else 0,
        "mae": sum(float(row["absolute_error"]) for row in valid) / len(valid) if valid else None,
        "rmse": math.sqrt(sum(float(row["squared_error"]) for row in valid) / len(valid)) if valid else None,
        "off_by_one_accuracy": sum(row["off_by_one"] for row in records) / len(records) if records else 0,
        "architecture": vars(runtime.architecture),
    }
    minimum_accuracy = float(config.get("minimum_calibration_accuracy", 0.5))
    maximum_invalid = float(config.get("maximum_calibration_invalid_rate", 0.1))
    calibration_passed = bool(records) and (
        float(summary["accuracy"]) >= minimum_accuracy
        and float(summary["invalid_rate"]) <= maximum_invalid
    )
    summary.update(
        {
            "calibration_passed": calibration_passed,
            "calibration_result": (
                "not assessed in smoke"
                if args.smoke
                else "passed" if calibration_passed else "failed calibration"
            ),
            "calibration_thresholds": {
                "minimum_accuracy": minimum_accuracy,
                "maximum_invalid_rate": maximum_invalid,
            },
        }
    )
    if not args.smoke and not calibration_passed:
        summary["label"] = "failed calibration"
    summary_path = args.output_dir / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_manifest(args.output_dir, config={**config, "smoke": args.smoke, "architecture": vars(runtime.architecture)}, seeds={"global": args.seed}, inputs=[args.config, Path(config["dataset"]), *referenced_image_paths(rows)], outputs=[output, summary_path], status="complete", repo_root=Path.cwd())
    print(json.dumps(summary, indent=2))
    if not args.smoke and not calibration_passed:
        raise SystemExit("counting behavioral calibration failed; causal scans are blocked")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}) or ["id"], delimiter="\t"); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
