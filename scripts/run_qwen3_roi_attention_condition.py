#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image
from tqdm import tqdm

from adapters.qwen_gaze_factory import make_roi_attention_adapter
from scripts.run_qwen3_attention_method_condition import _telemetry, _vlmbias_summary
from vlm_eval.datasets import load_examples
from vlm_eval.gaze_resume import ensure_resume_config, load_completed_keys
from vlm_eval.metrics import prediction_to_dict, score_response
from vlm_eval.overheat import maybe_pause
from vlm_eval.qwen3_roi_attention import (
    DEFAULT_GAZE_RANKING,
    DEFAULT_RUN_ROOT,
    read_jsonl,
    spatial_control_mask,
)


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Qwen3 spatial ROI-attention condition."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--gaze-ranking", type=Path, default=DEFAULT_GAZE_RANKING)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not 0 <= args.task_index < len(manifest["conditions"]):
        raise SystemExit(
            f"task index {args.task_index} is outside the condition manifest"
        )
    condition = manifest["conditions"][args.task_index]
    out_dir = args.run_root / manifest["stage"] / condition["name"]
    result = run_condition(
        condition=condition,
        stage=manifest["stage"],
        out_dir=out_dir,
        gaze_ranking=args.gaze_ranking,
        device_map=args.device_map,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2))


def run_condition(
    *,
    condition: dict[str, Any],
    stage: str,
    out_dir: Path,
    gaze_ranking: Path,
    device_map: str,
    resume: bool,
    adapter: Any | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "task": "qwen3_vlmbias_roi_attention",
        "model_id": MODEL_ID,
        "stage": stage,
        "condition": condition,
        "gaze_ranking": str(gaze_ranking),
        "roi_token_min_coverage": 0.05,
        "max_new_tokens": 16,
        "max_pixels": 1048576,
        "decode_only": False,
        "git_commit": _git_commit(),
    }
    ensure_resume_config(
        out_dir, config, resume=resume, artifact_name="predictions.jsonl"
    )
    if adapter is None:
        adapter = make_roi_attention_adapter(
            model_id=MODEL_ID,
            max_new_tokens=16,
            max_pixels=1048576,
            min_pixels=0,
            do_sample=bool(condition["do_sample"]),
            temperature=condition.get("temperature"),
            seed=int(condition["seed"]),
            device_map=device_map,
            prompt_mode="baseline",
            attention_alpha=float(condition["alpha"]),
            attention_controller="fixed",
            gaze_ranking_path=str(gaze_ranking),
            top_k_gaze=int(condition["head_count"]),
            head_selection=str(condition["head_selection"]),
            head_selection_seed=int(condition["head_seed"]),
            decode_only=False,
            roi_token_min_coverage=0.05,
        )
    predictions = _run_examples(
        adapter=adapter,
        condition=condition,
        dataset=Path(condition["dataset"]),
        out_path=out_dir / "predictions.jsonl",
        resume=resume,
    )
    result = {
        "valid": True,
        "stage": stage,
        "condition": condition,
        "provenance": {
            "git_commit": config["git_commit"],
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
        },
        "selected_heads": [list(head) for head in adapter._boosted_heads],
        "vlmbias": _vlmbias_summary(predictions),
        "attention_telemetry": _telemetry(predictions),
        "token_mask_telemetry": _token_mask_telemetry(predictions),
        "artifacts": {
            "config": str(out_dir / "experiment_config.json"),
            "predictions": str(out_dir / "predictions.jsonl"),
        },
        "errors": [],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _run_examples(
    *,
    adapter: Any,
    condition: dict[str, Any],
    dataset: Path,
    out_path: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    examples = load_examples(str(dataset))
    completed = load_completed_keys(out_path, ["example_id"]) if resume else set()
    mode = "a" if resume and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        for example in tqdm(examples, desc=condition["name"]):
            if (example.id,) in completed:
                continue
            metadata = example.metadata or {}
            region = str(condition["region"])
            pixel_mask = None
            if region in {"roi", "shifted_roi"}:
                mask_variant = str(condition.get("mask_variant") or "tight")
                mask_paths = metadata.get("roi_mask_paths") or {}
                mask_path = Path(
                    str(
                        mask_paths.get(mask_variant)
                        or metadata.get("roi_mask_path")
                        or ""
                    )
                )
                if not mask_path.is_file():
                    raise FileNotFoundError(
                        f"missing {mask_variant} ROI mask for {example.id}: {mask_path}"
                    )
                pixel_mask = Image.open(mask_path).convert("L")
                if region == "shifted_roi":
                    pixel_mask = spatial_control_mask(
                        pixel_mask, example.id, int(condition["seed"])
                    )
            adapter.set_attention_region(pixel_mask, region=region)
            maybe_pause()
            response = adapter.generate(example)
            maybe_pause()
            prediction = score_response(example, response)
            prediction = replace(
                prediction,
                metadata={
                    **(prediction.metadata or {}),
                    "generation": dict(adapter.last_generation_metadata or {}),
                },
            )
            handle.write(
                json.dumps(prediction_to_dict(prediction), ensure_ascii=False) + "\n"
            )
            handle.flush()
    return read_jsonl(out_path)


def _token_mask_telemetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = [
        (((row.get("metadata") or {}).get("generation") or {}).get("roi_tokens") or {})
        for row in rows
    ]
    counts = [
        float(row["n_target_tokens"])
        for row in records
        if row.get("n_target_tokens") is not None
    ]
    fractions = [
        float(row["target_token_fraction"])
        for row in records
        if row.get("target_token_fraction") is not None
    ]
    image_counts = [
        float(row["n_image_tokens"])
        for row in records
        if row.get("n_image_tokens") is not None
    ]
    context_counts = [
        float(row["n_context_tokens"])
        for row in records
        if row.get("n_context_tokens") is not None
    ]
    context_fractions = [
        float(row["context_token_fraction"])
        for row in records
        if row.get("context_token_fraction") is not None
    ]
    return {
        "n_examples": len(records),
        "n_nonempty_target_masks": sum(
            float(row.get("n_target_tokens", 0)) > 0 for row in records
        ),
        "mean_image_tokens": mean(image_counts) if image_counts else None,
        "mean_target_tokens": mean(counts) if counts else None,
        "mean_target_token_fraction": mean(fractions) if fractions else None,
        "minimum_target_tokens": min(counts) if counts else None,
        "maximum_target_tokens": max(counts) if counts else None,
        "mean_context_tokens": mean(context_counts) if context_counts else None,
        "mean_context_token_fraction": (
            mean(context_fractions) if context_fractions else None
        ),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    main()
