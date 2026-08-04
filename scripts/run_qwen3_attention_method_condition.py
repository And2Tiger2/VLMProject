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

from adapters.qwen_gaze_factory import make_image_attention_adapter
from vlm_eval.datasets import load_examples
from vlm_eval.gaze_resume import ensure_resume_config, load_completed_keys
from vlm_eval.metrics import prediction_to_dict, score_response, summarize
from vlm_eval.naturalbench import (
    NaturalBenchPrediction,
    extract_naturalbench_answer,
    load_naturalbench_calls,
    normalize_naturalbench_answer,
    prediction_to_dict as naturalbench_prediction_to_dict,
    summarize_naturalbench,
)
from vlm_eval.overheat import maybe_pause
from vlm_eval.qwen3_attention_methods import DEFAULT_RUN_ROOT
from vlm_eval.types import EvalExample


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
ATTENTION_KEYS = (
    "mean_image_attention_mass",
    "mean_boosted_layer_image_attention_mass",
    "mean_boosted_head_image_attention_mass",
    "mean_context_attention_mass",
    "mean_boosted_head_context_attention_mass",
    "mean_preboost_head_image_attention_mass",
    "mean_effective_alpha",
    "mean_alpha_cap_fraction",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one staged attention condition.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--gaze-ranking",
        default=(
            "segments/gaze_heads_qwen3_8b/runs/gaze_discovery_seed42_merged/"
            "gaze_head_ranking.json"
        ),
    )
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    conditions = manifest["conditions"]
    if not 0 <= args.task_index < len(conditions):
        raise SystemExit(
            f"task index {args.task_index} is outside 0..{len(conditions) - 1}"
        )
    condition = conditions[args.task_index]
    out_dir = args.run_root / manifest["stage"] / condition["name"]
    result = run_condition(
        condition=condition,
        stage=manifest["stage"],
        gaze_ranking=args.gaze_ranking,
        out_dir=out_dir,
        device_map=args.device_map,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2))


def run_condition(
    *,
    condition: dict[str, Any],
    stage: str,
    gaze_ranking: str,
    out_dir: Path,
    device_map: str,
    resume: bool,
    adapter: Any | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "task": "qwen3_attention_method_condition",
        "stage": stage,
        "condition": condition,
        "model_id": MODEL_ID,
        "gaze_ranking": gaze_ranking,
        "device_map": device_map,
        "max_new_tokens": 16,
        "prompt_mode": "baseline",
        "decode_only": False,
        "confidence_definition": (
            "geometric mean probability of the model's generated tokens "
            "under an unboosted baseline pass"
        ),
        "git_commit": _git_commit(),
    }
    for artifact in ("vlmbias.jsonl", "naturalbench.jsonl"):
        ensure_resume_config(
            out_dir,
            config,
            resume=resume,
            artifact_name=artifact,
        )
    if adapter is None:
        adapter = make_image_attention_adapter(
            model_id=MODEL_ID,
            max_new_tokens=16,
            max_pixels=1048576,
            min_pixels=0,
            do_sample=bool(condition["do_sample"]),
            temperature=condition.get("temperature"),
            seed=int(condition["seed"]),
            device_map=device_map,
            prompt_mode="baseline",
            attention_alpha=0.0,
            gaze_ranking_path=gaze_ranking,
            top_k_gaze=int(condition["head_count"]),
            head_selection=str(condition["head_selection"]),
            head_selection_seed=int(condition["head_seed"]),
            attention_controller="fixed",
            target_attention_mass=0.0,
            max_attention_alpha=float(condition["max_alpha"]),
            decode_only=False,
        )
    _apply_condition(adapter, condition)
    vlmbias_rows = _run_vlmbias(
        adapter=adapter,
        condition=condition,
        dataset=Path(condition["vlmbias_dataset"]),
        out_path=out_dir / "vlmbias.jsonl",
        resume=resume,
    )
    naturalbench_rows = _run_naturalbench(
        adapter=adapter,
        condition=condition,
        dataset=Path(condition["naturalbench_dataset"]),
        out_path=out_dir / "naturalbench.jsonl",
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
        "vlmbias": _vlmbias_summary(vlmbias_rows),
        "naturalbench": _naturalbench_summary(naturalbench_rows),
        "telemetry": {
            "vlmbias": _telemetry(vlmbias_rows),
            "naturalbench": _telemetry(naturalbench_rows),
        },
        "artifacts": {
            "config": str(out_dir / "experiment_config.json"),
            "vlmbias": str(out_dir / "vlmbias.jsonl"),
            "naturalbench": str(out_dir / "naturalbench.jsonl"),
        },
        "errors": [],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _apply_condition(adapter: Any, condition: dict[str, Any]) -> None:
    controller = str(condition["controller"])
    adapter.attention_controller = (
        "fixed" if controller == "confidence_gate" else controller
    )
    adapter.attention_alpha = (
        0.0 if controller == "confidence_gate" else float(condition["alpha"])
    )
    adapter.target_attention_mass = float(condition["target_mass"])
    adapter.max_attention_alpha = float(condition["max_alpha"])
    adapter.top_k_gaze = int(condition["head_count"])
    adapter.head_selection = str(condition["head_selection"])
    adapter.head_selection_seed = int(condition["head_seed"])
    adapter.seed = int(condition["seed"])
    adapter.do_sample = bool(condition["do_sample"])
    adapter.temperature = condition.get("temperature")
    adapter._configure_attention_modules()


def _generate(
    adapter: Any, example: EvalExample, condition: dict[str, Any]
) -> tuple[str, dict]:
    if condition["controller"] != "confidence_gate":
        response = adapter.generate(example)
        return response, dict(adapter.last_generation_metadata or {})

    adapter.attention_controller = "fixed"
    adapter.attention_alpha = 0.0
    adapter._configure_attention_modules()
    baseline_response = adapter.generate(example)
    baseline_metadata = dict(adapter.last_generation_metadata or {})
    confidence = (baseline_metadata.get("token_confidence") or {}).get(
        "geometric_mean_probability"
    )
    threshold = float(condition["confidence_threshold"])
    intervene = confidence is None or float(confidence) < threshold
    if intervene:
        adapter.attention_alpha = float(condition["alpha"])
        adapter._configure_attention_modules()
        response = adapter.generate(example)
        final_metadata = dict(adapter.last_generation_metadata or {})
    else:
        response = baseline_response
        final_metadata = baseline_metadata
    final_metadata["confidence_gate"] = {
        "threshold": threshold,
        "baseline_confidence": confidence,
        "intervened": intervene,
        "baseline_response": baseline_response,
    }
    return response, final_metadata


def _run_vlmbias(
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
        for example in tqdm(examples, desc=f"VLMBias {condition['name']}"):
            if (example.id,) in completed:
                continue
            maybe_pause()
            raw_response, generation = _generate(adapter, example, condition)
            maybe_pause()
            prediction = score_response(example, raw_response)
            prediction = replace(
                prediction,
                metadata={
                    **(prediction.metadata or {}),
                    "generation": generation,
                },
            )
            handle.write(
                json.dumps(prediction_to_dict(prediction), ensure_ascii=False) + "\n"
            )
            handle.flush()
    return _read_jsonl(out_path)


def _run_naturalbench(
    *,
    adapter: Any,
    condition: dict[str, Any],
    dataset: Path,
    out_path: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    calls = load_naturalbench_calls(str(dataset))
    completed = (
        load_completed_keys(out_path, ["group_id", "call_id"]) if resume else set()
    )
    mode = "a" if resume and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        for call in tqdm(calls, desc=f"NaturalBench {condition['name']}"):
            if (call.group_id, call.call_id) in completed:
                continue
            example = EvalExample(
                id=f"{call.group_id}_{call.call_id}",
                prompt=call.prompt,
                ground_truth=call.ground_truth,
                image=Image.open(call.image_path).convert("RGB"),
                image_path=call.image_path,
                topic="NaturalBench",
                sub_topic=call.question_type,
                metadata={
                    "group_id": call.group_id,
                    "question_id": call.question_id,
                    "image_id": call.image_id,
                    "source": call.source,
                },
            )
            maybe_pause()
            raw_response, generation = _generate(adapter, example, condition)
            maybe_pause()
            parsed = extract_naturalbench_answer(raw_response, call.question_type)
            prediction = NaturalBenchPrediction(
                group_id=call.group_id,
                call_id=call.call_id,
                question_id=call.question_id,
                image_id=call.image_id,
                question_type=call.question_type,
                prompt=call.prompt,
                ground_truth=call.ground_truth,
                raw_response=raw_response,
                parsed_answer=parsed,
                is_correct=normalize_naturalbench_answer(parsed)
                == normalize_naturalbench_answer(call.ground_truth),
                source=call.source,
            )
            row = naturalbench_prediction_to_dict(prediction)
            row["metadata"] = {"generation": generation}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    return _read_jsonl(out_path)


def _vlmbias_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [_prediction_from_dict(row) for row in rows]
    result = summarize(predictions)
    unique = {str(row.get("example_id")) for row in rows}
    result["n_unique"] = len(unique)
    result["duplicate_count"] = len(rows) - len(unique)
    result["invalid_rate"] = mean(
        not bool(str(row.get("parsed_answer", "")).strip()) for row in rows
    )
    for topic, metrics in result.get("by_topic", {}).items():
        topic_rows = [
            row for row in rows if str(row.get("topic") or "unknown") == topic
        ]
        metrics["invalid_rate"] = mean(
            not bool(str(row.get("parsed_answer", "")).strip()) for row in topic_rows
        )
    return result


def _naturalbench_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [
        NaturalBenchPrediction(
            **{
                key: row[key]
                for key in (
                    "group_id",
                    "call_id",
                    "question_id",
                    "image_id",
                    "question_type",
                    "prompt",
                    "ground_truth",
                    "raw_response",
                    "parsed_answer",
                    "is_correct",
                    "source",
                )
            }
        )
        for row in rows
    ]
    result = summarize_naturalbench(predictions)
    unique = {(str(row.get("group_id")), str(row.get("call_id"))) for row in rows}
    result["n_unique_calls"] = len(unique)
    result["duplicate_count"] = len(rows) - len(unique)
    result["invalid_rate"] = mean(
        not bool(str(row.get("parsed_answer", "")).strip()) for row in rows
    )
    return result


def _prediction_from_dict(row: dict[str, Any]):
    from vlm_eval.types import Prediction

    return Prediction(
        **{
            key: row.get(key)
            for key in (
                "example_id",
                "prompt",
                "ground_truth",
                "expected_bias",
                "raw_response",
                "parsed_answer",
                "is_correct",
                "is_bias_aligned_error",
                "topic",
                "sub_topic",
                "metadata",
            )
        }
    )


def _telemetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attentions = [
        (((row.get("metadata") or {}).get("generation") or {}).get("attention") or {})
        for row in rows
    ]
    gates = [
        (
            ((row.get("metadata") or {}).get("generation") or {}).get("confidence_gate")
            or {}
        )
        for row in rows
    ]
    output = {
        key: _mean_clean(
            [
                float(attention[key])
                for attention in attentions
                if attention.get(key) is not None
            ]
        )
        for key in ATTENTION_KEYS
    }
    gated = [gate for gate in gates if gate]
    output["confidence_gate_intervention_rate"] = (
        mean(bool(gate.get("intervened")) for gate in gated) if gated else None
    )
    output["mean_baseline_confidence"] = _mean_clean(
        [
            float(gate["baseline_confidence"])
            for gate in gated
            if gate.get("baseline_confidence") is not None
        ]
    )
    return output


def _mean_clean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    main()
