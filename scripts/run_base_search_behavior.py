#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from vlm_eval.mechanistic_heads.base_search import (
    assert_base_only,
    assert_unmodified_runtime,
    build_presence_probe,
    build_search_probe,
    configured_cues,
    exemplar_source_rows,
    find_exemplars,
    read_jsonl,
    seeded_group_sample,
)
from vlm_eval.mechanistic_heads.config import (
    add_standard_run_arguments,
    effective_limit,
    load_json_config,
    prepare_output_directory,
)
from vlm_eval.mechanistic_heads.likelihood import candidate_sequence_log_likelihood
from vlm_eval.mechanistic_heads.qwen3_runtime import Qwen3MechanisticRuntime
from vlm_eval.mechanistic_heads.reproducibility import referenced_image_paths, seed_everything, write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen-base visual search with text, exemplar, and control cues.")
    add_standard_run_arguments(parser)
    parser.add_argument("--device-map", default="cuda")
    args = parser.parse_args()
    config = load_json_config(args.config)
    assert_base_only(config)
    output = args.output_dir / "base_search_behavior.tsv"
    summary_path = args.output_dir / "summary.json"
    prepare_output_directory(
        args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
        known_outputs=(output.name, summary_path.name),
    )
    seed_everything(args.seed)
    all_rows = read_jsonl(Path(config["dataset"]))
    exemplars = find_exemplars(all_rows)
    split = str(config.get("behavior_split", "locked_test"))
    rows = [row for row in all_rows if row["split"] == split]
    limit = effective_limit(args, smoke_max=4)
    rows = seeded_group_sample(
        rows, limit=limit, seed=args.seed, purpose="base-search-behavior"
    )
    runtime = Qwen3MechanisticRuntime(
        model_id=str(config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")),
        device_map=args.device_map,
    )
    assert_unmodified_runtime(runtime)
    records: list[dict[str, Any]] = []
    for cue_mode in configured_cues(config):
        for row in rows:
            presence = build_presence_probe(row, cue_mode=cue_mode, exemplars=exemplars)
            records.append(score_probe(runtime, row, cue_mode, "presence", presence))
            if row.get("target_present"):
                localization = build_search_probe(row, cue_mode=cue_mode, exemplars=exemplars)
                records.append(score_probe(runtime, row, cue_mode, "localization", localization))
    write_tsv(output, records)
    summary = summarize(records, config=config, architecture=vars(runtime.architecture), smoke=args.smoke)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_manifest(
        args.output_dir,
        config={**config, "smoke": args.smoke, "architecture": vars(runtime.architecture), "base_model_only": True},
        seeds={"global": args.seed},
        inputs=[args.config, Path(config["dataset"]), *referenced_image_paths([*rows, *exemplar_source_rows(all_rows)])],
        outputs=[output, summary_path],
        status="complete",
        repo_root=Path.cwd(),
    )
    print(json.dumps(summary, indent=2))


def score_probe(runtime: Any, row: dict[str, Any], cue_mode: str, task: str, probe: Any) -> dict[str, Any]:
    inputs = runtime.prepare(probe.image, probe.prompt, prompt_mode="raw")
    scores: dict[str, float] = {}
    with runtime.torch.no_grad():
        for answer in probe.candidate_answers:
            result = candidate_sequence_log_likelihood(runtime.model, inputs, runtime.answer_token_ids(answer))
            scores[answer] = float(result.total_log_probability[0].detach().cpu())
    prediction = max(scores, key=scores.get)
    alternatives = [value for answer, value in scores.items() if answer != probe.answer]
    margin = scores[probe.answer] - max(alternatives)
    return {
        "id": row["id"],
        "group_id": row["group_id"],
        "split": row["split"],
        "cue_mode": cue_mode,
        "task": task,
        "expected": probe.answer,
        "prediction": prediction,
        "correct": int(prediction == probe.answer),
        "correct_log_likelihood_margin": margin,
        "candidate_log_likelihoods": json.dumps(scores, sort_keys=True),
        "target_present": int(bool(row.get("target_present"))),
    }


def summarize(records: list[dict[str, Any]], *, config: dict[str, Any], architecture: dict[str, Any], smoke: bool) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for cue in sorted({row["cue_mode"] for row in records}):
        for task in sorted({row["task"] for row in records}):
            group = [row for row in records if row["cue_mode"] == cue and row["task"] == task]
            if not group:
                continue
            by_condition[f"{cue}:{task}"] = {
                "n": len(group),
                "accuracy": sum(row["correct"] for row in group) / len(group),
                "mean_correct_log_likelihood_margin": sum(row["correct_log_likelihood_margin"] for row in group) / len(group),
            }
    return {
        "valid": True,
        "label": "instrumentation smoke test" if smoke else "frozen-base behavioral diagnostic",
        "base_model_only": True,
        "trained_checkpoint": None,
        "n_records": len(records),
        "by_condition": by_condition,
        "architecture": architecture,
        "interpretation": "Behavior is reported, not used to train or select any model weights.",
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}) or ["id"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
