#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from vlm_eval.mechanistic_heads.base_search import assert_base_only, assert_unmodified_runtime, build_search_probe, configured_cues, exemplar_source_rows, find_exemplars, read_jsonl
from vlm_eval.mechanistic_heads.causal import projected_head_scaling
from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, effective_limit, load_json_config, prepare_output_directory
from vlm_eval.mechanistic_heads.likelihood import candidate_sequence_log_likelihood
from vlm_eval.mechanistic_heads.preflight import require_completed_manifest
from vlm_eval.mechanistic_heads.qwen3_runtime import Qwen3MechanisticRuntime
from vlm_eval.mechanistic_heads.reproducibility import referenced_image_paths, seed_everything, write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Causally validate frozen-base search heads on locked scenes.")
    add_standard_run_arguments(parser)
    parser.add_argument("--device-map", default="cuda")
    args = parser.parse_args()
    config = load_json_config(args.config)
    assert_base_only(config)
    output = args.output_dir / "base_search_causal_validation.tsv"
    summary_path = args.output_dir / "summary.json"
    prepare_output_directory(
        args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
        known_outputs=(output.name, summary_path.name),
    )
    seed_everything(args.seed)
    ranking_path = Path(config["ranking"])
    require_completed_manifest(ranking_path.parent, expected_outputs=(ranking_path,), require_current_git=True)
    ranking_summary = json.loads((ranking_path.parent / "summary.json").read_text(encoding="utf-8"))
    controls = ranking_summary["controls"]
    head_sets = {
        name: {(int(row["layer"]), int(row["head"])): 0.0 for row in rows}
        for name, rows in controls.items()
    }
    all_rows = read_jsonl(Path(config["dataset"]))
    exemplars = find_exemplars(all_rows)
    rows = [
        row for row in all_rows
        if row["split"] == str(config.get("validation_split", "locked_test")) and row.get("target_present")
    ]
    limit = effective_limit(args, smoke_max=2)
    if limit is None:
        limit = int(config.get("validation_examples", 24))
    rows = rows[:limit]
    runtime = Qwen3MechanisticRuntime(
        model_id=str(config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")),
        device_map=args.device_map,
    )
    assert_unmodified_runtime(runtime)
    records: list[dict[str, Any]] = []
    for row in rows:
        for cue_mode in configured_cues(config):
            probe = build_search_probe(row, cue_mode=cue_mode, exemplars=exemplars)
            inputs = runtime.prepare(probe.image, probe.prompt, prompt_mode="raw")
            intact = score_candidates(runtime, inputs, probe.candidate_answers, probe.answer)
            records.append(record(row, cue_mode, "intact", probe.answer, intact, intact))
            for condition, scales in head_sets.items():
                with projected_head_scaling(runtime.model, scales):
                    intervention = score_candidates(runtime, inputs, probe.candidate_answers, probe.answer)
                records.append(record(row, cue_mode, condition, probe.answer, intervention, intact))
    write_tsv(output, records)
    summary = summarize(records, architecture=vars(runtime.architecture), controls=controls, smoke=args.smoke)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_manifest(
        args.output_dir,
        config={**config, "smoke": args.smoke, "architecture": vars(runtime.architecture), "base_model_only": True},
        seeds={"global": args.seed},
        inputs=[args.config, ranking_path, ranking_path.parent / "summary.json", Path(config["dataset"]), *referenced_image_paths([*rows, *exemplar_source_rows(all_rows)])],
        outputs=[output, summary_path],
        status="complete",
        repo_root=Path.cwd(),
    )
    print(json.dumps(summary, indent=2))


def score_candidates(runtime: Any, inputs: Any, answers: tuple[str, ...], correct: str) -> dict[str, Any]:
    scores: dict[str, float] = {}
    with runtime.torch.no_grad():
        for answer in answers:
            result = candidate_sequence_log_likelihood(runtime.model, inputs, runtime.answer_token_ids(answer))
            scores[answer] = float(result.total_log_probability[0].detach().cpu())
    prediction = max(scores, key=scores.get)
    incorrect = max(value for answer, value in scores.items() if answer != correct)
    return {"scores": scores, "prediction": prediction, "margin": scores[correct] - incorrect}


def record(row: dict[str, Any], cue_mode: str, condition: str, expected: str, result: dict[str, Any], intact: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "group_id": row["group_id"],
        "cue_mode": cue_mode,
        "condition": condition,
        "expected": expected,
        "prediction": result["prediction"],
        "correct": int(result["prediction"] == expected),
        "correct_log_likelihood_margin": result["margin"],
        "intact_margin": intact["margin"],
        "ablation_harm": intact["margin"] - result["margin"],
        "candidate_log_likelihoods": json.dumps(result["scores"], sort_keys=True),
    }


def summarize(records: list[dict[str, Any]], *, architecture: dict[str, Any], controls: dict[str, Any], smoke: bool) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for cue in sorted({row["cue_mode"] for row in records}):
        for condition in sorted({row["condition"] for row in records}):
            group = [row for row in records if row["cue_mode"] == cue and row["condition"] == condition]
            if not group:
                continue
            by_condition[f"{cue}:{condition}"] = {
                "n": len(group),
                "accuracy": sum(row["correct"] for row in group) / len(group),
                "mean_correct_log_likelihood_margin": sum(row["correct_log_likelihood_margin"] for row in group) / len(group),
                "mean_ablation_harm": sum(row["ablation_harm"] for row in group) / len(group),
            }
    search_harm = mean(row["ablation_harm"] for row in records if row["condition"] == "search_heads")
    image_harm = mean(row["ablation_harm"] for row in records if row["condition"] == "high_image_attention_control")
    random_harm = mean(row["ablation_harm"] for row in records if row["condition"] == "random_control")
    return {
        "valid": True,
        "label": "instrumentation smoke test" if smoke else "locked frozen-base causal validation",
        "base_model_only": True,
        "trained_checkpoint": None,
        "controls": controls,
        "by_condition": by_condition,
        "contrasts": {
            "search_minus_high_image_attention_harm": search_harm - image_harm,
            "search_minus_random_harm": search_harm - random_harm,
        },
        "architecture": architecture,
        "claim_gate": "Call heads search-causal only if their held-out ablation harm exceeds both controls and behavior is above chance.",
    }


def mean(values: Any) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}) or ["id"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
