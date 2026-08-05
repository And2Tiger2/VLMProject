#!/usr/bin/env python3
"""Locked top/bottom/random ablations for search, verification, and suppression heads."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from vlm_eval.mechanistic_heads.causal import candidate_margin, projected_head_scaling
from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, effective_limit, load_json_config, prepare_output_directory
from vlm_eval.mechanistic_heads.controls import layer_matched_control_draws, multivariate_matched_control_draws
from vlm_eval.mechanistic_heads.preflight import require_scientific_validation, validation_path_from_config
from vlm_eval.mechanistic_heads.qwen3_runtime import checkpoint_manifest_inputs, runtime_from_config
from vlm_eval.mechanistic_heads.reproducibility import seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.schema import read_paired_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate point-search head families with locked ablations.")
    add_standard_run_arguments(parser); parser.add_argument("--device-map", default="cuda")
    args = parser.parse_args(); config = load_json_config(args.config)
    if not args.smoke: require_scientific_validation(validation_path_from_config(config))
    output = args.output_dir / "point_head_ablation.tsv"
    prepare_output_directory(args.output_dir, resume=args.resume, overwrite=args.overwrite, known_outputs=(output.name, "summary.json"))
    seed_everything(args.seed)
    runtime = runtime_from_config(config, device_map=args.device_map)
    gaze = {(int(row["layer"]), int(row["head"])): float(row.get("score", row.get("gaze_score", 0))) for row in json.loads(Path(config["gaze_ranking"]).read_text(encoding="utf-8"))}
    general = {}
    general_path = Path(config["general_causal_importance"])
    if general_path.is_file():
        general = {(int(row["layer"]), int(row["head"])): float(row["general_causal_importance"]) for row in read_tsv(general_path)}
    elif not args.smoke:
        raise RuntimeError("full point-head validation requires general causal importance controls")
    rows: list[dict[str, Any]] = []
    inputs: list[Path] = [args.config, *checkpoint_manifest_inputs(config)]
    for study_index, study in enumerate(config["studies"]):
        score_path = Path(study["scores"]); pair_path = Path(study["paired_dataset"]); inputs.extend([score_path, pair_path])
        source_rows = read_tsv(score_path)
        ranking = aggregate_scores(source_rows, str(study["score_column"]))
        ordered = sorted(ranking, key=lambda head: abs(ranking[head]), reverse=True)
        k = min(2 if args.smoke else int(study.get("k", 30)), len(ordered)); selected = ordered[:k]; low = list(reversed(ordered))[:k]
        sets = {"top": selected, "bottom": low}
        n_draws=max(20,int(config.get("random_draws",20)))
        families={"layer":layer_matched_control_draws(selected,n_layers=runtime.architecture.n_layers,n_heads=runtime.architecture.n_heads,n_draws=n_draws,seed=args.seed+study_index)}
        if general:
            diagnostics=aggregate_diagnostics(source_rows,gaze,general)
            for offset,(name,columns) in enumerate({"image_attention":("image_attention",),"output_norm":("projected_output_norm",),"entropy":("attention_entropy",),"gaze":("gaze_score",),"general":("general_causal_importance",),"fully":("image_attention","projected_output_norm","attention_entropy","gaze_score","general_causal_importance")}.items(),start=1):
                families[name]=multivariate_matched_control_draws(selected,diagnostics,feature_names=columns,n_draws=n_draws,seed=args.seed+study_index*20+offset)
        for family,draws in families.items():
            if args.smoke:draws=draws[:1]
            for draw,heads in enumerate(draws):sets[f"{family}_random_{draw:02d}"]=heads
        pairs = [pair for pair in read_paired_jsonl(pair_path) if pair.split == str(study.get("split", "locked_test"))]
        limit = effective_limit(args)
        if limit is not None: pairs = pairs[:limit]
        for pair in pairs:
            image = Image.open(pair.recipient_image).convert("RGB")
            model_inputs = runtime.prepare(image, pair.recipient_prompt, prompt_mode="raw")
            if study.get("correct_answer_role", "recipient") == "donor":
                correct, alternative = pair.donor_answer, pair.recipient_answer
            else:
                correct, alternative = pair.recipient_answer, pair.donor_answer
            baseline, _ = candidate_margin(runtime, model_inputs, positive_answer=correct, negative_answer=alternative)
            rows.append({"study": study["name"], "pair_id": pair.pair_id, "head_set": "baseline", "n_heads": 0, "baseline_margin": baseline, "ablated_margin": baseline, "margin_change": 0.0, "preference_flip": 0})
            for set_name, heads in sets.items():
                with projected_head_scaling(runtime.model, {head: 0.0 for head in heads}):
                    ablated, _ = candidate_margin(runtime, model_inputs, positive_answer=correct, negative_answer=alternative)
                rows.append({"study": study["name"], "pair_id": pair.pair_id, "head_set": set_name, "n_heads": len(heads), "baseline_margin": baseline, "ablated_margin": ablated, "margin_change": ablated - baseline, "preference_flip": int(baseline >= 0 and ablated < 0)})
    write_tsv(output, rows)
    aggregate = summarize(rows)
    summary = {"valid": True, "label": "instrumentation smoke test" if args.smoke else "locked confirmation", "architecture": vars(runtime.architecture), "aggregate": aggregate, "control_policy": "20 layer-matched random draws per selected set in full mode"}
    summary_path = args.output_dir / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_manifest(args.output_dir, config={**config, "architecture": vars(runtime.architecture)}, seeds={"global": args.seed}, inputs=inputs, outputs=[output, summary_path], status="complete", repo_root=Path.cwd())
    print(json.dumps(summary, indent=2))


def aggregate_scores(rows: list[dict[str, str]], column: str) -> dict[tuple[int, int], float]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows: grouped[(int(row["layer"]), int(row["head"]))].append(float(row[column]))
    return {head: sum(values) / len(values) for head, values in grouped.items()}


def aggregate_diagnostics(rows,gaze,general):
    grouped=defaultdict(list)
    for row in rows:grouped[(int(row["layer"]),int(row["head"]))].append(row)
    result={}
    for head,group in grouped.items():
        result[head]={name:sum(float(row[name]) for row in group)/len(group) for name in ("image_attention","projected_output_norm","attention_entropy")};result[head]["gaze_score"]=gaze.get(head,0.0);result[head]["general_causal_importance"]=general[head]
    return result


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[(row["study"], row["head_set"])].append(row)
    return [{"study": study, "head_set": head_set, "n": len(group), "mean_baseline_margin": sum(float(row["baseline_margin"]) for row in group)/len(group), "mean_ablated_margin": sum(float(row["ablated_margin"]) for row in group)/len(group), "mean_margin_change": sum(float(row["margin_change"]) for row in group)/len(group), "preference_flip_rate": sum(int(row["preference_flip"]) for row in group)/len(group)} for (study, head_set), group in sorted(grouped.items())]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle: return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else ["study"],delimiter="\t");writer.writeheader();writer.writerows(rows)


if __name__ == "__main__": main()
