#!/usr/bin/env python3
"""Validate MACI detector-gated driving-head suppression on locked MMMC."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np

from vlm_eval.mechanistic_heads.causal import candidate_margin, capture_prefill, projected_head_scaling
from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, effective_limit, load_json_config, prepare_output_directory
from vlm_eval.mechanistic_heads.mmmc import MMMCImages
from vlm_eval.mechanistic_heads.preflight import require_scientific_validation, validation_path_from_config
from vlm_eval.mechanistic_heads.qwen3_runtime import Qwen3MechanisticRuntime
from vlm_eval.mechanistic_heads.reproducibility import seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.schema import read_paired_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare always/never/detector/confidence/random MACI interventions.")
    add_standard_run_arguments(parser); parser.add_argument("--device-map", default="cuda"); parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(); config = load_json_config(args.config)
    if not args.smoke: require_scientific_validation(validation_path_from_config(config))
    output = args.output_dir / "gated_interventions.tsv"; prepare_output_directory(args.output_dir, resume=args.resume, overwrite=args.overwrite, known_outputs=(output.name,))
    seed_everything(args.seed); runtime = Qwen3MechanisticRuntime(model_id=str(config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")), device_map=args.device_map)
    pairs = [pair for pair in read_paired_jsonl(Path(config["paired_dataset"])) if pair.split == str(config.get("split", "locked_test"))]
    limit = effective_limit(args)
    if limit is not None: pairs = pairs[:limit]
    detector = json.loads(Path(config["detector"]).read_text(encoding="utf-8")); ranking = read_tsv(Path(config["head_scores"])); driving = [(int(row["layer"]), int(row["head"])) for row in sorted(ranking, key=lambda row: float(row["mean_signed_intervention_score"]), reverse=True)[:int(config.get("driving_heads", 30))]]; resisting = [tuple(head) for head in detector["resisting_heads"]]
    images = MMMCImages(args.cache_dir); prepared = []
    for pair in pairs:
        image = images.resolve(pair.recipient_image); inputs = runtime.prepare(image, pair.recipient_prompt, prompt_mode="raw"); margin, scores = candidate_margin(runtime, inputs, positive_answer=pair.bias_answer, negative_answer=pair.correct_answer)
        capture = capture_prefill(runtime, image_path=image, prompt=pair.recipient_prompt, layers=sorted({layer for layer, _ in resisting}), to_cpu=True); vectors = [capture.store.raw_heads[layer][0, capture.prompt_length-1, head, :].float().detach().cpu().numpy() for layer, head in resisting]; feature = np.stack(vectors).mean(0); probability = detector_probability(feature, detector); prepared.append((pair, inputs, margin, scores, probability))
    detector_budget = sum(probability >= float(detector["threshold"]) for *_, probability in prepared); random_indices = set(random.Random(args.seed).sample(range(len(prepared)), detector_budget))
    rows = []
    for index, (pair, inputs, baseline_margin, baseline_scores, probability) in enumerate(prepared):
        pairwise_confidence = 1 / (1 + math.exp(-abs(baseline_margin)))
        decisions = {"never": False, "always": True, "detector_gated": probability >= float(detector["threshold"]), "confidence_gated": pairwise_confidence < float(config.get("confidence_threshold", 0.6)), "random_budget_matched": index in random_indices}
        for condition, intervene in decisions.items():
            with projected_head_scaling(runtime.model, {head: 0.0 for head in driving} if intervene else {}): margin, scores = candidate_margin(runtime, inputs, positive_answer=pair.bias_answer, negative_answer=pair.correct_answer)
            rows.append({"condition": condition, "pair_id": pair.pair_id, "intervened": int(intervene), "detector_probability": probability, "pairwise_confidence": pairwise_confidence, "hallucination_advantage": margin, "logp_hallucinated": scores["positive"], "logp_factual": scores["negative"], "margin_shift": margin-baseline_margin})
    write_tsv(output, rows); summary = summarize(rows); summary.update({"valid": True, "label": "instrumentation smoke test" if args.smoke else "locked confirmation", "architecture": vars(runtime.architecture)})
    summary_path = args.output_dir / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_manifest(args.output_dir, config={**config, "architecture": vars(runtime.architecture)}, seeds={"global": args.seed}, inputs=[args.config, Path(config["paired_dataset"]), Path(config["head_scores"]), Path(config["detector"])], outputs=[output, summary_path], status="complete", repo_root=Path.cwd()); print(json.dumps(summary, indent=2))


def detector_probability(feature: np.ndarray, detector: dict) -> float:
    scaled = (feature - np.asarray(detector["scaler_mean"])) / np.asarray(detector["scaler_scale"]); logit = float(scaled @ np.asarray(detector["coefficient"]) + np.asarray(detector["intercept"])[0]); return 1/(1+math.exp(-max(-30,min(30,logit))))
def summarize(rows: list[dict]) -> dict:
    result = {"conditions": {}}
    for condition in sorted({row["condition"] for row in rows}):
        group=[row for row in rows if row["condition"]==condition]; result["conditions"][condition]={"n":len(group),"intervention_rate":sum(row["intervened"] for row in group)/len(group),"mean_hallucination_advantage":sum(row["hallucination_advantage"] for row in group)/len(group),"mean_margin_shift":sum(row["margin_shift"] for row in group)/len(group)}
    return result
def read_tsv(path: Path) -> list[dict]:
    with path.open("r",encoding="utf-8") as handle:return list(csv.DictReader(handle,delimiter="\t"))
def write_tsv(path: Path,rows:list[dict])->None:
    with path.open("w",encoding="utf-8",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else ["empty"],delimiter="\t");writer.writeheader();writer.writerows(rows)
if __name__=="__main__":main()
