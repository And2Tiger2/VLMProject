#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Any

from vlm_eval.mechanistic_heads.causal import candidate_margin, projected_head_scaling
from vlm_eval.mechanistic_heads.controls import layer_matched_control_draws, multivariate_matched_control_draws
from vlm_eval.mechanistic_heads.config import (
    add_standard_run_arguments,
    effective_limit,
    load_json_config,
    prepare_output_directory,
)
from vlm_eval.mechanistic_heads.qwen3_runtime import Qwen3MechanisticRuntime
from vlm_eval.mechanistic_heads.preflight import require_calibration_report, require_current_artifact, require_scientific_validation, validation_path_from_config
from vlm_eval.mechanistic_heads.mmmc import MMMCImages
from vlm_eval.mechanistic_heads.reproducibility import seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.schema import read_paired_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate driving/resisting MACI heads by zero ablation.")
    add_standard_run_arguments(parser)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    config = load_json_config(args.config)
    require_current_artifact(Path(config["head_scores"]))
    if not args.smoke:
        require_scientific_validation(validation_path_from_config(config))
        if config.get("stability_report"):
            require_calibration_report(Path(config["stability_report"]), boolean_key="passes_stability_gate")
        if config.get("validation_ablation_report"):
            require_calibration_report(Path(config["validation_ablation_report"]))
    output = args.output_dir / "maci_ablation.tsv"
    prepare_output_directory(
        args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
        known_outputs=(output.name, "summary.json"),
    )
    seed_everything(args.seed)
    runtime = Qwen3MechanisticRuntime(
        model_id=str(config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")),
        device_map=args.device_map,
    )
    score_rows = _read_tsv(Path(config["head_scores"])); ranking = _ranking(score_rows)
    conditions = make_conditions(
        ranking,
        n_layers=runtime.architecture.n_layers,
        n_heads=runtime.architecture.n_heads,
        seed=args.seed,
        require_full_sets=not args.smoke,
    )
    if not bool(config.get("include_validation_k_sweep", True)):
        conditions = {name: heads for name, heads in conditions.items() if not name.startswith("validation_sweep_")}
    if bool(config.get("matched_control_distributions", False)):
        require_current_artifact(Path(config["general_causal_importance"]))
        conditions.update(matched_conditions(score_rows, ranking, config=config, n_layers=runtime.architecture.n_layers, n_heads=runtime.architecture.n_heads, seed=args.seed))
    if args.smoke:
        conditions = {key: value for key, value in conditions.items() if key in {"baseline", "driving_top30", "resisting_top40", "random_seed0"}}
    pairs = [
        pair
        for pair in read_paired_jsonl(Path(config["paired_dataset"]))
        if pair.split == str(config.get("split", "validation"))
    ]
    limit = effective_limit(args)
    if limit is None and config.get("max_examples") is not None:
        limit = int(config["max_examples"])
    if limit is not None:
        pairs = pairs[:limit]
    audit_path = Path(config["paired_dataset"]).with_name("audit.json")
    images = MMMCImages(args.cache_dir, audit_path=audit_path)
    rows = []
    for condition, heads in conditions.items():
        for pair in pairs:
            image = images.resolve(pair.recipient_image)
            inputs = runtime.prepare(image, pair.recipient_prompt, prompt_mode="raw")
            scales = {head: 0.0 for head in heads}
            with projected_head_scaling(runtime.model, scales):
                hallucination_advantage, scores = candidate_margin(
                    runtime,
                    inputs,
                    positive_answer=pair.bias_answer,
                    negative_answer=pair.correct_answer,
                )
                generated = None
                if should_generate(condition, config):
                    generated = runtime.model.generate(
                        **inputs, do_sample=False, max_new_tokens=32
                    )
            text = ""
            normalized = ""
            if generated is not None:
                continuation = generated[:, inputs.input_ids.shape[1] :]
                text = runtime.processor.batch_decode(
                    continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()
                normalized = text.casefold().strip(" .")
            rows.append(
                {
                    "condition": condition,
                    "pair_id": pair.pair_id,
                    "n_heads": len(heads),
                    "hallucination_advantage": hallucination_advantage,
                    "logp_hallucinated": scores["positive"],
                    "logp_factual": scores["negative"],
                    "generated": text,
                    "generation_scored": int(generated is not None),
                    "generated_factual": int(normalized == pair.correct_answer.casefold().strip(" .")) if generated is not None else "",
                    "generated_hallucinated": int(normalized == pair.bias_answer.casefold().strip(" .")) if generated is not None else "",
                }
            )
    _write_tsv(output, rows)
    aggregates = aggregate_conditions(rows)
    claim_checks = maci_claim_checks(
        aggregates,
        random_tolerance=float(config.get("random_control_margin_tolerance", 0.25)),
    )
    summary={
        "valid":True,
        "label":"instrumentation smoke test" if args.smoke else ("locked confirmation" if str(config.get("split"))=="locked_test" else "methods-based reproduction"),
        "calibration_result":"not assessed in smoke" if args.smoke else ("passed" if claim_checks["all_pass"] else "failed calibration"),
        "rows":len(rows),
        "conditions":len(conditions),
        "aggregate":aggregates,
        "claim_checks":claim_checks,
        "control_policy":"20 joint matches on layer, image attention, projected norm, entropy, gaze, and general causal importance; single-feature diagnostic families are opt-in",
        "generation_policy":"greedy generation for paper-style and five random controls; matched controls use complete candidate-sequence likelihood",
        "output":str(output),
        "architecture":vars(runtime.architecture),
    }
    summary_path=args.output_dir/"summary.json";summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    manifest_inputs=[args.config,Path(config["paired_dataset"]),audit_path,Path(config["head_scores"])]
    if bool(config.get("matched_control_distributions",False)):manifest_inputs.extend([Path(config["gaze_ranking"]),Path(config["general_causal_importance"])])
    for key in ("stability_report","validation_ablation_report"):
        if config.get(key):manifest_inputs.append(Path(config[key]))
    write_run_manifest(
        args.output_dir,
        config={**config, "conditions": {name: len(heads) for name, heads in conditions.items()},"architecture":vars(runtime.architecture)},
        seeds={"global": args.seed, **{f"random_{idx}": args.seed + idx for idx in range(5)}},
        inputs=manifest_inputs,
        outputs=[output,summary_path],
        status="complete",
        repo_root=Path.cwd(),
    )
    print(json.dumps(summary,indent=2))


def make_conditions(
    ranking: list[tuple[tuple[int, int], float]],
    *,
    n_layers: int,
    n_heads: int,
    seed: int,
    require_full_sets: bool = False,
) -> dict[str, list[tuple[int, int]]]:
    driving = [head for head, score in ranking if score > 0][:30]
    resisting = [head for head, score in reversed(ranking) if score < 0][:40]
    if require_full_sets and (len(driving) != 30 or len(resisting) != 40):
        raise RuntimeError(
            "paper-style MACI validation requires 30 positive driving heads and "
            f"40 negative resisting heads; found {len(driving)} and {len(resisting)}"
        )
    conditions = {
        "baseline": [],
        "driving_top30": driving,
        "resisting_top40": resisting,
        "joint_top30": driving[:15] + resisting[:15],
    }
    for k in (5, 10, 20, 30, 40, 50):
        conditions[f"validation_sweep_driving_k{k}"] = [head for head, score in ranking if score > 0][:k]
        conditions[f"validation_sweep_resisting_k{k}"] = [head for head, score in reversed(ranking) if score < 0][:k]
    forbidden = set(driving) | set(resisting)
    candidates = [
        (layer, head)
        for layer in range(n_layers)
        for head in range(n_heads)
        if (layer, head) not in forbidden
    ]
    for random_seed in range(5):
        conditions[f"random_seed{random_seed}"] = random.Random(
            seed + random_seed
        ).sample(candidates, 30)
    return conditions


def _ranking(rows: list[dict[str,str]]) -> list[tuple[tuple[int, int], float]]:
    ranking = [
        ((int(row["layer"]), int(row["head"])), float(row["mean_signed_intervention_score"]))
        for row in rows
    ]
    return sorted(ranking, key=lambda item: item[1], reverse=True)


def matched_conditions(rows,ranking,*,config,n_layers,n_heads,seed):
    gaze={(int(row["layer"]),int(row["head"])):float(row.get("score",row.get("gaze_score",0))) for row in json.loads(Path(config["gaze_ranking"]).read_text(encoding="utf-8"))};general={(int(row["layer"]),int(row["head"])):float(row["general_causal_importance"]) for row in _read_tsv(Path(config["general_causal_importance"]))}
    features={(int(row["layer"]),int(row["head"])):{"image_attention":float(row["image_attention"]),"projected_output_norm":float(row["projected_output_norm"]),"attention_entropy":float(row["attention_entropy"]),"gaze_score":gaze.get((int(row["layer"]),int(row["head"])),0),"general_causal_importance":general[(int(row["layer"]),int(row["head"]))]} for row in rows}
    result={};n_draws=max(20,int(config.get("control_draws",20)));sets={"driving":[head for head,score in ranking if score>0][:30],"resisting":[head for head,score in reversed(ranking) if score<0][:40]}
    diagnostic_families = bool(config.get("diagnostic_single_feature_controls", False))
    for role,selected in sets.items():
        families={"fully":multivariate_matched_control_draws(selected,features,n_draws=n_draws,seed=seed+6)}
        if diagnostic_families:
            families.update({"layer":layer_matched_control_draws(selected,n_layers=n_layers,n_heads=n_heads,n_draws=n_draws,seed=seed),"image":multivariate_matched_control_draws(selected,features,feature_names=("image_attention",),n_draws=n_draws,seed=seed+1),"norm":multivariate_matched_control_draws(selected,features,feature_names=("projected_output_norm",),n_draws=n_draws,seed=seed+2),"entropy":multivariate_matched_control_draws(selected,features,feature_names=("attention_entropy",),n_draws=n_draws,seed=seed+3),"gaze":multivariate_matched_control_draws(selected,features,feature_names=("gaze_score",),n_draws=n_draws,seed=seed+4),"general":multivariate_matched_control_draws(selected,features,feature_names=("general_causal_importance",),n_draws=n_draws,seed=seed+5)})
        for family,draws in families.items():
            for index,heads in enumerate(draws):result[f"{role}_control_{family}_{index:02d}"]=heads
    return result


def aggregate_conditions(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["condition"], []).append(row)
    return {
        condition: {
            "n": len(group),
            "mean_hallucination_advantage": sum(float(row["hallucination_advantage"]) for row in group) / len(group),
            "factual_generation_rate": generation_rate(group, "generated_factual"),
            "hallucinated_generation_rate": generation_rate(group, "generated_hallucinated"),
        }
        for condition, group in sorted(grouped.items())
    }


def generation_rate(rows, key):
    scored = [row for row in rows if int(row.get("generation_scored", 1))]
    return sum(int(row[key]) for row in scored) / len(scored) if scored else None


def should_generate(condition: str, config: dict) -> bool:
    prefixes = tuple(
        config.get(
            "generation_condition_prefixes",
            ("baseline", "driving_top", "resisting_top", "joint_top", "random_seed"),
        )
    )
    return condition.startswith(prefixes)


def maci_claim_checks(aggregates, *, random_tolerance):
    required = {"baseline", "driving_top30", "resisting_top40", "joint_top30"}
    if not required <= set(aggregates):
        return {"all_pass": False, "reason": "required paper-style conditions are missing"}
    baseline = aggregates["baseline"]["mean_hallucination_advantage"]
    driving = aggregates["driving_top30"]["mean_hallucination_advantage"]
    resisting = aggregates["resisting_top40"]["mean_hallucination_advantage"]
    joint = aggregates["joint_top30"]["mean_hallucination_advantage"]
    random_values = [
        value["mean_hallucination_advantage"]
        for name, value in aggregates.items()
        if name.startswith("random_seed")
    ]
    checks = {
        "driving_ablation_lowers_hallucination_advantage": driving < baseline,
        "resisting_ablation_raises_hallucination_advantage": resisting > baseline,
        "joint_lies_between_driving_and_resisting": min(driving, resisting) <= joint <= max(driving, resisting),
        "random_controls_near_baseline": bool(random_values) and max(abs(value - baseline) for value in random_values) <= random_tolerance,
    }
    return {
        **checks,
        "all_pass": all(checks.values()),
        "random_control_margin_tolerance": random_tolerance,
    }


def _read_tsv(path:Path)->list[dict[str,str]]:
    with path.open("r",encoding="utf-8") as handle:return list(csv.DictReader(handle,delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}) or ["condition"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
