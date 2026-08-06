#!/usr/bin/env python3
"""Locked top/bottom/random ablations for search, verification, and suppression heads."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from vlm_eval.mechanistic_heads.causal import candidate_margin, projected_head_scaling
from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, effective_limit, load_json_config, partitioned_limit, prepare_output_directory
from vlm_eval.mechanistic_heads.controls import layer_matched_control_draws, multivariate_matched_control_draws
from vlm_eval.mechanistic_heads.preflight import require_current_artifact, require_scientific_validation, validation_path_from_config
from vlm_eval.mechanistic_heads.qwen3_runtime import checkpoint_manifest_inputs, runtime_from_config
from vlm_eval.mechanistic_heads.reproducibility import referenced_image_paths, seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.schema import read_paired_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate point-search head families with locked ablations.")
    add_standard_run_arguments(parser); parser.add_argument("--device-map", default="cuda"); parser.add_argument("--checkpoint")
    args = parser.parse_args(); config = load_json_config(args.config)
    if not args.smoke: require_scientific_validation(validation_path_from_config(config))
    output = args.output_dir / "point_head_ablation.tsv"
    prepare_output_directory(args.output_dir, resume=args.resume, overwrite=args.overwrite, known_outputs=(output.name, "summary.json"))
    seed_everything(args.seed)
    runtime = runtime_from_config(config, device_map=args.device_map, checkpoint_override=args.checkpoint)
    gaze = {(int(row["layer"]), int(row["head"])): float(row.get("score", row.get("gaze_score", 0))) for row in json.loads(Path(config["gaze_ranking"]).read_text(encoding="utf-8"))}
    general = {}
    general_path = Path(config["general_causal_importance"])
    if general_path.is_file():
        require_current_artifact(general_path)
        general = {(int(row["layer"]), int(row["head"])): float(row["general_causal_importance"]) for row in read_tsv(general_path)}
    elif not args.smoke:
        raise RuntimeError("full point-head validation requires general causal importance controls")
    rows: list[dict[str, Any]] = []
    inputs: list[Path] = [args.config, *checkpoint_manifest_inputs(config, checkpoint_override=args.checkpoint)]
    cross_task_sets: dict[str, list[tuple[int, int]]] = {}
    required_head_counts: dict[str, int] = {}
    for study in config["studies"]:
        score_path = Path(study["scores"])
        require_current_artifact(score_path)
        source_rows = read_tsv(score_path)
        ranking = aggregate_scores(source_rows, str(study["score_column"]))
        bidirectional = bidirectional_positive_heads(source_rows)
        requested_k = 2 if args.smoke else int(study.get("k", 30))
        cross_task_sets[f"{study['name']}_top"] = select_positive_function_heads(
            ranking,
            requested_k=requested_k,
            allow_unsigned_fallback=args.smoke,
            eligible_heads=bidirectional,
        )
        required_head_counts[str(study["name"])] = requested_k
    studies = config["studies"]
    total_smoke_limit = effective_limit(args) if args.smoke else None
    for study_index, study in enumerate(studies):
        score_path = Path(study["scores"]); pair_path = Path(study["paired_dataset"]); inputs.extend([score_path, pair_path])
        require_current_artifact(score_path)
        source_rows = read_tsv(score_path)
        ranking = aggregate_scores(source_rows, str(study["score_column"]))
        ordered = sorted(ranking, key=lambda head: ranking[head], reverse=True)
        selected = cross_task_sets[f"{study['name']}_top"]
        k = len(selected)
        low = list(reversed(ordered))[:k]
        # Apply every discovered functional set to every task. This is the
        # causal task-by-head-set matrix needed for a real double dissociation.
        sets = dict(cross_task_sets)
        sets[f"{study['name']}_bottom"] = low
        n_draws=max(20,int(config.get("random_draws",20)))
        families={}
        if general:
            diagnostics=aggregate_diagnostics(source_rows,gaze,general)
            family_columns={"fully":("image_attention","projected_output_norm","attention_entropy","gaze_score","general_causal_importance")}
            if bool(config.get("diagnostic_single_feature_controls",False)):
                families["layer"]=layer_matched_control_draws(selected,n_layers=runtime.architecture.n_layers,n_heads=runtime.architecture.n_heads,n_draws=n_draws,seed=args.seed+study_index)
                family_columns.update({"image_attention":("image_attention",),"output_norm":("projected_output_norm",),"entropy":("attention_entropy",),"gaze":("gaze_score",),"general":("general_causal_importance",)})
            for offset,(name,columns) in enumerate(family_columns.items(),start=1):
                families[name]=multivariate_matched_control_draws(selected,diagnostics,feature_names=columns,n_draws=n_draws,seed=args.seed+study_index*20+offset)
        for family,draws in families.items():
            if args.smoke:draws=draws[:1]
            for draw,heads in enumerate(draws):sets[f"{study['name']}_{family}_random_{draw:02d}"]=heads
        pairs = [pair for pair in read_paired_jsonl(pair_path) if pair.split == str(study.get("split", "locked_test"))]
        limit = (
            partitioned_limit(total_smoke_limit, groups=len(studies), index=study_index)
            if args.smoke
            else effective_limit(args)
        )
        if limit is None and config.get("max_examples_per_study") is not None: limit=int(config["max_examples_per_study"])
        if limit is not None: pairs = pairs[:limit]
        inputs.extend(referenced_image_paths(pairs))
        for pair in pairs:
            image = Image.open(pair.recipient_image).convert("RGB")
            model_inputs = runtime.prepare(image, pair.recipient_prompt, prompt_mode="raw")
            if study.get("correct_answer_role", "recipient") == "donor":
                correct, alternative = pair.donor_answer, pair.recipient_answer
            else:
                correct, alternative = pair.recipient_answer, pair.donor_answer
            baseline, _ = candidate_margin(runtime, model_inputs, positive_answer=correct, negative_answer=alternative)
            generation = (
                generate_selection(
                    runtime,
                    model_inputs,
                    target_answer=correct,
                    decoy_answer=alternative,
                    max_new_tokens=int(config.get("generation_max_new_tokens", 8)),
                )
                if should_generate_selection(study["name"], "baseline", config)
                else {}
            )
            rows.append({"study": study["name"], "pair_id": pair.pair_id, "head_set": "baseline", "n_heads": 0, "baseline_margin": baseline, "ablated_margin": baseline, "margin_change": 0.0, "preference_flip": 0, **generation})
            for set_name, heads in sets.items():
                with projected_head_scaling(runtime.model, {head: 0.0 for head in heads}):
                    ablated, _ = candidate_margin(runtime, model_inputs, positive_answer=correct, negative_answer=alternative)
                    generation = (
                        generate_selection(
                            runtime,
                            model_inputs,
                            target_answer=correct,
                            decoy_answer=alternative,
                            max_new_tokens=int(config.get("generation_max_new_tokens", 8)),
                        )
                        if should_generate_selection(study["name"], set_name, config)
                        else {}
                    )
                rows.append({"study": study["name"], "pair_id": pair.pair_id, "head_set": set_name, "n_heads": len(heads), "baseline_margin": baseline, "ablated_margin": ablated, "margin_change": ablated - baseline, "preference_flip": int(baseline >= 0 and ablated < 0), **generation})
    write_tsv(output, rows)
    aggregate = summarize(rows)
    claim_checks = point_claim_checks(
        aggregate,
        required_head_counts=None if args.smoke else required_head_counts,
    )
    summary = {
        "valid": True,
        "label": (
            "instrumentation smoke test"
            if args.smoke
            else ("locked confirmation" if claim_checks["all_pass"] else "failed calibration")
        ),
        "calibration_result": (
            "not assessed in smoke"
            if args.smoke
            else ("passed" if claim_checks["all_pass"] else "failed calibration")
        ),
        "architecture": vars(runtime.architecture),
        "aggregate": aggregate,
        "claim_checks": claim_checks,
        "cross_task_head_sets": sorted(cross_task_sets),
        "required_head_counts": required_head_counts,
        "selected_head_counts": {
            name: len(heads) for name, heads in cross_task_sets.items()
        },
        "control_policy": "Every discovered point-function set is ablated on every locked task; each task also uses 20 joint matches on layer, image attention, projected norm, entropy, gaze, and general causal importance in full mode",
        "claim_gate": "Each named head family must harm its own task under ablation, exceed bottom/cross-task/jointly matched controls, and distractor-head ablation must increase generated decoy selections.",
    }
    summary_path = args.output_dir / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_manifest(args.output_dir, config={**config, "architecture": vars(runtime.architecture)}, seeds={"global": args.seed}, inputs=inputs, outputs=[output, summary_path], status="complete", repo_root=Path.cwd())
    print(json.dumps(summary, indent=2))


def aggregate_scores(rows: list[dict[str, str]], column: str) -> dict[tuple[int, int], float]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows: grouped[(int(row["layer"]), int(row["head"]))].append(float(row[column]))
    return {head: sum(values) / len(values) for head, values in grouped.items()}


def select_positive_function_heads(
    ranking: dict[tuple[int, int], float],
    *,
    requested_k: int,
    allow_unsigned_fallback: bool,
    eligible_heads: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Select heads with the causal direction defined by each scan.

    Negative scores represent an opposite role, not a stronger instance of
    the named function. Smoke may fall back to measured heads so the hook path
    is still exercised on a tiny, noisy sample; scientific runs may not.
    """

    if requested_k <= 0:
        raise ValueError("requested functional-head count must be positive")
    ordered = sorted(ranking, key=lambda head: ranking[head], reverse=True)
    positive = [
        head
        for head in ordered
        if ranking[head] > 0
        and (eligible_heads is None or head in eligible_heads)
    ]
    if allow_unsigned_fallback and len(positive) < requested_k:
        return ordered[:requested_k]
    return positive[:requested_k]


def bidirectional_positive_heads(
    rows: list[dict[str, str]],
) -> set[tuple[int, int]] | None:
    """Return heads whose mean donor effects have both expected directions.

    Search and verification scans emit forward and reverse shifts.  A positive
    symmetric mean can otherwise conceal a strong effect in only one direction.
    Distractor-suppression scans are matched ablations and have no such fields,
    so ``None`` means that the direction filter is not applicable.
    """

    if not rows or not all(
        "forward_margin_shift" in row and "reverse_margin_shift" in row
        for row in rows
    ):
        return None
    grouped: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: {"forward": [], "reverse": []}
    )
    for row in rows:
        head = (int(row["layer"]), int(row["head"]))
        grouped[head]["forward"].append(float(row["forward_margin_shift"]))
        grouped[head]["reverse"].append(float(row["reverse_margin_shift"]))
    return {
        head
        for head, values in grouped.items()
        if sum(values["forward"]) / len(values["forward"]) > 0
        and sum(values["reverse"]) / len(values["reverse"]) > 0
    }


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
    output = []
    for (study, head_set), group in sorted(grouped.items()):
        row = {"study": study, "head_set": head_set, "n": len(group), "n_heads": int(group[0].get("n_heads", 0)), "mean_baseline_margin": sum(float(value["baseline_margin"]) for value in group)/len(group), "mean_ablated_margin": sum(float(value["ablated_margin"]) for value in group)/len(group), "mean_margin_change": sum(float(value["margin_change"]) for value in group)/len(group), "preference_flip_rate": sum(int(value["preference_flip"]) for value in group)/len(group)}
        generated = [value for value in group if int(value.get("generation_scored", 0))]
        if generated:
            row.update(
                {
                    "target_selection_rate": sum(int(value["selected_target"]) for value in generated) / len(generated),
                    "decoy_selection_rate": sum(int(value["selected_decoy"]) for value in generated) / len(generated),
                    "invalid_generation_rate": sum(value["selection_state"] == "invalid" for value in generated) / len(generated),
                }
            )
        output.append(row)
    return output


def point_claim_checks(
    aggregate: list[dict[str, Any]],
    *,
    required_head_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Conservative locked gate for the three proposed point-head functions."""

    lookup = {
        (str(row["study"]), str(row["head_set"])): row for row in aggregate
    }
    studies = ("search", "verification", "distractor_suppression")
    per_study: dict[str, Any] = {}
    for study in studies:
        top = lookup.get((study, f"{study}_top"))
        bottom = lookup.get((study, f"{study}_bottom"))
        cross = [
            lookup.get((study, f"{other}_top"))
            for other in studies
            if other != study
        ]
        controls = [
            row
            for (row_study, head_set), row in lookup.items()
            if row_study == study and head_set.startswith(f"{study}_fully_random_")
        ]
        top_change = float(top["mean_margin_change"]) if top is not None else None
        bottom_change = float(bottom["mean_margin_change"]) if bottom is not None else None
        cross_changes = [
            float(row["mean_margin_change"]) for row in cross if row is not None
        ]
        control_changes = [float(row["mean_margin_change"]) for row in controls]
        checks = {
            "required_positive_head_count_available": (
                required_head_counts is None
                or (
                    top is not None
                    and int(top.get("n_heads", 0))
                    == int(required_head_counts[study])
                )
            ),
            "own_top_harms_margin": top_change is not None and top_change < 0,
            "top_exceeds_bottom": (
                top_change is not None
                and bottom_change is not None
                and abs(top_change) > abs(bottom_change)
            ),
            "double_dissociation": (
                top_change is not None
                and len(cross_changes) == len(studies) - 1
                and top_change < min(cross_changes)
            ),
            "beats_fully_matched_controls": (
                top_change is not None
                and len(control_changes) >= 20
                and top_change < empirical_quantile(control_changes, 0.05)
            ),
        }
        if study == "distractor_suppression":
            baseline = lookup.get((study, "baseline"))
            generated_controls = [
                row
                for row in controls
                if row.get("decoy_selection_rate") is not None
            ]
            top_decoy = top.get("decoy_selection_rate") if top is not None else None
            baseline_decoy = (
                baseline.get("decoy_selection_rate") if baseline is not None else None
            )
            checks["increases_generated_decoy_selections"] = (
                top_decoy is not None
                and baseline_decoy is not None
                and float(top_decoy) > float(baseline_decoy)
            )
            checks["decoy_effect_exceeds_generated_controls"] = (
                top_decoy is not None
                and len(generated_controls) >= 5
                and float(top_decoy)
                > max(float(row["decoy_selection_rate"]) for row in generated_controls)
            )
        per_study[study] = {
            **checks,
            "all_pass": all(checks.values()),
            "top_mean_margin_change": top_change,
            "bottom_mean_margin_change": bottom_change,
            "fully_matched_control_count": len(control_changes),
        }
    return {
        "per_study": per_study,
        "all_pass": all(row["all_pass"] for row in per_study.values()),
    }


def empirical_quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute an empirical quantile of no values")
    ordered = sorted(values)
    index = int((len(ordered) - 1) * quantile)
    return ordered[index]


def generate_selection(
    runtime: Any,
    model_inputs: Any,
    *,
    target_answer: str,
    decoy_answer: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    generated = runtime.model.generate(
        **model_inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    text = runtime.processor.batch_decode(
        generated[:, model_inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    parsed = parse_cell(text)
    target = parse_cell(target_answer)
    decoy = parse_cell(decoy_answer)
    if parsed is None:
        state = "invalid"
    elif parsed == target:
        state = "target"
    elif parsed == decoy:
        state = "decoy"
    else:
        state = "other"
    return {
        "generation_scored": 1,
        "generated_text": text,
        "parsed_cell": parsed,
        "selection_state": state,
        "selected_target": int(state == "target"),
        "selected_decoy": int(state == "decoy"),
    }


def should_generate_selection(study: str, head_set: str, config: dict[str, Any]) -> bool:
    if study != "distractor_suppression":
        return False
    if head_set in {
        "baseline",
        "distractor_suppression_top",
        "distractor_suppression_bottom",
    }:
        return True
    prefix = "distractor_suppression_fully_random_"
    if not head_set.startswith(prefix):
        return False
    draw = int(head_set.removeprefix(prefix))
    return draw < int(config.get("generation_random_control_draws", 5))


def parse_cell(text: str) -> int | None:
    match = re.search(r"\bcell\s*=\s*(\d{1,2})\b", text, flags=re.IGNORECASE)
    if match is None:
        return None
    value = int(match.group(1))
    return value if 0 <= value < 100 else None


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle: return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=sorted({key for row in rows for key in row}) or ["study"],delimiter="\t");writer.writeheader();writer.writerows(rows)


if __name__ == "__main__": main()
