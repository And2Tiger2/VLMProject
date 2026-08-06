#!/usr/bin/env python3
"""Locked necessity/sufficiency validation for selected counting head sets."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from vlm_eval.mechanistic_heads.causal import (
    candidate_margin,
    capture_prefill,
    projected_head_set_replacement,
    scope_positions,
)
from vlm_eval.mechanistic_heads.config import (
    add_standard_run_arguments,
    effective_limit,
    load_json_config,
    prepare_output_directory,
)
from vlm_eval.mechanistic_heads.preflight import (
    require_current_artifact,
    require_scientific_validation,
    validation_path_from_config,
)
from vlm_eval.mechanistic_heads.qwen3_runtime import Qwen3MechanisticRuntime
from vlm_eval.mechanistic_heads.reproducibility import referenced_image_paths, seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.schema import read_paired_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate count-head necessity and sufficiency on locked controls.")
    add_standard_run_arguments(parser)
    parser.add_argument("--device-map", default="cuda")
    args = parser.parse_args()
    config = load_json_config(args.config)
    require_current_artifact(Path(config["count_ranking"]))
    require_current_artifact(Path(config["controls"]))
    require_current_artifact(Path(config["stability_summary"]))
    if not args.smoke:
        require_scientific_validation(validation_path_from_config(config))
    output = args.output_dir / "count_head_validation.tsv"
    prepare_output_directory(args.output_dir, resume=args.resume, overwrite=args.overwrite, known_outputs=(output.name, "summary.json", "count_head_double_dissociation.png"))
    seed_everything(args.seed)
    runtime = Qwen3MechanisticRuntime(model_id=str(config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")), device_map=args.device_map)

    all_ranking = read_tsv(Path(config["count_ranking"]))
    ranking = [
        row
        for row in all_ranking
        if int(row.get("bidirectional_positive", 0)) == 1
        and float(row["count_causal_score"]) > 0
    ]
    if args.smoke and len(ranking) < 2:
        ranking = all_ranking
    ranking.sort(key=lambda row: float(row["count_causal_score"]), reverse=True)
    gaze_rows = json.loads(Path(config["gaze_ranking"]).read_text(encoding="utf-8"))
    gaze_heads = [(int(row["layer"]), int(row["head"])) for row in gaze_rows]
    controls = read_tsv(Path(config["controls"]))
    head_sets = build_head_sets(
        ranking,
        gaze_heads,
        controls,
        smoke=args.smoke,
        control_families=tuple(config.get("validation_control_families", ("fully_matched",))),
    )

    pairs = [pair for pair in read_paired_jsonl(Path(config["paired_dataset"])) if pair.split == str(config.get("split", "locked_test"))]
    # Every selected base pair may add one matched-sham pair below. Keep the
    # expanded smoke workload at or below eight examples.
    limit = effective_limit(args, smoke_max=4)
    if limit is None and config.get("max_examples") is not None:
        limit = int(config["max_examples"])
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        raise RuntimeError("locked validation contains no pairs")
    base_pairs = list(pairs)
    for pair in base_pairs:
        sham_images = pair.metadata.get("sham_images")
        if sham_images:
            pairs.append(
                replace(
                    pair,
                    pair_id=f"{pair.pair_id}-matched-sham",
                    donor_image=sham_images[1],
                    donor_prompt=pair.recipient_prompt,
                    metadata={
                        **pair.metadata,
                        "pair_type": "matched-sham",
                        "same_semantic_count": True,
                        "donor_visual_answer": pair.recipient_answer,
                        "candidate_contrast_answer": pair.donor_answer,
                        "sham_definition": (
                            "patch a non-task pixel edit of the recipient while retaining "
                            "the original donor answer only as the candidate-margin contrast"
                        ),
                    },
                )
            )
    all_layers = sorted({layer for heads in head_sets.values() for layer, _ in heads})
    reference_count = min(len(pairs), int(config.get("mean_reference_examples", 16)))
    mean_replacements = build_mean_replacements(runtime, pairs[:reference_count], all_layers, scope=str(config.get("patch_scope", "final_prompt_token")))

    rows: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs):
        donor = capture_prefill(runtime, image_path=pair.donor_image, prompt=pair.donor_prompt, layers=all_layers, to_cpu=True)
        recipient = capture_prefill(runtime, image_path=pair.recipient_image, prompt=pair.recipient_prompt, layers=all_layers, to_cpu=True)
        resample_pair = pairs[(pair_index + 1) % len(pairs)]
        resample = capture_prefill(runtime, image_path=resample_pair.recipient_image, prompt=resample_pair.recipient_prompt, layers=all_layers, to_cpu=True)
        positions = scope_positions(recipient, str(config.get("patch_scope", "final_prompt_token")))
        donor_positions = scope_positions(donor, str(config.get("patch_scope", "final_prompt_token")))
        resample_positions = scope_positions(resample, str(config.get("patch_scope", "final_prompt_token")))
        if not (len(positions) == len(donor_positions) == len(resample_positions)):
            raise RuntimeError(f"unaligned validation scope for {pair.pair_id}")
        baseline, _ = candidate_margin(runtime, recipient.inputs, positive_answer=pair.recipient_answer, negative_answer=pair.donor_answer)
        reverse_baseline, _ = candidate_margin(runtime, donor.inputs, positive_answer=pair.donor_answer, negative_answer=pair.recipient_answer)
        for set_name, heads in head_sets.items():
            interventions = {
                "zero": {head: None for head in heads},
                "mean": {head: mean_replacements[head] for head in heads},
                "resample": {head: resample.store.projected_heads[head[0]][0, resample_positions, head[1], :] for head in heads},
                "donor_patch": {head: donor.store.projected_heads[head[0]][0, donor_positions, head[1], :] for head in heads},
            }
            for intervention, replacements in interventions.items():
                with projected_head_set_replacement(runtime.model, replacements=replacements, recipient_projected=recipient.store.projected_heads, positions=positions):
                    margin, _ = candidate_margin(runtime, recipient.inputs, positive_answer=pair.recipient_answer, negative_answer=pair.donor_answer)
                rows.append(result_row(pair, set_name, len(heads), intervention, baseline, margin))
            reverse_replacements = {head: recipient.store.projected_heads[head[0]][0, positions, head[1], :] for head in heads}
            with projected_head_set_replacement(runtime.model, replacements=reverse_replacements, recipient_projected=donor.store.projected_heads, positions=donor_positions):
                reverse_margin, _ = candidate_margin(runtime, donor.inputs, positive_answer=pair.donor_answer, negative_answer=pair.recipient_answer)
            rows.append(result_row(pair, set_name, len(heads), "reverse_donor_patch", reverse_baseline, reverse_margin))

    write_tsv(output, rows)
    stability_summary = json.loads(
        Path(config["stability_summary"]).read_text(encoding="utf-8")
    )
    summary = summarize(
        rows,
        head_sets,
        runtime,
        stability_summary=stability_summary,
        matched_control_quantile=float(config.get("matched_control_quantile", 0.95)),
    )
    if args.smoke:
        summary["label"] = "instrumentation smoke test"
        summary["calibration_result"] = "not assessed in smoke"
    figure_path=args.output_dir/"count_head_double_dissociation.png";render_validation(summary["aggregate"],figure_path)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_manifest(args.output_dir, config={**config, "architecture": vars(runtime.architecture), "head_sets": {key: value for key, value in head_sets.items()}}, seeds={"global": args.seed}, inputs=[args.config, Path(config["count_ranking"]), Path(config["controls"]), Path(config["stability_summary"]), Path(config["paired_dataset"]), Path(config["gaze_ranking"]), *referenced_image_paths(pairs)], outputs=[output, summary_path,figure_path], status="complete", repo_root=Path.cwd())
    print(json.dumps(summary, indent=2))


def build_mean_replacements(runtime: Any, pairs: list[Any], layers: list[int], *, scope: str) -> dict[tuple[int, int], Any]:
    sums: dict[tuple[int, int], Any] = {}
    for pair in pairs:
        capture = capture_prefill(runtime, image_path=pair.recipient_image, prompt=pair.recipient_prompt, layers=layers, to_cpu=True)
        positions = scope_positions(capture, scope)
        for layer in layers:
            for head in range(runtime.architecture.n_heads):
                key = (layer, head)
                value = capture.store.projected_heads[layer][0, positions, head, :].detach().float().cpu()
                sums[key] = value if key not in sums else sums[key] + value
    return {key: value / len(pairs) for key, value in sums.items()}


def build_head_sets(ranking: list[dict[str, str]], gaze: list[tuple[int, int]], controls: list[dict[str, str]], *, smoke: bool, control_families: tuple[str, ...] = ("fully_matched",)) -> dict[str, list[tuple[int, int]]]:
    ks = (2,) if smoke else (10, 25, 50)
    result: dict[str, list[tuple[int, int]]] = {}
    for k in ks:
        result[f"count_top{k}"] = [(int(row["layer"]), int(row["head"])) for row in ranking[:k]]
        result[f"gaze_top{k}"] = gaze[:k]
        grouped: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
        for row in controls:
            if int(row["selected_k"]) == k and row["control_family"] in control_families:
                grouped[(row["control_family"], int(row["draw"]))].append((int(row["layer"]), int(row["head"])))
        max_draws = 1 if smoke else 20
        for (family, draw), heads in sorted(grouped.items()):
            if draw < max_draws:
                result[f"{family}_k{k}_draw{draw:02d}"] = heads
        low = [
            (int(row["layer"]), int(row["head"]))
            for row in controls
            if int(row["selected_k"]) == k and row["control_family"] == "low_count_score"
        ]
        if low:
            result[f"low_count_score_k{k}"] = low
    if smoke:
        allowed_layers: list[int] = []
        for heads in result.values():
            for layer, _head in heads:
                if layer not in allowed_layers:
                    allowed_layers.append(layer)
                if len(allowed_layers) == 2:
                    break
            if len(allowed_layers) == 2:
                break
        result = {
            name: [head for head in heads if head[0] in allowed_layers][:2]
            for name, heads in result.items()
        }
    return result


def result_row(pair: Any, set_name: str, n_heads: int, intervention: str, baseline: float, margin: float) -> dict[str, Any]:
    return {"pair_id": pair.pair_id, "group_id": pair.group_id, "pair_type": pair.metadata.get("pair_type"), "variant": pair.metadata.get("variant"), "position_variant": pair.metadata.get("position_variant"), "renderer_seed": pair.metadata.get("renderer_seed"), "head_set": set_name, "n_heads": n_heads, "intervention": intervention, "baseline_correct_minus_donor_margin": baseline, "intervened_correct_minus_donor_margin": margin, "margin_shift": margin - baseline, "answer_flip": int(baseline >= 0 and margin < 0)}


def summarize(
    rows: list[dict[str, Any]],
    head_sets: dict[str, list[tuple[int, int]]],
    runtime: Any,
    *,
    stability_summary: dict[str, Any],
    matched_control_quantile: float,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["head_set"], row["intervention"])].append(float(row["margin_shift"]))
    flip_grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        flip_grouped[(row["head_set"], row["intervention"])].append(int(row["answer_flip"]))
    aggregate = [{"head_set": head_set, "intervention": intervention, "n": len(values), "mean_margin_shift": sum(values) / len(values), "answer_flip_rate": sum(flip_grouped[(head_set, intervention)]) / len(values)} for (head_set, intervention), values in sorted(grouped.items())]
    claim_checks = count_claim_checks(
        rows,
        stability_passed=stability_summary.get("passes_stability_gate") is True,
        matched_control_quantile=matched_control_quantile,
    )
    return {"valid": True, "label": "locked confirmation" if claim_checks["all_pass"] else "failed calibration", "architecture": vars(runtime.architecture), "n_head_sets": len(head_sets), "n_rows": len(rows), "aggregate": aggregate, "control_policy": "Locked behavioral validation uses 20 jointly matched draws per k; separate single-feature control distributions remain in count_head_controls.tsv.", "claim_checks": claim_checks, "claim_gate": "Count heads are declared only when stability, bidirectionality, necessity, sufficiency, sham, relocation, answer-code, renderer-transfer, and matched-control checks all pass."}


def count_claim_checks(
    rows: list[dict[str, Any]],
    *,
    stability_passed: bool,
    matched_control_quantile: float,
) -> dict[str, Any]:
    if not 0 < matched_control_quantile < 1:
        raise ValueError("matched-control quantile must lie strictly between zero and one")

    def shifts(
        group: list[dict[str, Any]],
        *,
        intervention: str,
        pair_type: str | None = None,
        variant: str | None = None,
        position_variant: str | None = None,
        renderer_seeds: set[int] | None = None,
    ) -> list[float]:
        values = []
        for row in group:
            if row["intervention"] != intervention:
                continue
            if pair_type is not None and row.get("pair_type") != pair_type:
                continue
            if variant is not None and row.get("variant") != variant:
                continue
            if position_variant is not None and row.get("position_variant") != position_variant:
                continue
            if renderer_seeds is not None and int(row["renderer_seed"]) not in renderer_seeds:
                continue
            values.append(float(row["margin_shift"]))
        return values

    def negative(values: list[float]) -> bool:
        return bool(values) and sum(values) / len(values) < 0

    per_k: dict[str, Any] = {}
    selected_names = sorted(
        {row["head_set"] for row in rows if row["head_set"].startswith("count_top")},
        key=lambda value: int(value.removeprefix("count_top")),
    )
    for selected_name in selected_names:
        k = int(selected_name.removeprefix("count_top"))
        selected = [row for row in rows if row["head_set"] == selected_name]
        real = [row for row in selected if row.get("pair_type") == "constant-complexity"]
        sham = [row for row in selected if row.get("pair_type") == "matched-sham"]
        code = [row for row in selected if row.get("pair_type") == "randomized-answer-code"]
        renderer_seeds = sorted({int(row["renderer_seed"]) for row in real})
        renderer_halves = [set(renderer_seeds[::2]), set(renderer_seeds[1::2])]
        selected_donor = shifts(real, intervention="donor_patch")
        selected_effect = abs(sum(selected_donor) / len(selected_donor)) if selected_donor else 0.0
        control_effects = []
        control_prefix = f"fully_matched_k{k}_draw"
        for control_name in sorted(
            {row["head_set"] for row in rows if row["head_set"].startswith(control_prefix)}
        ):
            control = [
                row
                for row in rows
                if row["head_set"] == control_name
                and row.get("pair_type") == "constant-complexity"
            ]
            values = shifts(control, intervention="donor_patch")
            if values:
                control_effects.append(abs(sum(values) / len(values)))
        threshold = empirical_quantile(control_effects, matched_control_quantile)
        real_sham = shifts(sham, intervention="donor_patch")
        sham_effect = abs(sum(real_sham) / len(real_sham)) if real_sham else float("inf")
        checks = {
            "necessity_zero": negative(shifts(real, intervention="zero")),
            "necessity_mean": negative(shifts(real, intervention="mean")),
            "necessity_resample": negative(shifts(real, intervention="resample")),
            "sufficiency_donor_patch": negative(selected_donor),
            "bidirectional_reverse_patch": negative(
                shifts(real, intervention="reverse_donor_patch")
            ),
            "color_transfer": negative(
                shifts(real, intervention="donor_patch", variant="color")
            ),
            "shape_transfer": negative(
                shifts(real, intervention="donor_patch", variant="shape")
            ),
            "standard_position": negative(
                shifts(real, intervention="donor_patch", position_variant="standard")
            ),
            "relocated_position": negative(
                shifts(
                    real,
                    intervention="donor_patch",
                    position_variant="target_relocation",
                )
            ),
            "randomized_answer_code": negative(shifts(code, intervention="donor_patch")),
            "sham_smaller_than_real": bool(real_sham) and selected_effect > sham_effect,
            "renderer_seed_transfer": len(renderer_seeds) >= 2
            and all(
                negative(
                    shifts(real, intervention="donor_patch", renderer_seeds=half)
                )
                for half in renderer_halves
            ),
            "beats_matched_controls": threshold is not None
            and selected_effect > threshold,
        }
        per_k[str(k)] = {
            **checks,
            "all_pass": all(checks.values()),
            "selected_absolute_donor_effect": selected_effect,
            "sham_absolute_donor_effect": sham_effect if real_sham else None,
            "matched_control_quantile": matched_control_quantile,
            "matched_control_threshold": threshold,
            "n_matched_control_draws": len(control_effects),
            "n_renderer_seeds": len(renderer_seeds),
        }
    any_k_passes = any(value["all_pass"] for value in per_k.values())
    return {
        "stability_passed": stability_passed,
        "any_k_passes_locked_checks": any_k_passes,
        "all_pass": stability_passed and any_k_passes,
        "per_k": per_k,
    }


def empirical_quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(quantile * len(ordered) + 0.999999) - 1))
    return ordered[index]


def render_validation(rows,path):
    import matplotlib.pyplot as plt
    selected=[row for row in rows if (row["head_set"].startswith("count_top") or row["head_set"].startswith("gaze_top")) and row["intervention"] in {"zero","donor_patch"}]
    figure,axis=plt.subplots(figsize=(10,5.5),constrained_layout=True);labels=[f"{row['head_set']}\n{row['intervention']}" for row in selected];values=[float(row["mean_margin_shift"]) for row in selected];colors=["#d95f02" if row["head_set"].startswith("count") else "#31688e" for row in selected];axis.bar(labels,values,color=colors);axis.axhline(0,color="#555555",linewidth=.9);axis.set_ylabel("Mean correct-answer margin shift");axis.set_title("Causal double dissociation: count-ranked vs gaze-ranked heads",loc="left",fontweight="bold");axis.tick_params(axis="x",rotation=35,labelsize=8);figure.savefig(path,dpi=220,facecolor="white");plt.close(figure)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}) or ["pair_id"], delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
