#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from vlm_eval.mechanistic_heads.causal import capture_prefill
from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, effective_limit, load_json_config, partitioned_limit, prepare_output_directory
from vlm_eval.mechanistic_heads.mmmc import MMMCImages
from vlm_eval.mechanistic_heads.preflight import require_calibration_report, require_current_artifact, require_scientific_validation, validation_path_from_config
from vlm_eval.mechanistic_heads.qwen3_runtime import Qwen3MechanisticRuntime
from vlm_eval.mechanistic_heads.reproducibility import seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.schema import read_paired_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the L1 MACI conflict detector.")
    add_standard_run_arguments(parser)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    config = load_json_config(args.config)
    require_current_artifact(Path(config["head_scores"]))
    if not args.smoke:
        require_scientific_validation(validation_path_from_config(config))
        require_calibration_report(Path(config["stability_report"]), boolean_key="passes_stability_gate")
        require_calibration_report(Path(config["ablation_report"]))
    output = args.output_dir / "conflict_detector.json"
    prepare_output_directory(args.output_dir, resume=args.resume, overwrite=args.overwrite, known_outputs=(output.name,))
    seed_everything(args.seed)
    runtime = Qwen3MechanisticRuntime(model_id=str(config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")), device_map=args.device_map)
    resisting_k = min(2, int(config.get("resisting_heads", 40))) if args.smoke else int(config.get("resisting_heads", 40))
    resisting = load_resisting_heads(
        Path(config["head_scores"]), k=resisting_k, allow_unsigned=args.smoke
    )
    pairs = read_paired_jsonl(Path(config["paired_dataset"])); limit = effective_limit(args)
    by_split = {}
    for pair in pairs: by_split.setdefault(pair.split, []).append(pair)
    if limit is not None:
        split_names = tuple(name for name in ("prototype", "validation", "locked_test") if name in by_split)
        pairs = [
            pair
            for split_index, split in enumerate(split_names)
            for pair in by_split[split][
                : partitioned_limit(limit, groups=len(split_names), index=split_index)
            ]
        ]
    else:
        split_limits = config.get("max_pairs_by_split", {})
        pairs = [
            pair
            for split, rows in by_split.items()
            for pair in rows[: int(split_limits.get(split, len(rows)))]
        ]
    audit_path = Path(config["paired_dataset"]).with_name("audit.json")
    images = MMMCImages(args.cache_dir, audit_path=audit_path)
    features, labels, splits = [], [], []
    layers = sorted({layer for layer, _ in resisting})
    for pair in pairs:
        for role, reference, prompt, label in (("clean", pair.donor_image, pair.donor_prompt, 0), ("conflict", pair.recipient_image, pair.recipient_prompt, 1)):
            capture = capture_prefill(runtime, image_path=images.resolve(reference), prompt=prompt, layers=layers, to_cpu=True)
            vectors = [capture.store.raw_heads[layer][0, capture.prompt_length - 1, head, :].float().detach().cpu().numpy() for layer, head in resisting]
            features.append(np.stack(vectors).mean(axis=0)); labels.append(label); splits.append(pair.split)
    result = fit_detector(np.stack(features), np.asarray(labels), splits, seed=args.seed)
    result.update({"valid": True, "label": "instrumentation smoke test" if args.smoke else "methods-based reproduction", "resisting_heads": [list(head) for head in resisting], "feature": "mean last-prefill raw A_hV_h over resisting heads"})
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest_inputs = [
        args.config,
        Path(config["paired_dataset"]),
        audit_path,
        Path(config["head_scores"]),
    ]
    for key in ("stability_report", "ablation_report"):
        path = Path(config[key])
        if path.is_file():
            manifest_inputs.append(path)
    write_run_manifest(args.output_dir, config={**config, "architecture": vars(runtime.architecture)}, seeds={"global": args.seed}, inputs=manifest_inputs, outputs=[output], status="complete", repo_root=Path.cwd())
    print(json.dumps(result, indent=2))


def fit_detector(x: np.ndarray, y: np.ndarray, splits: list[str], *, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    masks = {name: np.asarray([value == name for value in splits]) for name in ("prototype", "validation", "locked_test")}
    scaler = StandardScaler().fit(x[masks["prototype"]])
    model = LogisticRegression(penalty="l1", solver="liblinear", random_state=seed, max_iter=2000).fit(scaler.transform(x[masks["prototype"]]), y[masks["prototype"]])
    validation_prob = model.predict_proba(scaler.transform(x[masks["validation"]]))[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    threshold = max(thresholds, key=lambda value: f1_score(y[masks["validation"]], validation_prob >= value))
    metrics = {}
    for split, mask in masks.items():
        probabilities = model.predict_proba(scaler.transform(x[mask]))[:, 1]
        metrics[split] = {"n": int(mask.sum()), "AUROC": float(roc_auc_score(y[mask], probabilities)), "AUPRC": float(average_precision_score(y[mask], probabilities)), "F1": float(f1_score(y[mask], probabilities >= threshold)), "intervention_rate": float((probabilities >= threshold).mean())}
    return {"threshold": float(threshold), "metrics": metrics, "coefficient": model.coef_[0].tolist(), "intercept": model.intercept_.tolist(), "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(), "nonzero_coefficients": int(np.count_nonzero(model.coef_))}


def load_resisting_heads(
    path: Path, *, k: int, allow_unsigned: bool = False
) -> list[tuple[int, int]]:
    with path.open("r", encoding="utf-8") as handle: rows = list(csv.DictReader(handle, delimiter="\t"))
    ranked = sorted(
        (
            row
            for row in rows
            if float(row["mean_signed_intervention_score"]) < 0
        ),
        key=lambda row: float(row["mean_signed_intervention_score"]),
    )
    if allow_unsigned and len(ranked) < k:
        ranked = sorted(
            rows, key=lambda row: float(row["mean_signed_intervention_score"])
        )
    heads = [(int(row["layer"]), int(row["head"])) for row in ranked[:k]]
    if len(heads) != k:
        role = "heads" if allow_unsigned else "negative resisting heads"
        raise RuntimeError(
            f"conflict detector requires {k} {role}; found {len(heads)}"
        )
    return heads


if __name__ == "__main__": main()
