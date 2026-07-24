from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


EXPERIMENT_VERSION = "qwen3_attention_methods_v1"
DEFAULT_EXPERIMENT_ROOT = Path(
    "segments/gaze_heads_qwen3_8b/experiments/attention_methods_v1"
)
DEFAULT_RUN_ROOT = Path(
    "segments/gaze_heads_qwen3_8b/runs/attention_methods_v1"
)
DEFAULT_REPORT_ROOT = Path(
    "segments/gaze_heads_qwen3_8b/reports/attention_methods_v1"
)
QUALIFICATION = {
    "max_vlmbias_accuracy_drop": 0.02,
    "max_naturalbench_g_acc_drop": 0.05,
    "max_invalid_rate_increase": 0.02,
}


def methodology() -> dict[str, Any]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "ranking": (
            "The seed-42 merged COMICS gaze ranking is fixed before benchmark "
            "evaluation; no VLMBias or NaturalBench labels enter head discovery."
        ),
        "controllers": {
            "baseline": "No image-attention logit bias (alpha = 0).",
            "fixed": (
                "Add one constant alpha to image-token logits in every selected "
                "head and every prefill/decode query."
            ),
            "target_mass": (
                "For each selected head and query, compute the smallest nonnegative "
                "logit shift that reaches the requested image-attention mass, "
                "capped at alpha = 5. The intervention never suppresses image mass."
            ),
            "confidence_gate": (
                "Run an unboosted greedy pass, compute the geometric mean chosen-token "
                "probability, and rerun with fixed alpha = 2 only below the threshold."
            ),
        },
        "development_sweep": {
            "mechanics_smoke": (
                "8 VLMBias rows and 2 NaturalBench groups across baseline, fixed "
                "alpha 2, target mass 0.7, and confidence gate 0.6"
            ),
            "vlmbias_examples": 100,
            "naturalbench_groups": 25,
            "split_seed": 2026,
            "controller_conditions": [row["name"] for row in controller_conditions()],
            "selection_objective": (
                "Minimize VLMBias bias_aligned_fraction after guardrails, then "
                "maximize VLMBias accuracy, NaturalBench G_Acc, and NaturalBench Acc."
            ),
            "guardrails": {
                "maximum_vlmbias_accuracy_drop": QUALIFICATION[
                    "max_vlmbias_accuracy_drop"
                ],
                "maximum_naturalbench_G_Acc_drop": QUALIFICATION[
                    "max_naturalbench_g_acc_drop"
                ],
                "maximum_invalid_rate_increase": QUALIFICATION[
                    "max_invalid_rate_increase"
                ],
            },
        },
        "head_sweep": {
            "eligible": (
                "global gaze top 10/50/100 and early/middle/late gaze top 50"
            ),
            "diagnostic_controls": (
                "two layer-matched random top-50 draws, a layer-matched lowest-score "
                "top-50 set, and a paper-style layer-20-to-35 random top-50 set"
            ),
            "policy": "Controls are never eligible to become the locked intervention.",
        },
        "confirmation": {
            "vlmbias_examples": 300,
            "naturalbench_groups": 75,
            "conditions": (
                "baseline plus the best development setting from fixed, target-mass, "
                "and confidence-gated families, all using the locked gaze-head set"
            ),
            "primary": True,
        },
        "robustness": {
            "default_seeds": [0, 1, 2],
            "sampling": "temperature 0.7 on all 400 VLMBias examples and 100 NaturalBench groups",
            "interpretation": (
                "Secondary robustness analysis; it includes development rows, so the "
                "held-out deterministic confirmation remains the primary result."
            ),
        },
    }


def controller_conditions() -> list[dict[str, Any]]:
    conditions = [_condition("baseline", controller="fixed", alpha=0.0)]
    conditions.extend(
        _condition(f"fixed_alpha{slug(alpha)}", controller="fixed", alpha=alpha)
        for alpha in (0.5, 1.0, 2.0, 5.0)
    )
    conditions.extend(
        _condition(
            f"target_mass{slug(target)}",
            controller="target_mass",
            target_mass=target,
        )
        for target in (0.5, 0.7, 0.9)
    )
    conditions.extend(
        _condition(
            f"confidence_gate{slug(threshold)}",
            controller="confidence_gate",
            alpha=2.0,
            confidence_threshold=threshold,
        )
        for threshold in (0.3, 0.6)
    )
    return conditions


def smoke_conditions() -> list[dict[str, Any]]:
    selected = [
        controller_conditions()[0],
        controller_conditions()[3],
        controller_conditions()[6],
        controller_conditions()[9],
    ]
    return [
        {
            **condition,
            "head_count": 0 if condition["name"] == "baseline" else 10,
        }
        for condition in selected
    ]


def head_conditions(controller_spec: dict[str, Any]) -> list[dict[str, Any]]:
    definitions = [
        ("global_top10", "gaze_global", 10, 0),
        ("global_top50", "gaze_global", 50, 0),
        ("global_top100", "gaze_global", 100, 0),
        ("early_top50", "gaze_early", 50, 0),
        ("middle_top50", "gaze_middle", 50, 0),
        ("late_top50", "gaze_late", 50, 0),
        ("layer_matched_random50_seed55", "layer_matched_random", 50, 55),
        ("layer_matched_random50_seed56", "layer_matched_random", 50, 56),
        ("layer_matched_low50", "layer_matched_low", 50, 0),
        ("paper_random50_seed55", "paper_random", 50, 55),
    ]
    return [
        {
            **controller_spec,
            "name": name,
            "head_selection": selection,
            "head_count": count,
            "head_seed": seed,
            "source_controller": controller_spec["name"],
        }
        for name, selection, count, seed in definitions
    ]


def confirmation_conditions(
    controller_selection: dict[str, Any], head_selection: dict[str, Any]
) -> list[dict[str, Any]]:
    chosen_head = head_selection["selected_head"]["spec"]
    baseline = _condition("baseline", controller="fixed", alpha=0.0)
    output = [baseline]
    for family in ("fixed", "target_mass", "confidence_gate"):
        source = controller_selection["selected_by_family"][family]["spec"]
        output.append(
            {
                **source,
                "name": f"confirm_{family}_{source['name']}_{chosen_head['name']}",
                "head_selection": chosen_head["head_selection"],
                "head_count": chosen_head["head_count"],
                "head_seed": chosen_head["head_seed"],
                "source_controller": source["name"],
                "source_head": chosen_head["name"],
            }
        )
    return output


def robustness_conditions(
    controller_selection: dict[str, Any],
    head_selection: dict[str, Any],
    seeds: list[int],
) -> list[dict[str, Any]]:
    base = confirmation_conditions(controller_selection, head_selection)
    return [
        {
            **condition,
            "name": f"{condition['name']}_seed{seed}",
            "seed": seed,
            "do_sample": True,
            "temperature": 0.7,
            "split": "all",
        }
        for seed in seeds
        for condition in base
    ]


def prepare_stratified_splits(
    *,
    vlmbias_path: Path,
    naturalbench_path: Path,
    out_dir: Path,
    vlmbias_dev: int = 100,
    naturalbench_dev: int = 25,
    vlmbias_smoke: int = 8,
    naturalbench_smoke: int = 2,
    seed: int = 2026,
) -> dict[str, Any]:
    if not 0 < vlmbias_smoke <= vlmbias_dev:
        raise ValueError("VLMBias smoke size must be within the development split")
    if not 0 < naturalbench_smoke <= naturalbench_dev:
        raise ValueError(
            "NaturalBench smoke size must be within the development split"
        )
    vlmbias_rows = read_jsonl(vlmbias_path)
    naturalbench_rows = read_jsonl(naturalbench_path)
    vlmbias_dev_ids = _stratified_ids(
        vlmbias_rows,
        n_select=vlmbias_dev,
        key=lambda row: str(row.get("topic", "unknown")),
        seed=seed,
    )
    naturalbench_dev_ids = _stratified_ids(
        naturalbench_rows,
        n_select=naturalbench_dev,
        key=lambda row: (
            str(row.get("question_type", "unknown")),
            str(row.get("source", "unknown")),
        ),
        seed=seed,
    )
    split_rows = {
        "smoke_vlmbias": [
            row
            for row in vlmbias_rows
            if str(row.get("id")) in vlmbias_dev_ids
        ][:vlmbias_smoke],
        "dev_vlmbias": [
            row for row in vlmbias_rows if str(row.get("id")) in vlmbias_dev_ids
        ],
        "confirm_vlmbias": [
            row for row in vlmbias_rows if str(row.get("id")) not in vlmbias_dev_ids
        ],
        "all_vlmbias": vlmbias_rows,
        "smoke_naturalbench": [
            row
            for row in naturalbench_rows
            if str(row.get("id")) in naturalbench_dev_ids
        ][:naturalbench_smoke],
        "dev_naturalbench": [
            row
            for row in naturalbench_rows
            if str(row.get("id")) in naturalbench_dev_ids
        ],
        "confirm_naturalbench": [
            row
            for row in naturalbench_rows
            if str(row.get("id")) not in naturalbench_dev_ids
        ],
        "all_naturalbench": naturalbench_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for label, rows in split_rows.items():
        source = vlmbias_path if label.endswith("vlmbias") else naturalbench_path
        adjusted = [_absolute_image_paths(row, source) for row in rows]
        path = out_dir / f"{label}.jsonl"
        write_jsonl(path, adjusted)
        paths[label] = str(path)
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "seed": seed,
        "source": {
            "vlmbias": str(vlmbias_path),
            "vlmbias_sha256": sha256(vlmbias_path),
            "naturalbench": str(naturalbench_path),
            "naturalbench_sha256": sha256(naturalbench_path),
        },
        "counts": {label: len(rows) for label, rows in split_rows.items()},
        "stratification": {
            "vlmbias": "topic",
            "naturalbench": "question_type x source",
        },
        "paths": paths,
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def write_condition_manifest(
    *,
    stage: str,
    conditions: list[dict[str, Any]],
    split_manifest: dict[str, Any],
    out_path: Path,
    seed: int = 0,
    do_sample: bool = False,
    temperature: float | None = None,
    split: str = "dev",
) -> dict[str, Any]:
    rows = []
    for condition in conditions:
        condition_split = str(condition.get("split", split))
        rows.append(
            {
                **condition,
                "seed": int(condition.get("seed", seed)),
                "do_sample": bool(condition.get("do_sample", do_sample)),
                "temperature": condition.get("temperature", temperature),
                "split": condition_split,
                "vlmbias_dataset": split_manifest["paths"][
                    f"{condition_split}_vlmbias"
                ],
                "naturalbench_dataset": split_manifest["paths"][
                    f"{condition_split}_naturalbench"
                ],
            }
        )
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": stage,
        "conditions": rows,
        "split_manifest": str(out_path.parent / "split_manifest.json"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _condition(
    name: str,
    *,
    controller: str,
    alpha: float = 0.0,
    target_mass: float = 0.0,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "controller": controller,
        "alpha": float(alpha),
        "target_mass": float(target_mass),
        "max_alpha": 5.0,
        "confidence_threshold": confidence_threshold,
        "head_selection": "gaze_global",
        "head_count": 0 if name == "baseline" else 100,
        "head_seed": 0,
    }


def _stratified_ids(
    rows: list[dict[str, Any]],
    *,
    n_select: int,
    key: Callable[[dict[str, Any]], Any],
    seed: int,
) -> set[str]:
    if not 0 < n_select < len(rows):
        raise ValueError("development split must be non-empty and smaller than data")
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    rng = random.Random(seed)
    for group_rows in groups.values():
        rng.shuffle(group_rows)
    raw = {
        group: n_select * len(group_rows) / len(rows)
        for group, group_rows in groups.items()
    }
    allocations = {group: int(value) for group, value in raw.items()}
    remaining = n_select - sum(allocations.values())
    order = sorted(
        groups,
        key=lambda group: (raw[group] - allocations[group], str(group)),
        reverse=True,
    )
    for group in order[:remaining]:
        allocations[group] += 1
    selected = {
        str(row.get("id"))
        for group, group_rows in groups.items()
        for row in group_rows[: allocations[group]]
    }
    if len(selected) != n_select:
        raise RuntimeError(f"selected {len(selected)} unique IDs; expected {n_select}")
    return selected


def _absolute_image_paths(row: dict[str, Any], source: Path) -> dict[str, Any]:
    output = dict(row)
    for key in ("image_path", "image_0_path", "image_1_path"):
        if output.get(key):
            path = Path(str(output[key]))
            if not path.is_absolute():
                path = (source.parent / path).resolve()
            output[key] = str(path)
    return output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")
