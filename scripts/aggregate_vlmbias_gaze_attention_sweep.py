from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any


VLMBIAS_METRICS = ("accuracy", "bias_aligned_fraction", "bias_aligned_error_rate", "error_rate")
NATURALBENCH_METRICS = ("Acc", "Q_Acc", "I_Acc", "G_Acc")
ATTENTION_METRICS = (
    "mean_image_attention_mass",
    "mean_boosted_layer_image_attention_mass",
    "mean_boosted_head_image_attention_mass",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate gaze-attention sweep summary JSON files across seeds.")
    parser.add_argument("--out-dir", default="segments/vlm_bias_attention/runs/vlmbias_gaze_attention_high_alpha_sweep")
    parser.add_argument("--expected-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--expected-top-ks", nargs="*", type=int, default=None)
    parser.add_argument("--expected-alphas", nargs="*", type=float, default=None)
    parser.add_argument("--strict", action="store_true", help="Fail if any expected summary file is missing.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    vlmbias_rows = _load_rows(out_dir, "vlmbias")
    naturalbench_rows = _load_rows(out_dir, "naturalbench")
    if not vlmbias_rows:
        raise SystemExit(f"No VLMBias summary files found under {out_dir / 'vlmbias'}")
    if not naturalbench_rows:
        raise SystemExit(f"No NaturalBench summary files found under {out_dir / 'naturalbench'}")

    _write_experiment_config(
        out_dir,
        rows=[*vlmbias_rows, *naturalbench_rows],
        expected_seeds=args.expected_seeds,
        expected_top_ks=args.expected_top_ks,
        expected_alphas=args.expected_alphas,
    )

    if args.expected_seeds is not None and args.expected_top_ks is not None and args.expected_alphas is not None:
        missing = _missing_conditions(
            rows=vlmbias_rows,
            expected_seeds=args.expected_seeds,
            expected_top_ks=args.expected_top_ks,
            expected_alphas=args.expected_alphas,
            benchmark="vlmbias",
        )
        missing += _missing_conditions(
            rows=naturalbench_rows,
            expected_seeds=args.expected_seeds,
            expected_top_ks=args.expected_top_ks,
            expected_alphas=args.expected_alphas,
            benchmark="naturalbench",
        )
        if missing:
            message = "Missing expected summaries:\n" + "\n".join(missing)
            if args.strict:
                raise SystemExit(message)
            print(message)

    _write_summary_tsv(vlmbias_rows, out_dir / "vlmbias_summary_by_seed.tsv", VLMBIAS_METRICS)
    _write_aggregate_tsv(vlmbias_rows, out_dir / "vlmbias_summary_aggregate.tsv", VLMBIAS_METRICS)
    _write_summary_tsv(naturalbench_rows, out_dir / "naturalbench_summary_by_seed.tsv", NATURALBENCH_METRICS)
    _write_aggregate_tsv(naturalbench_rows, out_dir / "naturalbench_summary_aggregate.tsv", NATURALBENCH_METRICS)
    print(f"Wrote aggregate TSVs under {out_dir}")


def _write_experiment_config(
    out_dir: Path,
    *,
    rows: list[dict[str, Any]],
    expected_seeds: list[int] | None,
    expected_top_ks: list[int] | None,
    expected_alphas: list[float] | None,
) -> None:
    config = dict(rows[0].get("_run_config") or {})
    seeds = expected_seeds if expected_seeds is not None else sorted({int(row["seed"]) for row in rows})
    top_ks = expected_top_ks if expected_top_ks is not None else sorted({int(row["top_k_gaze"]) for row in rows if int(row["top_k_gaze"]) > 0})
    alphas = expected_alphas if expected_alphas is not None else sorted({float(row["attention_alpha"]) for row in rows if float(row["attention_alpha"]) > 0})
    config.update(
        {
            "seeds": seeds,
            "top_ks": top_ks,
            "alphas": alphas,
            "conditions": _conditions(alphas, top_ks),
            "aggregated_from_summary_json": True,
        }
    )
    for transient_key in ("benchmark", "condition", "seed", "attention_alpha", "top_k_gaze"):
        config.pop(transient_key, None)
    _write_json(out_dir / "experiment_config.json", config)


def _load_rows(out_dir: Path, benchmark: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((out_dir / benchmark).glob("*.summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        run_config = summary.get("run_config") or {}
        condition = run_config.get("condition") or _condition_from_filename(path)
        seed = int(run_config.get("seed", _seed_from_filename(path)))
        row = {
            "benchmark": benchmark,
            "condition": condition,
            "seed": seed,
            "attention_alpha": float(run_config.get("attention_alpha", _alpha_from_condition(condition))),
            "top_k_gaze": int(run_config.get("top_k_gaze", _top_k_from_condition(condition))),
            "is_baseline": condition == "baseline",
            "out_path": str(path.with_suffix("").with_suffix(".jsonl")),
            "_run_config": run_config,
        }
        for key, value in summary.items():
            if isinstance(value, int | float) or value is None:
                row[key] = value
        rows.append(row)
    return rows


def _condition_from_filename(path: Path) -> str:
    name = path.name
    name = re.sub(r"^.*?_(vlmbias|naturalbench)_gaze_attention_", "", name)
    name = re.sub(r"_seed\d+\.summary\.json$", "", name)
    return name


def _seed_from_filename(path: Path) -> int:
    match = re.search(r"_seed(\d+)\.summary\.json$", path.name)
    if not match:
        raise ValueError(f"Cannot parse seed from {path}")
    return int(match.group(1))


def _top_k_from_condition(condition: str) -> int:
    if condition == "baseline":
        return 0
    match = re.search(r"gaze_top(\d+)_", condition)
    if not match:
        raise ValueError(f"Cannot parse top-k from {condition}")
    return int(match.group(1))


def _alpha_from_condition(condition: str) -> float:
    if condition == "baseline":
        return 0.0
    match = re.search(r"_alpha([0-9p]+)$", condition)
    if not match:
        raise ValueError(f"Cannot parse alpha from {condition}")
    return float(match.group(1).replace("p", "."))


def _missing_conditions(
    *,
    rows: list[dict[str, Any]],
    expected_seeds: list[int],
    expected_top_ks: list[int],
    expected_alphas: list[float],
    benchmark: str,
) -> list[str]:
    present = {(row["condition"], int(row["seed"])) for row in rows}
    missing = []
    for seed in expected_seeds:
        if ("baseline", seed) not in present:
            missing.append(f"{benchmark}: baseline seed{seed}")
        for alpha in expected_alphas:
            for top_k in expected_top_ks:
                condition = f"gaze_top{top_k}_alpha{_alpha_label(alpha)}"
                if (condition, seed) not in present:
                    missing.append(f"{benchmark}: {condition} seed{seed}")
    return missing


def _conditions(alphas: list[float], top_ks: list[int]) -> list[dict[str, Any]]:
    conditions = [
        {
            "condition": "baseline",
            "attention_alpha": 0.0,
            "top_k_gaze": 0,
            "is_baseline": True,
        }
    ]
    for alpha in alphas:
        for top_k in top_ks:
            conditions.append(
                {
                    "condition": f"gaze_top{top_k}_alpha{_alpha_label(alpha)}",
                    "attention_alpha": float(alpha),
                    "top_k_gaze": int(top_k),
                    "is_baseline": False,
                }
            )
    return conditions


def _alpha_label(alpha: float) -> str:
    return f"{alpha:g}".replace("-", "m").replace(".", "p")


def _write_summary_tsv(rows: list[dict[str, Any]], out_path: Path, metrics: tuple[str, ...]) -> None:
    fieldnames = [
        "benchmark",
        "condition",
        "seed",
        "attention_alpha",
        "top_k_gaze",
        "is_baseline",
        *metrics,
        *ATTENTION_METRICS,
        "out_path",
    ]
    _write_tsv(sorted(rows, key=lambda row: (row["seed"], row["top_k_gaze"], row["attention_alpha"])), out_path, fieldnames)


def _write_aggregate_tsv(rows: list[dict[str, Any]], out_path: Path, metrics: tuple[str, ...]) -> None:
    grouped: dict[tuple[str, float, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["condition"], float(row["attention_alpha"]), int(row["top_k_gaze"]))
        grouped.setdefault(key, []).append(row)

    aggregate_rows = []
    for (condition, attention_alpha, top_k_gaze), group_rows in sorted(grouped.items(), key=lambda item: (item[0][2], item[0][1], item[0][0])):
        aggregate_row = {
            "condition": condition,
            "attention_alpha": attention_alpha,
            "top_k_gaze": top_k_gaze,
            "seeds": ",".join(str(row["seed"]) for row in sorted(group_rows, key=lambda item: item["seed"])),
            "runs": len(group_rows),
        }
        for metric in (*metrics, *ATTENTION_METRICS):
            values = [float(row[metric]) for row in group_rows if row.get(metric) is not None and not _is_nan(row.get(metric))]
            aggregate_row[f"{metric}_mean"] = mean(values) if values else None
            aggregate_row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else None
        aggregate_rows.append(aggregate_row)

    fieldnames = ["condition", "attention_alpha", "top_k_gaze", "seeds", "runs"]
    for metric in (*metrics, *ATTENTION_METRICS):
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    _write_tsv(aggregate_rows, out_path, fieldnames)


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _write_tsv(rows: list[dict[str, Any]], out_path: Path, fieldnames: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_value(row.get(field)) for field in fieldnames})


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else str(value)


if __name__ == "__main__":
    main()
