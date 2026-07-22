from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast validation gates for the Qwen3 Gaze Heads pipeline.")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    discovery = subparsers.add_parser("discovery")
    discovery.add_argument("--run-dir", required=True)
    discovery.add_argument("--min-samples", type=int, default=10)
    discovery.add_argument("--min-routing-accuracy", type=float, default=0.5)

    static = subparsers.add_parser("static")
    static.add_argument("--run-dir", required=True)
    static.add_argument("--warn-empty-rate", type=float, default=0.05)
    static.add_argument("--max-empty-rate", type=float, default=0.50)
    static.add_argument("--min-target-mass", type=float, default=0.95)

    datasets = subparsers.add_parser("datasets")
    datasets.add_argument(
        "--discovery-root", default="segments/gaze_heads_qwen3_8b/data/discovery_comics"
    )
    datasets.add_argument("--eval-root", default="segments/gaze_heads_qwen3_8b/data/eval_comics")
    datasets.add_argument("--expected-eval-strips", type=int, default=500)
    datasets.add_argument("--vlmbias", default="segments/vlm_bias_attention/data/vlmbias_400.jsonl")
    datasets.add_argument(
        "--naturalbench", default="segments/vlm_bias_attention/data/naturalbench_100_groups.jsonl"
    )
    datasets.add_argument("--expected-vlmbias", type=int, default=400)
    datasets.add_argument("--expected-naturalbench-groups", type=int, default=100)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--run-dir", required=True)

    args = parser.parse_args()
    if args.stage == "discovery":
        report = validate_discovery(Path(args.run_dir), args.min_samples, args.min_routing_accuracy)
        report_path = Path(args.run_dir) / "validation.json"
    elif args.stage == "static":
        report = validate_static(
            Path(args.run_dir),
            args.max_empty_rate,
            args.min_target_mass,
            warn_empty_rate=args.warn_empty_rate,
        )
        report_path = Path(args.run_dir) / "validation.json"
    elif args.stage == "datasets":
        report = validate_datasets(
            Path(args.discovery_root),
            Path(args.eval_root),
            Path(args.vlmbias),
            Path(args.naturalbench),
            expected_eval_strips=args.expected_eval_strips,
            expected_vlmbias=args.expected_vlmbias,
            expected_naturalbench_groups=args.expected_naturalbench_groups,
        )
        report_path = Path("segments/gaze_heads_qwen3_8b/reports/dataset_validation.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        report = validate_benchmark(Path(args.run_dir))
        report_path = Path(args.run_dir) / "validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(2)


def validate_datasets(
    discovery_root: Path,
    eval_root: Path,
    vlmbias_path: Path,
    naturalbench_path: Path,
    *,
    expected_eval_strips: int = 500,
    expected_vlmbias: int = 400,
    expected_naturalbench_groups: int = 100,
) -> dict[str, Any]:
    errors = []
    discovery_names = _directory_names(discovery_root, "raw COMICS discovery root", errors)
    eval_names = _directory_names(eval_root, "evaluation comics root", errors)
    overlap = sorted(discovery_names & eval_names)
    if discovery_root.exists() and eval_root.exists() and discovery_root.resolve() == eval_root.resolve():
        errors.append("discovery and evaluation roots resolve to the same directory")
    if overlap:
        errors.append(f"{len(overlap)} comic IDs overlap between discovery and evaluation")

    eligible_raw_pages = _count_eligible_raw_pages(discovery_root) if discovery_root.is_dir() else 0
    if discovery_root.is_dir() and eligible_raw_pages == 0:
        errors.append("raw COMICS contains no page with at least six numbered panels")

    complete_eval_strips = _count_complete_eval_strips(eval_root) if eval_root.is_dir() else 0
    if complete_eval_strips < expected_eval_strips:
        errors.append(
            f"only {complete_eval_strips} complete six-panel evaluation strips; "
            f"require {expected_eval_strips}"
        )

    vlmbias_rows = _safe_read_jsonl(vlmbias_path, "VLMBias slice", errors)
    _validate_vlmbias(vlmbias_rows, vlmbias_path, expected_vlmbias, errors)
    naturalbench_rows = _safe_read_jsonl(naturalbench_path, "NaturalBench slice", errors)
    _validate_naturalbench(
        naturalbench_rows, naturalbench_path, expected_naturalbench_groups, errors
    )
    return {
        "valid": not errors,
        "stage": "datasets",
        "n_discovery_directories": len(discovery_names),
        "n_eligible_raw_pages": eligible_raw_pages,
        "n_evaluation_directories": len(eval_names),
        "n_complete_evaluation_strips": complete_eval_strips,
        "n_vlmbias_rows": len(vlmbias_rows),
        "n_naturalbench_groups": len(naturalbench_rows),
        "n_naturalbench_model_calls": len(naturalbench_rows) * 4,
        "n_overlap": len(overlap),
        "overlap_examples": overlap[:20],
        "errors": errors,
    }


def _directory_names(root: Path, label: str, errors: list[str]) -> set[str]:
    if not root.is_dir():
        errors.append(f"missing {label}: {root}")
        return set()
    return {path.name for path in root.iterdir() if path.is_dir()}


def _count_eligible_raw_pages(root: Path) -> int:
    # This corpus contains millions of panel files on shared storage. Dataset
    # validation only needs proof that the expected raw layout is present, so
    # stop at the first page with six numbered panels instead of traversing the
    # entire corpus.
    for comic_dir in (path for path in root.iterdir() if path.is_dir()):
        counts: Counter[int] = Counter()
        for image in comic_dir.iterdir():
            parts = image.stem.split("_")
            if not image.is_file() or len(parts) != 2:
                continue
            try:
                page, panel = (int(part) for part in parts)
            except ValueError:
                continue
            if panel >= 0:
                counts[page] += 1
                if counts[page] >= 6:
                    return 1
    return 0


def _count_complete_eval_strips(root: Path) -> int:
    suffixes = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    return sum(
        all(any((strip / f"p{panel}{suffix}").is_file() for suffix in suffixes) for panel in range(1, 7))
        for strip in root.iterdir()
        if strip.is_dir()
    )


def _safe_read_jsonl(path: Path, label: str, errors: list[str]) -> list[dict[str, Any]]:
    try:
        return _read_jsonl(path)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {label} {path}: {exc}")
        return []


def _validate_vlmbias(
    rows: list[dict[str, Any]], path: Path, expected: int, errors: list[str]
) -> None:
    if len(rows) != expected:
        errors.append(f"VLMBias has {len(rows)} rows; require exactly {expected}")
    ids = [str(row.get("id", "")) for row in rows]
    if any(not identifier for identifier in ids) or len(ids) != len(set(ids)):
        errors.append("VLMBias IDs must be non-empty and unique")
    for index, row in enumerate(rows):
        if not str(row.get("prompt", "")).strip() or not str(row.get("ground_truth", "")).strip():
            errors.append(f"VLMBias row {index} is missing its prompt or ground truth")
            break
        if not _referenced_image_exists(path, row.get("image_path")):
            errors.append(f"VLMBias row {index} references a missing image: {row.get('image_path')}")
            break


def _validate_naturalbench(
    rows: list[dict[str, Any]], path: Path, expected: int, errors: list[str]
) -> None:
    if len(rows) != expected:
        errors.append(f"NaturalBench has {len(rows)} groups; require exactly {expected}")
    ids = [str(row.get("id", "")) for row in rows]
    if any(not identifier for identifier in ids) or len(ids) != len(set(ids)):
        errors.append("NaturalBench group IDs must be non-empty and unique")
    answer_keys = {"q0_i0", "q0_i1", "q1_i0", "q1_i1"}
    for index, row in enumerate(rows):
        if not str(row.get("question_0", "")).strip() or not str(row.get("question_1", "")).strip():
            errors.append(f"NaturalBench group {index} is missing a question")
            break
        answers = row.get("answers") or {}
        if set(answers) != answer_keys or any(not str(value).strip() for value in answers.values()):
            errors.append(f"NaturalBench group {index} does not contain all four answers")
            break
        for key in ("image_0_path", "image_1_path"):
            if not _referenced_image_exists(path, row.get(key)):
                errors.append(f"NaturalBench group {index} references a missing image: {row.get(key)}")
                return


def _referenced_image_exists(dataset_path: Path, value: Any) -> bool:
    if not value:
        return False
    image = Path(str(value))
    return (image if image.is_absolute() else dataset_path.parent / image).is_file()


def validate_discovery(run_dir: Path, min_samples: int, min_routing_accuracy: float) -> dict[str, Any]:
    summary = _read_json(run_dir / "summary.json")
    routing = np.load(run_dir / "gaze_routing_accuracy.npy")
    errors = []
    valid_samples = int(summary.get("valid_samples", 0))
    max_routing = float(routing.max())
    if valid_samples < min_samples:
        errors.append(f"only {valid_samples} valid samples; require at least {min_samples}")
    if max_routing <= min_routing_accuracy:
        errors.append(
            f"best head routing accuracy {max_routing:.3f} is not above {min_routing_accuracy:.3f}; "
            "the ranking may reflect fixed spatial attention rather than queried-panel gaze"
        )
    return {
        "valid": not errors,
        "stage": "discovery",
        "valid_samples": valid_samples,
        "max_head_routing_accuracy": max_routing,
        "errors": errors,
    }


def validate_benchmark(run_dir: Path) -> dict[str, Any]:
    errors = []
    summaries = []
    for benchmark in ("vlmbias", "naturalbench"):
        paths = sorted((run_dir / benchmark).glob("*.summary.json"))
        if not paths:
            errors.append(f"no {benchmark} condition summaries")
        for path in paths:
            summary = _read_json(path)
            config = summary.get("run_config") or {}
            condition = str(config.get("condition", ""))
            is_baseline = condition == "baseline"
            boosted_mass = summary.get("mean_boosted_head_image_attention_mass")
            if not is_baseline and boosted_mass is None:
                errors.append(f"missing boosted-head attention telemetry: {path}")
            summaries.append(
                {
                    "benchmark": benchmark,
                    "path": str(path),
                    "condition": condition,
                    "seed": config.get("seed"),
                    "boosted_head_image_attention_mass": boosted_mass,
                }
            )
    return {
        "valid": not errors,
        "stage": "benchmark",
        "n_condition_summaries": len(summaries),
        "errors": errors,
        "summaries": summaries,
    }


def validate_static(
    run_dir: Path,
    max_empty_rate: float,
    min_target_mass: float,
    *,
    warn_empty_rate: float = 0.05,
) -> dict[str, Any]:
    rows = _read_jsonl(run_dir / "generations.jsonl")
    config_path = run_dir / "experiment_config.json"
    summary_path = run_dir / "summary.json"
    config = _read_json(config_path) if config_path.exists() else None
    summary = _read_json(summary_path) if summary_path.exists() else None
    errors = []
    warnings = []
    keys = [
        (str(row.get("strip_name")), str(row.get("condition")), int(row.get("target_panel", 0)))
        for row in rows
    ]
    duplicate_count = len(keys) - len(set(keys))
    if duplicate_count:
        errors.append(f"found {duplicate_count} duplicate (strip, condition, target-panel) rows")
    counts = Counter(str(row.get("condition")) for row in rows)
    empty = Counter(
        str(row.get("condition"))
        for row in rows
        if not str(row.get("generated_text", "") or "").strip()
    )
    empty_rates = {condition: empty[condition] / count for condition, count in sorted(counts.items())}
    if not rows:
        errors.append("no generation rows")
    expected_rows = None
    if config is not None and summary is not None:
        n_comics = int(summary.get("n_comics", 0))
        targets = max(1, min(int(config.get("targets_per_strip", 0)), 6))
        n_conditions = 2 + int(bool(config.get("include_all_heads", False)))
        expected_rows = n_comics * targets * n_conditions
        if len(rows) != expected_rows:
            errors.append(f"row count {len(rows)} does not match expected {expected_rows}")
        expected_per_condition = n_comics * targets
        if any(count != expected_per_condition for count in counts.values()) or len(counts) != n_conditions:
            errors.append(
                f"condition counts are incomplete: {dict(sorted(counts.items()))}; "
                f"expected {n_conditions} conditions with {expected_per_condition} rows each"
            )
    if any(rate > max_empty_rate for rate in empty_rates.values()):
        errors.append(f"empty-generation rate exceeds {max_empty_rate:.1%}: {empty_rates}")
    elif any(rate > warn_empty_rate for rate in empty_rates.values()):
        warnings.append(
            f"empty-generation rate exceeds warning threshold {warn_empty_rate:.1%}: {empty_rates}; "
            "empty outputs must be retained and scored as misses"
        )

    masses = []
    missing_mass_empty = 0
    missing_mass_nonempty = 0
    for row in rows:
        panel = int(row.get("target_panel", 0))
        attention = ((row.get("metadata") or {}).get("attention") or {})
        value = attention.get(f"mean_decode_panel_{panel}_attention_mass")
        if value is None:
            if str(row.get("generated_text", "") or "").strip():
                missing_mass_nonempty += 1
            else:
                # An immediate EOS has no decode attention step. Its missing
                # decode telemetry is expected and the empty output is scored
                # separately as a miss.
                missing_mass_empty += 1
        else:
            masses.append(float(value))
    if missing_mass_nonempty:
        errors.append(
            f"{missing_mass_nonempty}/{len(rows)} non-empty rows lack decode-step target-attention telemetry"
        )
    if masses and min(masses) < min_target_mass:
        errors.append(
            f"minimum decode target attention mass {min(masses):.6f} is below {min_target_mass:.6f}"
        )

    unique_by_condition = {
        condition: len({str(row.get("generated_text", "")).strip() for row in rows if row.get("condition") == condition})
        for condition in sorted(counts)
    }
    if any(unique <= 1 and counts[condition] > 1 for condition, unique in unique_by_condition.items()):
        warnings.append(f"one or more conditions produced identical text for every target: {unique_by_condition}")
    return {
        "valid": not errors,
        "stage": "static",
        "n_rows": len(rows),
        "expected_rows": expected_rows,
        "n_unique_rows": len(set(keys)),
        "duplicate_count": duplicate_count,
        "empty_rates": empty_rates,
        "target_attention_mass": {
            "n": len(masses),
            "minimum": min(masses) if masses else None,
            "mean": sum(masses) / len(masses) if masses else None,
            "missing_on_empty_outputs": missing_mass_empty,
            "missing_on_nonempty_outputs": missing_mass_nonempty,
        },
        "unique_outputs_by_condition": unique_by_condition,
        "errors": errors,
        "warnings": warnings,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
