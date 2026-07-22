from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from scripts.verify_neuronic_qwen3_prejudge import verify_prejudge


DEFAULT_SEGMENT_ROOT = Path("segments/gaze_heads_qwen3_8b")
EXPECTED_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_TOP_KS = [1, 10, 50, 100]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the provenance and contents of the completed Neuronic Qwen3 gaze-head "
            "artifacts, then create a compact bundle for inspection off-cluster."
        )
    )
    parser.add_argument("--segment-root", type=Path, default=DEFAULT_SEGMENT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path(".cache/audits"))
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--ranking-seed", type=int, default=42)
    parser.add_argument("--top-ks", nargs="+", type=int, default=DEFAULT_TOP_KS)
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--shard-size", type=int, default=50)
    parser.add_argument("--skip-bundle", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    if not (repo_root / ".git").exists():
        raise SystemExit("Run this command from the VLMProject repository root.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    preflight = verify_prejudge(
        segment_root=args.segment_root,
        seeds=args.seeds,
        base_seed=args.base_seed,
        top_ks=args.top_ks,
        shards=args.shards,
        shard_size=args.shard_size,
    )
    preflight_path = args.out_dir / "qwen3_prejudge.json"
    _write_json(preflight_path, preflight)

    audit = audit_artifacts(
        repo_root=repo_root,
        segment_root=args.segment_root,
        base_seed=args.base_seed,
        seeds=args.seeds,
        ranking_seed=args.ranking_seed,
        top_ks=args.top_ks,
        shards=args.shards,
        shard_size=args.shard_size,
    )
    audit["prejudge_valid"] = bool(preflight["valid"])
    if not preflight["valid"]:
        audit["errors"].extend(f"prejudge: {error}" for error in preflight["errors"])
        audit["valid"] = False
    audit_path = args.out_dir / "qwen3_artifact_audit.json"
    _write_json(audit_path, audit)

    bundle_result = None
    if not args.skip_bundle:
        bundle_path = args.out_dir / "qwen3_audit_bundle.tar.gz"
        bundle_result = create_audit_bundle(
            repo_root=repo_root,
            segment_root=args.segment_root,
            audit_paths=[preflight_path, audit_path],
            bundle_path=bundle_path,
            base_seed=args.base_seed,
            seeds=args.seeds,
            ranking_seed=args.ranking_seed,
            top_ks=args.top_ks,
            shards=args.shards,
            shard_size=args.shard_size,
        )
        _write_json(args.out_dir / "qwen3_audit_bundle_manifest.json", bundle_result)

    result = {
        "valid": audit["valid"],
        "prejudge_valid": audit["prejudge_valid"],
        "provenance_valid": audit["provenance_valid"],
        "audit": str(audit_path),
        "prejudge": str(preflight_path),
        "bundle": bundle_result,
        "errors": audit["errors"],
        "warnings": audit["warnings"],
    }
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


def audit_artifacts(
    *,
    repo_root: Path,
    segment_root: Path,
    base_seed: int,
    seeds: int,
    ranking_seed: int,
    top_ks: list[int],
    shards: int,
    shard_size: int,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    runs = segment_root / "runs"
    expected_comics = shards * shard_size
    seed_values = list(range(base_seed, base_seed + seeds))

    discovery: list[dict[str, Any]] = []
    ranking_heads: dict[int, set[str]] = {}
    for seed in seed_values:
        merged = runs / f"gaze_discovery_seed{seed}_merged"
        ranking_path = merged / "gaze_head_ranking.json"
        ranking = _load_json(ranking_path, errors, f"discovery seed {seed} ranking", default=[])
        if not isinstance(ranking, list):
            errors.append(f"discovery seed {seed}: ranking is not a JSON list")
            ranking = []
        if len(ranking) < max(top_ks):
            errors.append(
                f"discovery seed {seed}: ranking has {len(ranking)} heads; require {max(top_ks)}"
            )
        top_heads = {_head_id(row) for row in ranking[: max(top_ks)] if _is_head_row(row)}
        ranking_heads[seed] = top_heads

        shard_summaries = []
        for shard in range(shards):
            start = shard * shard_size
            summary_path = runs / f"gaze_discovery_seed{seed}_{start}_{shard_size}" / "summary.json"
            summary = _load_json(
                summary_path, errors, f"discovery seed {seed} shard {shard} summary"
            )
            if not summary:
                continue
            _require_equal(summary, "model_id", EXPECTED_MODEL, errors, str(summary_path))
            _require_equal(summary, "use_raw", True, errors, str(summary_path))
            _require_equal(summary, "start_comic_idx", start, errors, str(summary_path))
            _require_equal(summary, "max_comics", shard_size, errors, str(summary_path))
            _require_qwen3_path(summary.get("comics_root"), errors, str(summary_path))
            shard_summaries.append(
                {
                    "path": str(summary_path),
                    "valid_samples": summary.get("valid_samples"),
                    "model_id": summary.get("model_id"),
                    "use_raw": summary.get("use_raw"),
                }
            )
        merged_summary = _load_json(
            merged / "summary.json", errors, f"discovery seed {seed} merged summary"
        )
        if merged_summary and int(merged_summary.get("valid_samples", -1)) != expected_comics:
            errors.append(
                f"discovery seed {seed}: merged valid_samples is "
                f"{merged_summary.get('valid_samples')}; expected {expected_comics}"
            )
        discovery.append(
            {
                "seed": seed,
                "merged_dir": str(merged),
                "ranking_sha256": _sha256(ranking_path) if ranking_path.is_file() else None,
                "ranking_heads": len(ranking),
                "top_head": ranking[0] if ranking else None,
                "top_head_layers": dict(
                    sorted(Counter(int(row["layer"]) for row in ranking[:100] if _is_head_row(row)).items())
                ),
                "merged_valid_samples": merged_summary.get("valid_samples"),
                "shards_checked": len(shard_summaries),
            }
        )

    ranking_overlap = []
    for index, left in enumerate(seed_values):
        for right in seed_values[index + 1 :]:
            union = ranking_heads[left] | ranking_heads[right]
            overlap = ranking_heads[left] & ranking_heads[right]
            ranking_overlap.append(
                {
                    "left_seed": left,
                    "right_seed": right,
                    "top_k": max(top_ks),
                    "intersection": len(overlap),
                    "jaccard": len(overlap) / len(union) if union else None,
                }
            )

    static: list[dict[str, Any]] = []
    output_digests: dict[tuple[int, str], dict[int, str]] = defaultdict(dict)
    baseline_digests: dict[tuple[int, int], str] = {}
    for seed in seed_values:
        for top_k in top_ks:
            shard_configs = []
            for shard in range(shards):
                start = shard * shard_size
                shard_dir = runs / f"static_narration_seed{seed}_top{top_k}_{start}_{shard_size}"
                config_path = shard_dir / "experiment_config.json"
                config = _load_json(
                    config_path, errors, f"static seed {seed} top-{top_k} shard {shard} config"
                )
                if not config:
                    continue
                expected = {
                    "model_id": EXPECTED_MODEL,
                    "start_comic_idx": start,
                    "max_comics": shard_size,
                    "top_k_gaze": top_k,
                    "top_k_random": top_k,
                    "control_mode": "paper",
                    "targets_per_strip": 6,
                    "max_new_tokens": 100,
                    "swap_bias": 10000.0,
                    "decode_only": False,
                    "seed": seed,
                }
                for key, value in expected.items():
                    _require_equal(config, key, value, errors, str(config_path))
                _require_qwen3_path(config.get("comics_root"), errors, str(config_path))
                ranking_value = str(config.get("gaze_ranking", ""))
                expected_ranking = f"gaze_discovery_seed{ranking_seed}_merged/gaze_head_ranking.json"
                if expected_ranking not in ranking_value or "gaze_heads_qwen3_8b" not in ranking_value:
                    errors.append(
                        f"{config_path}: gaze_ranking={ranking_value!r}, expected Qwen3 seed "
                        f"{ranking_seed} merged ranking"
                    )
                if "qwen2.5" in json.dumps(config).lower() or "gaze_heads_qwen25" in json.dumps(config).lower():
                    errors.append(f"{config_path}: contains a Qwen2.5/Qwen25 provenance value")
                shard_configs.append(config)

            merged = runs / f"static_narration_seed{seed}_top{top_k}_merged_0_{expected_comics}"
            rows = _load_jsonl(
                merged / "generations.jsonl", errors, f"static seed {seed} top-{top_k} generations"
            )
            expected_rows = expected_comics * 6 * 2
            if len(rows) != expected_rows:
                errors.append(
                    f"static seed {seed} top-{top_k}: {len(rows)} rows; expected {expected_rows}"
                )
            conditions = Counter(str(row.get("condition")) for row in rows)
            empty = Counter(
                str(row.get("condition"))
                for row in rows
                if not str(row.get("generated_text", "") or "").strip()
            )
            for condition in conditions:
                output_digests[(top_k, condition)][seed] = _rows_digest(rows, condition)

            baselines: dict[str, str] = {}
            for row in rows:
                strip = str(row.get("strip_name", ""))
                baseline = str(row.get("baseline_text", "") or "")
                if strip in baselines and baselines[strip] != baseline:
                    errors.append(
                        f"static seed {seed} top-{top_k}: inconsistent baseline text within strip {strip}"
                    )
                    break
                baselines[strip] = baseline
            baseline_digests[(seed, top_k)] = _digest_json(sorted(baselines.items()))

            summary = _load_json(merged / "summary.json", errors, f"static {merged} summary")
            validation = _load_json(
                merged / "validation.json", errors, f"static {merged} validation"
            )
            control_condition = next(
                (condition for condition in conditions if condition.startswith("non_gaze_")), None
            )
            control_heads = _condition_size(control_condition)
            if top_k == 100 and control_heads != 100:
                warnings.append(
                    f"seed {seed} top-100 used {control_heads} non-gaze heads, not a matched 100-head "
                    "control; paper control_mode selects only the bottom 5% pool"
                )
            static.append(
                {
                    "seed": seed,
                    "top_k": top_k,
                    "merged_dir": str(merged),
                    "rows": len(rows),
                    "shard_configs_checked": len(shard_configs),
                    "conditions": dict(sorted(conditions.items())),
                    "empty_rates": {
                        condition: empty[condition] / count
                        for condition, count in sorted(conditions.items())
                    },
                    "control_heads": control_heads,
                    "validation_valid": validation.get("valid"),
                    "merged_sources": len(summary.get("source_summaries", [])),
                }
            )

    seed_output_comparison = []
    for (top_k, condition), digests in sorted(output_digests.items()):
        seed_output_comparison.append(
            {
                "top_k": top_k,
                "condition": condition,
                "seeds": sorted(digests),
                "all_outputs_exactly_equal": len(set(digests.values())) == 1,
                "digests": {str(seed): digest for seed, digest in sorted(digests.items())},
            }
        )

    warnings.extend(
        [
            "The artifact configs do not record the git commit or Slurm job ID used at generation "
            "time. This audit records the current checkout and artifact hashes, but cannot "
            "retroactively prove the generating commit.",
            "Generation is deterministic and all six targets are used. Gaze-condition outputs are "
            "therefore expected to repeat across seeds when they share the seed-42 ranking; only "
            "random non-gaze head selection can vary.",
            "The completed 72,000 rows are generations, not correctness labels. Do not interpret "
            "the pre-judge valid flag as evidence that static steering worked.",
        ]
    )

    smoke_path = (
        runs
        / f"static_narration_seed{base_seed}_top1_merged_0_{expected_comics}"
        / "kimi_judge_smoke_24_fast"
        / "aggregate_results.json"
    )
    kimi_smoke = _load_json(smoke_path, [], "Kimi smoke") if smoke_path.is_file() else None
    if kimi_smoke is None:
        warnings.append(f"Kimi 24-row smoke result not found at {smoke_path}")

    current_commit = _git(repo_root, "rev-parse", "HEAD")
    current_status = _git(repo_root, "status", "--short").splitlines()
    return {
        "valid": not errors,
        "stage": "artifact_provenance_audit",
        "provenance_valid": not errors,
        "expected_model": EXPECTED_MODEL,
        "current_checkout": {
            "commit": current_commit,
            "status_short": current_status,
            "note": "Current checkout only; not embedded in previously generated artifacts.",
        },
        "configuration": {
            "segment_root": str(segment_root),
            "base_seed": base_seed,
            "seeds": seeds,
            "ranking_seed": ranking_seed,
            "top_ks": top_ks,
            "shards": shards,
            "shard_size": shard_size,
        },
        "discovery": discovery,
        "ranking_overlap": ranking_overlap,
        "static": static,
        "seed_output_comparison": seed_output_comparison,
        "all_baseline_runs_exactly_equal": len(set(baseline_digests.values())) == 1,
        "baseline_digests": {
            f"seed{seed}_top{top_k}": value
            for (seed, top_k), value in sorted(baseline_digests.items())
        },
        "kimi_smoke": (
            {
                "path": str(smoke_path),
                "valid": kimi_smoke.get("valid"),
                "n_rows": kimi_smoke.get("n_rows"),
                "parse_failures": kimi_smoke.get("parse_failures"),
            }
            if kimi_smoke
            else None
        ),
        "errors": errors,
        "warnings": _deduplicate(warnings),
    }


def create_audit_bundle(
    *,
    repo_root: Path,
    segment_root: Path,
    audit_paths: list[Path],
    bundle_path: Path,
    base_seed: int,
    seeds: int,
    ranking_seed: int,
    top_ks: list[int],
    shards: int,
    shard_size: int,
) -> dict[str, Any]:
    expected_comics = shards * shard_size
    runs = segment_root / "runs"
    selected: set[Path] = set(audit_paths)

    for seed in range(base_seed, base_seed + seeds):
        selected.update(_regular_files(runs / f"gaze_discovery_seed{seed}_merged"))
        for shard in range(shards):
            start = shard * shard_size
            selected.add(
                runs / f"gaze_discovery_seed{seed}_{start}_{shard_size}" / "summary.json"
            )
        for top_k in top_ks:
            merged = runs / f"static_narration_seed{seed}_top{top_k}_merged_0_{expected_comics}"
            selected.add(merged / "summary.json")
            selected.add(merged / "validation.json")
            if top_k == 100:
                selected.update(_regular_files(merged))
            for shard in range(shards):
                start = shard * shard_size
                selected.add(
                    runs
                    / f"static_narration_seed{seed}_top{top_k}_{start}_{shard_size}"
                    / "experiment_config.json"
                )

    smoke = (
        runs
        / f"static_narration_seed{base_seed}_top1_merged_0_{expected_comics}"
        / "kimi_judge_smoke_24_fast"
    )
    selected.update(_regular_files(smoke))
    missing = sorted(str(path) for path in selected if not path.is_file())
    if missing:
        raise FileNotFoundError("Cannot build audit bundle; missing selected files: " + ", ".join(missing))

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "w:gz") as archive:
        for path in sorted(selected, key=str):
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(repo_root)
            except ValueError as exc:
                raise ValueError(f"Refusing to archive path outside repository: {path}") from exc
            archive.add(resolved, arcname=str(relative), recursive=False)
    return {
        "path": str(bundle_path),
        "bytes": bundle_path.stat().st_size,
        "sha256": _sha256(bundle_path),
        "files": len(selected),
        "includes": [
            "all three merged discovery runs and every discovery shard summary",
            "all static shard experiment configs plus every merged summary/validation",
            "complete merged top-100 generations for all seeds",
            "the Kimi 24-row smoke result when present",
        ],
        "excludes": ["datasets/images", "model weights", "Slurm logs", "non-top-100 generations"],
    }


def _load_json(
    path: Path, errors: list[str], label: str, *, default: Any = None
) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: cannot read {path}: {exc}")
        return {} if default is None else default


def _load_jsonl(path: Path, errors: list[str], label: str) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: cannot read {path}: {exc}")
        return []


def _require_equal(
    value: dict[str, Any], key: str, expected: Any, errors: list[str], source: str
) -> None:
    if value.get(key) != expected:
        errors.append(f"{source}: {key}={value.get(key)!r}; expected {expected!r}")


def _require_qwen3_path(value: Any, errors: list[str], source: str) -> None:
    text = str(value or "")
    if "gaze_heads_qwen3_8b" not in text or "gaze_heads_qwen25" in text:
        errors.append(f"{source}: expected a Qwen3 segment path, found {text!r}")


def _rows_digest(rows: list[dict[str, Any]], condition: str) -> str:
    values = sorted(
        (
            str(row.get("strip_name", "")),
            int(row.get("target_panel", 0)),
            str(row.get("generated_text", "") or ""),
        )
        for row in rows
        if str(row.get("condition")) == condition
    )
    return _digest_json(values)


def _digest_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _head_id(row: dict[str, Any]) -> str:
    return f"L{int(row['layer'])}H{int(row['head'])}"


def _is_head_row(value: Any) -> bool:
    return isinstance(value, dict) and "layer" in value and "head" in value


def _condition_size(condition: str | None) -> int | None:
    if not condition:
        return None
    try:
        return int(condition.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _regular_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return []
    return (path for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


if __name__ == "__main__":
    main()
