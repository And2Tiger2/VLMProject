from __future__ import annotations

import json
from pathlib import Path

from vlm_eval.mechanistic_heads.reproducibility import git_sha, sha256_file


REQUIRED_CHECKS = (
    "identity_patch",
    "projected_head_reconstruction",
    "self_subtraction_noop",
    "batched_serial_agreement",
    "teacher_forcing_likelihood",
    "token_spans",
    "attention_normalization",
    "generator_determinism",
    "split_leakage",
    "backend_equivalence",
    "cached_uncached_equivalence",
    "reproducibility_manifest",
)


def require_scientific_validation(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(
            f"scientific run refused: instrumentation validation is missing: {path}"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = report.get("checks", {})
    failed = [name for name in REQUIRED_CHECKS if checks.get(name) is not True]
    if not report.get("valid") or failed:
        raise RuntimeError(
            "scientific run refused: required instrumentation checks failed or are missing: "
            + ", ".join(failed)
        )
    try:
        require_completed_manifest(
            path.parent, expected_outputs=(path,), require_current_git=True
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "scientific run refused: instrumentation was not validated for the current Git SHA "
            f"and exact report bytes: {exc}"
        ) from exc
    return report


def validation_path_from_config(config: dict) -> Path:
    return Path(
        config.get(
            "instrumentation_validation",
            "segments/mechanistic_heads_qwen3_8b/reports/instrumentation/instrumentation_validation.json",
        )
    )


def require_completed_manifest(
    run_dir: Path, *, expected_outputs: tuple[Path, ...] = (), require_current_git: bool = False
) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"completed-run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"run is not complete: {manifest_path}")
    if require_current_git and manifest.get("git_sha") != git_sha(Path.cwd()):
        raise RuntimeError(f"run was not produced by the current Git SHA: {manifest_path}")
    hashes = manifest.get("output_sha256", {})
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError(f"run manifest has no output hashes: {manifest_path}")
    normalized = {str(Path(key).resolve()): value for key, value in hashes.items()}
    for value, expected in hashes.items():
        artifact = Path(value).resolve()
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise RuntimeError(f"completed-run output is missing or changed: {value}")
    for value, expected in manifest.get("input_sha256", {}).items():
        source = Path(value).resolve()
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"prepared-run input is missing or changed: {value}")
    for output in expected_outputs:
        resolved = output.resolve()
        expected = normalized.get(str(resolved))
        if expected is None:
            raise RuntimeError(f"required artifact is not hashed by {manifest_path}: {output}")
        if not resolved.is_file() or sha256_file(resolved) != expected:
            raise RuntimeError(f"required artifact is missing or modified: {output}")
    return manifest


def require_calibration_report(path: Path, *, boolean_key: str | None = None) -> dict:
    if not path.is_file():
        raise RuntimeError(f"required calibration report is missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("valid") is not True:
        raise RuntimeError(f"required calibration report is invalid: {path}")
    passed = (
        report.get(boolean_key) is True
        if boolean_key is not None
        else report.get("calibration_result") == "passed"
    )
    if not passed:
        detail = boolean_key or "calibration_result"
        raise RuntimeError(f"required calibration gate {detail} did not pass: {path}")
    require_completed_manifest(
        path.parent, expected_outputs=(path,), require_current_git=True
    )
    return report
