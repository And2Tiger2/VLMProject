from __future__ import annotations

import json
from pathlib import Path


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
    return report


def validation_path_from_config(config: dict) -> Path:
    return Path(
        config.get(
            "instrumentation_validation",
            "segments/mechanistic_heads_qwen3_8b/reports/instrumentation/instrumentation_validation.json",
        )
    )
