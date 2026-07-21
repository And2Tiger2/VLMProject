from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_completed_keys(path: Path, fields: Iterable[str]) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    field_list = list(fields)
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed.add(tuple(row.get(field) for field in field_list))
    return completed


def row_key(row: dict[str, Any], fields: Iterable[str]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in fields)


def ensure_resume_config(
    out_dir: Path,
    config: dict[str, Any],
    *,
    resume: bool,
    artifact_name: str,
    config_name: str = "experiment_config.json",
) -> Path:
    """Prevent resume from mixing rows produced by different experiments."""
    config_path = out_dir / config_name
    artifact_path = out_dir / artifact_name
    if resume and artifact_path.exists():
        if not config_path.exists():
            raise RuntimeError(
                f"Cannot safely resume {artifact_path}: {config_name} is missing. "
                "Use a new output directory for the corrected run."
            )
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            changed = sorted(
                key for key in set(existing) | set(config) if existing.get(key) != config.get(key)
            )
            raise RuntimeError(
                f"Cannot resume {artifact_path} with a different experiment configuration. "
                f"Changed fields: {changed}. Use a new output directory."
            )
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return config_path
