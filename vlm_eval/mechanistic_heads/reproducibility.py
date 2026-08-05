from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_paths(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in sorted(paths) if path.is_file()}


def git_sha(cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_snapshot() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in ("torch", "transformers", "datasets", "PIL", "numpy"):
        try:
            module = __import__(package)
            packages[package] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            packages[package] = "not-installed"
    cuda: dict[str, Any] = {"available": False}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "runtime": str(torch.version.cuda),
            "device_count": int(torch.cuda.device_count()),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURMD_NODENAME",
            )
        },
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_run_manifest(
    output_dir: Path,
    *,
    config: dict[str, Any],
    seeds: dict[str, int],
    inputs: Iterable[Path],
    outputs: Iterable[Path] = (),
    status: str = "started",
    repo_root: Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(repo_root),
        "config": config,
        "seeds": seeds,
        "environment": environment_snapshot(),
        "input_sha256": hash_paths(inputs),
        "output_sha256": hash_paths(outputs),
        "resume_marker": str(output_dir / "resume.marker.json"),
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "resume.marker.json").write_text(
        json.dumps(
            {
                "status": status,
                "updated_at_utc": manifest["timestamp_utc"],
                "git_sha": manifest["git_sha"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
