#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run(run_dir: Path, *, repo: Path, newer_than_epoch: float | None) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing run manifest: {manifest_path}")
    if newer_than_epoch is not None and manifest_path.stat().st_mtime < newer_than_epoch:
        raise RuntimeError(f"run manifest was not refreshed by this job: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(
            f"run manifest is not complete ({manifest.get('status')!r}): {manifest_path}"
        )
    current_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if manifest.get("git_sha") != current_sha:
        raise RuntimeError(
            f"run Git SHA {manifest.get('git_sha')} != current checkout {current_sha}"
        )
    output_hashes = manifest.get("output_sha256")
    if not isinstance(output_hashes, dict) or not output_hashes:
        raise RuntimeError(f"run manifest declares no hashed outputs: {manifest_path}")
    bad = []
    input_hashes = manifest.get("input_sha256")
    if not isinstance(input_hashes, dict) or not input_hashes:
        raise RuntimeError(f"run manifest declares no hashed inputs: {manifest_path}")
    for value, expected in input_hashes.items():
        path = Path(value)
        if not path.is_absolute():
            path = repo / path
        if not path.is_file():
            bad.append(f"missing-input:{value}")
        elif sha256_file(path) != expected:
            bad.append(f"input-hash-mismatch:{value}")
    for value, expected in output_hashes.items():
        path = Path(value)
        if not path.is_absolute():
            path = repo / path
        if not path.is_file():
            bad.append(f"missing:{value}")
        elif sha256_file(path) != expected:
            bad.append(f"hash-mismatch:{value}")
        elif path.suffix in {".tsv", ".jsonl"} and not _has_data_record(path):
            bad.append(f"empty-data:{value}")
        elif path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                bad.append(f"invalid-json:{value}")
            else:
                if isinstance(payload, dict) and payload.get("valid") is False:
                    bad.append(f"invalid-report:{value}")
    if bad:
        raise RuntimeError("invalid declared outputs: " + ", ".join(bad))
    return {
        "valid": True,
        "run_dir": str(run_dir),
        "git_sha": current_sha,
        "n_outputs": len(output_hashes),
    }


def _has_data_record(path: Path) -> bool:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".tsv":
        return len(lines) >= 2
    return bool(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a completed mechanistic run contract.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--newer-than-epoch", type=float)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_run(
                args.run_dir,
                repo=args.repo.resolve(),
                newer_than_epoch=args.newer_than_epoch,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
