#!/usr/bin/env python3
"""Find or recoverably archive mechanistic outputs from another code revision."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess


def current_git_sha(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()


def stale_output_dirs(repo: Path) -> list[Path]:
    segment = repo / "segments/mechanistic_heads_qwen3_8b"
    current = current_git_sha(repo)
    candidates: set[Path] = set()
    for root_name in ("runs", "reports", "checkpoints"):
        root = segment / root_name
        if not root.is_dir():
            continue
        for manifest_path in root.rglob("run_manifest.json"):
            if "archive" in manifest_path.parts or "slurm" in manifest_path.parts:
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("git_sha") != current:
                candidates.add(manifest_path.parent)
        for checkpoint in root.rglob("*.checkpoint.jsonl"):
            if "archive" in checkpoint.parts or "slurm" in checkpoint.parts:
                continue
            meta_path = checkpoint.with_suffix(checkpoint.suffix + ".meta.json")
            if not meta_path.is_file():
                candidates.add(checkpoint.parent)
                continue
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata.get("git_sha") != current:
                candidates.add(checkpoint.parent)
        for context_path in root.rglob("training_context.json"):
            if "archive" in context_path.parts or "slurm" in context_path.parts:
                continue
            context = json.loads(context_path.read_text(encoding="utf-8"))
            if context.get("git_sha") != current:
                candidates.add(context_path.parent)
    # If a completed aggregate directory is stale, moving it also moves its
    # stale layer children. Keep only the outermost candidate.
    return sorted(
        [path for path in candidates if not any(parent in candidates for parent in path.parents)],
        key=lambda path: str(path),
    )


def archive(repo: Path, paths: list[Path], *, execute: bool) -> Path | None:
    if not paths:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination_root = (
        repo
        / "segments/mechanistic_heads_qwen3_8b/archive"
        / f"stale-{stamp}-{current_git_sha(repo)[:8]}"
    )
    for source in paths:
        relative = source.relative_to(repo / "segments/mechanistic_heads_qwen3_8b")
        destination = destination_root / relative
        print(f"{'archive' if execute else 'would archive'} {source} -> {destination}")
        if execute:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
    return destination_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recoverably archive stale mechanistic result directories before a new run."
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--execute", action="store_true", help="Perform moves; default is a dry run.")
    args = parser.parse_args()
    repo = args.repo.resolve()
    paths = stale_output_dirs(repo)
    destination = archive(repo, paths, execute=args.execute)
    print(
        json.dumps(
            {
                "valid": True,
                "execute": args.execute,
                "n_stale_directories": len(paths),
                "archive_root": str(destination) if destination else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
