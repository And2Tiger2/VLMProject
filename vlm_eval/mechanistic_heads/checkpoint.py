from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

from vlm_eval.mechanistic_heads.reproducibility import git_sha


class JsonlCheckpoint:
    """Append-only, fsync-backed scientific scan checkpoint."""

    def __init__(self, path: Path, *, key: Callable[[dict[str, Any]], tuple[Any, ...]], resume: bool, context: dict[str, Any] | None = None) -> None:
        self.path = path
        self.key = key
        self.meta_path = path.with_suffix(path.suffix + ".meta.json")
        self.metadata = {"schema_version": 1, "git_sha": git_sha(Path.cwd()), "context": context or {}}
        self.rows: list[dict[str, Any]] = []
        if resume and path.is_file():
            if not self.meta_path.is_file():
                raise RuntimeError(f"checkpoint metadata is missing: {self.meta_path}")
            observed = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if observed != self.metadata:
                raise RuntimeError(
                    f"checkpoint context does not match this run: {self.meta_path}"
                )
            self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif path.exists():
            raise FileExistsError(f"checkpoint exists without --resume: {path}")
        elif resume and self.meta_path.exists():
            raise RuntimeError(f"checkpoint metadata exists without checkpoint rows: {self.meta_path}")
        self.completed = {key(row) for row in self.rows}
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            self.meta_path.write_text(json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8")
            path.touch()

    def missing(self, key: tuple[Any, ...]) -> bool:
        return key not in self.completed

    def append(self, rows: Iterable[dict[str, Any]]) -> None:
        fresh = []
        pending = set(self.completed)
        for row in rows:
            row_key = self.key(row)
            if row_key not in pending:
                fresh.append(row)
                pending.add(row_key)
        if not fresh:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            for row in fresh:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                self.completed.add(self.key(row))
            handle.flush()
            os.fsync(handle.fileno())
        self.rows.extend(fresh)
