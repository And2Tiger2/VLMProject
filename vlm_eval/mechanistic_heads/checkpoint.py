from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable


class JsonlCheckpoint:
    """Append-only, fsync-backed scientific scan checkpoint."""

    def __init__(self, path: Path, *, key: Callable[[dict[str, Any]], tuple[Any, ...]], resume: bool) -> None:
        self.path = path
        self.key = key
        self.rows: list[dict[str, Any]] = []
        if resume and path.is_file():
            self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif path.exists():
            raise FileExistsError(f"checkpoint exists without --resume: {path}")
        self.completed = {key(row) for row in self.rows}
        path.parent.mkdir(parents=True, exist_ok=True)

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
