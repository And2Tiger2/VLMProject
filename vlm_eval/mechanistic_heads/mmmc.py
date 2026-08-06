from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MMMCImages:
    """Lazy resolver for `hf://ustc-zhangzm/MMMC/<split>/<index>` images."""

    def __init__(self, cache_dir: Path | None, *, audit_path: Path | None = None) -> None:
        from datasets import load_dataset

        self.dataset = load_dataset(
            "ustc-zhangzm/MMMC", cache_dir=str(cache_dir) if cache_dir else None
        )
        self.fingerprints = {
            split: str(rows._fingerprint) for split, rows in self.dataset.items()
        }
        if audit_path is not None:
            if not audit_path.is_file():
                raise RuntimeError(f"MMMC preparation audit is missing: {audit_path}")
            expected = json.loads(audit_path.read_text(encoding="utf-8")).get(
                "dataset_fingerprints"
            )
            if not isinstance(expected, dict) or not expected:
                raise RuntimeError(
                    "MMMC preparation audit has no dataset fingerprints; rerun preparation"
                )
            if expected != self.fingerprints:
                raise RuntimeError(
                    f"MMMC cache fingerprint mismatch: {self.fingerprints} != {expected}"
                )

    def resolve(self, reference: str) -> Any:
        prefix = "hf://ustc-zhangzm/MMMC/"
        if not reference.startswith(prefix):
            raise ValueError(f"not an MMMC reference: {reference}")
        split, index = reference[len(prefix) :].split("/", 1)
        return self.dataset[split][int(index)]["image"].convert("RGB")
