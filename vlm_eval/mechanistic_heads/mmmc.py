from __future__ import annotations

from pathlib import Path
from typing import Any


class MMMCImages:
    """Lazy resolver for `hf://ustc-zhangzm/MMMC/<split>/<index>` images."""

    def __init__(self, cache_dir: Path | None) -> None:
        from datasets import load_dataset

        self.dataset = load_dataset(
            "ustc-zhangzm/MMMC", cache_dir=str(cache_dir) if cache_dir else None
        )

    def resolve(self, reference: str) -> Any:
        prefix = "hf://ustc-zhangzm/MMMC/"
        if not reference.startswith(prefix):
            raise ValueError(f"not an MMMC reference: {reference}")
        split, index = reference[len(prefix) :].split("/", 1)
        return self.dataset[split][int(index)]["image"].convert("RGB")
