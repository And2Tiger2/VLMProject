from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from PIL import Image

from vlm_eval.types import EvalExample


def load_examples(
    source: str,
    split: str = "test",
    limit: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> list[EvalExample]:
    path = Path(source)
    if path.exists():
        examples = list(_load_local(path))
    else:
        examples = list(_load_huggingface(source, split))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(examples)
    if limit is not None:
        examples = examples[:limit]
    return examples


def _load_local(path: Path) -> Iterable[EvalExample]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        rows = data["examples"] if isinstance(data, dict) and "examples" in data else data
    else:
        raise ValueError(f"Unsupported local dataset format: {path}")

    for idx, row in enumerate(rows):
        yield _row_to_example(row, idx=idx, base_dir=path.parent)


def _load_huggingface(dataset_name: str, split: str) -> Iterable[EvalExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependencies with `uv sync` before loading Hugging Face datasets.") from exc

    dataset = load_dataset(dataset_name, split=split)
    for idx, row in enumerate(dataset):
        yield _row_to_example(row, idx=idx)


def _row_to_example(row: dict, idx: int, base_dir: Path | None = None) -> EvalExample:
    image = row.get("image")
    image_path = row.get("image_path")
    resolved_path = None

    if image is None and image_path:
        resolved_path = Path(image_path)
        if base_dir is not None and not resolved_path.is_absolute():
            resolved_path = base_dir / resolved_path
        image = Image.open(resolved_path).convert("RGB")

    metadata = row.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {"raw_metadata": metadata}

    return EvalExample(
        id=str(row.get("ID") or row.get("id") or idx),
        prompt=str(row["prompt"]),
        ground_truth=str(row["ground_truth"]),
        expected_bias=str(row["expected_bias"]) if row.get("expected_bias") is not None else None,
        image=image,
        image_path=resolved_path,
        topic=row.get("topic"),
        sub_topic=row.get("sub_topic"),
        metadata=metadata if isinstance(metadata, dict) else None,
    )

