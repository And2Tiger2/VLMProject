from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


ImageLike = Any


@dataclass(frozen=True)
class EvalExample:
    id: str
    prompt: str
    ground_truth: str
    expected_bias: str | None = None
    image: ImageLike | None = None
    image_path: Path | None = None
    topic: str | None = None
    sub_topic: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class Prediction:
    example_id: str
    prompt: str
    ground_truth: str
    expected_bias: str | None
    raw_response: str
    parsed_answer: str
    is_correct: bool
    is_bias_aligned_error: bool
    topic: str | None = None
    sub_topic: str | None = None
    metadata: dict[str, Any] | None = None


class VLMAdapter(Protocol):
    name: str

    def generate(self, example: EvalExample) -> str:
        """Return a model response for one image-question example."""

