from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid half-open token span [{self.start}, {self.end})")

    def indices(self) -> list[int]:
        return list(range(self.start, self.end))


@dataclass(frozen=True)
class TokenSpans:
    system: TokenSpan
    visual: TokenSpan
    image_end: TokenSpan
    user_prompt: TokenSpan
    final_prompt: TokenSpan
    generated_answer: TokenSpan
    sequence_length: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_partition_bounds(self) -> None:
        for span in (
            self.system,
            self.visual,
            self.image_end,
            self.user_prompt,
            self.final_prompt,
            self.generated_answer,
        ):
            if span.end > self.sequence_length:
                raise AssertionError(f"span {span} exceeds sequence length")
        if self.visual.end > self.image_end.start:
            raise AssertionError("visual span overlaps image-end token")
        if self.user_prompt.end > self.final_prompt.start:
            raise AssertionError("user prompt overlaps final assistant/prompt token")
        if self.final_prompt.end > self.generated_answer.start:
            raise AssertionError("prefill overlaps generated answer")


def contiguous_span(indices: Any, *, name: str) -> TokenSpan:
    values = [int(value) for value in indices]
    if not values:
        raise ValueError(f"no {name} tokens found")
    expected = list(range(values[0], values[-1] + 1))
    if values != expected:
        raise ValueError(f"{name} tokens are not one contiguous span")
    return TokenSpan(values[0], values[-1] + 1)


def trace_qwen3_token_spans(
    input_ids: Any,
    *,
    image_token_id: int,
    image_end_token_id: int,
    system_end: int,
    user_start: int,
    final_prompt_start: int,
    prompt_length: int | None = None,
    answer_length: int = 0,
) -> TokenSpans:
    """Trace asserted Qwen spans using template-derived text boundaries."""

    ids = [int(value) for value in input_ids]
    prompt_length = len(ids) if prompt_length is None else int(prompt_length)
    visual_positions = [idx for idx, token in enumerate(ids[:prompt_length]) if token == image_token_id]
    visual = contiguous_span(visual_positions, name="visual")
    ends = [
        idx
        for idx, token in enumerate(ids[:prompt_length])
        if token == image_end_token_id and idx >= visual.end
    ]
    if not ends:
        raise ValueError("no image-end token found after visual span")
    spans = TokenSpans(
        system=TokenSpan(0, system_end),
        visual=visual,
        image_end=TokenSpan(ends[0], ends[0] + 1),
        user_prompt=TokenSpan(user_start, final_prompt_start),
        final_prompt=TokenSpan(final_prompt_start, prompt_length),
        generated_answer=TokenSpan(prompt_length, prompt_length + answer_length),
        sequence_length=prompt_length + answer_length,
    )
    spans.assert_partition_bounds()
    return spans


def locate_subsequence(sequence: list[int], subsequence: list[int]) -> tuple[int, int]:
    if not subsequence:
        raise ValueError("cannot locate an empty token subsequence")
    matches = [
        start for start in range(len(sequence) - len(subsequence) + 1)
        if sequence[start : start + len(subsequence)] == subsequence
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one prompt-token match, found {len(matches)}")
    return matches[0], matches[0] + len(subsequence)
