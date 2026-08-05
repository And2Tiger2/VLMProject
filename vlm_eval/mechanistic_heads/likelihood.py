from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SequenceScore:
    token_log_probabilities: Any
    total_log_probability: Any
    mean_log_probability: Any


def score_answer_from_logits(
    logits: Any,
    answer_token_ids: Any,
    *,
    prompt_length: int,
) -> SequenceScore:
    """Teacher-forced causal score summed over every answer token."""

    import torch

    if answer_token_ids.ndim == 1:
        answer_token_ids = answer_token_ids.unsqueeze(0)
    answer_length = answer_token_ids.shape[1]
    first_logit = prompt_length - 1
    selected_logits = logits[:, first_logit : first_logit + answer_length, :]
    if selected_logits.shape[1] != answer_length:
        raise ValueError("logits do not cover the complete answer sequence")
    log_probs = torch.log_softmax(selected_logits.float(), dim=-1)
    token_scores = log_probs.gather(-1, answer_token_ids.unsqueeze(-1)).squeeze(-1)
    return SequenceScore(
        token_log_probabilities=token_scores,
        total_log_probability=token_scores.sum(dim=-1),
        mean_log_probability=token_scores.mean(dim=-1),
    )


def candidate_sequence_log_likelihood(
    model: Any,
    model_inputs: dict[str, Any],
    answer_token_ids: Any,
) -> SequenceScore:
    """Append answer tokens to one prefill and run exact teacher forcing."""

    import torch

    input_ids = model_inputs["input_ids"]
    if answer_token_ids.ndim == 1:
        answer_token_ids = answer_token_ids.unsqueeze(0)
    if answer_token_ids.shape[0] == 1 and input_ids.shape[0] > 1:
        answer_token_ids = answer_token_ids.expand(input_ids.shape[0], -1)
    if answer_token_ids.shape[0] != input_ids.shape[0]:
        raise ValueError("candidate answer batch does not match model input batch")
    answer_token_ids = answer_token_ids.to(input_ids.device)
    prompt_length = input_ids.shape[1]
    combined = torch.cat([input_ids, answer_token_ids], dim=1)
    kwargs = dict(model_inputs)
    kwargs["input_ids"] = combined
    if "attention_mask" in kwargs:
        extension = kwargs["attention_mask"].new_ones(
            kwargs["attention_mask"].shape[0], answer_token_ids.shape[1]
        )
        kwargs["attention_mask"] = torch.cat([kwargs["attention_mask"], extension], dim=1)
    # Position IDs/cache positions are model-derived for the longer sequence.
    kwargs.pop("position_ids", None)
    kwargs.pop("cache_position", None)
    outputs = model(**kwargs, use_cache=False, return_dict=True)
    return score_answer_from_logits(
        outputs.logits, answer_token_ids, prompt_length=prompt_length
    )
