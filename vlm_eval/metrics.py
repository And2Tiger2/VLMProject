from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Iterable

from vlm_eval.answer import extract_answer, normalize_answer
from vlm_eval.types import EvalExample, Prediction


def score_response(example: EvalExample, raw_response: str) -> Prediction:
    parsed = extract_answer(raw_response)
    truth = normalize_answer(example.ground_truth)
    expected_bias = normalize_answer(example.expected_bias) if example.expected_bias is not None else None
    is_correct = parsed == truth
    is_bias_aligned_error = bool(expected_bias) and not is_correct and parsed == expected_bias

    return Prediction(
        example_id=example.id,
        prompt=example.prompt,
        ground_truth=example.ground_truth,
        expected_bias=example.expected_bias,
        raw_response=raw_response,
        parsed_answer=parsed,
        is_correct=is_correct,
        is_bias_aligned_error=is_bias_aligned_error,
        topic=example.topic,
        sub_topic=example.sub_topic,
        metadata=example.metadata,
    )


def summarize(predictions: Iterable[Prediction]) -> dict:
    preds = list(predictions)
    total = len(preds)
    correct = sum(pred.is_correct for pred in preds)
    errors = total - correct
    bias_errors = sum(pred.is_bias_aligned_error for pred in preds)

    summary = {
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "error_rate": errors / total if total else 0.0,
        "bias_aligned_error_rate": bias_errors / errors if errors else 0.0,
        "bias_aligned_fraction": bias_errors / total if total else 0.0,
    }

    by_topic = defaultdict(list)
    for pred in preds:
        by_topic[pred.topic or "unknown"].append(pred)
    summary["by_topic"] = {
        topic: {
            "n": len(topic_preds),
            "accuracy": sum(p.is_correct for p in topic_preds) / len(topic_preds),
            "bias_aligned_error_rate": (
                sum(p.is_bias_aligned_error for p in topic_preds)
                / max(1, sum(not p.is_correct for p in topic_preds))
            ),
        }
        for topic, topic_preds in sorted(by_topic.items())
    }
    return summary


def prediction_to_dict(prediction: Prediction) -> dict:
    return asdict(prediction)

