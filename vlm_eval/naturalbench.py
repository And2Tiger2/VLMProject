from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from vlm_eval.overheat import maybe_pause
from vlm_eval.types import EvalExample, VLMAdapter


SUFFIX_FOR_VQA = {
    "yes_no": "Please answer Yes or No.",
    "multiple_choice": "Please output only the letter corresponding to the correct option: A or B.",
}


@dataclass(frozen=True)
class NaturalBenchCall:
    group_id: str
    call_id: str
    question_id: str
    image_id: str
    question_type: str
    prompt: str
    ground_truth: str
    image_path: Path
    source: str | None = None


@dataclass(frozen=True)
class NaturalBenchPrediction:
    group_id: str
    call_id: str
    question_id: str
    image_id: str
    question_type: str
    prompt: str
    ground_truth: str
    raw_response: str
    parsed_answer: str
    is_correct: bool
    source: str | None = None


def load_naturalbench_calls(source: str, limit_groups: int | None = None) -> list[NaturalBenchCall]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(
            f"NaturalBench evaluator expects a local slice JSONL. Create one with "
            f"`uv run python scripts/make_naturalbench_slice.py --out {source}`."
        )

    groups = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                groups.append(json.loads(line))

    if limit_groups is not None:
        groups = groups[:limit_groups]

    calls = []
    for group in groups:
        calls.extend(_group_to_calls(group, path.parent))
    return calls


def evaluate_naturalbench(calls: Iterable[NaturalBenchCall], adapter: VLMAdapter) -> Iterable[NaturalBenchPrediction]:
    for call in calls:
        example = EvalExample(
            id=call.call_id,
            prompt=call.prompt,
            ground_truth=call.ground_truth,
            image=Image.open(call.image_path).convert("RGB"),
            image_path=call.image_path,
            topic="NaturalBench",
            sub_topic=call.question_type,
            metadata={
                "group_id": call.group_id,
                "question_id": call.question_id,
                "image_id": call.image_id,
                "source": call.source,
            },
        )
        maybe_pause()
        raw_response = adapter.generate(example)
        maybe_pause()
        parsed_answer = extract_naturalbench_answer(raw_response, call.question_type)
        yield NaturalBenchPrediction(
            group_id=call.group_id,
            call_id=call.call_id,
            question_id=call.question_id,
            image_id=call.image_id,
            question_type=call.question_type,
            prompt=call.prompt,
            ground_truth=call.ground_truth,
            raw_response=raw_response,
            parsed_answer=parsed_answer,
            is_correct=normalize_naturalbench_answer(parsed_answer) == normalize_naturalbench_answer(call.ground_truth),
            source=call.source,
        )


def summarize_naturalbench(predictions: Iterable[NaturalBenchPrediction]) -> dict:
    preds = list(predictions)
    groups: dict[str, dict[str, NaturalBenchPrediction]] = {}
    for pred in preds:
        groups.setdefault(pred.group_id, {})[pred.call_id] = pred

    complete_groups = {
        group_id: group_preds
        for group_id, group_preds in groups.items()
        if {"q0_i0", "q0_i1", "q1_i0", "q1_i1"} <= set(group_preds)
    }

    call_count = len(preds)
    correct_calls = sum(pred.is_correct for pred in preds)
    group_count = len(complete_groups)
    q_correct = 0
    i_correct = 0
    g_correct = 0

    for group_preds in complete_groups.values():
        q0_ok = group_preds["q0_i0"].is_correct and group_preds["q0_i1"].is_correct
        q1_ok = group_preds["q1_i0"].is_correct and group_preds["q1_i1"].is_correct
        i0_ok = group_preds["q0_i0"].is_correct and group_preds["q1_i0"].is_correct
        i1_ok = group_preds["q0_i1"].is_correct and group_preds["q1_i1"].is_correct
        q_correct += int(q0_ok) + int(q1_ok)
        i_correct += int(i0_ok) + int(i1_ok)
        g_correct += int(q0_ok and q1_ok and i0_ok and i1_ok)

    return {
        "n_groups": group_count,
        "n_model_calls": call_count,
        "Acc": correct_calls / call_count if call_count else 0.0,
        "Q_Acc": q_correct / (2 * group_count) if group_count else 0.0,
        "I_Acc": i_correct / (2 * group_count) if group_count else 0.0,
        "G_Acc": g_correct / group_count if group_count else 0.0,
        "by_question_type": _summarize_by_key(preds, "question_type"),
        "by_source": _summarize_by_key(preds, "source"),
    }


def prediction_to_dict(prediction: NaturalBenchPrediction) -> dict:
    return asdict(prediction)


def extract_naturalbench_answer(response: str, question_type: str) -> str:
    if question_type == "yes_no":
        yes_pos = _word_position(response, "yes")
        no_pos = _word_position(response, "no")
        if yes_pos == -1 and no_pos == -1:
            return ""
        if yes_pos != -1 and no_pos != -1:
            return "Yes" if yes_pos < no_pos else "No"
        return "No" if yes_pos == -1 else "Yes"

    if question_type == "multiple_choice":
        a_pos = _word_position(response, "A")
        b_pos = _word_position(response, "B")
        if a_pos == -1 and b_pos == -1:
            return ""
        if a_pos != -1 and b_pos != -1:
            return "A" if a_pos < b_pos else "B"
        return "B" if a_pos == -1 else "A"

    raise ValueError(f"Unsupported NaturalBench question type: {question_type}")


def normalize_naturalbench_answer(answer: str) -> str:
    value = answer.strip().lower()
    if value in {"yes", "y"}:
        return "yes"
    if value in {"no", "n"}:
        return "no"
    if value in {"a", "option a"}:
        return "a"
    if value in {"b", "option b"}:
        return "b"
    return value


def _group_to_calls(group: dict, base_dir: Path) -> list[NaturalBenchCall]:
    question_type = group["question_type"]
    suffix = SUFFIX_FOR_VQA[question_type]
    image_paths = {
        "i0": base_dir / group["image_0_path"],
        "i1": base_dir / group["image_1_path"],
    }
    questions = {
        "q0": group["question_0"],
        "q1": group["question_1"],
    }
    answers = group["answers"]

    calls = []
    for question_id in ("q0", "q1"):
        for image_id in ("i0", "i1"):
            call_id = f"{question_id}_{image_id}"
            calls.append(
                NaturalBenchCall(
                    group_id=str(group["id"]),
                    call_id=call_id,
                    question_id=question_id,
                    image_id=image_id,
                    question_type=question_type,
                    prompt=f"{questions[question_id]}\n{suffix}",
                    ground_truth=answers[call_id],
                    image_path=image_paths[image_id],
                    source=group.get("source"),
                )
            )
    return calls


def _word_position(string: str, word: str) -> int:
    match = re.search(r"\b" + re.escape(word) + r"\b", string, re.IGNORECASE)
    return match.start() if match else -1


def _summarize_by_key(predictions: list[NaturalBenchPrediction], key: str) -> dict:
    groups: dict[str, list[NaturalBenchPrediction]] = {}
    for pred in predictions:
        value = getattr(pred, key) or "unknown"
        groups.setdefault(value, []).append(pred)

    return {
        value: {
            "n_model_calls": len(group_preds),
            "Acc": sum(pred.is_correct for pred in group_preds) / len(group_preds),
        }
        for value, group_preds in sorted(groups.items())
    }
