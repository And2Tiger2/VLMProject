from vlm_eval.answer import extract_answer, normalize_answer
from vlm_eval.metrics import score_response, summarize
from vlm_eval.naturalbench import (
    NaturalBenchPrediction,
    extract_naturalbench_answer,
    summarize_naturalbench,
)
from vlm_eval.types import EvalExample
import pytest

from adapters.qwen25_vl_two_pass_unified import answer_prompt_text, description_prompt


def test_extracts_answer_prefix_and_number_words():
    assert extract_answer("Answer: Four.") == "4"
    assert extract_answer("I think there are 5 legs.") == "5"
    assert normalize_answer("Yes!") == "yes"


def test_bias_aligned_error_rate_is_conditioned_on_errors():
    examples = [
        EvalExample(id="1", prompt="p", ground_truth="4", expected_bias="3"),
        EvalExample(id="2", prompt="p", ground_truth="no", expected_bias="yes"),
        EvalExample(id="3", prompt="p", ground_truth="2", expected_bias="1"),
    ]
    predictions = [
        score_response(examples[0], "Answer: 4"),
        score_response(examples[1], "Answer: yes"),
        score_response(examples[2], "Answer: 7"),
    ]
    summary = summarize(predictions)
    assert summary["accuracy"] == 1 / 3
    assert summary["bias_aligned_error_rate"] == 1 / 2


def test_unified_two_pass_prompt_contract():
    question = "Count the stars. Answer with a number in curly brackets, e.g., {9}."
    evidence = "There are three visible stars."

    with pytest.raises(ValueError):
        description_prompt(question, "direct")
    assert "Do not infer what is normally true" in description_prompt(question, "counterfactual")
    assert "Return only this JSON object" in description_prompt(question, "structured")
    assert "common assumption might be misleading" in description_prompt(question, "verification")

    original_image = answer_prompt_text(question, evidence, answer_include_image=True, variant="original")
    original_description_only = answer_prompt_text(
        question,
        evidence,
        answer_include_image=False,
        variant="original",
    )
    assert original_image == original_description_only
    assert "Use the visual evidence, not what is normally true." in original_image
    direct_original = answer_prompt_text(
        question,
        "",
        answer_include_image=True,
        variant="original",
        include_visual_evidence=False,
    )
    assert "Visual evidence:" not in direct_original
    assert "Use the visual evidence, not what is normally true." in direct_original

    image_aware = answer_prompt_text(question, evidence, answer_include_image=True, variant="image_aware")
    description_only = answer_prompt_text(question, evidence, answer_include_image=False, variant="image_aware")
    assert image_aware != description_only
    assert "Use the visual evidence and the image" in image_aware
    assert "Use only the visual evidence above" in description_only


def test_naturalbench_answer_extraction():
    assert extract_naturalbench_answer("Answer: Yes", "yes_no") == "Yes"
    assert extract_naturalbench_answer("The correct option is B.", "multiple_choice") == "B"


def test_naturalbench_group_metrics():
    preds = [
        _nb_pred("g0", "q0_i0", True),
        _nb_pred("g0", "q0_i1", True),
        _nb_pred("g0", "q1_i0", True),
        _nb_pred("g0", "q1_i1", False),
    ]
    summary = summarize_naturalbench(preds)
    assert summary["Acc"] == 3 / 4
    assert summary["Q_Acc"] == 1 / 2
    assert summary["I_Acc"] == 1 / 2
    assert summary["G_Acc"] == 0


def _nb_pred(group_id: str, call_id: str, is_correct: bool) -> NaturalBenchPrediction:
    question_id, image_id = call_id.split("_")
    return NaturalBenchPrediction(
        group_id=group_id,
        call_id=call_id,
        question_id=question_id,
        image_id=image_id,
        question_type="yes_no",
        prompt="p",
        ground_truth="Yes",
        raw_response="Yes" if is_correct else "No",
        parsed_answer="Yes" if is_correct else "No",
        is_correct=is_correct,
    )
