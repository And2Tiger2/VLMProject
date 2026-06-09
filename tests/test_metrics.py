from vlm_eval.answer import extract_answer, normalize_answer
from vlm_eval.metrics import score_response, summarize
from vlm_eval.naturalbench import (
    NaturalBenchPrediction,
    extract_naturalbench_answer,
    summarize_naturalbench,
)
from vlm_eval.types import EvalExample


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
