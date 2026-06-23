from adapters.paligemma2 import _blank_image, format_paligemma_prompt


def test_format_paligemma_prompt_baseline() -> None:
    prompt = format_paligemma_prompt("What color is the object?", "baseline")

    assert prompt == "What color is the object?\nAnswer with only the final answer."


def test_format_paligemma_prompt_counterfactual_warning() -> None:
    prompt = format_paligemma_prompt("What is shown?", "counterfactual_warning")

    assert "counterfactual" in prompt
    assert "Do not rely on what is normally true" in prompt
    assert prompt.endswith("Answer with only the final answer.")


def test_format_paligemma_prompt_raw() -> None:
    assert format_paligemma_prompt("What is shown?", "raw") == "What is shown?"


def test_blank_image_control_shape() -> None:
    image = _blank_image(448)

    assert image.mode == "RGB"
    assert image.size == (448, 448)
