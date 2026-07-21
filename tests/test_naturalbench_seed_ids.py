from __future__ import annotations

from pathlib import Path

from PIL import Image

from vlm_eval.naturalbench import NaturalBenchCall, evaluate_naturalbench


class _RecordingAdapter:
    def __init__(self) -> None:
        self.ids: list[str] = []

    def generate(self, example) -> str:
        self.ids.append(example.id)
        return "Yes"


def test_adapter_seed_ids_include_naturalbench_group(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), "white").save(image_path)
    calls = [
        NaturalBenchCall(
            group_id=group_id,
            call_id="q0_i0",
            question_id="q0",
            image_id="i0",
            question_type="yes_no",
            prompt="Is it white?",
            ground_truth="Yes",
            image_path=image_path,
        )
        for group_id in ("group-a", "group-b")
    ]
    adapter = _RecordingAdapter()

    list(evaluate_naturalbench(calls, adapter))

    assert adapter.ids == ["group-a_q0_i0", "group-b_q0_i0"]
