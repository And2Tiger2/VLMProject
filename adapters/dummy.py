from __future__ import annotations

from dataclasses import dataclass

from vlm_eval.types import EvalExample


@dataclass
class DummyAdapter:
    """Deterministic adapter for smoke-testing metrics and file outputs."""

    mode: str = "bias"
    name: str = "dummy"

    def generate(self, example: EvalExample) -> str:
        if self.mode == "truth":
            answer = example.ground_truth
        elif self.mode == "empty":
            answer = ""
        else:
            answer = example.expected_bias or example.ground_truth
        return f"Answer: {answer}"


def make_adapter(mode: str = "bias") -> DummyAdapter:
    return DummyAdapter(mode=mode, name=f"dummy-{mode}")

