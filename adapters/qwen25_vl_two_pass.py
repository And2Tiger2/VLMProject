from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from adapters.qwen25_vl import Qwen25VLAdapter, _resolve_image, _stable_int
from vlm_eval.types import EvalExample


@dataclass
class Qwen25VLTwoPassAdapter(Qwen25VLAdapter):
    description_max_new_tokens: int = 80
    answer_include_image: bool = True

    @property
    def eval_config(self) -> dict[str, Any]:
        config = super().eval_config
        config.update(
            {
                "description_max_new_tokens": self.description_max_new_tokens,
                "answer_include_image": self.answer_include_image,
            }
        )
        return config

    def generate(self, example: EvalExample) -> str:
        if self.seed is not None:
            self._torch.manual_seed(self.seed + _stable_int(example.id))

        image = _resolve_image(example) if self.include_image else None
        description_prompt = (
            "Describe only the visual evidence needed to answer this question. "
            "Do not answer the question.\n"
            f"Question: {example.prompt}"
        )
        visual_evidence = self._generate_text(
            description_prompt,
            image,
            prompt_mode="raw",
            max_new_tokens=self.description_max_new_tokens,
        )

        answer_image = image if self.answer_include_image else None
        answer_prompt = (
            f"{example.prompt}\n"
            f"Visual evidence: {visual_evidence}\n"
            "Use the visual evidence, not what is normally true. "
            "Write 'Final answer:' followed by the answer."
        )
        answer = self._generate_text(answer_prompt, answer_image, prompt_mode="raw")
        return f"Visual evidence: {visual_evidence}\nFinal answer: {_strip_answer_label(answer)}"


def make_adapter(
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    max_new_tokens: int = 16,
    max_pixels: int = 1048576,
    min_pixels: int = 0,
    do_sample: bool = True,
    temperature: float | None = 0.7,
    top_p: float | None = None,
    top_k: int | None = None,
    include_image: bool = True,
    answer_include_image: bool = True,
    seed: int | None = 0,
    device_map: str = "auto",
    description_max_new_tokens: int = 80,
) -> Qwen25VLTwoPassAdapter:
    adapter = Qwen25VLTwoPassAdapter(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        include_image=include_image,
        answer_include_image=answer_include_image,
        seed=seed,
        device_map=device_map,
        prompt_mode="raw",
        description_max_new_tokens=description_max_new_tokens,
    )
    adapter.name = f"{adapter.model_id}-two-pass"
    return adapter


def _strip_answer_label(text: str) -> str:
    match = re.search(r"(?:final\s+answer|answer)\s*[:\-]\s*(.+)", text.strip(), re.I | re.S)
    return match.group(1).strip() if match else text.strip()
