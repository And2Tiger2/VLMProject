from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from adapters.qwen25_vl import Qwen25VLAdapter, _resolve_image, _stable_int
from vlm_eval.types import EvalExample


@dataclass
class Qwen25VLTwoPassVariantAdapter(Qwen25VLAdapter):
    description_max_new_tokens: int = 80
    answer_include_image: bool = True
    description_prompt_variant: str = "neutral"
    last_generation_metadata: dict[str, Any] | None = None

    @property
    def eval_config(self) -> dict[str, Any]:
        config = super().eval_config
        config.update(
            {
                "description_max_new_tokens": self.description_max_new_tokens,
                "answer_include_image": self.answer_include_image,
                "description_prompt_variant": self.description_prompt_variant,
                "fast_mode": True,
            }
        )
        return config

    def generate(self, example: EvalExample) -> str:
        self.last_generation_metadata = None
        if self.seed is not None:
            self._torch.manual_seed(self.seed + _stable_int(example.id))

        image = _resolve_image(example) if self.include_image else None
        visual_evidence = self._generate_text(
            _description_prompt(example.prompt, self.description_prompt_variant),
            image,
            prompt_mode="raw",
            max_new_tokens=self.description_max_new_tokens,
        )

        answer_image = image if self.answer_include_image else None
        answer = self._generate_text(
            _answer_prompt(example.prompt, visual_evidence, self.answer_include_image),
            answer_image,
            prompt_mode="raw",
        )
        cleaned_answer = _strip_answer_label(answer)
        self.last_generation_metadata = {
            "description_prompt_variant": self.description_prompt_variant,
            "answer_include_image": self.answer_include_image,
            "visual_evidence": visual_evidence,
            "fast_mode": True,
        }
        return f"Visual evidence: {visual_evidence}\nFinal answer: {cleaned_answer}"


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
    prompt_mode: str = "raw",
    description_max_new_tokens: int = 80,
    description_prompt_variant: str = "neutral",
) -> Qwen25VLTwoPassVariantAdapter:
    adapter = Qwen25VLTwoPassVariantAdapter(
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
        prompt_mode=prompt_mode,
        description_max_new_tokens=description_max_new_tokens,
        description_prompt_variant=description_prompt_variant,
    )
    suffix = "image-again" if answer_include_image else "description-only"
    adapter.name = f"{adapter.model_id}-two-pass-fast-{description_prompt_variant}-{suffix}"
    return adapter


def _description_prompt(question: str, variant: str) -> str:
    if variant == "neutral":
        prefix = "Describe only the visual evidence needed to answer this question. Do not answer the question."
    elif variant == "counterfactual":
        prefix = (
            "The image may contradict common assumptions. Describe the visible evidence needed to answer this "
            "question. Do not infer what is normally true. Do not answer the question."
        )
    elif variant == "structured":
        prefix = (
            'Return only this JSON object: {"visible_evidence": "...", '
            '"relevant_count_or_attribute": "...", "uncertainty": "low|medium|high"}. '
            "Do not answer the question."
        )
    elif variant == "verification":
        prefix = (
            "Describe the visual evidence needed to answer the question. Then state what common assumption might "
            "be misleading. Do not give the final answer."
        )
    else:
        raise ValueError(f"Unknown description_prompt_variant: {variant}")
    return f"{prefix}\nQuestion: {question}"


def _answer_prompt(question: str, visual_evidence: str, answer_include_image: bool) -> str:
    if answer_include_image:
        instruction = (
            "Use the visual evidence and the image to answer. The image may contradict common assumptions. "
            "Write 'Final answer:' followed by the answer."
        )
    else:
        instruction = (
            "Use only the visual evidence above. Do not rely on what is normally true. "
            "Write 'Final answer:' followed by the answer."
        )
    return f"Question: {question}\nVisual evidence: {visual_evidence}\n{instruction}"


def _strip_answer_label(text: str) -> str:
    match = re.search(r"(?:final\s+answer|answer)\s*[:\-]\s*(.+)", text.strip(), re.I | re.S)
    return match.group(1).strip() if match else text.strip()
