from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from adapters.paligemma2 import DEFAULT_MODEL_ID, PaliGemma2Adapter, _blank_image
from adapters.qwen25_vl import _resolve_image, _stable_int
from adapters.qwen25_vl_two_pass_unified import (
    ANSWER_PROMPT_VARIANTS,
    DESCRIPTION_PROMPT_VARIANTS,
    answer_prompt_text,
    description_prompt,
    prompt_manifest,
)
from vlm_eval.types import EvalExample


@dataclass
class PaliGemma2TwoPassUnifiedAdapter(PaliGemma2Adapter):
    description_max_new_tokens: int = 80
    answer_include_image: bool = True
    description_prompt_variant: str = "original"
    answer_prompt_variant: str = "original"
    last_generation_metadata: dict[str, Any] | None = None

    @property
    def eval_config(self) -> dict[str, Any]:
        config = super().eval_config
        config.update(
            {
                "description_max_new_tokens": self.description_max_new_tokens,
                "answer_include_image": self.answer_include_image,
                "answer_no_image_behavior": "blank_image" if not self.answer_include_image else None,
                "description_prompt_variant": self.description_prompt_variant,
                "answer_prompt_variant": self.answer_prompt_variant,
            }
        )
        return config

    def generate(self, example: EvalExample) -> str:
        self.last_generation_metadata = None
        if self.seed is not None:
            self._torch.manual_seed(self.seed + _stable_int(example.id))

        image = _resolve_image(example) if self.include_image else _blank_image(self.blank_image_size)
        description_prompt_value = None
        if self.description_prompt_variant == "direct":
            visual_evidence = ""
        else:
            description_prompt_value = description_prompt(example.prompt, self.description_prompt_variant)
            visual_evidence = self._generate_text(
                description_prompt_value,
                image,
                max_new_tokens=self.description_max_new_tokens,
            )

        answer_image = image if self.answer_include_image else _blank_image(self.blank_image_size)
        answer_prompt = answer_prompt_text(
            example.prompt,
            visual_evidence,
            answer_include_image=self.answer_include_image,
            variant=self.answer_prompt_variant,
            include_visual_evidence=self.description_prompt_variant != "direct",
        )
        raw_answer = self._generate_text(answer_prompt, answer_image)
        final_answer = _strip_answer_label(raw_answer)

        self.last_generation_metadata = {
            "description_prompt_variant": self.description_prompt_variant,
            "answer_prompt_variant": self.answer_prompt_variant,
            "answer_include_image": self.answer_include_image,
            "answer_no_image_behavior": "blank_image" if not self.answer_include_image else None,
            "description_prompt": description_prompt_value,
            "description_was_run": self.description_prompt_variant != "direct",
            "answer_prompt": answer_prompt,
            "visual_evidence": visual_evidence,
            "raw_answer": raw_answer,
            "final_answer": final_answer,
        }
        return f"Visual evidence: {visual_evidence}\nFinal answer: {final_answer}"


def make_adapter(
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 16,
    do_sample: bool = True,
    temperature: float | None = 0.7,
    top_p: float | None = None,
    top_k: int | None = None,
    include_image: bool = True,
    answer_include_image: bool = True,
    seed: int | None = 0,
    device_map: str = "auto",
    prompt_mode: str = "raw",
    blank_image_size: int = 448,
    description_max_new_tokens: int = 80,
    description_prompt_variant: str = "original",
    answer_prompt_variant: str = "original",
) -> PaliGemma2TwoPassUnifiedAdapter:
    adapter = PaliGemma2TwoPassUnifiedAdapter(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        include_image=include_image,
        answer_include_image=answer_include_image,
        seed=seed,
        device_map=device_map,
        prompt_mode=prompt_mode,
        blank_image_size=blank_image_size,
        description_max_new_tokens=description_max_new_tokens,
        description_prompt_variant=description_prompt_variant,
        answer_prompt_variant=answer_prompt_variant,
    )
    suffix = "image-again" if answer_include_image else "blank-image"
    adapter.name = f"{adapter.model_id}-two-pass-{description_prompt_variant}-{answer_prompt_variant}-{suffix}"
    return adapter


def _strip_answer_label(text: str) -> str:
    match = re.search(r"(?:final\s+answer|answer)\s*[:\-]\s*(.+)", text.strip(), re.I | re.S)
    return match.group(1).strip() if match else text.strip()
