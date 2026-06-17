from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from adapters.qwen25_vl import _resolve_image, _stable_int
from adapters.qwen25_vl_attention import (
    Qwen25VLAttentionAdapter,
    _language_layers,
    _summarize_attention_records,
)
from vlm_eval.types import EvalExample


QUESTION_START = "<question>"
QUESTION_END = "</question>"
EVIDENCE_START = "<visual_evidence>"
EVIDENCE_END = "</visual_evidence>"


@dataclass
class Qwen25VLTwoPassAttentionAdapter(Qwen25VLAttentionAdapter):
    description_max_new_tokens: int = 80
    answer_include_image: bool = True
    description_prompt_variant: str = "neutral"
    _phase: str = "idle"

    @property
    def eval_config(self) -> dict[str, Any]:
        config = super().eval_config
        config.update(
            {
                "description_max_new_tokens": self.description_max_new_tokens,
                "answer_include_image": self.answer_include_image,
                "description_prompt_variant": self.description_prompt_variant,
            }
        )
        return config

    def generate(self, example: EvalExample) -> str:
        if self.seed is not None:
            self._torch.manual_seed(self.seed + _stable_int(example.id))

        image = _resolve_image(example) if self.include_image else None
        self._phase = "description"
        visual_evidence = self._generate_text(
            _description_prompt(example.prompt, self.description_prompt_variant),
            image,
            prompt_mode="raw",
            max_new_tokens=self.description_max_new_tokens,
        )

        answer_image = image if self.answer_include_image else None
        self._phase = "answer"
        answer_prompt = _answer_prompt(example.prompt, visual_evidence, self.answer_include_image)
        answer = self._generate_text(answer_prompt, answer_image, prompt_mode="raw")
        self._phase = "idle"
        cleaned_answer = _strip_answer_label(answer)

        if self.last_generation_metadata is not None:
            self.last_generation_metadata.update(
                {
                    "description_prompt_variant": self.description_prompt_variant,
                    "answer_include_image": self.answer_include_image,
                    "visual_evidence": visual_evidence,
                }
            )
        return f"Visual evidence: {visual_evidence}\nFinal answer: {cleaned_answer}"

    def _before_generate(self, inputs: Any, *, include_image: bool) -> None:
        self._attention_records.clear()
        self.last_generation_metadata = None

        masks: dict[str, Any] = {}
        image_token_id = getattr(self._model.config, "image_token_id", None)
        if self._phase == "answer" and include_image and image_token_id is not None:
            image_mask = inputs.input_ids[0].eq(image_token_id).detach()
            if bool(image_mask.any()):
                masks["image"] = image_mask

        if self._phase == "answer":
            input_ids = inputs.input_ids[0].detach()
            question_mask = _span_mask(self._processor.tokenizer, input_ids, QUESTION_START, QUESTION_END)
            description_mask = _span_mask(self._processor.tokenizer, input_ids, EVIDENCE_START, EVIDENCE_END)
            if question_mask is not None:
                masks["question"] = question_mask
            if description_mask is not None:
                masks["description"] = description_mask

        for layer in _language_layers(self._model):
            layer.self_attn._vlm_image_token_mask = masks.get("image")
            layer.self_attn._vlm_attention_masks = masks

    def _after_generate(self) -> None:
        if self._phase != "answer":
            self.last_generation_metadata = None
            return

        self.last_generation_metadata = {
            "attention": _summarize_attention_records(self._attention_records, self._boosted_layers),
            "attention_alpha": self.attention_alpha,
            "layer_selection": self.layer_selection,
        }


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
    description_prompt_variant: str = "neutral",
) -> Qwen25VLTwoPassAttentionAdapter:
    adapter = Qwen25VLTwoPassAttentionAdapter(
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
        attention_alpha=0.0,
        layer_selection="baseline",
        description_max_new_tokens=description_max_new_tokens,
        description_prompt_variant=description_prompt_variant,
    )
    suffix = "image-again" if answer_include_image else "description-only"
    adapter.name = f"{adapter.model_id}-two-pass-attention-{description_prompt_variant}-{suffix}"
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

    return (
        f"{QUESTION_START}\n{question}\n{QUESTION_END}\n"
        f"{EVIDENCE_START}\n{visual_evidence}\n{EVIDENCE_END}\n"
        f"{instruction}"
    )


def _span_mask(tokenizer: Any, input_ids: Any, start_marker: str, end_marker: str) -> Any | None:
    start_ids = tokenizer.encode(start_marker, add_special_tokens=False)
    end_ids = tokenizer.encode(end_marker, add_special_tokens=False)
    tokens = input_ids.tolist()
    start = _find_subsequence(tokens, start_ids)
    if start is None:
        return None
    content_start = start + len(start_ids)
    end = _find_subsequence(tokens, end_ids, start=content_start)
    if end is None or end <= content_start:
        return None

    import torch

    mask = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    mask[content_start:end] = True
    return mask


def _find_subsequence(tokens: list[int], pattern: list[int], start: int = 0) -> int | None:
    if not pattern:
        return None
    last = len(tokens) - len(pattern)
    for idx in range(start, last + 1):
        if tokens[idx : idx + len(pattern)] == pattern:
            return idx
    return None


def _strip_answer_label(text: str) -> str:
    match = re.search(r"(?:final\s+answer|answer)\s*[:\-]\s*(.+)", text.strip(), re.I | re.S)
    return match.group(1).strip() if match else text.strip()
