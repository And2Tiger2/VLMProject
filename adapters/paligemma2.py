from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from adapters.qwen25_vl import _resolve_image, _stable_int
from vlm_eval.types import EvalExample


DEFAULT_MODEL_ID = "google/paligemma2-3b-mix-448"


@dataclass
class PaliGemma2Adapter:
    model_id: str = DEFAULT_MODEL_ID
    max_new_tokens: int = 16
    do_sample: bool = True
    temperature: float | None = 0.7
    top_p: float | None = None
    top_k: int | None = None
    include_image: bool = True
    seed: int | None = 0
    device_map: str = "auto"
    prompt_mode: str = "baseline"
    blank_image_size: int = 448
    name: str = "paligemma2"

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
        except ImportError as exc:
            raise RuntimeError("Install optional PaliGemma dependencies with `uv sync --extra paligemma`.") from exc

        self._torch = torch
        self._processor = PaliGemmaProcessor.from_pretrained(self.model_id)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=_resolve_device_map(self.device_map, torch),
        )
        self.name = self.model_id if self.include_image else f"{self.model_id}-blank-image"

    @property
    def eval_config(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "include_image": self.include_image,
            "seed": self.seed,
            "device_map": self.device_map,
            "prompt_mode": self.prompt_mode,
            "blank_image_size": self.blank_image_size,
            "no_image_behavior": "blank_image" if not self.include_image else None,
        }

    def generate(self, example: EvalExample) -> str:
        if self.seed is not None:
            self._torch.manual_seed(self.seed + _stable_int(example.id))

        image = _resolve_image(example) if self.include_image else _blank_image(self.blank_image_size)
        prompt = format_paligemma_prompt(example.prompt, self.prompt_mode)
        return self._generate_text(prompt, image)

    def _generate_text(self, prompt: str, image: Any, max_new_tokens: int | None = None) -> str:
        inputs = self._processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)
        input_length = inputs["input_ids"].shape[-1]
        generated = self._model.generate(**inputs, **self._generation_kwargs(max_new_tokens=max_new_tokens))
        decoded = self._processor.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()

    def _generation_kwargs(self, max_new_tokens: int | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        return kwargs


def format_paligemma_prompt(prompt: str, prompt_mode: str) -> str:
    if prompt_mode == "baseline":
        suffix = "Answer with only the final answer."
    elif prompt_mode == "visual_first":
        suffix = "Answer based only on the visible image. Answer with only the final answer."
    elif prompt_mode == "counterfactual_warning":
        suffix = (
            "The image may be unusual or counterfactual. Do not rely on what is normally true. "
            "Answer based only on the visible image. Answer with only the final answer."
        )
    elif prompt_mode == "raw":
        suffix = ""
    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    if not suffix:
        return prompt
    return f"{prompt}\n{suffix}"


def make_adapter(
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 16,
    do_sample: bool = True,
    temperature: float | None = 0.7,
    top_p: float | None = None,
    top_k: int | None = None,
    include_image: bool = True,
    seed: int | None = 0,
    device_map: str = "auto",
    prompt_mode: str = "baseline",
    blank_image_size: int = 448,
) -> PaliGemma2Adapter:
    return PaliGemma2Adapter(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        include_image=include_image,
        seed=seed,
        device_map=device_map,
        prompt_mode=prompt_mode,
        blank_image_size=blank_image_size,
    )


def _blank_image(size: int) -> Image.Image:
    return Image.new("RGB", (size, size), "white")


def _resolve_device_map(device_map: str, torch: Any) -> Any:
    if device_map in {"cuda", "cuda:0"}:
        if not torch.cuda.is_available():
            raise RuntimeError("device_map='cuda' requested, but torch.cuda.is_available() is false.")
        return {"": "cuda:0"}
    return device_map
