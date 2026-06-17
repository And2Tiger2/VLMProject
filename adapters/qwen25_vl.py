from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vlm_eval.types import EvalExample


@dataclass
class Qwen25VLAdapter:
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_new_tokens: int = 16
    max_pixels: int = 1048576
    min_pixels: int = 0
    do_sample: bool = True
    temperature: float | None = 0.7
    top_p: float | None = None
    top_k: int | None = None
    include_image: bool = True
    seed: int | None = 0
    device_map: str = "auto"
    prompt_mode: str = "baseline"
    name: str = "qwen2.5-vl"

    def __post_init__(self) -> None:
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Install optional Qwen dependencies with `uv sync --extra qwen`.") from exc

        self._torch = torch
        self._process_vision_info = process_vision_info
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device_map,
        )
        self.name = self.model_id

    @property
    def eval_config(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "max_new_tokens": self.max_new_tokens,
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_pixels,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "include_image": self.include_image,
            "seed": self.seed,
            "device_map": self.device_map,
            "prompt_mode": self.prompt_mode,
        }

    def generate(self, example: EvalExample) -> str:
        if self.seed is not None:
            self._torch.manual_seed(self.seed + _stable_int(example.id))
        image = _resolve_image(example) if self.include_image else None
        return self._generate_text(example.prompt, image)

    def _generate_text(
        self,
        prompt: str,
        image: Any | None,
        *,
        prompt_mode: str | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        include_image = image is not None
        messages = [
            {
                "role": "user",
                "content": _message_content(
                    image,
                    prompt,
                    self.max_pixels,
                    self.min_pixels,
                    include_image,
                    prompt_mode or self.prompt_mode,
                ),
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages) if include_image else (None, None)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        self._before_generate(inputs, include_image=include_image)
        generated = self._model.generate(**inputs, **self._generation_kwargs(max_new_tokens=max_new_tokens))
        generated_trimmed = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        decoded = self._processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self._after_generate()
        return decoded[0].strip()

    def _before_generate(self, inputs: Any, *, include_image: bool) -> None:
        return None

    def _after_generate(self) -> None:
        return None

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


def _resolve_image(example: EvalExample) -> Any:
    if example.image is not None:
        return example.image
    if example.image_path is not None:
        return str(example.image_path)
    raise ValueError(f"Example {example.id} has no image.")


def _image_content(image: Any, max_pixels: int, min_pixels: int) -> dict[str, Any]:
    content = {"type": "image", "image": image}
    if max_pixels > 0:
        content["max_pixels"] = max_pixels
    if min_pixels > 0:
        content["min_pixels"] = min_pixels
    return content


def _message_content(
    image: Any,
    prompt: str,
    max_pixels: int,
    min_pixels: int,
    include_image: bool,
    prompt_mode: str,
) -> list[dict[str, Any]]:
    content = []
    if include_image:
        content.append(_image_content(image, max_pixels, min_pixels))
    content.append({"type": "text", "text": _format_prompt(prompt, prompt_mode)})
    return content


def _format_prompt(prompt: str, prompt_mode: str) -> str:
    if prompt_mode == "baseline":
        suffix = "Answer with only the final answer."
    elif prompt_mode == "visual_first":
        suffix = "Look carefully at the image. Answer using only the visible image. Answer with only the final answer."
    elif prompt_mode == "counterfactual_warning":
        suffix = (
            "The image may be unusual or counterfactual. Do not rely on what is normally true. "
            "Answer based only on the visible image. Answer with only the final answer."
        )
    elif prompt_mode == "describe_then_answer":
        suffix = (
            "First write one short sentence describing the relevant visual evidence. "
            "Then write 'Final answer:' followed by the answer."
        )
    elif prompt_mode == "raw":
        suffix = ""
    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    if not suffix:
        return prompt
    return f"{prompt}\n{suffix}"


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
    seed: int | None = 0,
    device_map: str = "auto",
    prompt_mode: str = "baseline",
) -> Qwen25VLAdapter:
    return Qwen25VLAdapter(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        include_image=include_image,
        seed=seed,
        device_map=device_map,
        prompt_mode=prompt_mode,
    )


def _stable_int(value: str) -> int:
    total = 0
    for char in value:
        total = (total * 33 + ord(char)) % 1_000_000
    return total
