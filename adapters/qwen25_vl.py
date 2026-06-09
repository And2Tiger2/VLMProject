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
    device_map: str = "auto"
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

    def generate(self, example: EvalExample) -> str:
        image = _resolve_image(example)
        messages = [
            {
                "role": "user",
                "content": [
                    _image_content(image, self.max_pixels, self.min_pixels),
                    {"type": "text", "text": _format_prompt(example.prompt)},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_trimmed = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        decoded = self._processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()


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


def _format_prompt(prompt: str) -> str:
    return f"{prompt}\nAnswer with only the final answer."


def make_adapter(
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    max_new_tokens: int = 16,
    max_pixels: int = 1048576,
    min_pixels: int = 0,
    device_map: str = "auto",
) -> Qwen25VLAdapter:
    return Qwen25VLAdapter(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        device_map=device_map,
    )
