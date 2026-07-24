from __future__ import annotations

from dataclasses import dataclass

from typing import Any

from adapters.qwen25_vl import (
    Qwen25VLAdapter,
    _message_content,
    _resolve_device_map,
    _resolve_torch_dtype,
)


@dataclass
class Qwen3VLAdapter(Qwen25VLAdapter):
    """Qwen3-VL equivalent of the shared Qwen VLM evaluation adapter."""

    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    name: str = "qwen3-vl"

    def __post_init__(self) -> None:
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Install Qwen3-VL dependencies with `uv sync --extra qwen`.") from exc

        self._torch = torch
        self._process_vision_info = process_vision_info
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=_resolve_torch_dtype(torch),
            device_map=_resolve_device_map(self.device_map, torch),
        )
        self._model.eval()
        self.name = self.model_id

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
        inputs = prepare_qwen3_inputs(self._processor, messages, self._model.device)
        self._before_generate(inputs, include_image=include_image)
        generated = self._model.generate(**inputs, **self._generation_kwargs(max_new_tokens=max_new_tokens))
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        self._after_generate_output(generated, inputs)
        generated_trimmed = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, sequences)
        ]
        decoded = self._processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self._after_generate()
        return decoded[0].strip()


def prepare_qwen3_inputs(processor: Any, messages: list[dict[str, Any]], device: Any) -> Any:
    """Use the native Qwen3-VL multimodal chat-template path from the reference repo."""
    return processor.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
