from __future__ import annotations

from dataclasses import dataclass

from typing import Any

from adapters.qwen25_vl import _resolve_device_map, _resolve_torch_dtype
from adapters.qwen25_vl_gaze import (
    ATTENTION_IMPL_NAME,
    Qwen25VLGazeAdapter,
    _gaze_attention_forward,
    _image_content,
)
from adapters.qwen3_vl import prepare_qwen3_inputs


@dataclass
class Qwen3VLGazeAdapter(Qwen25VLGazeAdapter):
    """Panel-level Gaze Heads discovery and steering for Qwen3-VL."""

    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"

    def __post_init__(self) -> None:
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            from transformers.models.qwen3_vl import modeling_qwen3_vl
        except ImportError as exc:
            raise RuntimeError("Install Qwen3-VL dependencies with `uv sync --extra qwen`.") from exc

        if ATTENTION_IMPL_NAME not in modeling_qwen3_vl.ALL_ATTENTION_FUNCTIONS:
            modeling_qwen3_vl.ALL_ATTENTION_FUNCTIONS.register(ATTENTION_IMPL_NAME, _gaze_attention_forward)
        self._torch = torch
        self._process_vision_info = process_vision_info
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=_resolve_torch_dtype(torch),
            device_map=_resolve_device_map(self.device_map, torch),
            attn_implementation=self.attention_impl,
        )
        self._model.eval()
        self.name = f"{self.model_id}-gaze"
        self._configure_attention_modules()

    def prepare_inputs(self, image: Any, prompt: str) -> Any:
        messages = [
            {
                "role": "user",
                "content": [
                    _image_content(image, self.max_pixels, self.min_pixels),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return prepare_qwen3_inputs(self._processor, messages, self._model.device)
