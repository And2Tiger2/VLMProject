from __future__ import annotations

from dataclasses import dataclass

from adapters.qwen25_vl import _resolve_device_map, _resolve_torch_dtype
from adapters.qwen25_vl_gaze_attention import (
    ATTENTION_IMPL_NAME,
    Qwen25VLGazeAttentionAdapter,
    _gaze_image_bias_attention_forward,
)


@dataclass
class Qwen3VLGazeAttentionAdapter(Qwen25VLGazeAttentionAdapter):
    """Image-token alpha boosting on discovered Qwen3-VL gaze heads."""

    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    gaze_ranking_path: str = "segments/gaze_heads_qwen3_8b/runs/gaze_discovery_seed42_merged/gaze_head_ranking.json"
    decode_only: bool = False

    def __post_init__(self) -> None:
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            from transformers.models.qwen3_vl import modeling_qwen3_vl
        except ImportError as exc:
            raise RuntimeError("Install Qwen3-VL dependencies with `uv sync --extra qwen`.") from exc

        if ATTENTION_IMPL_NAME not in modeling_qwen3_vl.ALL_ATTENTION_FUNCTIONS:
            modeling_qwen3_vl.ALL_ATTENTION_FUNCTIONS.register(
                ATTENTION_IMPL_NAME, _gaze_image_bias_attention_forward
            )
        self._torch = torch
        self._process_vision_info = process_vision_info
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=_resolve_torch_dtype(torch),
            device_map=_resolve_device_map(self.device_map, torch),
            attn_implementation=ATTENTION_IMPL_NAME,
        )
        self._model.eval()
        self.name = f"{self.model_id}-gaze-attention-top{self.top_k_gaze}-alpha{self.attention_alpha}"
        self._configure_attention_modules()
