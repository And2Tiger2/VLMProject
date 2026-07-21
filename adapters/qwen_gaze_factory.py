from __future__ import annotations

from typing import Any


QWEN3_GAZE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def is_qwen3_vl(model_id: str) -> bool:
    return "qwen3-vl" in model_id.lower()


def make_panel_gaze_adapter(*, model_id: str, **kwargs: Any):
    if is_qwen3_vl(model_id):
        from adapters.qwen3_vl_gaze import Qwen3VLGazeAdapter

        return Qwen3VLGazeAdapter(model_id=model_id, **kwargs)
    from adapters.qwen25_vl_gaze import Qwen25VLGazeAdapter

    return Qwen25VLGazeAdapter(model_id=model_id, **kwargs)


def make_qwen_adapter(*, model_id: str, **kwargs: Any):
    if is_qwen3_vl(model_id):
        from adapters.qwen3_vl import Qwen3VLAdapter

        return Qwen3VLAdapter(model_id=model_id, **kwargs)
    from adapters.qwen25_vl import Qwen25VLAdapter

    return Qwen25VLAdapter(model_id=model_id, **kwargs)


def make_image_attention_adapter(*, model_id: str, **kwargs: Any):
    if is_qwen3_vl(model_id):
        from adapters.qwen3_vl_gaze_attention import Qwen3VLGazeAttentionAdapter

        return Qwen3VLGazeAttentionAdapter(model_id=model_id, **kwargs)
    from adapters.qwen25_vl_gaze_attention import Qwen25VLGazeAttentionAdapter

    return Qwen25VLGazeAttentionAdapter(model_id=model_id, **kwargs)
