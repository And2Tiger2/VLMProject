from __future__ import annotations

from adapters.qwen25_vl import Qwen25VLAdapter


EVAL_PRESET = {
    "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "max_new_tokens": 16,
    "max_pixels": 1048576,
    "min_pixels": 0,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": None,
    "top_k": None,
    "include_image": False,
    "seed": 0,
    "device_map": "auto",
}


def make_adapter() -> Qwen25VLAdapter:
    adapter = Qwen25VLAdapter(**EVAL_PRESET)
    adapter.name = f"{adapter.model_id}-no-image-temp0.7"
    return adapter

