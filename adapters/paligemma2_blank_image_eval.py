from __future__ import annotations

from adapters.paligemma2 import DEFAULT_MODEL_ID, PaliGemma2Adapter


EVAL_PRESET = {
    "model_id": DEFAULT_MODEL_ID,
    "max_new_tokens": 16,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": None,
    "top_k": None,
    "include_image": False,
    "seed": 0,
    "device_map": "auto",
    "prompt_mode": "baseline",
    "blank_image_size": 448,
}


def make_adapter() -> PaliGemma2Adapter:
    return PaliGemma2Adapter(**EVAL_PRESET)
