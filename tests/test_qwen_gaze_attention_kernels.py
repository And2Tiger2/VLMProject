from types import SimpleNamespace

import torch

from adapters.qwen25_vl_gaze import _gaze_attention_forward
from adapters.qwen25_vl_gaze_attention import _gaze_image_bias_attention_forward


def _qkv(query_len: int, key_len: int = 3):
    query = torch.zeros(1, 2, query_len, 4)
    key = torch.zeros(1, 2, key_len, 4)
    value = torch.randn(1, 2, key_len, 4)
    return query, key, value


def test_panel_bias_full_sequence_and_decode_only_modes() -> None:
    query, key, value = _qkv(query_len=2)
    target = torch.tensor([True, False, False])
    base = {
        "num_key_value_groups": 1,
        "training": False,
        "_vlm_gaze_bias_by_head": {
            0: {"boost_mask": target, "suppress_mask": ~target, "swap_bias": 100.0}
        },
        "_vlm_gaze_dynamic_bias_by_head": {},
        "_vlm_gaze_attention_records": [],
        "_vlm_gaze_attention_masks": {},
        "_vlm_gaze_track_heads": [0],
        "_vlm_gaze_layer_idx": 0,
    }
    full_module = SimpleNamespace(**base, _vlm_gaze_decode_only=False)
    _, full_weights = _gaze_attention_forward(full_module, query, key, value, None, 0.5)
    assert torch.all(full_weights[0, 0, :, 0] > 0.999)

    decode_module = SimpleNamespace(**base, _vlm_gaze_decode_only=True)
    _, prefill_weights = _gaze_attention_forward(decode_module, query, key, value, None, 0.5)
    assert torch.allclose(prefill_weights[0, 0], torch.full((2, 3), 1 / 3))


def test_image_alpha_full_sequence_biases_every_prefill_query() -> None:
    query, key, value = _qkv(query_len=3)
    image_mask = torch.tensor([True, False, False])
    module = SimpleNamespace(
        num_key_value_groups=1,
        training=False,
        _vlm_gaze_image_token_mask=image_mask,
        _vlm_gaze_image_attention_alpha=100.0,
        _vlm_gaze_image_head_indices=[1],
        _vlm_gaze_image_decode_only=False,
        _vlm_gaze_image_attention_records=[],
        _vlm_gaze_image_layer_idx=0,
    )
    _, weights = _gaze_image_bias_attention_forward(module, query, key, value, None, 0.5)
    assert torch.all(weights[0, 1, :, 0] > 0.999)
    assert torch.allclose(weights[0, 0], torch.full((3, 3), 1 / 3))


def test_image_alpha_decode_only_leaves_prefill_unchanged() -> None:
    query, key, value = _qkv(query_len=2)
    module = SimpleNamespace(
        num_key_value_groups=1,
        training=False,
        _vlm_gaze_image_token_mask=torch.tensor([True, False, False]),
        _vlm_gaze_image_attention_alpha=100.0,
        _vlm_gaze_image_head_indices=[1],
        _vlm_gaze_image_decode_only=True,
        _vlm_gaze_image_attention_records=[],
        _vlm_gaze_image_layer_idx=0,
    )
    _, weights = _gaze_image_bias_attention_forward(module, query, key, value, None, 0.5)
    assert torch.allclose(weights, torch.full((1, 2, 2, 3), 1 / 3))
