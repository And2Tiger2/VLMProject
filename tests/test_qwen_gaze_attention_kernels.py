from types import SimpleNamespace

import pytest
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


def test_target_mass_controller_reaches_requested_mass_without_touching_other_heads() -> None:
    query, key, value = _qkv(query_len=3)
    module = SimpleNamespace(
        num_key_value_groups=1,
        training=False,
        _vlm_gaze_image_token_mask=torch.tensor([True, False, False]),
        _vlm_gaze_image_attention_alpha=0.0,
        _vlm_gaze_image_attention_controller="target_mass",
        _vlm_gaze_image_target_attention_mass=0.7,
        _vlm_gaze_image_max_attention_alpha=5.0,
        _vlm_gaze_image_head_indices=[1],
        _vlm_gaze_image_decode_only=False,
        _vlm_gaze_image_attention_records=[],
        _vlm_gaze_image_layer_idx=0,
    )
    _, weights = _gaze_image_bias_attention_forward(
        module, query, key, value, None, 0.5
    )
    assert torch.allclose(weights[0, 1, :, 0], torch.full((3,), 0.7), atol=1e-5)
    assert torch.allclose(weights[0, 0], torch.full((3, 3), 1 / 3))
    record = module._vlm_gaze_image_attention_records[0]
    assert record["preboost_head_image_attention_mass"] == pytest.approx(1 / 3)
    assert 0.0 < record["mean_effective_alpha"] < 5.0
    assert record["alpha_cap_fraction"] == 0.0


def test_target_mass_controller_respects_alpha_cap() -> None:
    query, key, value = _qkv(query_len=1)
    module = SimpleNamespace(
        num_key_value_groups=1,
        training=False,
        _vlm_gaze_image_token_mask=torch.tensor([True, False, False]),
        _vlm_gaze_image_attention_alpha=0.0,
        _vlm_gaze_image_attention_controller="target_mass",
        _vlm_gaze_image_target_attention_mass=0.99,
        _vlm_gaze_image_max_attention_alpha=0.1,
        _vlm_gaze_image_head_indices=[1],
        _vlm_gaze_image_decode_only=False,
        _vlm_gaze_image_attention_records=[],
        _vlm_gaze_image_layer_idx=0,
    )
    _, weights = _gaze_image_bias_attention_forward(
        module, query, key, value, None, 0.5
    )
    assert weights[0, 1, 0, 0] < 0.99
    assert module._vlm_gaze_image_attention_records[0][
        "alpha_cap_fraction"
    ] == pytest.approx(1.0)
