import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
import torch

from adapters.qwen3_vl_roi_attention import Qwen3VLROIAttentionAdapter
from vlm_eval.qwen3_roi_attention import (
    confirm_conditions,
    head_conditions,
    pixel_mask_to_visual_tokens,
    prepare_roi_splits,
    spatial_control_mask,
    tune_conditions,
)


def test_pixel_mask_maps_to_merged_qwen_grid_in_row_major_order() -> None:
    array = np.zeros((80, 120), dtype=np.uint8)
    array[:40, :60] = 255
    tokens, metadata = pixel_mask_to_visual_tokens(
        Image.fromarray(array), [1, 8, 12], spatial_merge_size=2
    )
    expected = np.zeros((4, 6), dtype=bool)
    expected[:2, :3] = True
    assert np.array_equal(tokens, expected.reshape(-1))
    assert metadata["merged_grid_hw"] == [4, 6]
    assert metadata["n_image_tokens"] == 24
    assert metadata["n_target_tokens"] == 6
    assert metadata["target_token_indices"] == [0, 1, 2, 6, 7, 8]


def test_tiny_nonempty_mask_always_selects_at_least_one_token() -> None:
    array = np.zeros((100, 100), dtype=np.uint8)
    array[1, 1] = 255
    tokens, metadata = pixel_mask_to_visual_tokens(
        Image.fromarray(array),
        [1, 10, 10],
        spatial_merge_size=2,
        min_token_coverage=0.9,
    )
    assert tokens.sum() == 1
    assert metadata["n_target_tokens"] == 1


def test_bad_grid_or_merge_is_rejected() -> None:
    mask = Image.new("L", (10, 10), 255)
    with pytest.raises(ValueError, match="divisible"):
        pixel_mask_to_visual_tokens(mask, [1, 7, 8], spatial_merge_size=2)


def test_adapter_installs_sparse_roi_mask_at_qwen_placeholder_positions() -> None:
    class Inputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

    layers = [SimpleNamespace(self_attn=SimpleNamespace()) for _ in range(2)]
    adapter = object.__new__(Qwen3VLROIAttentionAdapter)
    adapter._torch = torch
    adapter._model = SimpleNamespace(
        config=SimpleNamespace(image_token_id=99),
        model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)),
    )
    adapter._processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
    adapter._attention_records = []
    adapter._token_confidence_metadata = {}
    adapter.last_generation_metadata = None
    array = np.zeros((40, 40), dtype=np.uint8)
    array[:20, :20] = 255
    adapter._pending_roi_mask = Image.fromarray(array)
    adapter._attention_region = "roi"
    adapter._roi_token_metadata = {}
    adapter.roi_token_min_coverage = 0.05
    inputs = Inputs(
        input_ids=torch.tensor([[10, 99, 99, 99, 99, 11]]),
        image_grid_thw=torch.tensor([[1, 4, 4]]),
    )

    adapter._before_generate(inputs, include_image=True)

    expected = torch.tensor([False, True, False, False, False, False])
    for layer in layers:
        assert torch.equal(layer.self_attn._vlm_gaze_image_token_mask, expected)
    assert adapter._roi_token_metadata["target_token_indices"] == [0]


def test_spatial_control_preserves_area_and_is_deterministic() -> None:
    array = np.zeros((20, 30), dtype=np.uint8)
    array[2:5, 3:9] = 255
    mask = Image.fromarray(array)
    first = np.asarray(spatial_control_mask(mask, "example", 7))
    second = np.asarray(spatial_control_mask(mask, "example", 7))
    assert np.array_equal(first, second)
    assert np.count_nonzero(first) == np.count_nonzero(array)
    assert not np.array_equal(first, array)


def test_sweep_contains_expected_alpha_and_head_controls() -> None:
    tune = tune_conditions()
    roi_alphas = {row["alpha"] for row in tune if row["region"] == "roi"}
    assert roi_alphas == {0.5, 1.0, 2.0, 5.0}
    heads = head_conditions(2.0)
    assert any(
        row["head_count"] == 100 and row["head_selection"] == "gaze_global"
        for row in heads
    )
    assert any(
        row["head_count"] == 100 and row["head_selection"] == "layer_matched_random"
        for row in heads
    )
    confirm = confirm_conditions(tune[2], heads[2])
    assert {row["region"] for row in confirm} == {"full_image", "roi", "shifted_roi"}


def test_split_keeps_shared_mask_prompt_variants_together(tmp_path) -> None:
    source = tmp_path / "vlmbias.jsonl"
    roi_root = tmp_path / "roi"
    roi_root.mkdir()
    source_rows = [
        {"id": "a1", "image_path": "a.png", "metadata": {}},
        {"id": "a2", "image_path": "a.png", "metadata": {}},
        {"id": "b1", "image_path": "b.png", "metadata": {}},
        {"id": "c1", "image_path": "c.png", "metadata": {}},
    ]
    accepted = [
        {
            "id": "a1",
            "topic": "Animals",
            "covered_local_ids": ["a1", "a2"],
            "artifacts": {"mask_path": "masks/a.png"},
            "mask_stats": {"clean_mask_fraction": 0.1},
        },
        {
            "id": "b1",
            "topic": "Flags",
            "covered_local_ids": ["b1"],
            "artifacts": {"mask_path": "masks/b.png"},
            "mask_stats": {"clean_mask_fraction": 0.2},
        },
        {
            "id": "c1",
            "topic": "Chess Pieces",
            "covered_local_ids": ["c1"],
            "artifacts": {"mask_path": "masks/c.png"},
            "mask_stats": {"clean_mask_fraction": 0.3},
        },
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8"
    )
    (roi_root / "accepted.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in accepted), encoding="utf-8"
    )

    manifest = prepare_roi_splits(
        vlmbias_path=source,
        roi_root=roi_root,
        out_dir=tmp_path / "experiment",
        dev_groups=2,
        smoke_groups=1,
        seed=7,
    )
    dev = {
        row["metadata"]["roi_group_id"]
        for row in _read_jsonl(tmp_path / "experiment/dev_vlmbias_roi.jsonl")
    }
    confirm = {
        row["metadata"]["roi_group_id"]
        for row in _read_jsonl(tmp_path / "experiment/confirm_vlmbias_roi.jsonl")
    }
    assert dev.isdisjoint(confirm)
    assert manifest["group_counts"] == {
        "smoke": 1,
        "dev": 2,
        "confirm": 1,
        "all": 3,
    }
    memberships = {
        row["id"]: split
        for split, path in (
            ("dev", tmp_path / "experiment/dev_vlmbias_roi.jsonl"),
            ("confirm", tmp_path / "experiment/confirm_vlmbias_roi.jsonl"),
        )
        for row in _read_jsonl(path)
    }
    assert memberships["a1"] == memberships["a2"]


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]
