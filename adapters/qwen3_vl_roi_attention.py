from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from adapters.qwen25_vl_gaze_attention import _language_layers
from adapters.qwen3_vl_gaze_attention import Qwen3VLGazeAttentionAdapter
from vlm_eval.qwen3_roi_attention import pixel_mask_to_visual_tokens


@dataclass
class Qwen3VLROIAttentionAdapter(Qwen3VLGazeAttentionAdapter):
    """Qwen3 gaze-head boosting restricted to a per-example spatial ROI."""

    roi_token_min_coverage: float = 0.05
    roi_attention_bias: float | None = None
    context_attention_bias: float = 0.0
    _pending_roi_mask: Image.Image | None = field(default=None, init=False)
    _attention_region: str = field(default="full_image", init=False)
    _roi_token_metadata: dict[str, Any] = field(default_factory=dict, init=False)

    @property
    def eval_config(self) -> dict[str, Any]:
        config = super().eval_config
        config.update(
            {
                "roi_token_min_coverage": self.roi_token_min_coverage,
                "roi_attention_bias": self.roi_attention_bias,
                "context_attention_bias": self.context_attention_bias,
            }
        )
        return config

    def set_attention_region(
        self, mask: Image.Image | None, *, region: str = "roi"
    ) -> None:
        if region not in {"roi", "shifted_roi", "full_image"}:
            raise ValueError(f"unknown attention region: {region}")
        if region != "full_image" and mask is None:
            raise ValueError(f"{region} requires a pixel mask")
        self._pending_roi_mask = mask.copy() if mask is not None else None
        self._attention_region = region

    def _before_generate(self, inputs: Any, *, include_image: bool) -> None:
        super()._before_generate(inputs, include_image=include_image)
        self._roi_token_metadata.clear()
        if self._attention_region == "full_image" or self._pending_roi_mask is None:
            image_token_id = getattr(self._model.config, "image_token_id", None)
            image_count = (
                int(inputs.input_ids[0].eq(image_token_id).sum().item())
                if include_image and image_token_id is not None
                else 0
            )
            self._roi_token_metadata.update(
                {
                    "region": "full_image",
                    "n_image_tokens": image_count,
                    "n_target_tokens": image_count,
                    "target_token_indices": list(range(image_count)),
                    "target_token_fraction": 1.0 if image_count else 0.0,
                }
            )
            return

        image_token_id = getattr(self._model.config, "image_token_id", None)
        if image_token_id is None:
            raise RuntimeError("Qwen3 config does not expose image_token_id")
        all_image_mask = inputs.input_ids[0].eq(image_token_id).detach()
        grid = inputs["image_grid_thw"][0].detach().cpu().tolist()
        merge_size = int(getattr(self._processor.image_processor, "merge_size", 0))
        local_mask, metadata = pixel_mask_to_visual_tokens(
            self._pending_roi_mask,
            grid,
            spatial_merge_size=merge_size,
            min_token_coverage=self.roi_token_min_coverage,
        )
        if int(all_image_mask.sum().item()) != int(local_mask.size):
            raise RuntimeError(
                "ROI token count does not match Qwen image placeholders: "
                f"roi={local_mask.size}, placeholders={int(all_image_mask.sum().item())}, "
                f"grid={grid}, merge={merge_size}"
            )
        target = self._torch.zeros_like(all_image_mask, dtype=self._torch.bool)
        target[all_image_mask] = self._torch.as_tensor(
            local_mask, dtype=self._torch.bool, device=target.device
        )
        context = all_image_mask & ~target
        token_bias = None
        if self.roi_attention_bias is not None:
            token_bias = self._torch.zeros_like(target, dtype=self._torch.float32)
            token_bias[target] = float(self.roi_attention_bias)
            token_bias[context] = float(self.context_attention_bias)
        for layer in _language_layers(self._model):
            layer.self_attn._vlm_gaze_image_token_mask = target
            if token_bias is not None:
                layer.self_attn._vlm_gaze_image_context_token_mask = context
                layer.self_attn._vlm_gaze_image_token_bias = token_bias
        self._roi_token_metadata.update(
            {
                "region": self._attention_region,
                "roi_attention_bias": self.roi_attention_bias,
                "context_attention_bias": self.context_attention_bias,
                "n_context_tokens": int(all_image_mask.sum().item())
                - int(metadata["n_target_tokens"]),
                "context_token_fraction": 1.0
                - float(metadata["target_token_fraction"]),
                **metadata,
            }
        )

    def _after_generate(self) -> None:
        super()._after_generate()
        assert self.last_generation_metadata is not None
        self.last_generation_metadata["roi_tokens"] = dict(self._roi_token_metadata)
