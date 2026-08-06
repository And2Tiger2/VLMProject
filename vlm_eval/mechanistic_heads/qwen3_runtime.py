from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adapters.qwen25_vl import (
    _message_content,
    _resolve_device_map,
    _resolve_torch_dtype,
)
from adapters.qwen3_vl import prepare_qwen3_inputs
from vlm_eval.mechanistic_heads.capture import (
    MECHANISTIC_ATTENTION_IMPL,
    get_language_layers,
    register_qwen3_mechanistic_attention,
)
from vlm_eval.mechanistic_heads.token_spans import TokenSpan, TokenSpans, contiguous_span, locate_subsequence


EXPECTED_LANGUAGE_LAYERS = 36
EXPECTED_LANGUAGE_HEADS = 32


@dataclass(frozen=True)
class Qwen3Architecture:
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    attention_bias: bool

    @property
    def n_language_heads_total(self) -> int:
        return self.n_layers * self.n_heads


def inspect_qwen3_architecture(model: Any, *, enforce_expected: bool = True) -> Qwen3Architecture:
    text_config = getattr(model.config, "text_config", None)
    if text_config is None:
        raise RuntimeError("Qwen3 model config has no text_config")
    layers = get_language_layers(model)
    if not layers:
        raise RuntimeError("Qwen3 model has no language layers")
    first = layers[0].self_attn
    architecture = Qwen3Architecture(
        n_layers=len(layers),
        n_heads=int(text_config.num_attention_heads),
        n_kv_heads=int(text_config.num_key_value_heads),
        head_dim=int(first.head_dim),
        hidden_size=int(text_config.hidden_size),
        attention_bias=first.o_proj.bias is not None,
    )
    if first.o_proj.in_features != architecture.n_heads * architecture.head_dim:
        raise RuntimeError("o_proj input width does not match heads * head_dim")
    for index, layer in enumerate(layers):
        if layer.self_attn.o_proj.in_features != first.o_proj.in_features:
            raise RuntimeError(f"layer {index} has inconsistent o_proj input width")
    if enforce_expected and (
        architecture.n_layers != EXPECTED_LANGUAGE_LAYERS
        or architecture.n_heads != EXPECTED_LANGUAGE_HEADS
    ):
        raise RuntimeError(
            "unexpected Qwen3-VL-8B language architecture: "
            f"{architecture.n_layers} layers x {architecture.n_heads} heads; "
            f"expected {EXPECTED_LANGUAGE_LAYERS} x {EXPECTED_LANGUAGE_HEADS}"
        )
    return architecture


class Qwen3MechanisticRuntime:
    """Pinned Qwen3 loader using the repository's exact processor path."""

    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        device_map: str = "cuda",
        attention_backend: str = MECHANISTIC_ATTENTION_IMPL,
        adapter_path: str | None = None,
        max_pixels: int = 1048576,
        min_pixels: int = 0,
        enforce_expected_architecture: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Install Qwen3 dependencies with `uv sync --extra qwen`.") from exc
        if attention_backend == MECHANISTIC_ATTENTION_IMPL:
            register_qwen3_mechanistic_attention()
        resolved_adapter = adapter_path
        candidate_path = Path(model_id)
        if resolved_adapter is None and candidate_path.joinpath("adapter_config.json").is_file():
            resolved_adapter = model_id
            try:
                from peft import PeftConfig
            except ImportError as exc:
                raise RuntimeError("Adapter checkpoints require `uv sync --extra mechanistic`.") from exc
            model_id = str(PeftConfig.from_pretrained(resolved_adapter).base_model_name_or_path)
        self.torch = torch
        self.model_id = model_id
        self.adapter_path = resolved_adapter
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.processor = AutoProcessor.from_pretrained(model_id)
        # The mechanistic implementation captures language-head A_h V_h and
        # post-W_O contributions.  Loading it as the global implementation
        # also replaces the vision transformer's normal SDPA attention, which
        # is unnecessary and makes otherwise identical image batches follow a
        # different numerical path.  Load the repository-normal backend for
        # every component, then opt only the language stack into capture.
        loader_attention_backend = (
            "sdpa"
            if attention_backend == MECHANISTIC_ATTENTION_IMPL
            else attention_backend
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=_resolve_torch_dtype(torch),
            device_map=_resolve_device_map(device_map, torch),
            attn_implementation=loader_attention_backend,
        )
        self.adapter_merged = False
        if resolved_adapter is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Adapter checkpoints require `uv sync --extra mechanistic`.") from exc
            adapted = PeftModel.from_pretrained(self.model, resolved_adapter)
            # Exact post-W_O attribution must use the effective projection
            # matrix, including the LoRA delta.  PEFT's unmerged Linear wrapper
            # exposes the frozen base ``weight`` while applying the adapter in
            # a separate forward branch, which would make per-head projection
            # incomplete.  Safe-merging is inference-equivalent and leaves one
            # concrete weight matrix for every mechanistic hook.
            self.model = adapted.merge_and_unload(safe_merge=True)
            self.adapter_merged = True
        if attention_backend == MECHANISTIC_ATTENTION_IMPL:
            text_config = getattr(self.model.config, "text_config", None)
            vision_config = getattr(self.model.config, "vision_config", None)
            if text_config is None or vision_config is None:
                raise RuntimeError("Qwen3-VL config is missing text/vision subconfigs")
            text_config._attn_implementation = MECHANISTIC_ATTENTION_IMPL
            if vision_config._attn_implementation != loader_attention_backend:
                raise RuntimeError("Qwen3-VL vision backend was unexpectedly changed")
        self.model.eval()
        self.architecture = inspect_qwen3_architecture(
            self.model, enforce_expected=enforce_expected_architecture
        )

    def prepare(self, image: Any, prompt: str, *, prompt_mode: str = "raw") -> Any:
        messages = [
            {
                "role": "user",
                "content": _message_content(
                    image,
                    prompt,
                    self.max_pixels,
                    self.min_pixels,
                    image is not None,
                    prompt_mode,
                ),
            }
        ]
        return prepare_qwen3_inputs(self.processor, messages, self.model.device)

    def answer_token_ids(self, answer: str) -> Any:
        encoded = self.processor.tokenizer(
            answer, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        if encoded.shape[1] == 0:
            raise ValueError("candidate answer tokenizes to an empty sequence")
        return encoded.to(self.model.device)

    def trace_spans(self, inputs: Any, prompt: str, *, answer_length: int = 0) -> TokenSpans:
        ids = [int(value) for value in inputs.input_ids[0].tolist()]
        prompt_ids = self.processor.tokenizer(prompt, add_special_tokens=False).input_ids
        user_start, user_end = locate_subsequence(ids, [int(value) for value in prompt_ids])
        image_positions = [index for index, token in enumerate(ids) if token == int(self.model.config.image_token_id)]
        visual = contiguous_span(image_positions, name="visual")
        end_id = getattr(self.model.config, "vision_end_token_id", None)
        if end_id is None:
            end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        ends = [index for index, token in enumerate(ids) if token == int(end_id) and index >= visual.end]
        if not ends:
            raise ValueError("no image-end token after the visual span")
        # The repository's exact conversation template supplies no system
        # message, so its system-token span is explicitly empty.
        spans = TokenSpans(
            system=TokenSpan(0, 0),
            visual=visual,
            image_end=TokenSpan(ends[0], ends[0] + 1),
            user_prompt=TokenSpan(user_start, user_end),
            final_prompt=TokenSpan(user_end, len(ids)),
            generated_answer=TokenSpan(len(ids), len(ids) + answer_length),
            sequence_length=len(ids) + answer_length,
        )
        spans.assert_partition_bounds()
        return spans


def runtime_from_config(
    config: dict[str, Any],
    *,
    device_map: str,
    checkpoint_override: str | None = None,
) -> Qwen3MechanisticRuntime:
    """Load either the base/full-weight checkpoint or a PEFT adapter config."""

    base_model = str(config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct"))
    checkpoint = checkpoint_override or config.get("adapter_path") or config.get("checkpoint")
    if checkpoint is None:
        return Qwen3MechanisticRuntime(model_id=base_model, device_map=device_map)
    checkpoint_path = Path(str(checkpoint))
    checkpoint_text = str(checkpoint)
    if (
        checkpoint_text.startswith(("segments/", "./", "../", "/"))
        and not checkpoint_path.exists()
    ):
        raise FileNotFoundError(
            f"configured point-model checkpoint does not exist: {checkpoint_path}; "
            "run the Point-Answer training calibration first"
        )
    if checkpoint_path.is_dir():
        from vlm_eval.mechanistic_heads.preflight import require_completed_manifest

        identity_files = checkpoint_manifest_inputs(
            config, checkpoint_override=str(checkpoint_path)
        )
        if not identity_files:
            raise RuntimeError(
                f"configured point-model checkpoint has no identity files: {checkpoint_path}"
            )
        require_completed_manifest(
            checkpoint_path,
            expected_outputs=tuple(identity_files),
            require_current_git=True,
        )
    if checkpoint_path.joinpath("adapter_config.json").is_file():
        return Qwen3MechanisticRuntime(
            model_id=base_model,
            adapter_path=str(checkpoint_path),
            device_map=device_map,
        )
    return Qwen3MechanisticRuntime(model_id=str(checkpoint), device_map=device_map)


def checkpoint_manifest_inputs(
    config: dict[str, Any], *, checkpoint_override: str | None = None
) -> list[Path]:
    """Return compact checkpoint identity files for reproducibility hashing."""

    checkpoint = checkpoint_override or config.get("adapter_path") or config.get("checkpoint")
    if checkpoint is None:
        return []
    root = Path(str(checkpoint))
    if not root.is_dir():
        return []
    names = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "config.json",
        "model.safetensors.index.json",
        "training_summary.json",
        "training_context.json",
    )
    return [root / name for name in names if (root / name).is_file()]
