from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adapters.paligemma2 import DEFAULT_MODEL_ID, PaliGemma2Adapter, _resolve_device_map


ATTENTION_IMPL_NAME = "vlm_image_token_bias"


@dataclass
class PaliGemma2AttentionAdapter(PaliGemma2Adapter):
    attention_alpha: float = 0.0
    layer_selection: str = "baseline"
    last_generation_metadata: dict[str, Any] | None = None
    _attention_records: list[dict[str, float | int]] = field(default_factory=list, init=False)
    _boosted_layers: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
        except ImportError as exc:
            raise RuntimeError("Install optional PaliGemma dependencies with `uv sync --extra paligemma`.") from exc

        _register_attention_impls()
        self._torch = torch
        self._processor = PaliGemmaProcessor.from_pretrained(self.model_id)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=_resolve_device_map(self.device_map, torch),
            attn_implementation=ATTENTION_IMPL_NAME,
        )
        self.name = f"{self.model_id}-attention-{self.layer_selection}-alpha{self.attention_alpha}"
        self._configure_attention_modules()

    @property
    def eval_config(self) -> dict[str, Any]:
        config = super().eval_config
        config.update(
            {
                "attention_alpha": self.attention_alpha,
                "layer_selection": self.layer_selection,
                "boosted_layers": sorted(self._boosted_layers),
                "attention_impl": ATTENTION_IMPL_NAME,
            }
        )
        return config

    def _configure_attention_modules(self) -> None:
        layers = _language_layers(self._model)
        self._boosted_layers = _select_layers(len(layers), self.layer_selection)
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            attn._vlm_layer_idx = layer_idx
            attn._vlm_attention_alpha = self.attention_alpha if layer_idx in self._boosted_layers else 0.0
            attn._vlm_attention_records = self._attention_records
            attn._vlm_image_token_mask = None
            attn._vlm_attention_masks = {}

    def _generate_text(self, prompt: str, image: Any, max_new_tokens: int | None = None) -> str:
        inputs = self._processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)
        input_length = inputs["input_ids"].shape[-1]
        self._before_generate(inputs)
        generated = self._model.generate(**inputs, **self._generation_kwargs(max_new_tokens=max_new_tokens))
        decoded = self._processor.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self._after_generate()
        return decoded[0].strip()

    def _before_generate(self, inputs: Any) -> None:
        self._attention_records.clear()
        self.last_generation_metadata = None
        image_token_id = getattr(self._model.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(self._model.config, "image_token_index", None)
        image_mask = inputs.input_ids[0].eq(image_token_id).detach() if image_token_id is not None else None

        for layer in _language_layers(self._model):
            layer.self_attn._vlm_image_token_mask = image_mask
            layer.self_attn._vlm_attention_masks = {"image": image_mask} if image_mask is not None else {}

    def _after_generate(self) -> None:
        self.last_generation_metadata = {
            "attention": _summarize_attention_records(self._attention_records, self._boosted_layers),
            "attention_alpha": self.attention_alpha,
            "layer_selection": self.layer_selection,
        }


def make_adapter(
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 16,
    do_sample: bool = True,
    temperature: float | None = 0.7,
    top_p: float | None = None,
    top_k: int | None = None,
    include_image: bool = True,
    seed: int | None = 0,
    device_map: str = "auto",
    prompt_mode: str = "baseline",
    blank_image_size: int = 448,
    attention_alpha: float = 0.0,
    layer_selection: str = "baseline",
) -> PaliGemma2AttentionAdapter:
    return PaliGemma2AttentionAdapter(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        include_image=include_image,
        seed=seed,
        device_map=device_map,
        prompt_mode=prompt_mode,
        blank_image_size=blank_image_size,
        attention_alpha=attention_alpha,
        layer_selection=layer_selection,
    )


def _register_attention_impls() -> None:
    for module_name in (
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
    ):
        try:
            module = __import__(module_name, fromlist=["ALL_ATTENTION_FUNCTIONS"])
        except ImportError:
            continue
        registry = getattr(module, "ALL_ATTENTION_FUNCTIONS", None)
        if registry is not None and ATTENTION_IMPL_NAME not in registry:
            registry.register(ATTENTION_IMPL_NAME, _image_bias_attention_forward)


def _image_bias_attention_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    softcap: float | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    import torch
    from torch import nn

    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    if scaling is None:
        scaling = module.head_dim**-0.5

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    image_mask = _token_mask_for_key_length(getattr(module, "_vlm_image_token_mask", None), key_states.shape[-2])
    query_slice = slice(-1, None) if query.shape[-2] > 1 else slice(None)
    alpha = float(getattr(module, "_vlm_attention_alpha", 0.0))
    if image_mask is not None and alpha != 0.0:
        attn_weights[..., query_slice, image_mask] += alpha

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    records = getattr(module, "_vlm_attention_records", None)
    named_masks = getattr(module, "_vlm_attention_masks", {})
    if records is not None and (image_mask is not None or named_masks):
        record = {
            "layer_idx": int(getattr(module, "_vlm_layer_idx", -1)),
            "query_len": int(query.shape[-2]),
            "key_len": int(key_states.shape[-2]),
        }
        if image_mask is not None:
            image_mass = attn_weights[..., query_slice, image_mask].sum(dim=-1).float().mean()
            record["image_attention_mass"] = float(image_mass.detach().cpu())
        for name, token_mask in named_masks.items():
            mask = _token_mask_for_key_length(token_mask, key_states.shape[-2])
            if mask is not None:
                mass = attn_weights[..., query_slice, mask].sum(dim=-1).float().mean()
                record[f"{name}_attention_mass"] = float(mass.detach().cpu())
        if any(key.endswith("_attention_mass") for key in record):
            records.append(record)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _repeat_kv(hidden_states: Any, n_rep: int) -> Any:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _token_mask_for_key_length(token_mask: Any | None, key_len: int) -> Any | None:
    if token_mask is None or not bool(token_mask.any()):
        return None
    if token_mask.numel() == key_len:
        return token_mask
    if token_mask.numel() > key_len:
        return token_mask[:key_len]

    import torch

    padding = torch.zeros(key_len - token_mask.numel(), dtype=torch.bool, device=token_mask.device)
    return torch.cat([token_mask, padding], dim=0)


def _select_layers(num_layers: int, layer_selection: str) -> set[int]:
    if layer_selection == "baseline":
        return set()
    if layer_selection == "all":
        return set(range(num_layers))
    if layer_selection == "last50":
        return set(range(num_layers // 2, num_layers))
    if layer_selection == "last25":
        return set(range((3 * num_layers) // 4, num_layers))
    raise ValueError(f"Unknown layer_selection: {layer_selection}")


def _language_layers(model: Any) -> Any:
    if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "model"):
        language_model = model.model.language_model.model
        if hasattr(language_model, "layers"):
            return language_model.layers
    raise AttributeError("Could not find PaliGemma language layers.")


def _summarize_attention_records(records: list[dict[str, float | int]], boosted_layers: set[int]) -> dict[str, Any]:
    mass_keys = sorted(
        {
            key
            for record in records
            for key in record
            if key.endswith("_attention_mass")
        }
    )
    all_masses = [float(record["image_attention_mass"]) for record in records if "image_attention_mass" in record]
    boosted_masses = [
        float(record["image_attention_mass"])
        for record in records
        if int(record["layer_idx"]) in boosted_layers and "image_attention_mass" in record
    ]
    by_layer: dict[int, list[float]] = {}
    for record in records:
        if "image_attention_mass" in record:
            by_layer.setdefault(int(record["layer_idx"]), []).append(float(record["image_attention_mass"]))

    summary = {
        "n_records": len(records),
        "mean_image_attention_mass": _mean(all_masses),
        "mean_boosted_layer_image_attention_mass": _mean(boosted_masses),
        "by_layer": {
            str(layer_idx): {
                "n_records": len(values),
                "mean_image_attention_mass": _mean(values),
            }
            for layer_idx, values in sorted(by_layer.items())
        },
    }
    for key in mass_keys:
        summary[f"mean_{key}"] = _mean([float(record[key]) for record in records if key in record])
    return summary


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
