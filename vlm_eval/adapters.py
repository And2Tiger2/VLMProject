from __future__ import annotations

import importlib

from vlm_eval.types import VLMAdapter


def load_adapter(spec: str) -> VLMAdapter:
    """Load an adapter from `module:factory` or `module:factory,arg=value`."""
    target, _, arg_text = spec.partition(",")
    module_name, sep, factory_name = target.partition(":")
    if not sep:
        raise ValueError("Adapter spec must look like `module:factory`.")

    kwargs = {}
    if arg_text:
        for pair in arg_text.split(","):
            key, value = pair.split("=", 1)
            kwargs[key] = _coerce_value(value)

    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    adapter = factory(**kwargs)
    if not hasattr(adapter, "generate"):
        raise TypeError(f"Adapter {spec!r} does not define generate(example).")
    return adapter


def _coerce_value(value: str):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        return value

