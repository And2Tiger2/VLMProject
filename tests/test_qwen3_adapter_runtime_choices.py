from __future__ import annotations

from adapters.qwen25_vl import _resolve_torch_dtype
from adapters.qwen3_vl import prepare_qwen3_inputs


class _Cuda:
    def __init__(self, *, available: bool, bf16: bool) -> None:
        self._available = available
        self._bf16 = bf16

    def is_available(self) -> bool:
        return self._available

    def is_bf16_supported(self) -> bool:
        return self._bf16


class _Torch:
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"

    def __init__(self, *, available: bool, bf16: bool) -> None:
        self.cuda = _Cuda(available=available, bf16=bf16)


class _Batch:
    def __init__(self) -> None:
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _Processor:
    def __init__(self) -> None:
        self.args = None
        self.kwargs = None
        self.batch = _Batch()

    def apply_chat_template(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self.batch


def test_dtype_uses_fp16_when_neuronic_gpu_has_no_bf16() -> None:
    assert _resolve_torch_dtype(_Torch(available=True, bf16=False)) == "float16"
    assert _resolve_torch_dtype(_Torch(available=True, bf16=True)) == "bfloat16"
    assert _resolve_torch_dtype(_Torch(available=False, bf16=False)) == "float32"


def test_qwen3_input_path_matches_native_reference_chat_template() -> None:
    processor = _Processor()
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

    batch = prepare_qwen3_inputs(processor, messages, "cuda:0")

    assert processor.args == ([messages],)
    assert processor.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    assert batch.device == "cuda:0"
