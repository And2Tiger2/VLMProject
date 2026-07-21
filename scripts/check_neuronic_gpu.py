from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail fast if a Slurm GPU is unusable for Qwen3-VL 8B eager attention.")
    parser.add_argument("--min-memory-gb", type=float, default=20.0)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit(
            f"Expected exactly one allocated CUDA device; available={torch.cuda.is_available()} "
            f"count={torch.cuda.device_count()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )
    properties = torch.cuda.get_device_properties(0)
    memory_gb = float(properties.total_memory) / (1024**3)
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
    report = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": properties.name,
        "total_memory_gb": memory_gb,
        "bf16_supported": bf16_supported,
        "compute_dtype": str(compute_dtype),
        "torch": torch.__version__,
    }
    print(json.dumps(report, indent=2), flush=True)
    if memory_gb < args.min_memory_gb:
        raise SystemExit(
            f"Allocated GPU has {memory_gb:.1f} GiB; require at least {args.min_memory_gb:.1f} GiB. "
            "Resubmit with an appropriate Neuronic constraint or explicitly lower --min-gpu-memory-gb."
        )
    # Touch the device immediately so GPU-usage watchdogs see real work before
    # the larger model is deserialized from shared storage.
    tensor = torch.randn((1024, 1024), device="cuda", dtype=compute_dtype)
    result = tensor @ tensor
    torch.cuda.synchronize()
    print(f"cuda_smoke_mean={result.float().mean().item():.6f}", flush=True)


if __name__ == "__main__":
    main()
