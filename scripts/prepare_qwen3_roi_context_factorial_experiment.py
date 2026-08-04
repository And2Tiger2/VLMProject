#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm_eval.qwen3_high_bias_roi_attention import DEFAULT_ROI_ROOT, DEFAULT_VLMBIAS
from vlm_eval.qwen3_roi_context_factorial import (
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_STRENGTH,
    prepare_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the Qwen3 ROI/context 2x2 factorial."
    )
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--vlmbias", type=Path, default=DEFAULT_VLMBIAS)
    parser.add_argument("--roi-root", type=Path, default=DEFAULT_ROI_ROOT)
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    args = parser.parse_args()
    manifest = prepare_experiment(
        vlmbias_path=args.vlmbias,
        roi_root=args.roi_root,
        experiment_root=args.experiment_root,
        strength=args.strength,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
