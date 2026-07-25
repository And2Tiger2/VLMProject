#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm_eval.qwen3_gaze_specificity import (
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_REPORT_ROOT,
    SOURCE_REPORT_ROOT,
    methodology,
    prepare_stage,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare one rigorous Qwen3 gaze-specificity stage."
    )
    parser.add_argument(
        "stage", choices=["repair", "controls", "tune", "final", "robustness"]
    )
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--source-report-root", type=Path, default=SOURCE_REPORT_ROOT)
    parser.add_argument(
        "--vlmbias",
        type=Path,
        default=Path("segments/vlm_bias_attention/data/vlmbias_400.jsonl"),
    )
    parser.add_argument(
        "--naturalbench",
        type=Path,
        default=Path("segments/vlm_bias_attention/data/naturalbench_100_groups.jsonl"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    args.experiment_root.mkdir(parents=True, exist_ok=True)
    (args.experiment_root / "methodology.json").write_text(
        json.dumps(methodology(), indent=2), encoding="utf-8"
    )
    manifest = prepare_stage(
        stage=args.stage,
        vlmbias_path=args.vlmbias,
        naturalbench_path=args.naturalbench,
        experiment_root=args.experiment_root,
        source_report_root=args.source_report_root,
        report_root=args.report_root,
        seeds=args.seeds,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
