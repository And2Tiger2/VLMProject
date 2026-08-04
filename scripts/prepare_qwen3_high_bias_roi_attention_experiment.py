#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm_eval.qwen3_high_bias_roi_attention import (
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_REPORT_ROOT,
    DEFAULT_ROI_ROOT,
    DEFAULT_VLMBIAS,
    confirm_conditions,
    head_conditions,
    methodology,
    prepare_splits,
    smoke_conditions,
    tune_conditions,
    write_stage_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare one high-bias-topic Qwen3 VLMBias ROI-attention stage."
    )
    parser.add_argument("stage", choices=["smoke", "tune", "heads", "confirm"])
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--vlmbias", type=Path, default=DEFAULT_VLMBIAS)
    parser.add_argument("--roi-root", type=Path, default=DEFAULT_ROI_ROOT)
    parser.add_argument("--dev-groups", type=int, default=40)
    parser.add_argument("--smoke-groups", type=int, default=8)
    args = parser.parse_args()

    splits = prepare_splits(
        vlmbias_path=args.vlmbias,
        roi_root=args.roi_root,
        out_dir=args.experiment_root,
        dev_groups=args.dev_groups,
        smoke_groups=args.smoke_groups,
    )
    (args.experiment_root / "methodology.json").write_text(
        json.dumps(methodology(), indent=2) + "\n", encoding="utf-8"
    )
    if args.stage == "smoke":
        conditions, split = smoke_conditions(), "smoke"
    elif args.stage == "tune":
        conditions, split = tune_conditions(), "dev"
    elif args.stage == "heads":
        tune = _read_json(args.report_root / "tune" / "selection.json")
        selected = tune["selected"]["spec"]
        conditions, split = (
            head_conditions(float(selected["alpha"]), str(selected["mask_variant"])),
            "dev",
        )
    else:
        tune = _read_json(args.report_root / "tune" / "selection.json")
        heads = _read_json(args.report_root / "heads" / "selection.json")
        conditions = confirm_conditions(
            tune["selected"]["spec"], heads["selected"]["spec"]
        )
        split = "confirm"
    manifest = write_stage_manifest(
        stage=args.stage,
        conditions=conditions,
        split=split,
        split_manifest=splits,
        path=args.experiment_root / f"{args.stage}_manifest.json",
    )
    print(json.dumps(manifest, indent=2))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
