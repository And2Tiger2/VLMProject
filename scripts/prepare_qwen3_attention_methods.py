#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm_eval.qwen3_attention_methods import (
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_REPORT_ROOT,
    confirmation_conditions,
    controller_conditions,
    head_conditions,
    methodology,
    prepare_stratified_splits,
    robustness_conditions,
    smoke_conditions,
    write_condition_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic splits and one stage manifest."
    )
    parser.add_argument(
        "stage", choices=["smoke", "controller", "heads", "confirm", "robustness"]
    )
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--vlmbias",
        type=Path,
        default=Path("segments/vlm_bias_attention/data/vlmbias_400.jsonl"),
    )
    parser.add_argument(
        "--naturalbench",
        type=Path,
        default=Path(
            "segments/vlm_bias_attention/data/naturalbench_100_groups.jsonl"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    split_manifest = prepare_stratified_splits(
        vlmbias_path=args.vlmbias,
        naturalbench_path=args.naturalbench,
        out_dir=args.experiment_root,
    )
    (args.experiment_root / "methodology.json").write_text(
        json.dumps(methodology(), indent=2), encoding="utf-8"
    )
    if args.stage == "smoke":
        conditions = smoke_conditions()
        split = "smoke"
    elif args.stage == "controller":
        conditions = controller_conditions()
        split = "dev"
    else:
        controller_selection = _read_json(
            args.report_root / "controller" / "selection.json"
        )
        if args.stage == "heads":
            conditions = head_conditions(
                controller_selection["selected_overall"]["spec"]
            )
            split = "dev"
        else:
            head_selection = _read_json(
                args.report_root / "heads" / "selection.json"
            )
            if args.stage == "confirm":
                conditions = confirmation_conditions(
                    controller_selection, head_selection
                )
                split = "confirm"
            else:
                conditions = robustness_conditions(
                    controller_selection, head_selection, args.seeds
                )
                split = "all"
    manifest = write_condition_manifest(
        stage=args.stage,
        conditions=conditions,
        split_manifest=split_manifest,
        out_path=args.experiment_root / f"{args.stage}_manifest.json",
        split=split,
    )
    print(json.dumps(manifest, indent=2))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
