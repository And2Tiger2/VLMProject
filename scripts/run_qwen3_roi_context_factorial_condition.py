#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapters.qwen_gaze_factory import make_roi_attention_adapter
from scripts.run_qwen3_roi_attention_condition import MODEL_ID, run_condition
from vlm_eval.qwen3_roi_attention import DEFAULT_GAZE_RANKING
from vlm_eval.qwen3_roi_context_factorial import DEFAULT_RUN_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Qwen3 ROI/context factorial arm."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--gaze-ranking", type=Path, default=DEFAULT_GAZE_RANKING)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("stage") != "factorial":
        raise SystemExit("Expected a factorial-stage manifest")
    if not 0 <= args.task_index < len(manifest["conditions"]):
        raise SystemExit("Task index is outside the factorial manifest")
    condition = manifest["conditions"][args.task_index]
    adapter = make_roi_attention_adapter(
        model_id=MODEL_ID,
        max_new_tokens=16,
        max_pixels=1048576,
        min_pixels=0,
        do_sample=False,
        temperature=None,
        seed=int(condition["seed"]),
        device_map=args.device_map,
        prompt_mode="baseline",
        attention_alpha=0.0,
        attention_controller="fixed",
        gaze_ranking_path=str(args.gaze_ranking),
        top_k_gaze=int(condition["head_count"]),
        head_selection=str(condition["head_selection"]),
        head_selection_seed=int(condition["head_seed"]),
        decode_only=False,
        roi_token_min_coverage=0.05,
        roi_attention_bias=float(condition["roi_attention_bias"]),
        context_attention_bias=float(condition["context_attention_bias"]),
    )
    out_dir = args.run_root / "factorial" / condition["name"]
    result = run_condition(
        condition=condition,
        stage="factorial",
        out_dir=out_dir,
        gaze_ranking=args.gaze_ranking,
        device_map=args.device_map,
        resume=args.resume,
        adapter=adapter,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
