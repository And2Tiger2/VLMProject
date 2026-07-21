from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from adapters.qwen_gaze_factory import QWEN3_GAZE_MODEL, make_panel_gaze_adapter
from scripts.run_qwen25_gaze_static_narration import load_head_ranking, sample_non_gaze_heads
from vlm_eval.gaze_comics import DEFAULT_N_PANELS, build_strip, list_comic_dirs
from vlm_eval.gaze_resume import ensure_resume_config, load_completed_keys, row_key


PROMPT = (
    "Please describe what happens in each panel, in order. "
    "Start each panel description with 'Panel 1:', 'Panel 2:', and so on."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Track Qwen-VL gaze-head panel attention during narration.")
    parser.add_argument("--comics-root", default="segments/gaze_heads_qwen3_8b/data/eval_comics")
    parser.add_argument("--gaze-ranking", default="segments/gaze_heads_qwen3_8b/runs/gaze_discovery_seed42_merged/gaze_head_ranking.json")
    parser.add_argument("--out-dir", default="segments/gaze_heads_qwen3_8b/runs/narration_trajectory")
    parser.add_argument("--model-id", default=QWEN3_GAZE_MODEL)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-comics", type=int, default=5)
    parser.add_argument("--start-comic-idx", type=int, default=0)
    parser.add_argument("--comic-name", default="")
    parser.add_argument("--top-k-gaze", type=int, default=100)
    parser.add_argument("--control-heads", type=int, default=100)
    parser.add_argument("--nongaze-percentile", type=float, default=5.0)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--min-pixels", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gap", type=int, default=6)
    parser.add_argument("--resume", action="store_true", help="Append to trajectories.jsonl and skip completed rows.")
    args = parser.parse_args()

    ranking_path = Path(args.gaze_ranking)
    if not ranking_path.exists():
        raise FileNotFoundError(f"Missing gaze ranking at {ranking_path}. Run discover_qwen25_gaze_heads.py first.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comic_dirs = list_comic_dirs(Path(args.comics_root), n_panels=DEFAULT_N_PANELS)
    if args.comic_name:
        comic_dirs = [comic_dir for comic_dir in comic_dirs if comic_dir.name == args.comic_name]
    if args.start_comic_idx > 0:
        comic_dirs = comic_dirs[args.start_comic_idx :]
    if args.max_comics > 0:
        comic_dirs = comic_dirs[: args.max_comics]
    if not comic_dirs:
        raise FileNotFoundError(f"No valid six-panel comics found under {args.comics_root}.")

    adapter = make_panel_gaze_adapter(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        do_sample=False,
        temperature=None,
        device_map=args.device_map,
        prompt_mode="raw",
    )
    n_layers, n_heads, _ = adapter.model_dims()
    gaze_heads = load_head_ranking(ranking_path, args.top_k_gaze)
    gaze_scores_path = ranking_path.parent / "gaze_scores.npy"
    gaze_scores = np.load(gaze_scores_path) if gaze_scores_path.exists() else None
    cutoff = float(np.percentile(gaze_scores, args.nongaze_percentile)) if gaze_scores is not None else None
    control_heads = sample_non_gaze_heads(
        n_layers=n_layers,
        n_heads=n_heads,
        exclude=set(gaze_heads),
        n_select=args.control_heads,
        seed=args.seed,
        scores=gaze_scores,
        max_score=cutoff,
    )

    conditions = {
        f"gaze_top{args.top_k_gaze}": gaze_heads,
        f"non_gaze_{len(control_heads)}": control_heads,
    }
    experiment_config = {
        "task": "narration_trajectory",
        "model_id": args.model_id,
        "comics_root": str(Path(args.comics_root)),
        "gaze_ranking": str(ranking_path),
        "start_comic_idx": args.start_comic_idx,
        "max_comics": args.max_comics,
        "comic_name": args.comic_name,
        "top_k_gaze": args.top_k_gaze,
        "control_heads": args.control_heads,
        "nongaze_percentile": args.nongaze_percentile,
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
        "seed": args.seed,
        "gap": args.gap,
        "prompt": PROMPT,
    }
    ensure_resume_config(out_dir, experiment_config, resume=args.resume, artifact_name="trajectories.jsonl")
    trajectories_path = out_dir / "trajectories.jsonl"
    key_fields = ["strip_name", "condition"]
    completed = load_completed_keys(trajectories_path, key_fields) if args.resume else set()
    skipped = 0
    mode = "a" if args.resume else "w"
    with trajectories_path.open(mode, encoding="utf-8") as handle:
        for comic_dir in tqdm(comic_dirs, desc="Narration trajectories"):
            strip = build_strip(comic_dir, n_panels=DEFAULT_N_PANELS, gap=args.gap)
            inputs = adapter.prepare_inputs(strip.strip, PROMPT)
            panel_masks = adapter.panel_token_masks(inputs, strip.panel_widths, DEFAULT_N_PANELS)
            for condition, heads in conditions.items():
                row_stub = {"strip_name": strip.name, "condition": condition}
                if row_key(row_stub, key_fields) in completed:
                    skipped += 1
                    continue
                text, trajectory = adapter.generate_with_panel_tracking(
                    inputs,
                    panel_masks,
                    heads,
                    max_new_tokens=args.max_new_tokens,
                )
                handle.write(
                    json.dumps(
                        {
                            "task": "narration_trajectory",
                            **row_stub,
                            "strip_name": strip.name,
                            "comic_dir": str(comic_dir),
                            "prompt": PROMPT,
                            "generated_text": text,
                            "trajectory": trajectory,
                        }
                    )
                    + "\n"
                )
                handle.flush()
                completed.add(row_key(row_stub, key_fields))

    summary = {
        "task": "narration_trajectory",
        "model_id": args.model_id,
        "comics_root": args.comics_root,
        "gaze_ranking": str(ranking_path),
        "start_comic_idx": args.start_comic_idx,
        "n_comics": len(comic_dirs),
        "conditions": sorted(conditions),
        "nongaze_score_cutoff": cutoff,
        "experiment_config": str(out_dir / "experiment_config.json"),
        "trajectories": str(trajectories_path),
        "resume": args.resume,
        "skipped_existing_rows": skipped,
        "note": "Trajectory panel masses are averaged across selected heads and layers for each decode step.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote trajectories to {trajectories_path}")


if __name__ == "__main__":
    main()
