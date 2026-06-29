from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from adapters.qwen25_vl_gaze import Qwen25VLGazeAdapter
from scripts.run_qwen25_gaze_static_narration import load_head_ranking
from vlm_eval.gaze_comics import DEFAULT_N_PANELS, build_strip, list_comic_dirs
from vlm_eval.gaze_resume import load_completed_keys, row_key


PROMPT = (
    "Please describe what happens in each panel, in order. "
    "Start each panel description with 'Panel 1:', 'Panel 2:', and so on."
)


def deranged_schedule(n_panels: int, segment_tokens: int, rng: np.random.RandomState) -> list[tuple[int, int]]:
    default = list(range(n_panels))
    while True:
        order = default[:]
        rng.shuffle(order)
        if order != default:
            break
    return [(idx * segment_tokens, panel_idx) for idx, panel_idx in enumerate(order)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL dynamic GazeHeads narration steering.")
    parser.add_argument("--comics-root", default="segments/gaze_heads_qwen25/data/comics")
    parser.add_argument("--gaze-ranking", default="segments/gaze_heads_qwen25/runs/gaze_discovery/gaze_head_ranking.json")
    parser.add_argument("--out-dir", default="segments/gaze_heads_qwen25/runs/dynamic_narration")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-comics", type=int, default=0)
    parser.add_argument("--start-comic-idx", type=int, default=0)
    parser.add_argument("--comic-name", default="")
    parser.add_argument("--top-k-gaze", type=int, default=100)
    parser.add_argument("--segment-tokens", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--swap-bias", type=float, default=10000.0)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--min-pixels", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gap", type=int, default=6)
    parser.add_argument("--resume", action="store_true", help="Append to generations.jsonl and skip completed rows.")
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

    adapter = Qwen25VLGazeAdapter(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        do_sample=False,
        temperature=None,
        device_map=args.device_map,
        prompt_mode="raw",
    )
    gaze_heads = load_head_ranking(ranking_path, args.top_k_gaze)
    rng = np.random.RandomState(args.seed)

    generations_path = out_dir / "generations.jsonl"
    key_fields = ["strip_name", "condition"]
    completed = load_completed_keys(generations_path, key_fields) if args.resume else set()
    skipped = 0
    mode = "a" if args.resume else "w"
    with generations_path.open(mode, encoding="utf-8") as handle:
        for comic_dir in tqdm(comic_dirs, desc="Dynamic gaze steering"):
            strip = build_strip(comic_dir, n_panels=DEFAULT_N_PANELS, gap=args.gap)
            condition = f"gaze_top{args.top_k_gaze}"
            row_stub = {"strip_name": strip.name, "condition": condition}
            if row_key(row_stub, key_fields) in completed:
                skipped += 1
                continue
            inputs = adapter.prepare_inputs(strip.strip, PROMPT)
            panel_masks = adapter.panel_token_masks(inputs, strip.panel_widths, DEFAULT_N_PANELS)
            baseline = adapter.generate_unsteered(inputs, max_new_tokens=args.max_new_tokens)
            schedule = deranged_schedule(DEFAULT_N_PANELS, args.segment_tokens, rng)
            text = adapter.generate_steered_dynamic(
                inputs,
                panel_masks,
                gaze_heads,
                schedule,
                max_new_tokens=args.max_new_tokens,
                swap_bias=args.swap_bias,
            )
            handle.write(
                json.dumps(
                    {
                        "task": "dynamic_narration",
                        **row_stub,
                        "strip_name": strip.name,
                        "comic_dir": str(comic_dir),
                        "prompt": PROMPT,
                        "baseline_text": baseline,
                        "generated_text": text,
                        "schedule": [
                            {"start_decode_step": start_step, "target_panel": panel_idx + 1}
                            for start_step, panel_idx in schedule
                        ],
                        "segment_tokens": args.segment_tokens,
                        "metadata": adapter.last_generation_metadata,
                    }
                )
                + "\n"
            )
            completed.add(row_key(row_stub, key_fields))

    summary = {
        "task": "dynamic_narration",
        "model_id": args.model_id,
        "comics_root": args.comics_root,
        "gaze_ranking": str(ranking_path),
        "start_comic_idx": args.start_comic_idx,
        "n_comics": len(comic_dirs),
        "top_k_gaze": args.top_k_gaze,
        "segment_tokens": args.segment_tokens,
        "generations": str(generations_path),
        "resume": args.resume,
        "skipped_existing_rows": skipped,
        "note": "Dynamic scoring needs segment-level panel matching; this file records schedule/text for later judging.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote generations to {generations_path}")


if __name__ == "__main__":
    main()
