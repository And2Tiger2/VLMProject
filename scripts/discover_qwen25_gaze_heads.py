from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from adapters.qwen25_vl_gaze import Qwen25VLGazeAdapter
from vlm_eval.gaze_comics import DEFAULT_N_PANELS, build_strip, list_comic_dirs


NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def panel_query_prompt(panel_index: int, n_panels: int = DEFAULT_N_PANELS) -> str:
    count = NUMBER_WORDS.get(n_panels, str(n_panels))
    return (
        f"Look carefully at this {count}-panel comic strip. "
        f"What is happening in the {ordinal(panel_index)} panel from the left? "
        "Answer briefly."
    )


def ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def rank_heads_by_score(scores: np.ndarray) -> list[dict[str, float]]:
    ranked = []
    for layer_idx in range(scores.shape[0]):
        for head_idx in range(scores.shape[1]):
            ranked.append({"layer": int(layer_idx), "head": int(head_idx), "score": float(scores[layer_idx, head_idx])})
    ranked.sort(key=lambda row: row["score"], reverse=True)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover Qwen2.5-VL gaze heads on six-panel comic strips.")
    parser.add_argument("--comics-root", default="segments/gaze_heads_qwen25/data/comics")
    parser.add_argument("--out-dir", default="segments/gaze_heads_qwen25/runs/gaze_discovery")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--start-comic-idx", type=int, default=0)
    parser.add_argument("--max-comics", type=int, default=0)
    parser.add_argument("--n-panels", type=int, default=DEFAULT_N_PANELS)
    parser.add_argument("--target-height", type=int, default=256)
    parser.add_argument("--gap", type=int, default=6)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--min-pixels", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comic_dirs = list_comic_dirs(Path(args.comics_root), n_panels=args.n_panels)
    if args.start_comic_idx > 0:
        comic_dirs = comic_dirs[args.start_comic_idx :]
    limit = args.max_comics if args.max_comics > 0 else args.n_samples
    comic_dirs = comic_dirs[:limit]
    if not comic_dirs:
        raise FileNotFoundError(
            f"No valid {args.n_panels}-panel comics found under {args.comics_root}. "
            "Run scripts/download_gaze_comics.py first or provide --comics-root."
        )

    adapter = Qwen25VLGazeAdapter(
        model_id=args.model_id,
        max_new_tokens=1,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        do_sample=False,
        temperature=None,
        device_map=args.device_map,
        prompt_mode="raw",
    )
    n_layers, n_heads, _ = adapter.model_dims()

    gaze_sum = np.zeros((args.n_panels, n_layers, n_heads, args.n_panels), dtype=np.float64)
    valid_samples = 0
    sampled_names = []

    for comic_dir in tqdm(comic_dirs, desc="Discovering gaze heads"):
        strip = build_strip(comic_dir, n_panels=args.n_panels, target_height=args.target_height, gap=args.gap)
        per_prompt = []
        ok = True
        for panel_idx in range(args.n_panels):
            prompt = panel_query_prompt(panel_idx + 1, n_panels=args.n_panels)
            try:
                inputs = adapter.prepare_inputs(strip.strip, prompt)
                panel_masks = adapter.panel_token_masks(inputs, strip.panel_widths, args.n_panels)
                per_prompt.append(adapter.collect_panel_attention(inputs, panel_masks))
            except Exception as exc:
                print(f"Skipping {strip.name} panel {panel_idx + 1}: {exc}", flush=True)
                ok = False
                break
        if not ok:
            continue
        gaze_sum += np.stack(per_prompt, axis=0)
        valid_samples += 1
        sampled_names.append(strip.name)

    if valid_samples == 0:
        raise RuntimeError("No valid comics were processed.")

    mean_panel_attention = gaze_sum / float(valid_samples)
    gaze_scores = np.zeros((n_layers, n_heads), dtype=np.float64)
    for panel_idx in range(args.n_panels):
        gaze_scores += mean_panel_attention[panel_idx, :, :, panel_idx]
    gaze_scores /= float(args.n_panels)

    ranking = rank_heads_by_score(gaze_scores)
    np.save(out_dir / "gaze_sum.npy", gaze_sum)
    np.save(out_dir / "mean_panel_attention.npy", mean_panel_attention)
    np.save(out_dir / "gaze_scores.npy", gaze_scores)
    (out_dir / "gaze_head_ranking.json").write_text(json.dumps(ranking, indent=2))
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "comics_root": args.comics_root,
                "start_comic_idx": args.start_comic_idx,
                "max_comics": args.max_comics,
                "n_requested": len(comic_dirs),
                "valid_samples": valid_samples,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "top_head": ranking[0],
                "sampled_names": sampled_names,
            },
            indent=2,
        )
    )
    print(f"Processed {valid_samples} comics.")
    print(f"Wrote gaze ranking to {out_dir / 'gaze_head_ranking.json'}")


if __name__ == "__main__":
    main()
