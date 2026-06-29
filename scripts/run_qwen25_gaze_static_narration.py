from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from adapters.qwen25_vl_gaze import Qwen25VLGazeAdapter
from vlm_eval.gaze_comics import DEFAULT_N_PANELS, build_strip, list_comic_dirs
from vlm_eval.gaze_resume import load_completed_keys, row_key


PROMPT = "What is happening in this panel of the comic strip?"


def load_head_ranking(path: Path, top_k: int) -> list[tuple[int, int]]:
    ranking = json.loads(path.read_text())
    return [(int(row["layer"]), int(row["head"])) for row in ranking[:top_k]]


def sample_non_gaze_heads(
    n_layers: int,
    n_heads: int,
    exclude: set[tuple[int, int]],
    n_select: int,
    seed: int,
    scores: np.ndarray | None = None,
    max_score: float | None = None,
) -> list[tuple[int, int]]:
    preferred = []
    fallback = []
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            head = (layer_idx, head_idx)
            if head in exclude:
                continue
            score = float(scores[layer_idx, head_idx]) if scores is not None else None
            if score is not None and max_score is not None and score > max_score:
                fallback.append((score, head))
            else:
                preferred.append(head)
    rng = np.random.RandomState(seed)
    rng.shuffle(preferred)
    if len(preferred) >= n_select:
        return preferred[:n_select]
    if scores is None:
        return preferred[: min(n_select, len(preferred))]
    fallback = sorted(fallback, key=lambda item: item[0])
    candidates = [*preferred, *[head for _, head in fallback]]
    return candidates[: min(n_select, len(candidates))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL static GazeHeads steering generations.")
    parser.add_argument("--comics-root", default="segments/gaze_heads_qwen25/data/comics")
    parser.add_argument("--gaze-ranking", default="segments/gaze_heads_qwen25/runs/gaze_discovery/gaze_head_ranking.json")
    parser.add_argument("--out-dir", default="segments/gaze_heads_qwen25/runs/static_narration")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-comics", type=int, default=0)
    parser.add_argument("--start-comic-idx", type=int, default=0)
    parser.add_argument("--comic-name", default="")
    parser.add_argument("--top-k-gaze", type=int, default=100)
    parser.add_argument("--top-k-random", type=int, default=100)
    parser.add_argument("--nongaze-percentile", type=float, default=5.0)
    parser.add_argument("--targets-per-strip", type=int, default=DEFAULT_N_PANELS)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--swap-bias", type=float, default=10000.0)
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--include-all-heads", action="store_true")
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
    n_layers, n_heads, _ = adapter.model_dims()

    gaze_heads = load_head_ranking(ranking_path, args.top_k_gaze)
    scores_path = ranking_path.parent / "gaze_scores.npy"
    gaze_scores = np.load(scores_path) if scores_path.exists() else None
    cutoff = float(np.percentile(gaze_scores, args.nongaze_percentile)) if gaze_scores is not None else None
    random_heads = sample_non_gaze_heads(
        n_layers=n_layers,
        n_heads=n_heads,
        exclude=set(gaze_heads),
        n_select=args.top_k_random,
        seed=args.seed,
        scores=gaze_scores,
        max_score=cutoff,
    )

    conditions = {
        f"gaze_top{args.top_k_gaze}": gaze_heads,
        f"non_gaze_{len(random_heads)}": random_heads,
    }
    if args.include_all_heads:
        conditions["all_heads"] = [(layer_idx, head_idx) for layer_idx in range(n_layers) for head_idx in range(n_heads)]

    rng = np.random.RandomState(args.seed)
    generations_path = out_dir / "generations.jsonl"
    key_fields = ["strip_name", "condition", "target_panel"]
    completed = load_completed_keys(generations_path, key_fields) if args.resume else set()
    skipped = 0
    mode = "a" if args.resume else "w"
    with generations_path.open(mode, encoding="utf-8") as handle:
        for comic_dir in tqdm(comic_dirs, desc="Static gaze steering"):
            strip = build_strip(comic_dir, n_panels=DEFAULT_N_PANELS, gap=args.gap)
            inputs = adapter.prepare_inputs(strip.strip, PROMPT)
            panel_masks = adapter.panel_token_masks(inputs, strip.panel_widths, DEFAULT_N_PANELS)
            baseline = adapter.generate_unsteered(inputs, max_new_tokens=args.max_new_tokens)

            if args.targets_per_strip >= DEFAULT_N_PANELS:
                targets = list(range(DEFAULT_N_PANELS))
            else:
                targets = rng.choice(DEFAULT_N_PANELS, size=max(1, args.targets_per_strip), replace=False).tolist()

            for condition, heads in conditions.items():
                for target_panel in targets:
                    row_stub = {
                        "strip_name": strip.name,
                        "condition": condition,
                        "target_panel": target_panel + 1,
                    }
                    if row_key(row_stub, key_fields) in completed:
                        skipped += 1
                        continue
                    text = adapter.generate_steered(
                        inputs,
                        panel_masks,
                        heads,
                        target_panel,
                        max_new_tokens=args.max_new_tokens,
                        swap_bias=args.swap_bias,
                        decode_only=args.decode_only,
                    )
                    handle.write(
                        json.dumps(
                            {
                                **row_stub,
                                "strip_name": strip.name,
                                "comic_dir": str(comic_dir),
                                "prompt": PROMPT,
                                "baseline_text": baseline,
                                "generated_text": text,
                                "metadata": adapter.last_generation_metadata,
                            }
                        )
                        + "\n"
                    )
                    completed.add(row_key(row_stub, key_fields))

    summary = {
        "model_id": args.model_id,
        "comics_root": args.comics_root,
        "gaze_ranking": str(ranking_path),
        "start_comic_idx": args.start_comic_idx,
        "n_comics": len(comic_dirs),
        "conditions": sorted(conditions),
        "nongaze_score_cutoff": cutoff,
        "generations": str(generations_path),
        "resume": args.resume,
        "skipped_existing_rows": skipped,
        "note": "This script writes generation records. Judge scoring can be applied later without rerunning Qwen.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote generations to {generations_path}")


if __name__ == "__main__":
    main()
