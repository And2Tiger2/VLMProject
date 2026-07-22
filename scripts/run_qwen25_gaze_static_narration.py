from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from adapters.qwen_gaze_factory import QWEN3_GAZE_MODEL, make_panel_gaze_adapter
from vlm_eval.gaze_comics import DEFAULT_N_PANELS, build_strip, list_comic_dirs
from vlm_eval.gaze_resume import ensure_resume_config, load_completed_keys, row_key


PROMPT = "What is happening in this panel of the comic strip?"
DEFAULT_DECODE_ONLY = False
PAPER_CONTROL_MIN_LAYER = 20
PAPER_CONTROL_MAX_LAYER = 35


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
    backfill: bool = False,
    min_layer: int = 0,
    max_layer: int | None = None,
) -> list[tuple[int, int]]:
    if max_layer is None:
        max_layer = n_layers - 1
    if min_layer < 0 or max_layer >= n_layers or min_layer > max_layer:
        raise ValueError(
            f"invalid layer range [{min_layer}, {max_layer}] for {n_layers} layers"
        )
    preferred = []
    fallback = []
    for layer_idx in range(min_layer, max_layer + 1):
        for head_idx in range(n_heads):
            head = (layer_idx, head_idx)
            if head in exclude:
                continue
            score = float(scores[layer_idx, head_idx]) if scores is not None else None
            if score is not None and max_score is not None and score > max_score:
                fallback.append((score, head))
            else:
                preferred.append(head)
    import torch

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(preferred), generator=generator).tolist()
    preferred = [preferred[index] for index in permutation]
    if len(preferred) >= n_select:
        return preferred[:n_select]
    if not backfill or scores is None:
        return preferred[: min(n_select, len(preferred))]
    fallback = sorted(fallback, key=lambda item: item[0])
    candidates = [*preferred, *[head for _, head in fallback]]
    return candidates[: min(n_select, len(candidates))]


def select_control_heads(
    *,
    control_mode: str,
    n_layers: int,
    n_heads: int,
    gaze_heads: list[tuple[int, int]],
    n_select: int,
    seed: int,
    gaze_scores: np.ndarray | None,
    nongaze_percentile: float,
) -> tuple[list[tuple[int, int]], float | None]:
    """Select the static-narration control heads.

    The paper control is exactly K uniformly sampled non-gaze heads from
    inclusive layers 20--35. ``bottom5`` preserves the current public
    repository's bottom-five-percent implementation as an explicit ablation.
    ``matched`` preserves the prior exact-K global low-score fallback.
    """
    cutoff = (
        float(np.percentile(gaze_scores, nongaze_percentile))
        if gaze_scores is not None
        else None
    )
    common = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "exclude": set(gaze_heads),
        "n_select": n_select,
        "seed": seed,
    }
    if control_mode == "paper":
        heads = sample_non_gaze_heads(
            **common,
            min_layer=PAPER_CONTROL_MIN_LAYER,
            max_layer=PAPER_CONTROL_MAX_LAYER,
        )
        if len(heads) != n_select:
            raise RuntimeError(
                f"paper control requested {n_select} heads but only {len(heads)} are available "
                f"in layers {PAPER_CONTROL_MIN_LAYER}--{PAPER_CONTROL_MAX_LAYER}"
            )
        return heads, None
    if gaze_scores is None:
        raise FileNotFoundError(
            f"control mode {control_mode!r} requires gaze_scores.npy next to the ranking"
        )
    heads = sample_non_gaze_heads(
        **common,
        scores=gaze_scores,
        max_score=cutoff,
        backfill=control_mode == "matched",
    )
    return heads, cutoff


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen-VL static Gaze Heads steering generations.")
    parser.add_argument("--comics-root", default="segments/gaze_heads_qwen3_8b/data/eval_comics")
    parser.add_argument(
        "--gaze-ranking",
        default="segments/gaze_heads_qwen3_8b/runs/gaze_discovery_seed42_merged/gaze_head_ranking.json",
    )
    parser.add_argument("--out-dir", default="segments/gaze_heads_qwen3_8b/runs/static_narration")
    parser.add_argument("--model-id", default=QWEN3_GAZE_MODEL)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-comics", type=int, default=0)
    parser.add_argument("--start-comic-idx", type=int, default=0)
    parser.add_argument("--comic-name", default="")
    parser.add_argument("--top-k-gaze", type=int, default=100)
    parser.add_argument("--top-k-random", type=int, default=100)
    parser.add_argument("--nongaze-percentile", type=float, default=5.0)
    parser.add_argument(
        "--control-mode",
        choices=["paper", "bottom5", "matched"],
        default="paper",
        help=(
            "paper samples exactly K non-gaze heads uniformly from layers 20--35; bottom5 "
            "preserves the public repository's bottom-score-percentile ablation; matched "
            "backfills that low-score pool to exactly K."
        ),
    )
    parser.add_argument(
        "--condition-set",
        choices=["both", "gaze", "control"],
        default="both",
        help="Generate both conditions, gaze only, or control only.",
    )
    parser.add_argument("--targets-per-strip", type=int, default=DEFAULT_N_PANELS)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--swap-bias", type=float, default=10000.0)
    parser.add_argument(
        "--decode-only",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DECODE_ONLY,
        help=(
            "Apply steering only during autoregressive decoding. The default is full-sequence "
            "steering (prefill and decode), matching the official static-narration protocol."
        ),
    )
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
    scores_path = ranking_path.parent / "gaze_scores.npy"
    gaze_scores = np.load(scores_path) if scores_path.exists() else None
    random_heads, cutoff = select_control_heads(
        control_mode=args.control_mode,
        n_layers=n_layers,
        n_heads=n_heads,
        gaze_heads=gaze_heads,
        n_select=args.top_k_random,
        seed=args.seed,
        gaze_scores=gaze_scores,
        nongaze_percentile=args.nongaze_percentile,
    )

    control_label = f"non_gaze_{args.control_mode}_{len(random_heads)}"
    all_conditions = {
        f"gaze_top{args.top_k_gaze}": gaze_heads,
        control_label: random_heads,
    }
    if args.condition_set == "both":
        conditions = all_conditions
    elif args.condition_set == "gaze":
        conditions = {f"gaze_top{args.top_k_gaze}": gaze_heads}
    else:
        conditions = {control_label: random_heads}
    if args.include_all_heads:
        conditions["all_heads"] = [(layer_idx, head_idx) for layer_idx in range(n_layers) for head_idx in range(n_heads)]

    experiment_config = {
        "task": "static_narration",
        "model_id": args.model_id,
        "comics_root": str(Path(args.comics_root)),
        "gaze_ranking": str(ranking_path),
        "start_comic_idx": args.start_comic_idx,
        "max_comics": args.max_comics,
        "comic_name": args.comic_name,
        "top_k_gaze": args.top_k_gaze,
        "top_k_random": args.top_k_random,
        "nongaze_percentile": args.nongaze_percentile,
        "control_mode": args.control_mode,
        "paper_control_layers": [PAPER_CONTROL_MIN_LAYER, PAPER_CONTROL_MAX_LAYER],
        "condition_set": args.condition_set,
        "condition_labels": sorted(conditions),
        "selected_gaze_heads": [list(head) for head in gaze_heads],
        "selected_control_heads": [list(head) for head in random_heads],
        "targets_per_strip": args.targets_per_strip,
        "max_new_tokens": args.max_new_tokens,
        "swap_bias": args.swap_bias,
        "decode_only": args.decode_only,
        "include_all_heads": args.include_all_heads,
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
        "seed": args.seed,
        "gap": args.gap,
        "prompt": PROMPT,
        "git_commit": _git_commit(),
    }
    ensure_resume_config(out_dir, experiment_config, resume=args.resume, artifact_name="generations.jsonl")
    provenance = {
        "git_commit": experiment_config["git_commit"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

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
                    handle.flush()
                    completed.add(row_key(row_stub, key_fields))

    summary = {
        "model_id": args.model_id,
        "comics_root": args.comics_root,
        "gaze_ranking": str(ranking_path),
        "start_comic_idx": args.start_comic_idx,
        "n_comics": len(comic_dirs),
        "conditions": sorted(conditions),
        "condition_set": args.condition_set,
        "selected_control_heads": [list(head) for head in random_heads],
        "nongaze_score_cutoff": cutoff,
        "control_mode": args.control_mode,
        "decode_only": args.decode_only,
        "swap_bias": args.swap_bias,
        "experiment_config": str(out_dir / "experiment_config.json"),
        "generations": str(generations_path),
        "resume": args.resume,
        "skipped_existing_rows": skipped,
        "note": "This script writes generation records. Judge scoring can be applied later without rerunning Qwen.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote generations to {generations_path}")


if __name__ == "__main__":
    main()
