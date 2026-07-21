from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from vlm_eval.gaze_comics import DEFAULT_N_PANELS, build_strip
from vlm_eval.gaze_judge import (
    DEFAULT_JUDGE_MODEL,
    aggregate_dynamic_judgments,
    aggregate_judgments,
    judge_match_target_panel_anthropic,
    split_dynamic_segments,
    steered_matches_baseline,
)
from vlm_eval.gaze_resume import ensure_resume_config, load_completed_keys, row_key


STATIC_KEY_FIELDS = ["strip_name", "condition", "target_panel"]
DYNAMIC_KEY_FIELDS = ["strip_name", "condition"]
JUDGMENT_SCHEMA_VERSION = 2


def _score_dynamic_row(row: dict, args: argparse.Namespace, strip_cache: dict[Path, object]) -> dict:
    schedule = row.get("schedule") or []
    target_panels = [int(item["target_panel"]) for item in schedule]
    if not target_panels:
        raise ValueError(f"Dynamic row for {row.get('strip_name')} is missing schedule targets.")

    explicit_segments = row.get("segments")
    if isinstance(explicit_segments, list) and explicit_segments:
        segments = [str(segment) for segment in explicit_segments]
    else:
        segments = split_dynamic_segments(row.get("generated_text", ""), len(target_panels))
    segments = segments[: len(target_panels)]
    if len(segments) < len(target_panels):
        segments.extend([""] * (len(target_panels) - len(segments)))

    segment_judgments = []
    if args.judge == "anthropic":
        comic_dir = Path(row["comic_dir"])
        if comic_dir not in strip_cache:
            strip_cache[comic_dir] = build_strip(comic_dir, n_panels=args.n_panels)
        strip = strip_cache[comic_dir]
    else:
        strip = None

    for segment_idx, (segment_text, target_panel) in enumerate(zip(segments, target_panels)):
        if args.judge == "baseline-only":
            judgment = {
                "matched_panel": None,
                "is_junk": False,
                "correct": False,
                "matches_baseline": False,
                "reasoning": "baseline-only mode does not assign dynamic segment panel matches",
            }
        else:
            judgment = judge_match_target_panel_anthropic(
                strip_image=strip.strip,
                segment_text=segment_text,
                baseline_text=None,
                target_panel=target_panel,
                n_panels=args.n_panels,
                model_name=args.claude_model,
                treat_baseline_match_as_miss=False,
            )
        judgment = dict(judgment)
        judgment.update(
            {
                "segment_index": segment_idx,
                "target_panel": target_panel,
                "segment_text": segment_text,
            }
        )
        segment_judgments.append(judgment)

    out = dict(row)
    out["segments"] = segments
    out["segment_judgments"] = segment_judgments
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score GazeHeads generation JSONL with panel-match judgments.")
    parser.add_argument("--generations", default="segments/gaze_heads_qwen25/runs/static_narration/generations.jsonl")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--judge", choices=["anthropic", "baseline-only"], default="baseline-only")
    parser.add_argument("--claude-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--n-panels", type=int, default=DEFAULT_N_PANELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Append to judgments.jsonl and skip completed rows.")
    args = parser.parse_args()

    generations_path = Path(args.generations)
    out_dir = Path(args.out_dir) if args.out_dir else generations_path.parent

    result = score_generations(
        generations_path=generations_path,
        out_dir=out_dir,
        judge=args.judge,
        claude_model=args.claude_model,
        n_panels=args.n_panels,
        seed=args.seed,
        resume=args.resume,
    )
    print(f"Wrote judgments to {result['judgments']}")
    print(f"Wrote aggregate results to {result['aggregate']}")


def score_generations(
    *,
    generations_path: Path,
    out_dir: Path,
    judge: str = "baseline-only",
    claude_model: str = DEFAULT_JUDGE_MODEL,
    n_panels: int = DEFAULT_N_PANELS,
    seed: int = 42,
    resume: bool = False,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in generations_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    judged_path = out_dir / "judgments.jsonl"
    ensure_resume_config(
        out_dir,
        _score_config(generations_path, judge, claude_model, n_panels, seed),
        resume=resume,
        artifact_name="judgments.jsonl",
        config_name="judgment_config.json",
    )
    is_dynamic_run = bool(rows) and all(row.get("task") == "dynamic_narration" for row in rows)
    key_fields = DYNAMIC_KEY_FIELDS if is_dynamic_run else STATIC_KEY_FIELDS
    existing_rows = _read_jsonl(judged_path) if resume and judged_path.exists() else []
    completed = load_completed_keys(judged_path, key_fields) if resume else set()
    new_rows = []
    strip_cache = {}
    score_args = argparse.Namespace(judge=judge, claude_model=claude_model, n_panels=n_panels)

    mode = "a" if resume else "w"
    with judged_path.open(mode, encoding="utf-8") as handle:
        for row in tqdm(rows, desc="Scoring generations"):
            if row_key(row, key_fields) in completed:
                continue
            if row.get("task") == "dynamic_narration":
                judged_row = _score_dynamic_row(row, score_args, strip_cache)
            else:
                judged_row = _score_static_or_vqa_row(row, score_args, strip_cache)
            handle.write(json.dumps(judged_row) + "\n")
            handle.flush()
            new_rows.append(judged_row)
            completed.add(row_key(judged_row, key_fields))

    judged_rows = [*existing_rows, *new_rows] if resume else new_rows
    aggregate_key, aggregate = _aggregate_judged_rows(judged_rows, seed=seed)
    aggregate_path = out_dir / "aggregate_results.json"
    aggregate_path.write_text(
        json.dumps(
            {
                "judge": judge,
                "generations": str(generations_path),
                "resume": resume,
                "n_rows": len(judged_rows),
                "n_new_rows": len(new_rows),
                "n_skipped_existing_rows": len(rows) - len(new_rows),
                aggregate_key: aggregate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "judgments": str(judged_path),
        "aggregate": str(aggregate_path),
        "n_rows": len(judged_rows),
        "n_new_rows": len(new_rows),
    }


def _score_static_or_vqa_row(row: dict, args: argparse.Namespace, strip_cache: dict[Path, object]) -> dict:
    if args.judge == "baseline-only":
        generated_text = str(row.get("generated_text", "") or "")
        is_empty = not generated_text.strip()
        matches_baseline = (not is_empty) and steered_matches_baseline(generated_text, row.get("baseline_text"))
        judgment = {
            "matched_panel": None,
            "is_junk": is_empty,
            "correct": False,
            "matches_baseline": matches_baseline,
            "reasoning": (
                "generated answer is empty"
                if is_empty
                else "baseline-only mode marks identical-to-baseline generations as no steering effect; "
                "it does not assign panel matches"
            ),
        }
    else:
        comic_dir = Path(row["comic_dir"])
        if comic_dir not in strip_cache:
            strip_cache[comic_dir] = build_strip(comic_dir, n_panels=args.n_panels)
        strip = strip_cache[comic_dir]
        judgment = judge_match_target_panel_anthropic(
            strip_image=strip.strip,
            segment_text=row.get("generated_text", ""),
            baseline_text=row.get("baseline_text"),
            target_panel=int(row["target_panel"]),
            n_panels=args.n_panels,
            model_name=args.claude_model,
        )
    out = dict(row)
    out["judgment"] = judgment
    return out


def _aggregate_judged_rows(judged_rows: list[dict], *, seed: int) -> tuple[str, dict]:
    if judged_rows and all(row.get("task") == "dynamic_narration" for row in judged_rows):
        return "dynamic_aggregate", aggregate_dynamic_judgments(judged_rows, seed=seed)
    return "aggregate", aggregate_judgments(
        [row for row in judged_rows if row.get("task") != "dynamic_narration"],
        seed=seed,
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _score_config(
    generations_path: Path,
    judge: str,
    claude_model: str,
    n_panels: int,
    seed: int,
) -> dict[str, object]:
    return {
        "task": "score_gaze_generations",
        "judgment_schema_version": JUDGMENT_SCHEMA_VERSION,
        "generations": str(generations_path),
        "judge": judge,
        "claude_model": claude_model,
        "n_panels": n_panels,
        "seed": seed,
    }


if __name__ == "__main__":
    main()
