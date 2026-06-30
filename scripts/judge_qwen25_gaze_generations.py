from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from adapters.qwen25_vl import Qwen25VLAdapter
from vlm_eval.gaze_comics import DEFAULT_N_PANELS, build_strip
from vlm_eval.gaze_judge import aggregate_judgments, steered_matches_baseline
from vlm_eval.gaze_resume import load_completed_keys, row_key
from vlm_eval.types import EvalExample


KEY_FIELDS = ["strip_name", "condition", "target_panel"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge static GazeHeads generations with Qwen2.5-VL forced panel matching.")
    parser.add_argument("--generations", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--min-pixels", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--n-panels", type=int, default=DEFAULT_N_PANELS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    result = judge_generations_qwen(
        generations_path=Path(args.generations),
        out_dir=Path(args.out_dir) if args.out_dir else Path(args.generations).parent / "qwen_judge",
        model_id=args.model_id,
        device_map=args.device_map,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        max_new_tokens=args.max_new_tokens,
        n_panels=args.n_panels,
        limit=args.limit,
        seed=args.seed,
        resume=args.resume,
    )
    print(f"Wrote judgments to {result['judgments']}")
    print(f"Wrote aggregate results to {result['aggregate']}")


def judge_generations_qwen(
    *,
    generations_path: Path,
    out_dir: Path,
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device_map: str = "auto",
    max_pixels: int = 1048576,
    min_pixels: int = 0,
    max_new_tokens: int = 64,
    n_panels: int = DEFAULT_N_PANELS,
    limit: int = 0,
    seed: int = 42,
    resume: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in generations_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit > 0:
        rows = rows[:limit]

    adapter = Qwen25VLAdapter(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        do_sample=False,
        temperature=None,
        device_map=device_map,
        prompt_mode="raw",
        seed=seed,
    )

    judgments_path = out_dir / "judgments.jsonl"
    existing_rows = _read_jsonl(judgments_path) if resume and judgments_path.exists() else []
    completed = load_completed_keys(judgments_path, KEY_FIELDS) if resume else set()
    strip_cache: dict[Path, Any] = {}
    new_rows = []

    for row in tqdm(rows, desc="Qwen forced-choice judging"):
        if row_key(row, KEY_FIELDS) in completed:
            continue
        comic_dir = Path(row["comic_dir"])
        if comic_dir not in strip_cache:
            strip_cache[comic_dir] = build_strip(comic_dir, n_panels=n_panels)
        judgment = judge_row_with_qwen(adapter, row, strip_cache[comic_dir].strip, n_panels=n_panels)
        judged = dict(row)
        judged["judgment"] = judgment
        new_rows.append(judged)
        completed.add(row_key(judged, KEY_FIELDS))

    with judgments_path.open("a" if resume else "w", encoding="utf-8") as handle:
        for row in new_rows:
            handle.write(json.dumps(row) + "\n")

    judged_rows = [*existing_rows, *new_rows] if resume else new_rows
    aggregate = aggregate_judgments(judged_rows, seed=seed)
    aggregate_path = out_dir / "aggregate_results.json"
    aggregate_path.write_text(
        json.dumps(
            {
                "judge": "qwen2.5-vl",
                "judge_model": model_id,
                "generations": str(generations_path),
                "resume": resume,
                "n_rows": len(judged_rows),
                "n_new_rows": len(new_rows),
                "n_skipped_existing_rows": len(rows) - len(new_rows),
                "aggregate": aggregate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"judgments": str(judgments_path), "aggregate": str(aggregate_path), "n_rows": len(judged_rows)}


def judge_row_with_qwen(adapter: Qwen25VLAdapter, row: dict[str, Any], strip_image: Any, *, n_panels: int) -> dict[str, Any]:
    generated_text = row.get("generated_text", "")
    baseline_text = row.get("baseline_text")
    if steered_matches_baseline(generated_text, baseline_text):
        return {
            "matched_panel": None,
            "is_junk": False,
            "correct": False,
            "matches_baseline": True,
            "raw_judge_text": "",
            "reasoning": "steered text is essentially identical to baseline",
        }

    prompt = forced_choice_prompt(generated_text, baseline_text, n_panels=n_panels)
    raw = adapter.generate(EvalExample(id=_example_id(row), image=strip_image, prompt=prompt, ground_truth=""))
    parsed = parse_qwen_panel_judgment(raw, n_panels=n_panels)
    matched_panel = parsed["matched_panel"]
    is_junk = bool(parsed["is_junk"])
    return {
        "matched_panel": matched_panel,
        "is_junk": is_junk,
        "correct": (not is_junk and matched_panel == int(row["target_panel"])),
        "matches_baseline": False,
        "raw_judge_text": raw,
        "reasoning": parsed["reasoning"],
    }


def forced_choice_prompt(generated_text: str, baseline_text: str | None, *, n_panels: int) -> str:
    baseline = f'Baseline answer: "{baseline_text}"\n' if baseline_text else ""
    return (
        f"This is a {n_panels}-panel comic strip. The panels are numbered 1 to {n_panels} from left to right.\n\n"
        f"{baseline}"
        f'Answer to judge: "{generated_text}"\n\n'
        "Ignore any panel number mentioned inside the answer. Match by visual content only.\n"
        f"Choose exactly one panel number from 1 to {n_panels} whose visual content the answer best describes.\n"
        "If the answer is empty, incoherent, repetitive, only symbols, or not about the comic, mark it as junk.\n\n"
        "Return only compact JSON with this schema:\n"
        '{"matched_panel": <integer or null>, "is_junk": <true or false>, "reasoning": "<short reason>"}'
    )


def _example_id(row: dict[str, Any]) -> str:
    return "::".join(str(value) for value in row_key(row, KEY_FIELDS))


def parse_qwen_panel_judgment(text: str, *, n_panels: int) -> dict[str, Any]:
    parsed = _extract_json_object(text)
    if parsed is not None:
        panel = _coerce_panel(parsed.get("matched_panel"), n_panels=n_panels)
        is_junk = bool(parsed.get("is_junk", panel is None))
        return {
            "matched_panel": None if is_junk else panel,
            "is_junk": is_junk,
            "reasoning": str(parsed.get("reasoning", "")),
        }

    lowered = text.lower()
    if "junk" in lowered or "incoherent" in lowered or "empty" in lowered:
        return {"matched_panel": None, "is_junk": True, "reasoning": "judge marked output as junk"}
    match = re.search(r"\b(?:panel\s*)?([1-6])\b", text, flags=re.IGNORECASE)
    panel = _coerce_panel(match.group(1), n_panels=n_panels) if match else None
    return {
        "matched_panel": panel,
        "is_junk": panel is None,
        "reasoning": "fallback parsed panel number" if panel is not None else "could not parse judge output",
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates = [text]
    if "```" in text:
        candidates.extend(part.strip() for part in text.split("```") if part.strip())
    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _coerce_panel(value: Any, *, n_panels: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        panel = int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        if not match:
            return None
        panel = int(match.group(0))
    if 1 <= panel <= n_panels:
        return panel
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
