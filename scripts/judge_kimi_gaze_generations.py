from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from vlm_eval.gaze_comics import DEFAULT_N_PANELS, build_strip
from vlm_eval.gaze_judge import aggregate_judgments, steered_matches_baseline
from vlm_eval.gaze_resume import ensure_resume_config, load_completed_keys, row_key


DEFAULT_KIMI_MODEL = "moonshotai/Kimi-VL-A3B-Instruct"
DEFAULT_KIMI_REVISION = "cc6452511d00c99f3b3bed213e96ab7802c415c8"
KEY_FIELDS = ["strip_name", "condition", "target_panel"]
JUDGMENT_SCHEMA_VERSION = 3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge static Gaze Heads generations with local Kimi-VL forced panel matching."
    )
    parser.add_argument("--generations", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_KIMI_MODEL)
    parser.add_argument("--revision", default=DEFAULT_KIMI_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--n-panels", type=int, default=DEFAULT_N_PANELS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-parse-failure-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    result = judge_generations_kimi(
        generations_path=Path(args.generations),
        out_dir=Path(args.out_dir),
        model_id=args.model_id,
        revision=args.revision,
        max_new_tokens=args.max_new_tokens,
        n_panels=args.n_panels,
        limit=args.limit,
        max_parse_failure_rate=args.max_parse_failure_rate,
        seed=args.seed,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


def judge_generations_kimi(
    *,
    generations_path: Path,
    out_dir: Path,
    model_id: str = DEFAULT_KIMI_MODEL,
    revision: str = DEFAULT_KIMI_REVISION,
    max_new_tokens: int = 64,
    n_panels: int = DEFAULT_N_PANELS,
    limit: int = 0,
    max_parse_failure_rate: float = 0.01,
    seed: int = 42,
    resume: bool = False,
    generator: Any | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(generations_path)
    if limit > 0:
        rows = rows[:limit]

    config = {
        "task": "judge_gaze_generations_kimi",
        "judgment_schema_version": JUDGMENT_SCHEMA_VERSION,
        "generations": str(generations_path),
        "model_id": model_id,
        "revision": revision,
        "max_new_tokens": max_new_tokens,
        "n_panels": n_panels,
        "limit": limit,
        "max_parse_failure_rate": max_parse_failure_rate,
        "seed": seed,
    }
    judgments_path = out_dir / "judgments.jsonl"
    ensure_resume_config(
        out_dir,
        config,
        resume=resume,
        artifact_name="judgments.jsonl",
        config_name="judgment_config.json",
    )

    existing_rows = _read_jsonl(judgments_path) if resume and judgments_path.exists() else []
    completed = load_completed_keys(judgments_path, KEY_FIELDS) if resume else set()
    pending_rows = [row for row in rows if row_key(row, KEY_FIELDS) not in completed]
    if pending_rows and generator is None:
        generator = KimiVLGenerator(
            model_id=model_id,
            revision=revision,
            max_new_tokens=max_new_tokens,
        )

    strip_cache: dict[Path, Any] = {}
    new_rows: list[dict[str, Any]] = []
    with judgments_path.open("a" if resume else "w", encoding="utf-8") as handle:
        for row in tqdm(pending_rows, desc="Kimi-VL forced-choice judging"):
            comic_dir = Path(row["comic_dir"])
            if comic_dir not in strip_cache:
                strip_cache[comic_dir] = build_strip(comic_dir, n_panels=n_panels).strip
            judgment = judge_row_with_kimi(
                generator,
                row,
                strip_image=strip_cache[comic_dir],
                n_panels=n_panels,
            )
            judged = dict(row)
            judged["judgment"] = judgment
            handle.write(json.dumps(judged) + "\n")
            handle.flush()
            new_rows.append(judged)

    judged_rows = [*existing_rows, *new_rows] if resume else new_rows
    duplicate_count = len(judged_rows) - len(
        {row_key(row, KEY_FIELDS) for row in judged_rows}
    )
    parse_failures = sum(
        bool((row.get("judgment") or {}).get("parse_failed")) for row in judged_rows
    )
    expected_rows = len(rows)
    errors: list[str] = []
    if len(judged_rows) != expected_rows:
        errors.append(f"judged {len(judged_rows)} rows; expected {expected_rows}")
    if duplicate_count:
        errors.append(f"found {duplicate_count} duplicate judgment keys")
    parse_failure_rate = parse_failures / len(judged_rows) if judged_rows else 0.0
    warnings: list[str] = []
    if parse_failure_rate > max_parse_failure_rate:
        errors.append(
            f"Kimi parse-failure rate {parse_failure_rate:.2%} exceeds "
            f"{max_parse_failure_rate:.2%} ({parse_failures}/{len(judged_rows)})"
        )
    elif parse_failures:
        warnings.append(
            f"{parse_failures}/{len(judged_rows)} Kimi responses could not be parsed and were scored as misses"
        )

    aggregate_path = out_dir / "aggregate_results.json"
    payload = {
        "valid": not errors,
        "judge": "kimi-vl",
        "judge_model": model_id,
        "judge_revision": revision,
        "generations": str(generations_path),
        "resume": resume,
        "n_rows": len(judged_rows),
        "expected_rows": expected_rows,
        "n_new_rows": len(new_rows),
        "n_skipped_existing_rows": expected_rows - len(new_rows),
        "duplicate_count": duplicate_count,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failure_rate,
        "errors": errors,
        "warnings": warnings,
        "aggregate": aggregate_judgments(judged_rows, seed=seed),
    }
    aggregate_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "valid": not errors,
        "judgments": str(judgments_path),
        "aggregate": str(aggregate_path),
        "n_rows": len(judged_rows),
        "n_new_rows": len(new_rows),
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failure_rate,
        "errors": errors,
        "warnings": warnings,
    }


class KimiVLGenerator:
    def __init__(self, *, model_id: str, revision: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("Kimi-VL judging requires an allocated CUDA GPU.")
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_source = resolve_local_snapshot(model_id, revision)
        source_kwargs = (
            {}
            if model_source != model_id
            else {"revision": revision}
        )
        self.processor = AutoProcessor.from_pretrained(
            model_source,
            trust_remote_code=True,
            local_files_only=True,
            **source_kwargs,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
            device_map={"": "cuda:0"},
            **source_kwargs,
        ).eval()

    def generate(self, image: Any, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ""},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        input_len = int(inputs["input_ids"].shape[-1])
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated = generated[:, input_len:]
        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def judge_row_with_kimi(
    generator: Any,
    row: dict[str, Any],
    strip_image: Any,
    *,
    n_panels: int,
) -> dict[str, Any]:
    generated_text = str(row.get("generated_text", "") or "")
    baseline_text = row.get("baseline_text")
    if not generated_text.strip():
        return {
            "matched_panel": None,
            "is_junk": True,
            "correct": False,
            "matches_baseline": False,
            "parse_failed": False,
            "raw_judge_text": "",
            "reasoning": "generated answer is empty",
        }
    if steered_matches_baseline(generated_text, baseline_text):
        return {
            "matched_panel": None,
            "is_junk": False,
            "correct": False,
            "matches_baseline": True,
            "parse_failed": False,
            "raw_judge_text": "",
            "reasoning": "steered text is essentially identical to baseline",
        }

    prompt = forced_choice_prompt(generated_text, baseline_text, n_panels=n_panels)
    raw = generator.generate(strip_image, prompt)
    parsed = parse_panel_judgment(raw, n_panels=n_panels)
    matched_panel = parsed["matched_panel"]
    is_junk = bool(parsed["is_junk"])
    return {
        "matched_panel": matched_panel,
        "is_junk": is_junk,
        "correct": (not is_junk and matched_panel == int(row["target_panel"])),
        "matches_baseline": False,
        "parse_failed": bool(parsed["parse_failed"]),
        "raw_judge_text": raw,
        "reasoning": parsed["reasoning"],
    }


def forced_choice_prompt(generated_text: str, baseline_text: str | None, *, n_panels: int) -> str:
    baseline = f'Baseline answer: "{baseline_text}"\n' if baseline_text else ""
    return (
        f"This is a {n_panels}-panel comic strip. The panels are numbered 1 to {n_panels} "
        "from left to right.\n\n"
        f"{baseline}"
        f'Answer to judge: "{generated_text}"\n\n'
        "Ignore any panel number mentioned inside the answer. Match by visual content only.\n"
        f"Choose exactly one panel number from 1 to {n_panels} whose visual content the answer "
        "best describes.\n"
        "If the answer is empty, incoherent, repetitive, only symbols, or not about the comic, "
        "mark it as junk.\n\n"
        "Return only compact JSON with this schema:\n"
        '{"matched_panel": <integer or null>, "is_junk": <true or false>, '
        '"reasoning": "<short reason>"}'
    )


def parse_panel_judgment(text: str, *, n_panels: int) -> dict[str, Any]:
    parsed = _extract_json_object(text)
    if parsed is not None:
        panel = _coerce_panel(parsed.get("matched_panel"), n_panels=n_panels)
        is_junk = bool(parsed.get("is_junk", panel is None))
        return {
            "matched_panel": None if is_junk else panel,
            "is_junk": is_junk,
            "parse_failed": (not is_junk and panel is None),
            "reasoning": str(parsed.get("reasoning", "")),
        }

    # Accept an unambiguous short fallback such as "panel 2", but flag every
    # other non-JSON response so a silently broken model/template cannot pass.
    match = re.fullmatch(r"\s*(?:panel\s*)?([1-6])[.!]?\s*", text, flags=re.IGNORECASE)
    panel = _coerce_panel(match.group(1), n_panels=n_panels) if match else None
    return {
        "matched_panel": panel,
        "is_junk": panel is None,
        "parse_failed": panel is None,
        "reasoning": "fallback parsed panel number" if panel is not None else "could not parse judge output",
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    if "```" in text:
        candidates.extend(part.strip() for part in text.split("```") if part.strip())
    brace_match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))
    for candidate in candidates:
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _coerce_panel(value: Any, *, n_panels: int) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        panel = int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        if not match:
            return None
        panel = int(match.group(0))
    return panel if 1 <= panel <= n_panels else None


def resolve_local_snapshot(model_id: str, revision: str) -> str:
    """Return the exact cached snapshot path, avoiding conflicting legacy caches."""
    explicit = Path(model_id)
    if explicit.is_dir():
        return str(explicit)
    hub_cache_value = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache_value:
        hub_cache = Path(hub_cache_value)
    else:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        hub_cache = hf_home / "hub"
    repo_cache_name = "models--" + model_id.replace("/", "--")
    snapshot = hub_cache / repo_cache_name / "snapshots" / revision
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(
            f"Pinned offline snapshot is missing: {snapshot}. "
            "Run bash scripts/setup_neuronic_kimi_judge.sh on the login node."
        )
    return str(snapshot)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
