from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any

import numpy as np


DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"


def normalize_for_match(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^panel\s*\d+\s*:\s*", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def steered_matches_baseline(steered: str, baseline: str | None, jaccard_threshold: float = 0.85) -> bool:
    if baseline is None:
        return False
    a = normalize_for_match(steered)
    b = normalize_for_match(baseline)
    if not a or not b:
        return False
    if a == b:
        return True
    a_set = set(a.split())
    b_set = set(b.split())
    if not a_set or not b_set:
        return False
    return len(a_set & b_set) / len(a_set | b_set) >= jaccard_threshold


def judge_match_target_panel_anthropic(
    strip_image,
    segment_text: str,
    baseline_text: str | None,
    target_panel: int,
    n_panels: int = 6,
    model_name: str = DEFAULT_JUDGE_MODEL,
    treat_baseline_match_as_miss: bool = True,
) -> dict[str, Any]:
    if not str(segment_text or "").strip():
        return {
            "matched_panel": None,
            "is_junk": True,
            "correct": False,
            "matches_baseline": False,
            "reasoning": "generated answer is empty",
        }
    if treat_baseline_match_as_miss and steered_matches_baseline(segment_text, baseline_text):
        return {
            "matched_panel": None,
            "is_junk": False,
            "correct": False,
            "matches_baseline": True,
            "reasoning": "steered text is essentially identical to baseline",
        }

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is required for --judge anthropic.")

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    image_b64 = _encode_image_base64(strip_image)
    baseline_blurb = f'Baseline answer: "{baseline_text}"\n\n' if baseline_text else ""
    response = _create_with_retry(
        client,
        model=model_name,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"This is a {n_panels}-panel comic strip, with panels numbered "
                            f"1 to {n_panels} from left to right.\n\n"
                            f"{baseline_blurb}"
                            f'Steered answer: "{segment_text}"\n\n'
                            "Ignore any panel numbering inside the answer. Match by visual content. "
                            f"Pick exactly one panel, an integer 1..{n_panels}, whose visual content "
                            "the answer best describes. If the answer is incoherent, repetitive, empty, "
                            "or only labels/numbers, set is_junk=true and matched_panel=null.\n\n"
                            "Return ONLY JSON like: "
                            '{"matched_panel": <integer or null>, "is_junk": <true/false>, '
                            '"reasoning": "<one sentence>"}'
                        ),
                    },
                ],
            }
        ],
    )
    response_text = response.content[0].text.strip()
    try:
        result = _extract_json(response_text)
    except ValueError:
        return {
            "matched_panel": None,
            "is_junk": True,
            "correct": False,
            "matches_baseline": False,
            "reasoning": "judge response was not valid JSON",
            "raw_judge_text": response_text,
        }
    matched_panel = _coerce_panel(result.get("matched_panel"), n_panels=n_panels)
    is_junk = bool(result.get("is_junk", False))
    return {
        "matched_panel": matched_panel,
        "is_junk": is_junk,
        "correct": (not is_junk and matched_panel == int(target_panel)),
        "matches_baseline": False,
        "reasoning": str(result.get("reasoning", "")),
    }


def bootstrap_ci(outcomes: list[bool], n_bootstrap: int = 10000, ci: float = 0.95, seed: int = 42) -> dict[str, float]:
    if not outcomes:
        return {"accuracy": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}

    values = np.array(outcomes, dtype=np.float64)
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(values), size=(n_bootstrap, len(values)))
    boot_means = values[indices].mean(axis=1)
    alpha = 1.0 - ci
    return {
        "accuracy": float(values.mean()),
        "ci_low": float(np.percentile(boot_means, 100.0 * alpha / 2.0)),
        "ci_high": float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0))),
        "n": int(len(values)),
    }


def aggregate_judgments(rows: list[dict[str, Any]], *, seed: int = 42) -> dict[str, Any]:
    by_condition: dict[str, list[bool]] = {}
    per_panel: dict[str, dict[int, list[bool]]] = {}
    junk_counts: dict[str, int] = {}
    baseline_match_counts: dict[str, int] = {}

    for row in rows:
        condition = str(row["condition"])
        judgment = row.get("judgment") or {}
        outcome = bool(judgment.get("correct", False))
        target_panel = int(row["target_panel"])

        by_condition.setdefault(condition, []).append(outcome)
        per_panel.setdefault(condition, {}).setdefault(target_panel, []).append(outcome)
        if judgment.get("is_junk"):
            junk_counts[condition] = junk_counts.get(condition, 0) + 1
        if judgment.get("matches_baseline"):
            baseline_match_counts[condition] = baseline_match_counts.get(condition, 0) + 1

    aggregate = {}
    for condition, outcomes in sorted(by_condition.items()):
        aggregate[condition] = {
            "overall": bootstrap_ci(outcomes, seed=seed),
            "per_panel": {
                str(panel): bootstrap_ci(panel_outcomes, seed=seed)
                for panel, panel_outcomes in sorted(per_panel[condition].items())
            },
            "junk_count": int(junk_counts.get(condition, 0)),
            "baseline_match_count": int(baseline_match_counts.get(condition, 0)),
        }
    return aggregate


def split_dynamic_segments(text: str, n_segments: int) -> list[str]:
    """Split a dynamic narration into schedule-aligned text chunks.

    Prefer explicit "Panel N:" markers because they mirror the prompt. If the
    model omits markers, fall back to paragraph/sentence chunks and finally to
    roughly even character chunks. This is less exact than token-time
    segmentation, but it makes saved full-text generations judgeable without
    rerunning Qwen.
    """
    text = text.strip()
    if n_segments <= 0:
        return []
    if not text:
        return [""] * n_segments

    marker_pattern = re.compile(r"(?i)(?=\b(?:\*\*)?panel\s*\d+\s*:)")
    marker_parts = [part.strip() for part in marker_pattern.split(text) if part.strip()]
    if len(marker_parts) >= n_segments:
        return _pad_or_trim(marker_parts, n_segments)

    paragraph_parts = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if len(paragraph_parts) >= n_segments:
        return _pad_or_trim(paragraph_parts, n_segments)

    sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(sentence_parts) >= n_segments:
        return _pad_or_trim(sentence_parts, n_segments)

    return _split_evenly(text, n_segments)


def aggregate_dynamic_judgments(rows: list[dict[str, Any]], *, seed: int = 42) -> dict[str, Any]:
    by_condition: dict[str, list[bool]] = {}
    rho_by_condition: dict[str, list[float]] = {}
    junk_counts: dict[str, int] = {}

    for row in rows:
        condition = str(row["condition"])
        segment_rows = row.get("segment_judgments") or []
        outcomes = [bool(item.get("correct", False)) for item in segment_rows]
        by_condition.setdefault(condition, []).extend(outcomes)
        junk_counts[condition] = junk_counts.get(condition, 0) + sum(1 for item in segment_rows if item.get("is_junk"))

        targets = [int(item["target_panel"]) for item in segment_rows if item.get("matched_panel") is not None]
        matched = [int(item["matched_panel"]) for item in segment_rows if item.get("matched_panel") is not None]
        if len(targets) >= 3:
            rho_by_condition.setdefault(condition, []).append(spearman_rho(targets, matched))

    aggregate = {}
    for condition, outcomes in sorted(by_condition.items()):
        rhos = rho_by_condition.get(condition, [])
        aggregate[condition] = {
            "per_segment_accuracy": bootstrap_ci(outcomes, seed=seed),
            "spearman_rho_mean": float(np.mean(rhos)) if rhos else 0.0,
            "spearman_rho_std": float(np.std(rhos)) if rhos else 0.0,
            "n_strips_for_rho": len(rhos),
            "junk_segments": int(junk_counts.get(condition, 0)),
        }
    return aggregate


def spearman_rho(xs: list[int], ys: list[int]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x = _average_ranks(np.array(xs, dtype=np.float64))
    y = _average_ranks(np.array(ys, dtype=np.float64))
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = float(np.sqrt((x_centered**2).sum() * (y_centered**2).sum()))
    if denom == 0.0:
        return 0.0
    return float((x_centered * y_centered).sum() / denom)


def _encode_image_base64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


def _pad_or_trim(parts: list[str], n_segments: int) -> list[str]:
    out = parts[:n_segments]
    if len(out) < n_segments:
        out.extend([""] * (n_segments - len(out)))
    return out


def _split_evenly(text: str, n_segments: int) -> list[str]:
    if n_segments <= 1:
        return [text]
    chunk_size = max(1, int(np.ceil(len(text) / n_segments)))
    chunks = [text[idx * chunk_size : (idx + 1) * chunk_size].strip() for idx in range(n_segments)]
    return chunks


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _create_with_retry(client, *, retries: int = 6, base_delay: float = 2.0, **kwargs):
    last_exc = None
    for attempt in range(retries):
        try:
            return client.messages.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            transient = any(marker in msg for marker in ["overload", "rate", "429", "529", "timeout", "503"])
            if not transient:
                raise
            time.sleep(base_delay * (2**attempt))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Anthropic request failed without an exception.")


def _extract_json(response_text: str) -> dict[str, Any]:
    for pattern in [r"\{[\s\S]*?\}", r"\{[\s\S]*\}"]:
        for match in re.finditer(pattern, response_text):
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    raise ValueError("Judge response did not contain a valid JSON object.")


def _coerce_panel(value: Any, n_panels: int) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        panel = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= panel <= n_panels:
        return panel
    return None
