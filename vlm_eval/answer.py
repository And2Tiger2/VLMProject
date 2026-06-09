from __future__ import annotations

import re
import string


_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}


def normalize_answer(text: object) -> str:
    """Normalize short benchmark answers for exact matching."""
    value = str(text).strip().lower()
    value = value.replace("\n", " ")
    value = value.translate(str.maketrans("", "", string.punctuation.replace(".", "")))
    value = re.sub(r"\s+", " ", value).strip()
    return _NUMBER_WORDS.get(value, value)


def extract_answer(response: str) -> str:
    """Extract a concise answer from a free-form model response.

    This intentionally avoids LLM judging. It handles common "Answer: X" forms,
    yes/no responses, and integer counting answers before falling back to the
    final short phrase.
    """
    text = response.strip()
    if not text:
        return ""

    answer_match = re.search(r"(?:final\s+answer|answer)\s*[:\-]\s*([^\n.]+)", text, re.I)
    if answer_match:
        return normalize_answer(answer_match.group(1))

    normalized = normalize_answer(text)
    yes_no = re.search(r"\b(yes|no)\b", normalized)
    if yes_no:
        return yes_no.group(1)

    number = re.search(r"\b\d+(?:\.\d+)?\b", normalized)
    if number:
        return number.group(0)

    tokens = normalized.split()
    return " ".join(tokens[-4:])

