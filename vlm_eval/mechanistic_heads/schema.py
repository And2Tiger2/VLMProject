from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PairedExample:
    """Serializable donor/recipient example used by every causal study."""

    pair_id: str
    group_id: str
    donor_image: str
    recipient_image: str
    donor_prompt: str
    recipient_prompt: str
    donor_answer: str
    recipient_answer: str
    correct_answer: str | None = None
    bias_answer: str | None = None
    donor_mask: str | None = None
    recipient_mask: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    split: str = "unspecified"
    generator_seed: int = 0
    source_id: str = ""

    def __post_init__(self) -> None:
        required = {
            "pair_id": self.pair_id,
            "group_id": self.group_id,
            "donor_image": self.donor_image,
            "recipient_image": self.recipient_image,
            "donor_prompt": self.donor_prompt,
            "recipient_prompt": self.recipient_prompt,
            "donor_answer": self.donor_answer,
            "recipient_answer": self.recipient_answer,
            "split": self.split,
            "source_id": self.source_id,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"PairedExample has empty required fields: {missing}")
        if self.donor_answer == self.recipient_answer and not self.metadata.get(
            "allow_equal_answers", False
        ):
            raise ValueError(
                "donor_answer and recipient_answer are equal; mark an intentional "
                "sham with metadata.allow_equal_answers=true"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "PairedExample":
        return cls(**row)


def write_paired_jsonl(path: Path, examples: Iterable[PairedExample]) -> None:
    rows = [example.to_dict() for example in examples]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_paired_jsonl(path: Path) -> list[PairedExample]:
    return [
        PairedExample.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def assert_no_group_leakage(examples: Iterable[PairedExample]) -> None:
    """Refuse any group/source/template that appears in more than one split."""

    owners: dict[tuple[str, str], str] = {}
    failures: list[str] = []
    for example in examples:
        keys = {
            ("group_id", example.group_id),
            ("source_id", example.source_id),
        }
        template = example.metadata.get("template_id")
        page = example.metadata.get("page_id")
        if template is not None:
            keys.add(("template_id", str(template)))
        if page is not None:
            keys.add(("page_id", str(page)))
        for key in keys:
            previous = owners.setdefault(key, example.split)
            if previous != example.split:
                failures.append(f"{key[0]}={key[1]}: {previous} vs {example.split}")
    if failures:
        raise ValueError("split leakage detected: " + "; ".join(sorted(set(failures))))
