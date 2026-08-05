from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def add_standard_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite an existing non-resume output directory (default: false).",
    )


def load_json_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return data


def prepare_output_directory(
    path: Path, *, resume: bool, overwrite: bool, known_outputs: tuple[str, ...]
) -> None:
    existing = [path / name for name in known_outputs if (path / name).exists()]
    if existing and not resume and not overwrite:
        raise FileExistsError(
            "output artifacts already exist; use --resume or explicit --overwrite: "
            + ", ".join(map(str, existing))
        )
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    path.mkdir(parents=True, exist_ok=True)


def effective_limit(args: argparse.Namespace, *, smoke_max: int = 8) -> int | None:
    if args.smoke:
        return min(args.limit, smoke_max) if args.limit is not None else smoke_max
    return args.limit


def parse_layer_spec(value: str | None, *, n_layers: int) -> list[int] | None:
    if value is None: return None
    layers: set[int] = set()
    for part in value.split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1)); layers.update(range(start, end + 1))
        else: layers.add(int(part))
    if not layers or min(layers) < 0 or max(layers) >= n_layers:
        raise ValueError(f"layer specification {value!r} is outside 0..{n_layers-1}")
    return sorted(layers)
