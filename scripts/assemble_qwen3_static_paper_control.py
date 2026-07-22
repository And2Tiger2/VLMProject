from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PAPER_MIN_LAYER = 20
PAPER_MAX_LAYER = 35


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble saved gaze generations with newly generated paper-control shards "
            "without rerunning the gaze condition."
        )
    )
    parser.add_argument("--existing-gaze-run", type=Path, required=True)
    parser.add_argument("--control-shard-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--expected-comics", type=int, required=True)
    args = parser.parse_args()

    result = assemble_paper_control(
        existing_gaze_run=args.existing_gaze_run,
        control_shard_runs=args.control_shard_runs,
        out_dir=args.out_dir,
        top_k=args.top_k,
        expected_comics=args.expected_comics,
    )
    print(json.dumps(result, indent=2))


def assemble_paper_control(
    *,
    existing_gaze_run: Path,
    control_shard_runs: list[Path],
    out_dir: Path,
    top_k: int,
    expected_comics: int,
) -> dict[str, Any]:
    if top_k <= 0 or expected_comics <= 0:
        raise ValueError("top_k and expected_comics must be positive")
    gaze_condition = f"gaze_top{top_k}"
    control_condition = f"non_gaze_paper_{top_k}"
    gaze_path = existing_gaze_run / "generations.jsonl"
    gaze_rows_all = _read_jsonl(gaze_path)

    control_rows: list[dict[str, Any]] = []
    shard_configs: list[dict[str, Any]] = []
    for run in control_shard_runs:
        config = _read_json(run / "experiment_config.json")
        _validate_control_config(config, top_k=top_k, expected_condition=control_condition)
        shard_configs.append(config)
        rows = _read_jsonl(run / "generations.jsonl")
        if any(str(row.get("condition")) != control_condition for row in rows):
            raise RuntimeError(f"{run} contains a condition other than {control_condition}")
        control_rows.extend(rows)

    if not shard_configs:
        raise RuntimeError("no paper-control shards supplied")
    selected_controls = shard_configs[0]["selected_control_heads"]
    if any(config["selected_control_heads"] != selected_controls for config in shard_configs[1:]):
        raise RuntimeError("paper-control head selection differs across shards")

    control_keys = [_key(row) for row in control_rows]
    if len(control_keys) != len(set(control_keys)):
        raise RuntimeError("paper-control shards contain duplicate (strip, target-panel) rows")
    expected_per_condition = expected_comics * 6
    if len(control_rows) != expected_per_condition:
        raise RuntimeError(
            f"paper control has {len(control_rows)} rows; expected {expected_per_condition}"
        )
    control_key_set = set(control_keys)
    gaze_rows = [
        row
        for row in gaze_rows_all
        if str(row.get("condition")) == gaze_condition and _key(row) in control_key_set
    ]
    gaze_by_key = {_key(row): row for row in gaze_rows}
    control_by_key = {_key(row): row for row in control_rows}
    if set(gaze_by_key) != control_key_set:
        missing = sorted(control_key_set - set(gaze_by_key))[:5]
        raise RuntimeError(
            f"saved gaze run does not cover every paper-control key; examples: {missing}"
        )

    paired_rows: list[dict[str, Any]] = []
    for key in sorted(control_key_set):
        paired_rows.extend([gaze_by_key[key], control_by_key[key]])

    out_dir.mkdir(parents=True, exist_ok=True)
    generations_path = out_dir / "generations.jsonl"
    control_path = out_dir / "control_generations.jsonl"
    _write_jsonl(generations_path, paired_rows)
    _write_jsonl(control_path, control_rows)
    first = shard_configs[0]
    experiment_config = {
        "task": "static_narration_paper_replication",
        "model_id": first["model_id"],
        "comics_root": first["comics_root"],
        "gaze_ranking": first["gaze_ranking"],
        "top_k_gaze": top_k,
        "top_k_random": top_k,
        "control_mode": "paper",
        "paper_control_layers": [PAPER_MIN_LAYER, PAPER_MAX_LAYER],
        "condition_set": "both",
        "condition_labels": [gaze_condition, control_condition],
        "selected_gaze_heads": first["selected_gaze_heads"],
        "selected_control_heads": selected_controls,
        "targets_per_strip": 6,
        "max_new_tokens": first["max_new_tokens"],
        "swap_bias": first["swap_bias"],
        "decode_only": first["decode_only"],
        "include_all_heads": False,
        "seed": first["seed"],
        "prompt": first["prompt"],
        "source_gaze_run": str(existing_gaze_run),
        "source_control_shards": [str(path) for path in control_shard_runs],
        "generation_git_commit": first.get("git_commit"),
    }
    (out_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2), encoding="utf-8"
    )
    summary = {
        "stage": "static",
        "paper_replication": True,
        "n_comics": expected_comics,
        "n_rows": len(paired_rows),
        "conditions": [gaze_condition, control_condition],
        "existing_gaze_run": str(existing_gaze_run),
        "control_shard_runs": [str(path) for path in control_shard_runs],
        "generations": str(generations_path),
        "control_generations": str(control_path),
        "generations_sha256": _sha256(generations_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _validate_control_config(config: dict[str, Any], *, top_k: int, expected_condition: str) -> None:
    if config.get("control_mode") != "paper" or config.get("condition_set") != "control":
        raise RuntimeError("control shard is not a paper control-only run")
    if config.get("paper_control_layers") != [PAPER_MIN_LAYER, PAPER_MAX_LAYER]:
        raise RuntimeError("control shard does not use inclusive layers 20--35")
    labels = config.get("condition_labels")
    if labels != [expected_condition]:
        raise RuntimeError(f"control shard labels are {labels!r}; expected {[expected_condition]!r}")
    heads = [tuple(int(value) for value in head) for head in config.get("selected_control_heads", [])]
    if len(heads) != top_k or len(set(heads)) != top_k:
        raise RuntimeError(f"paper control must contain exactly {top_k} unique heads")
    if any(not (PAPER_MIN_LAYER <= layer <= PAPER_MAX_LAYER) for layer, _ in heads):
        raise RuntimeError("paper control contains a head outside layers 20--35")
    gaze = {tuple(int(value) for value in head) for head in config.get("selected_gaze_heads", [])}
    if gaze.intersection(heads):
        raise RuntimeError("paper control overlaps the selected gaze heads")


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("strip_name")), int(row.get("target_panel", 0))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
