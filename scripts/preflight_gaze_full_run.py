from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from scripts.plan_gaze_full_run import shard_suffixes


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for the Qwen2.5-VL GazeHeads full replication.")
    parser.add_argument("--segment-root", default="segments/gaze_heads_qwen25")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--total-comics", type=int, default=500)
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--judge", choices=["baseline-only", "anthropic"], default="anthropic")
    parser.add_argument("--require-discovery", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.segment_root)
    report = preflight(
        root=root,
        model_id=args.model_id,
        total_comics=args.total_comics,
        shard_size=args.shard_size,
        judge=args.judge,
        require_discovery=args.require_discovery,
    )
    out_path = Path(args.out) if args.out else root / "reports" / "full_run_preflight.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote preflight report to {out_path}")
    for check in report["checks"]:
        print(f"{check['name']:24s} {'OK' if check['ok'] else 'FAIL'} {check['message']}")
    if not report["ok"]:
        raise SystemExit(1)


def preflight(
    *,
    root: Path,
    model_id: str,
    total_comics: int,
    shard_size: int,
    judge: str,
    require_discovery: bool = False,
) -> dict[str, Any]:
    checks = [
        _check_segment_layout(root),
        _check_dataset(root / "data" / "comics", expected=total_comics),
        _check_python_dependencies(),
        _check_model_cache(model_id),
        _check_judge(judge),
        _check_shards(total_comics, shard_size),
        _check_discovery(root / "runs" / "gaze_discovery", required=require_discovery),
    ]
    return {
        "segment_root": str(root),
        "model_id": model_id,
        "total_comics": total_comics,
        "shard_size": shard_size,
        "judge": judge,
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
    }


def _check_segment_layout(root: Path) -> dict[str, Any]:
    required = [root / "papers" / "GazeHeads.pdf", root / "data", root / "runs", root / "reports"]
    missing = [str(path) for path in required if not path.exists()]
    return _check("segment_layout", not missing, "all segment directories/files present" if not missing else f"missing {missing}")


def _check_dataset(comics_root: Path, *, expected: int) -> dict[str, Any]:
    n_comics = 0
    if comics_root.exists():
        for comic_dir in comics_root.iterdir():
            if comic_dir.is_dir() and all((comic_dir / f"p{idx}.png").exists() for idx in range(1, 7)):
                n_comics += 1
    ok = n_comics >= expected
    return _check("dataset", ok, f"{n_comics} valid six-panel comics found, expected at least {expected}", n_comics=n_comics)


def _check_python_dependencies() -> dict[str, Any]:
    packages = ["torch", "transformers", "qwen_vl_utils", "accelerate", "numpy"]
    missing = [package for package in packages if importlib.util.find_spec(package) is None]
    return _check(
        "python_dependencies",
        not missing,
        "all required Qwen packages importable" if not missing else f"missing packages: {missing}",
        missing=missing,
    )


def _check_model_cache(model_id: str) -> dict[str, Any]:
    cache_roots = [
        Path(os.environ.get("HF_HOME", "")) if os.environ.get("HF_HOME") else None,
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    model_cache_name = "models--" + model_id.replace("/", "--")
    matches = [str(root / model_cache_name) for root in cache_roots if root is not None and (root / model_cache_name).exists()]
    return _check(
        "model_cache",
        bool(matches),
        f"found cached model at {matches[0]}" if matches else f"no local cache found for {model_id}",
        paths=matches,
    )


def _check_judge(judge: str) -> dict[str, Any]:
    if judge != "anthropic":
        return _check("judge", True, "baseline-only judge selected")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_package = importlib.util.find_spec("anthropic") is not None
    return _check(
        "judge",
        has_key and has_package,
        "Anthropic key and package available"
        if has_key and has_package
        else f"anthropic package available={has_package}, ANTHROPIC_API_KEY set={has_key}",
        anthropic_package=has_package,
        has_api_key=has_key,
    )


def _check_shards(total_comics: int, shard_size: int) -> dict[str, Any]:
    try:
        shards = shard_suffixes(total_comics, shard_size)
    except ValueError as exc:
        return _check("shards", False, str(exc))
    covered = sum(count for _, count, _ in shards)
    contiguous = all(start == idx * shard_size for idx, (start, _, _) in enumerate(shards))
    ok = covered == total_comics and contiguous
    return _check("shards", ok, f"{len(shards)} shards cover {covered}/{total_comics} comics", shards=shards)


def _check_discovery(discovery_dir: Path, *, required: bool) -> dict[str, Any]:
    files = ["gaze_head_ranking.json", "gaze_scores.npy", "mean_panel_attention.npy", "summary.json"]
    missing = [name for name in files if not (discovery_dir / name).exists()]
    ok = not missing or not required
    if not missing:
        message = "canonical discovery artifacts present"
    elif required:
        message = f"missing required discovery artifacts: {missing}"
    else:
        message = f"canonical discovery not present yet; discovery stage will create {missing}"
    return _check("discovery", ok, message, required=required, missing=missing)


def _check(name: str, ok: bool, message: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "message": message, **extra}


if __name__ == "__main__":
    main()
