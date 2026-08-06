#!/usr/bin/env python3
"""Build a task-agnostic causal-importance diagnostic for matched controls."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, load_json_config, prepare_output_directory
from vlm_eval.mechanistic_heads.preflight import require_current_artifact
from vlm_eval.mechanistic_heads.reproducibility import write_run_manifest


def main() -> None:
    parser=argparse.ArgumentParser(description="Aggregate cross-task general head causal importance.");add_standard_run_arguments(parser);args=parser.parse_args();config=load_json_config(args.config)
    output=args.output_dir/"general_head_importance.tsv";prepare_output_directory(args.output_dir,resume=args.resume,overwrite=args.overwrite,known_outputs=(output.name,"summary.json"))
    families={};inputs=[args.config]
    for source in config["sources"]:
        path=Path(source["path"])
        if not path.is_file():continue
        require_current_artifact(path)
        inputs.append(path);families[source["name"]]=aggregate(path,source["column"],source.get("contrast"))
    minimum=int(config.get("minimum_families",2))
    if len(families)<minimum:raise RuntimeError(f"general causal importance requires {minimum} completed score families; found {len(families)}")
    family_heads = {name: set(values) for name, values in families.items()}
    reference_name = next(iter(family_heads))
    mismatched = {
        name: {
            "missing": len(family_heads[reference_name] - heads),
            "extra": len(heads - family_heads[reference_name]),
        }
        for name, heads in family_heads.items()
        if heads != family_heads[reference_name]
    }
    if mismatched:
        raise RuntimeError(
            "general causal importance requires identical measured head coverage "
            f"across families; reference={reference_name}, mismatches={mismatched}"
        )
    universe=sorted(set.intersection(*(set(values) for values in families.values())))
    normalized={name:percentile_abs(values,universe) for name,values in families.items()}
    rows=[{"layer":head[0],"head":head[1],"general_causal_importance":sum(normalized[name][head] for name in normalized)/len(normalized),"n_score_families":len(normalized),**{f"{name}_absolute_percentile":normalized[name][head] for name in normalized}} for head in universe]
    expected_heads = int(config.get("expected_heads", 64 if args.smoke else 1152))
    valid = bool(rows) and len(rows) == expected_heads
    write_tsv(output,rows);summary={"valid":valid,"label":"instrumentation smoke test" if args.smoke else "methods-based reproduction","n_heads":len(rows),"expected_heads":expected_heads,"families":sorted(families),"definition":"mean within-family percentile of absolute held-out causal score"}
    summary_path=args.output_dir/"summary.json";summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8");write_run_manifest(args.output_dir,config={**config,"smoke":args.smoke},seeds={"render":args.seed},inputs=inputs,outputs=[output,summary_path],status="complete" if summary["valid"] else "failed",repo_root=Path.cwd());print(json.dumps(summary,indent=2))
    if not summary["valid"]:raise SystemExit(1)


def aggregate(path:Path,column:str,contrast:str|None)->dict[tuple[int,int],float]:
    with path.open("r",encoding="utf-8") as handle:rows=list(csv.DictReader(handle,delimiter="\t"))
    grouped=defaultdict(list)
    for row in rows:
        if contrast is not None and row.get("contrast")!=contrast:continue
        if row.get("layer") in (None,"") or row.get("head") in (None,"") or row.get(column) in (None,""):continue
        grouped[(int(row["layer"]),int(row["head"]))].append(float(row[column]))
    return {head:sum(values)/len(values) for head,values in grouped.items()}


def percentile_abs(values:dict[tuple[int,int],float],universe:list[tuple[int,int]])->dict[tuple[int,int],float]:
    ordered=sorted(universe,key=lambda head:(abs(values[head]),head));denominator=max(1,len(ordered)-1);return {head:index/denominator for index,head in enumerate(ordered)}
def write_tsv(path:Path,rows:list[dict[str,Any]])->None:
    with path.open("w",encoding="utf-8",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=sorted({key for row in rows for key in row}) or ["layer"],delimiter="\t");writer.writeheader();writer.writerows(rows)
if __name__=="__main__":main()
