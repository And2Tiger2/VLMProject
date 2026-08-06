#!/usr/bin/env python3
"""Split-half/cross-seed sign and rank stability for MACI heads."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, load_json_config, prepare_output_directory
from vlm_eval.mechanistic_heads.reproducibility import write_run_manifest


def main()->None:
    parser=argparse.ArgumentParser(description="Analyze MACI signed-head stability.");add_standard_run_arguments(parser);args=parser.parse_args();config=load_json_config(args.config);output=args.output_dir/"maci_head_stability.tsv";prepare_output_directory(args.output_dir,resume=args.resume,overwrite=args.overwrite,known_outputs=(output.name,"summary.json"))
    source=Path(config["per_example_scores"]);rows=[row for row in read_tsv(source) if row.get("excluded","").casefold() not in {"true","1"}];groups=sorted({row["group_id"] for row in rows});halves=(set(groups[::2]),set(groups[1::2]));scores=[aggregate([row for row in rows if row["group_id"] in half]) for half in halves];comparison=compare(scores[0],scores[1],name="split_half",k=int(config.get("top_k",40)));comparisons=[comparison];inputs=[args.config,source]
    for index,path_value in enumerate(config.get("repeat_per_example_scores",[])):
        path=Path(path_value)
        if path.is_file():inputs.append(path);comparisons.append(compare(aggregate(rows),aggregate([row for row in read_tsv(path) if row.get("excluded","").casefold() not in {"true","1"}]),name=f"cross_seed_{index}",k=int(config.get("top_k",40))))
    write_tsv(output,comparisons);minimum_rho=float(config.get("minimum_spearman",.5));minimum_sign=float(config.get("minimum_sign_agreement",.7));passes=all(row["spearman_rho"]>=minimum_rho and row["sign_agreement"]>=minimum_sign for row in comparisons) and len(comparisons)>=1
    summary={"valid":True,"label":"methods-based reproduction" if passes else "failed calibration","passes_stability_gate":passes,"thresholds":{"minimum_spearman":minimum_rho,"minimum_sign_agreement":minimum_sign},"comparisons":comparisons,"claim_gate":"Driving/resisting labels remain provisional unless this gate and locked ablations pass."};summary_path=args.output_dir/"summary.json";summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8");write_run_manifest(args.output_dir,config=config,seeds={"analysis":args.seed},inputs=inputs,outputs=[output,summary_path],status="complete",repo_root=Path.cwd());print(json.dumps(summary,indent=2))


def aggregate(rows:list[dict[str,str]])->dict[tuple[int,int],float]:
    grouped=defaultdict(list)
    for row in rows:grouped[(int(row["layer"]),int(row["head"]))].append(float(row["signed_intervention_score"]))
    return {head:sum(values)/len(values) for head,values in grouped.items()}
def compare(left,right,*,name,k):
    common=sorted(set(left)&set(right));lr=ranks([left[head] for head in common]);rr=ranks([right[head] for head in common]);rho=correlation(lr,rr);sign=sum((left[head]>=0)==(right[head]>=0) for head in common)/len(common);ld=set(sorted(common,key=lambda head:left[head],reverse=True)[:k]);rd=set(sorted(common,key=lambda head:right[head],reverse=True)[:k]);lrset=set(sorted(common,key=lambda head:left[head])[:k]);rrset=set(sorted(common,key=lambda head:right[head])[:k]);return {"comparison":name,"n_heads":len(common),"spearman_rho":rho,"sign_agreement":sign,"driving_top_k_overlap":len(ld&rd),"resisting_top_k_overlap":len(lrset&rrset),"k":k}
def ranks(values):
    order=sorted(range(len(values)),key=lambda idx:values[idx]);result=[0]*len(values)
    for rank,index in enumerate(order):result[index]=rank
    return result
def correlation(left,right):
    import math
    lm=sum(left)/len(left);rm=sum(right)/len(right);num=sum((a-lm)*(b-rm) for a,b in zip(left,right));den=math.sqrt(sum((a-lm)**2 for a in left)*sum((b-rm)**2 for b in right));return num/den if den else 0.0
def read_tsv(path):
    with path.open("r",encoding="utf-8") as handle:return list(csv.DictReader(handle,delimiter="\t"))
def write_tsv(path,rows):
    with path.open("w",encoding="utf-8",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=sorted({key for row in rows for key in row}) or ["comparison"],delimiter="\t");writer.writeheader();writer.writerows(rows)
if __name__=="__main__":main()
