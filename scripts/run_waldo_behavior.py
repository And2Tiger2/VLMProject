#!/usr/bin/env python3
"""Behavioral calibration for original synthetic Waldo-like tasks."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, effective_limit, load_json_config, prepare_output_directory
from vlm_eval.mechanistic_heads.preflight import require_scientific_validation, validation_path_from_config
from vlm_eval.mechanistic_heads.qwen3_runtime import checkpoint_manifest_inputs, runtime_from_config
from vlm_eval.mechanistic_heads.reproducibility import referenced_image_paths, seed_everything, write_run_manifest


CELL_RE = re.compile(r"(?:cell|candidate)\s*=\s*(\d{1,2})", re.I)
POINT_RE = re.compile(r"point\s*=\s*\((0(?:\.\d+)?|1(?:\.0+)?),(0(?:\.\d+)?|1(?:\.0+)?)\)", re.I)


def main() -> None:
    parser=argparse.ArgumentParser(description="Evaluate synthetic Waldo-like localization and verification tasks.")
    add_standard_run_arguments(parser);parser.add_argument("--device-map",default="cuda");parser.add_argument("--checkpoint")
    args=parser.parse_args();config=load_json_config(args.config)
    if not args.smoke:require_scientific_validation(validation_path_from_config(config))
    output=args.output_dir/"waldo_behavior.tsv"
    prepare_output_directory(args.output_dir,resume=args.resume,overwrite=args.overwrite,known_outputs=(output.name,"summary.json"));seed_everything(args.seed)
    dataset=read_jsonl(Path(config["dataset"]));dataset=[row for row in dataset if row["split"]==str(config.get("split","locked_test"))];limit=effective_limit(args)
    if limit is not None:dataset=dataset[:limit]
    runtime=runtime_from_config(config,device_map=args.device_map,checkpoint_override=args.checkpoint)
    rows=[]
    for example in dataset:
        tasks=task_specs(example)
        for task,image_path,prompt,expected in tasks:
            inputs=runtime.prepare(Image.open(image_path).convert("RGB"),prompt,prompt_mode="raw")
            with runtime.torch.no_grad():generated=runtime.model.generate(**inputs,do_sample=False,max_new_tokens=int(config.get("max_new_tokens",32)))
            text=runtime.processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
            parsed,correct,error=parse_result(task,text,expected)
            rows.append({"id":example["id"],"task":task,"target_present":int(example["target_present"]),"target_scale":example["metadata"]["target_scale"],"scene_zoom":example["metadata"].get("scene_zoom",1.0),"clutter":len(example["objects"])-int(example["target_present"]),"distractor_similarity":example["metadata"].get("distractor_similarity"),"occluded":int(example["metadata"].get("occluded",False)),"prompt_wording_variant":example["metadata"]["prompt_wording_variant"],"expected":expected,"output":text,"parsed":parsed,"correct":int(correct),"localization_error":error})
    by_task=summarize(rows);minimums={str(key):float(value) for key,value in config.get("minimum_calibration_accuracy",{}).items()};calibration_checks={task:task in by_task and float(by_task[task]["accuracy"])>=minimum for task,minimum in minimums.items()};calibration_passed=bool(calibration_checks) and all(calibration_checks.values())
    write_tsv(output,rows);summary={"valid":True,"label":"instrumentation smoke test" if args.smoke else ("modified replication" if calibration_passed else "failed calibration"),"n_examples":len(dataset),"by_task":by_task,"minimum_calibration_accuracy":minimums,"calibration_checks":calibration_checks,"calibration_passed":calibration_passed,"architecture":vars(runtime.architecture),"deviation":"original non-copyright target and deterministic text output replace real Waldo/HTML boxes"}
    summary_path=args.output_dir/"summary.json";summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8");write_run_manifest(args.output_dir,config={**config,"smoke":args.smoke,"architecture":vars(runtime.architecture)},seeds={"global":args.seed},inputs=[args.config,Path(config["dataset"]),*referenced_image_paths(dataset),*checkpoint_manifest_inputs(config,checkpoint_override=args.checkpoint)],outputs=[output,summary_path],status="complete",repo_root=Path.cwd());print(json.dumps(summary,indent=2))
    if not args.smoke and not calibration_passed:
        raise SystemExit("Waldo-like behavioral calibration failed; causal scans are blocked")


def task_specs(row:dict[str,Any])->list[tuple[str,str,str,str]]:
    cell="absent" if row["target_cell"] is None else f"cell={int(row['target_cell']):02d}"
    point=row["tasks"]["normalized_point"]
    candidates=row["metadata"]["four_candidate_cells"]
    candidate_prompt="Candidate cells are "+", ".join(f"{index}={cell:02d}" for index,cell in enumerate(candidates))+". Which candidate is the four-feature target? Answer candidate=N, or absent."
    return [
        ("invisible_grid",row["image_path"],row["prompts"][row["metadata"]["prompt_wording_variant"]%2],cell),
        ("visible_grid_ocr",row["metadata"]["visible_grid_image"],"Read the visible 10x10 grid label at the target. Answer cell=NN.",cell),
        ("normalized_point",row["image_path"],"Give the normalized center of the four-feature target. Answer point=(x,y), or absent.",point),
        ("four_candidate_selection",row["image_path"],candidate_prompt,row["tasks"]["four_candidate_selection"]),
        ("presence",row["image_path"],row["prompts"][2],row["tasks"]["presence"]),
    ]


def parse_result(task:str,text:str,expected:str)->tuple[str,bool,float|str]:
    normalized=text.casefold().strip(" .")
    if expected=="absent":return ("absent" if "absent" in normalized else normalized,"absent" in normalized,"")
    if task=="presence":return normalized,normalized==expected,""
    if task=="normalized_point":
        actual=POINT_RE.search(text);truth=POINT_RE.search(expected)
        if not actual or not truth:return "",False,""
        ax,ay=float(actual.group(1)),float(actual.group(2));tx,ty=float(truth.group(1)),float(truth.group(2));error=((ax-tx)**2+(ay-ty)**2)**.5;return f"point=({ax:.3f},{ay:.3f})",error<=.05,error
    actual=CELL_RE.search(text);truth=CELL_RE.search(expected)
    if not actual or not truth:return "",False,""
    return actual.group(1),int(actual.group(1))==int(truth.group(1)),abs(int(actual.group(1))-int(truth.group(1)))


def summarize(rows:list[dict[str,Any]])->dict[str,Any]:
    result={}
    for task in sorted({row["task"] for row in rows}):
        group=[row for row in rows if row["task"]==task];errors=[float(row["localization_error"]) for row in group if row["localization_error"]!=""]
        result[task]={"n":len(group),"accuracy":sum(row["correct"] for row in group)/len(group),"mean_localization_error":sum(errors)/len(errors) if errors else None}
    return result


def read_jsonl(path:Path)->list[dict[str,Any]]:return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def write_tsv(path:Path,rows:list[dict[str,Any]])->None:
    with path.open("w",encoding="utf-8",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=sorted({key for row in rows for key in row}) or ["id"],delimiter="\t");writer.writeheader();writer.writerows(rows)
if __name__=="__main__":main()
