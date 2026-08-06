#!/usr/bin/env python3
"""Locked VLMBias role-aware intervention and NaturalBench retention."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
import numpy as np

from vlm_eval.datasets import load_examples
from vlm_eval.metrics import prediction_to_dict, score_response, summarize
from vlm_eval.naturalbench import NaturalBenchPrediction, extract_naturalbench_answer, load_naturalbench_calls, normalize_naturalbench_answer, summarize_naturalbench
from vlm_eval.mechanistic_heads.causal import candidate_margin, capture_prefill, projected_head_scaling
from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, effective_limit, load_json_config, prepare_output_directory
from vlm_eval.mechanistic_heads.controls import layer_matched_control_draws, multivariate_matched_control_draws
from vlm_eval.mechanistic_heads.preflight import require_scientific_validation, validation_path_from_config
from vlm_eval.mechanistic_heads.qwen3_runtime import Qwen3MechanisticRuntime
from vlm_eval.mechanistic_heads.reproducibility import seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.schema import read_paired_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate signed heads on locked VLMBias and NaturalBench.")
    add_standard_run_arguments(parser); parser.add_argument("--device-map", default="cuda")
    args=parser.parse_args(); config=load_json_config(args.config)
    if not args.smoke: require_scientific_validation(validation_path_from_config(config))
    output=args.output_dir/"vlmbias_predictions.jsonl"; prepare_output_directory(args.output_dir,resume=args.resume,overwrite=args.overwrite,known_outputs=(output.name,))
    seed_everything(args.seed); runtime=Qwen3MechanisticRuntime(model_id=str(config.get("model_id","Qwen/Qwen3-VL-8B-Instruct")),device_map=args.device_map)
    score_rows=read_tsv(Path(config["head_scores"])); contrast=str(config.get("selection_contrast","semantic_prior")); selected_rows=[row for row in score_rows if row["contrast"]==contrast]
    selected_rows.sort(key=lambda row:float(row["mean_signed_score"]),reverse=True); driving=[(int(row["layer"]),int(row["head"])) for row in selected_rows if float(row["mean_signed_score"])>0][:int(config.get("driving_k",30))]; resisting=[(int(row["layer"]),int(row["head"])) for row in reversed(selected_rows) if float(row["mean_signed_score"])<0][:int(config.get("resisting_k",40))]
    detector = json.loads(Path(config["conflict_detector"]).read_text(encoding="utf-8")) if config.get("conflict_detector") and Path(config["conflict_detector"]).is_file() else None
    conditions=build_conditions(driving,resisting,score_rows,config,args.seed,runtime.architecture.n_layers,runtime.architecture.n_heads,have_detector=detector is not None,include_controls=not args.smoke,require_external_general=not args.smoke)
    if args.smoke: conditions={key:value for key,value in conditions.items() if key in {"baseline","driving_suppress","resisting_amplify","joint_role_aware","conflict_gated"}}
    locked_ids={pair.pair_id.split("-",1)[1] for pair in read_paired_jsonl(Path(config["paired_contrasts"])) if pair.split==str(config.get("split","locked_test")) and pair.metadata["contrast"]==contrast}
    examples=[example for example in load_examples(str(config["vlmbias_dataset"])) if example.id in locked_ids]; limit=effective_limit(args)
    if limit is None and config.get("max_examples") is not None: limit=int(config["max_examples"])
    if limit is not None: examples=examples[:limit]
    rows=[]; baseline_state={}
    for condition,(scales,gate) in conditions.items():
        for example in examples:
            image = example.image or Image.open(example.image_path).convert("RGB")
            inputs=runtime.prepare(image,example.prompt,prompt_mode="raw"); margin,_=candidate_margin(runtime,inputs,positive_answer=example.expected_bias,negative_answer=example.ground_truth)
            detector_probability_value = conflict_probability(runtime, image, example.prompt, detector) if gate == "detector" else None
            intervene = gate == "always" or (gate == "detector" and detector_probability_value >= float(detector["threshold"]))
            with projected_head_scaling(runtime.model,scales if intervene else {}):
                post_margin,_=candidate_margin(runtime,inputs,positive_answer=example.expected_bias,negative_answer=example.ground_truth)
                generated=runtime.model.generate(**inputs,do_sample=False,max_new_tokens=int(config.get("max_new_tokens",16)))
            text=runtime.processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip(); prediction=score_response(example,text); state=prediction_state(prediction)
            if condition=="baseline": baseline_state[example.id]=state
            rows.append({"condition":condition,"intervened":int(intervene),"conflict_probability":detector_probability_value,"baseline_bias_minus_correct_margin":margin,"bias_minus_correct_margin":post_margin,"margin_shift":post_margin-margin,"state":state,**prediction_to_dict(prediction)})
    output.write_text("".join(json.dumps(row,ensure_ascii=False,default=str)+"\n" for row in rows),encoding="utf-8")
    summaries={}; transitions={}
    for condition in conditions:
        group=[row for row in rows if row["condition"]==condition]; predictions=[dict_to_prediction(row) for row in group]; summaries[condition]={**summarize(predictions),"invalid_rate":sum(row["state"]=="invalid" for row in group)/len(group) if group else 0,"unconditional_bias_answer_rate":sum(row["state"]=="bias" for row in group)/len(group) if group else 0,"mean_bias_minus_correct_margin":sum(row["bias_minus_correct_margin"] for row in group)/len(group) if group else None,"mean_margin_shift":sum(row["margin_shift"] for row in group)/len(group) if group else None,"intervention_rate":sum(row["intervened"] for row in group)/len(group) if group else 0}; counts=Counter(f"{baseline_state.get(row['example_id'],'missing')}->{row['state']}" for row in group);transitions[condition]={key:counts.get(key,0) for key in ("bias->correct","bias->other_wrong","correct->bias","correct->other_wrong")};transitions[condition]["all"]=dict(counts)
    naturalbench={}; naturalbench_input_paths=[]
    if config.get("naturalbench_dataset"):
        calls=load_naturalbench_calls(str(config["naturalbench_dataset"]),limit_groups=(2 if args.smoke else config.get("naturalbench_limit_groups")))
        naturalbench_input_paths=[Path(call.image_path) for call in calls]
        for condition,(scales,gate) in conditions.items():
            if not should_run_naturalbench(condition, config):
                continue
            predictions=[]
            for call in calls:
                image = Image.open(call.image_path).convert("RGB")
                inputs=runtime.prepare(image,call.prompt,prompt_mode="raw")
                probability = conflict_probability(runtime, image, call.prompt, detector) if gate == "detector" else None
                intervene = gate == "always" or (gate == "detector" and probability >= float(detector["threshold"]))
                with projected_head_scaling(runtime.model,scales if intervene else {}): generated=runtime.model.generate(**inputs,do_sample=False,max_new_tokens=8)
                text=runtime.processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip(); parsed=extract_naturalbench_answer(text,call.question_type); predictions.append(NaturalBenchPrediction(call.group_id,call.call_id,call.question_id,call.image_id,call.question_type,call.prompt,call.ground_truth,text,parsed,normalize_naturalbench_answer(parsed)==normalize_naturalbench_answer(call.ground_truth),call.source))
            naturalbench[condition]=summarize_naturalbench(predictions)
    summary={"valid":True,"label":"instrumentation smoke test" if args.smoke else "locked confirmation","selection_contrast":contrast,"n_locked_examples":len(examples),"head_sets":{"driving":driving,"resisting":resisting},"vlmbias":summaries,"transitions":transitions,"naturalbench":naturalbench,"control_policy":"20 joint matches on layer, image attention, projected norm, entropy, gaze, and general causal importance; NaturalBench retention is evaluated on core interventions only","architecture":vars(runtime.architecture)}
    summary_path=args.output_dir/"summary.json";summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    manifest_inputs=[args.config,Path(config["head_scores"]),Path(config["paired_contrasts"]),Path(config["vlmbias_dataset"]),*[Path(example.image_path) for example in examples if example.image_path],*naturalbench_input_paths]
    for key in ("naturalbench_dataset","gaze_ranking","conflict_detector","general_causal_importance"):
        if config.get(key):manifest_inputs.append(Path(config[key]))
    write_run_manifest(args.output_dir,config={**config,"architecture":vars(runtime.architecture)},seeds={"global":args.seed},inputs=manifest_inputs,outputs=[output,summary_path],status="complete",repo_root=Path.cwd());print(json.dumps(summary,indent=2))


def build_conditions(driving,resisting,rows,config,seed,n_layers,n_heads,*,have_detector,include_controls,require_external_general):
    conditions={"baseline":({},"never"),"driving_suppress":({head:0.0 for head in driving},"always"),"resisting_amplify":({head:float(config.get("resisting_scale",2.0)) for head in resisting},"always"),"joint_role_aware":({**{head:0.0 for head in driving},**{head:float(config.get("resisting_scale",2.0)) for head in resisting}},"always")}
    if have_detector: conditions["conflict_gated"] = ({**{head:0.0 for head in driving},**{head:float(config.get("resisting_scale",2.0)) for head in resisting}},"detector")
    if not include_controls:
        return conditions
    gaze={(int(row["layer"]),int(row["head"])):float(row.get("score",0)) for row in json.loads(Path(config["gaze_ranking"]).read_text(encoding="utf-8"))}; features={}
    by_head=defaultdict(list)
    for row in rows:by_head[(int(row["layer"]),int(row["head"]))].append(row)
    for head,group in by_head.items():features[head]={"image_attention":avg(group,"image_attention"),"projected_output_norm":avg(group,"projected_output_norm"),"attention_entropy":avg(group,"attention_entropy"),"gaze_score":gaze.get(head,0),"general_causal_importance":sum(abs(float(row["mean_signed_score"])) for row in group)/len(group)}
    general_path=Path(config.get("general_causal_importance",""))
    if general_path.is_file():
        for row in read_tsv(general_path):features[(int(row["layer"]),int(row["head"]))]["general_causal_importance"]=float(row["general_causal_importance"])
    elif require_external_general:raise RuntimeError("locked VLMBias validation requires cross-task general causal importance controls")
    draws=max(20,int(config.get("control_draws",20)))
    diagnostic_families=bool(config.get("diagnostic_single_feature_controls",False))
    for role,selected,scale in (("driving",driving,0.0),("resisting",resisting,float(config.get("resisting_scale",2.0)))):
        families={"fully":multivariate_matched_control_draws(selected,features,n_draws=draws,seed=seed+6)}
        if diagnostic_families:
            families.update({"layer":layer_matched_control_draws(selected,n_layers=n_layers,n_heads=n_heads,n_draws=draws,seed=seed),"image":multivariate_matched_control_draws(selected,features,feature_names=("image_attention",),n_draws=draws,seed=seed+1),"norm":multivariate_matched_control_draws(selected,features,feature_names=("projected_output_norm",),n_draws=draws,seed=seed+2),"entropy":multivariate_matched_control_draws(selected,features,feature_names=("attention_entropy",),n_draws=draws,seed=seed+3),"gaze":multivariate_matched_control_draws(selected,features,feature_names=("gaze_score",),n_draws=draws,seed=seed+4),"general":multivariate_matched_control_draws(selected,features,feature_names=("general_causal_importance",),n_draws=draws,seed=seed+5)})
        for family,control_draws in families.items():
            for index,heads in enumerate(control_draws):conditions[f"control_{role}_{family}_{index:02d}"]=({head:scale for head in heads},"always")
    return conditions
def conflict_probability(runtime, image, prompt, detector):
    if detector is None:return None
    resisting=[tuple(head) for head in detector["resisting_heads"]];capture=capture_prefill(runtime,image_path=image,prompt=prompt,layers=sorted({layer for layer,_ in resisting}),to_cpu=True);vectors=[capture.store.raw_heads[layer][0,capture.prompt_length-1,head,:].float().detach().cpu().numpy() for layer,head in resisting];feature=np.stack(vectors).mean(0);scaled=(feature-np.asarray(detector["scaler_mean"]))/np.asarray(detector["scaler_scale"]);logit=float(scaled@np.asarray(detector["coefficient"])+np.asarray(detector["intercept"])[0]);return 1/(1+math.exp(-max(-30,min(30,logit))))
def prediction_state(prediction):
    if not prediction.parsed_answer:return "invalid"
    if prediction.is_correct:return "correct"
    if prediction.is_bias_aligned_error:return "bias"
    return "other_wrong"
def dict_to_prediction(row):
    from vlm_eval.types import Prediction
    fields=Prediction.__dataclass_fields__;return Prediction(**{key:row[key] for key in fields})
def avg(rows,key):return sum(float(row[key]) for row in rows)/len(rows)
def should_run_naturalbench(condition,config):return condition.startswith(tuple(config.get("naturalbench_condition_prefixes",("baseline","driving_suppress","resisting_amplify","joint_role_aware","conflict_gated"))))
def read_tsv(path):
    with path.open("r",encoding="utf-8") as handle:return list(csv.DictReader(handle,delimiter="\t"))
if __name__=="__main__":main()
