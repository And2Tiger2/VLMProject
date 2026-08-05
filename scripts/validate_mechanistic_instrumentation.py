#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from vlm_eval.mechanistic_heads.causal import batched_candidate_margin, batched_projected_head_patch, candidate_margin, capture_prefill, projected_head_patch, repeat_model_inputs
from vlm_eval.mechanistic_heads.capture import MECHANISTIC_ATTENTION_IMPL, Qwen3CaptureHooks
from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, load_json_config, prepare_output_directory
from vlm_eval.mechanistic_heads.patching import reconstruct_attention_output
from vlm_eval.mechanistic_heads.preflight import REQUIRED_CHECKS
from vlm_eval.mechanistic_heads.qwen3_runtime import Qwen3MechanisticRuntime
from vlm_eval.mechanistic_heads.reproducibility import seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.synthetic import render_syndot, syndot_positions
from vlm_eval.mechanistic_heads.token_spans import TokenSpan, TokenSpans, contiguous_span


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mandatory Qwen3 mechanistic instrumentation checks.")
    add_standard_run_arguments(parser)
    parser.add_argument("--device-map", default="cuda")
    args = parser.parse_args()
    config = load_json_config(args.config)
    report_path = args.output_dir / "instrumentation_validation.json"
    prepare_output_directory(args.output_dir, resume=args.resume, overwrite=args.overwrite, known_outputs=(report_path.name,))
    seed_everything(args.seed)
    fixture = args.output_dir / "fixture_syndot.png"
    render_syndot(4, syndot_positions(args.seed, "instrumentation")).save(fixture)
    runtime = Qwen3MechanisticRuntime(model_id=str(config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")), device_map=args.device_map)
    checks, diagnostics = validate_runtime(runtime, fixture)
    # Pure CPU requirements are exercised by the named test module. Run it in
    # the same environment so a green report cannot omit those checks.
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_mechanistic_heads_common.py",
            "tests/test_mechanistic_synthetic_generators.py",
        ],
        text=True,
        capture_output=True,
    )
    pure_ok = completed.returncode == 0
    for name in ("self_subtraction_noop", "teacher_forcing_likelihood", "generator_determinism", "split_leakage", "reproducibility_manifest"):
        checks[name] = pure_ok
    diagnostics["cpu_test_output"] = completed.stdout + completed.stderr
    report = {"valid": all(checks.get(name) is True for name in REQUIRED_CHECKS), "label": "instrumentation smoke test", "architecture": vars(runtime.architecture), "checks": checks, "diagnostics": diagnostics, "errors": []}
    if not report["valid"]:
        report["errors"] = [name for name in REQUIRED_CHECKS if checks.get(name) is not True]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_run_manifest(args.output_dir, config={**config, "smoke": True, "architecture": vars(runtime.architecture)}, seeds={"global": args.seed}, inputs=[args.config], outputs=[report_path], status="complete" if report["valid"] else "failed", repo_root=Path.cwd())
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


def validate_runtime(runtime, fixture: Path):
    import torch

    checks = {name: False for name in REQUIRED_CHECKS}
    diagnostics = {}
    prompt = "What is the number of black dots? Answer with the number only."
    inputs = runtime.prepare(Image.open(fixture).convert("RGB"), prompt, prompt_mode="raw")
    layers = [0]
    capture = capture_prefill(runtime, image_path=fixture, prompt=prompt, layers=layers)
    layer = runtime.model.model.language_model.layers[0]
    reconstructed = reconstruct_attention_output(capture.store.projected_heads[0], layer.self_attn.o_proj.bias)
    observed = capture.store.attention_outputs[0]
    reconstruction_error = float((reconstructed - observed).float().abs().max().detach().cpu())
    checks["projected_head_reconstruction"] = reconstruction_error < 2e-3
    diagnostics["projected_head_reconstruction_max_abs"] = reconstruction_error
    probabilities = capture.store.attention_probabilities[0]
    normalization_error = float((probabilities.float().sum(-1) - 1).abs().max().detach().cpu())
    checks["attention_normalization"] = normalization_error < 1e-5
    diagnostics["attention_normalization_max_abs"] = normalization_error
    base, _ = candidate_margin(runtime, inputs, positive_answer="4", negative_answer="5")
    with projected_head_patch(runtime.model, layer_idx=0, head_idx=0, donor_projected=capture.store.projected_heads[0], recipient_projected=capture.store.projected_heads[0], positions=[capture.prompt_length - 1]):
        identity, _ = candidate_margin(runtime, inputs, positive_answer="4", negative_answer="5")
    checks["identity_patch"] = abs(identity - base) < 1e-5
    diagnostics["identity_patch_margin_delta"] = identity - base
    donor_fixture = fixture.with_name("fixture_syndot_donor.png")
    render_syndot(5, syndot_positions(17, "instrumentation-donor")).save(donor_fixture)
    donor = capture_prefill(runtime, image_path=donor_fixture, prompt=prompt, layers=layers)
    if donor.prompt_length != capture.prompt_length:
        raise RuntimeError("instrumentation donor/recipient prefill lengths differ")
    serial = []
    for head in (0, 1):
        with projected_head_patch(runtime.model, layer_idx=0, head_idx=head, donor_projected=donor.store.projected_heads[0], recipient_projected=capture.store.projected_heads[0], positions=[capture.prompt_length - 1]):
            value, _ = candidate_margin(runtime, inputs, positive_answer="5", negative_answer="4")
        serial.append(value)
    repeated = repeat_model_inputs(inputs, 2)
    with batched_projected_head_patch(runtime.model, layer_idx=0, head_indices=[0, 1], donor_projected=donor.store.projected_heads[0], recipient_projected=capture.store.projected_heads[0], positions=[capture.prompt_length - 1]):
        batched, _ = batched_candidate_margin(runtime, repeated, positive_answer="5", negative_answer="4")
    batch_error = max(abs(serial[index] - float(batched[index].detach().cpu())) for index in range(2))
    checks["batched_serial_agreement"] = batch_error < 1e-5
    diagnostics["batched_serial_margin_max_abs"] = batch_error
    spans = runtime.trace_spans(inputs, prompt)
    spans.assert_partition_bounds(); checks["token_spans"] = True; diagnostics["token_spans"] = spans.to_dict()
    # Compare the custom eager implementation with the repository loader's
    # normal SDPA backend using the same weights and exact prepared tensors.
    config_obj = layer.self_attn.config
    original_backend = config_obj._attn_implementation
    with torch.no_grad():
        config_obj._attn_implementation = "sdpa"
        normal_logits = runtime.model(**inputs, use_cache=False, return_dict=True).logits
        config_obj._attn_implementation = MECHANISTIC_ATTENTION_IMPL
        custom_logits = runtime.model(**inputs, use_cache=False, return_dict=True).logits
        config_obj._attn_implementation = original_backend
    backend_error = float((normal_logits.float() - custom_logits.float()).abs().max().detach().cpu())
    checks["backend_equivalence"] = backend_error < 5e-3
    diagnostics["backend_max_abs_logit_delta"] = backend_error
    # Cached versus uncached next-token logits for one tested answer step.
    first_token = runtime.answer_token_ids(" one")[:, :1]
    combined = torch.cat([inputs.input_ids, first_token], dim=1)
    full_kwargs = dict(inputs); full_kwargs["input_ids"] = combined
    if "attention_mask" in full_kwargs:
        full_kwargs["attention_mask"] = torch.cat([full_kwargs["attention_mask"], full_kwargs["attention_mask"].new_ones((1, 1))], dim=1)
    with torch.no_grad():
        full = runtime.model(**full_kwargs, use_cache=False, return_dict=True).logits[:, -1, :]
        prefill = runtime.model(**inputs, use_cache=True, return_dict=True)
        prepared = runtime.model.prepare_inputs_for_generation(combined, past_key_values=prefill.past_key_values, attention_mask=full_kwargs.get("attention_mask"), pixel_values=inputs.get("pixel_values"), image_grid_thw=inputs.get("image_grid_thw"), use_cache=True)
        cached = runtime.model(**prepared, return_dict=True).logits[:, -1, :]
    cache_error = float((full.float() - cached.float()).abs().max().detach().cpu())
    checks["cached_uncached_equivalence"] = cache_error < 5e-3
    diagnostics["cached_uncached_max_abs_logit_delta"] = cache_error
    return checks, diagnostics


if __name__ == "__main__":
    main()
