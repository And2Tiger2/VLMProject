#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from vlm_eval.mechanistic_heads.causal import batched_candidate_margin, batched_projected_head_patch, candidate_margin, capture_prefill, projected_head_patch, repeat_model_inputs
from vlm_eval.mechanistic_heads.capture import MECHANISTIC_ATTENTION_IMPL
from vlm_eval.mechanistic_heads.config import add_standard_run_arguments, load_json_config, prepare_output_directory
from vlm_eval.mechanistic_heads.patching import reconstruct_attention_output
from vlm_eval.mechanistic_heads.preflight import REQUIRED_CHECKS
from vlm_eval.mechanistic_heads.qwen3_runtime import Qwen3MechanisticRuntime
from vlm_eval.mechanistic_heads.reproducibility import seed_everything, write_run_manifest
from vlm_eval.mechanistic_heads.synthetic import render_syndot, syndot_positions


def backend_equivalence_passes(
    *,
    eager_custom_error: float,
    eager_custom_tolerance: float,
    custom_candidate_margin: float,
    sdpa_candidate_margin: float,
    greedy_agreement: bool,
) -> tuple[bool, bool]:
    """Validate capture fidelity without imposing SDPA's BF16 reduction path.

    The mechanistic implementation intentionally follows Hugging Face eager
    attention because it must materialize probabilities and per-head values.
    It must therefore reproduce eager logits to numerical precision. SDPA uses
    a different fused BF16 reduction and can move logit magnitudes materially
    while preserving inference decisions. Treat SDPA as a behavioral parity
    check: it must preserve both the greedy token and the ordering of the two
    tested answer candidates. Keep the magnitude delta as telemetry rather
    than disguising backend arithmetic as a capture error.
    """

    candidate_order_agreement = (
        custom_candidate_margin > 0 and sdpa_candidate_margin > 0
    ) or (
        custom_candidate_margin < 0 and sdpa_candidate_margin < 0
    )
    passed = (
        eager_custom_error <= eager_custom_tolerance
        and candidate_order_agreement
        and greedy_agreement
    )
    return passed, candidate_order_agreement


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

    def bf16_aware_tolerance(tensor, *, units: float, floor: float) -> float:
        """Absolute tolerance scaled to the tensor's storage precision."""

        if not tensor.dtype.is_floating_point:
            return floor
        scale = max(1.0, float(tensor.float().abs().max().detach().cpu()))
        return max(floor, units * float(torch.finfo(tensor.dtype).eps) * scale)

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
    reconstruction_tolerance = bf16_aware_tolerance(
        observed, units=2.0, floor=2e-5
    )
    checks["projected_head_reconstruction"] = reconstruction_error <= reconstruction_tolerance
    diagnostics["projected_head_reconstruction_max_abs"] = reconstruction_error
    diagnostics["projected_head_reconstruction_tolerance"] = reconstruction_tolerance
    probabilities = capture.store.attention_probabilities[0]
    normalization_error = float((probabilities.float().sum(-1) - 1).abs().max().detach().cpu())
    normalization_tolerance = bf16_aware_tolerance(
        probabilities, units=1.0, floor=1e-6
    )
    checks["attention_normalization"] = normalization_error <= normalization_tolerance
    diagnostics["attention_normalization_max_abs"] = normalization_error
    diagnostics["attention_normalization_tolerance"] = normalization_tolerance
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
    repeated = repeat_model_inputs(inputs, 2)
    serial = []
    for head in (0, 1):
        with projected_head_patch(runtime.model, layer_idx=0, head_idx=head, donor_projected=donor.store.projected_heads[0], recipient_projected=capture.store.projected_heads[0], positions=[capture.prompt_length - 1]):
            values, _ = batched_candidate_margin(
                runtime, repeated, positive_answer="5", negative_answer="4"
            )
        serial.append(float(values[head].detach().cpu()))
    with batched_projected_head_patch(runtime.model, layer_idx=0, head_indices=[0, 1], donor_projected=donor.store.projected_heads[0], recipient_projected=capture.store.projected_heads[0], positions=[capture.prompt_length - 1]):
        batched, _ = batched_candidate_margin(runtime, repeated, positive_answer="5", negative_answer="4")
    batch_error = max(abs(serial[index] - float(batched[index].detach().cpu())) for index in range(2))
    checks["batched_serial_agreement"] = batch_error < 5e-3
    diagnostics["batched_serial_margin_max_abs"] = batch_error
    spans = runtime.trace_spans(inputs, prompt)
    spans.assert_partition_bounds()
    checks["token_spans"] = True
    diagnostics["token_spans"] = spans.to_dict()
    # First prove that our capture implementation is equivalent to HF eager,
    # then compare its tested answer margin and greedy token with the
    # repository-normal SDPA backend. Raw maxima over every vocabulary item at
    # every visual position are not a meaningful inference equivalence test.
    config_obj = layer.self_attn.config
    original_backend = config_obj._attn_implementation
    try:
        with torch.no_grad():
            config_obj._attn_implementation = "eager"
            eager_logits = runtime.model(**inputs, use_cache=False, return_dict=True).logits
            config_obj._attn_implementation = MECHANISTIC_ATTENTION_IMPL
            custom_logits = runtime.model(**inputs, use_cache=False, return_dict=True).logits
            config_obj._attn_implementation = "sdpa"
            normal_logits = runtime.model(**inputs, use_cache=False, return_dict=True).logits
    finally:
        config_obj._attn_implementation = original_backend
    eager_custom_error = float(
        (eager_logits.float() - custom_logits.float()).abs().max().detach().cpu()
    )
    next_custom = custom_logits[:, -1, :].float()
    next_normal = normal_logits[:, -1, :].float()
    candidate_ids = torch.cat(
        [runtime.answer_token_ids("4")[:, :1], runtime.answer_token_ids("5")[:, :1]],
        dim=1,
    )[0]
    custom_log_probs = torch.log_softmax(next_custom, dim=-1)
    normal_log_probs = torch.log_softmax(next_normal, dim=-1)
    custom_margin = custom_log_probs[0, candidate_ids[0]] - custom_log_probs[0, candidate_ids[1]]
    normal_margin = normal_log_probs[0, candidate_ids[0]] - normal_log_probs[0, candidate_ids[1]]
    backend_margin_error = float((custom_margin - normal_margin).abs().detach().cpu())
    backend_greedy_agreement = bool(
        next_custom.argmax(-1).equal(next_normal.argmax(-1))
    )
    # These two paths call the same eager equations, so unlike the SDPA
    # comparison they should be bitwise identical.
    eager_custom_tolerance = 1e-5
    backend_passed, backend_candidate_order_agreement = backend_equivalence_passes(
        eager_custom_error=eager_custom_error,
        eager_custom_tolerance=eager_custom_tolerance,
        custom_candidate_margin=float(custom_margin.detach().cpu()),
        sdpa_candidate_margin=float(normal_margin.detach().cpu()),
        greedy_agreement=backend_greedy_agreement,
    )
    checks["backend_equivalence"] = backend_passed
    diagnostics["eager_custom_max_abs_logit_delta"] = eager_custom_error
    diagnostics["eager_custom_tolerance"] = eager_custom_tolerance
    diagnostics["normal_custom_candidate_margin_delta"] = backend_margin_error
    diagnostics["custom_candidate_margin"] = float(custom_margin.detach().cpu())
    diagnostics["normal_candidate_margin"] = float(normal_margin.detach().cpu())
    diagnostics["normal_custom_candidate_order_agreement"] = (
        backend_candidate_order_agreement
    )
    diagnostics["normal_custom_greedy_agreement"] = backend_greedy_agreement
    diagnostics["normal_custom_last_token_max_abs_logit_delta"] = float(
        (next_normal - next_custom).abs().max().detach().cpu()
    )
    # Cached and uncached scoring of the same first answer token. This tests
    # use_cache without conflating it with an additional decoding position.
    with torch.no_grad():
        uncached_logits = runtime.model(
            **inputs, use_cache=False, return_dict=True
        ).logits[:, -1, :].float()
        cached_logits = runtime.model(
            **inputs, use_cache=True, return_dict=True
        ).logits[:, -1, :].float()
    uncached_score = torch.log_softmax(uncached_logits, dim=-1)[0, candidate_ids[0]]
    cached_score = torch.log_softmax(cached_logits, dim=-1)[0, candidate_ids[0]]
    cache_score_error = float((uncached_score - cached_score).abs().detach().cpu())
    cache_greedy_agreement = bool(
        uncached_logits.argmax(-1).equal(cached_logits.argmax(-1))
    )
    checks["cached_uncached_equivalence"] = (
        cache_score_error < 1e-5 and cache_greedy_agreement
    )
    diagnostics["cached_uncached_answer_log_probability_delta"] = cache_score_error
    diagnostics["cached_uncached_greedy_agreement"] = cache_greedy_agreement
    return checks, diagnostics


if __name__ == "__main__":
    main()
