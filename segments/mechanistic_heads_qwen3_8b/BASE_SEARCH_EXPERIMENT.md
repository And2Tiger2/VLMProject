# Frozen-base visual-search head experiment

This experiment diagnoses the unmodified `Qwen/Qwen3-VL-8B-Instruct` model. It does not train a LoRA, load an adapter, or use a trained checkpoint as a gate.

## Question

Do language-attention heads route answer queries toward the visual location of a searched-for object, and are those heads causally important for choosing the correct location?

## Design

1. Reuse the deterministic synthetic crowded scenes and their exact object/candidate masks. Discovery uses `train` template groups; causal validation uses disjoint `locked_test` groups.
2. Test three cues: a text feature description, a separately sampled picture of the target, and a visually similar but incorrectly bound exemplar control.
3. Discover heads as in the gaze-head experiment: measure correct-region attention density minus the mean of three equal-area candidate distractors, plus four-way attention-routing accuracy. No behavior labels update model weights.
4. Rank only on text and correct-exemplar discovery examples. Keep the incorrect exemplar as a cue-specificity control.
5. On locked scenes, jointly zero the top discovered heads and measure the change in correct-answer log-likelihood margin and accuracy. Compare against equal-size high-image-attention and deterministic random head sets.
6. For multi-seed replication, sample discovery and validation rows deterministically by seed while balancing template groups. Store each seed in an isolated namespace, then report pairwise top-head overlap, full-ranking Spearman correlation, consensus heads, and per-seed causal effects.

Behavioral localization and presence checks are reported separately. Poor behavior limits causal interpretation but does not turn into a training request.

## Claim boundary

A passive attention ranking is evidence of routing, not causality. A head set should be described as search-causal only if every replication seed has positive absolute locked-set ablation harm, that harm exceeds both controls, and the intact base model performs above four-way chance.
