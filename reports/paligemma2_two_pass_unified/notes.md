# PaliGemma2 Unified Two-Pass Experiment

Model: `google/paligemma2-3b-mix-448`

Dataset: `data/vlmbias_400.jsonl`

Seeds: `0,1,2,3,4`

PaliGemma requires an image input, so the `description_only` answer mode uses a blank white image in pass 2.

## Main Readout

- Direct image-again baseline-like condition: accuracy `0.143`, bias-aligned fraction `0.319`.
- Best description-only accuracy: `direct` with accuracy `0.047`.
- Lowest description-only bias: `original` with bias-aligned fraction `0.021`.
- Lowest image-again bias: `verification` with bias-aligned fraction `0.231`.

## Interpretation

For PaliGemma2, image-again conditions preserve more accuracy but remain substantially more bias-aligned. Blank-image description-only conditions sharply reduce bias-aligned answers, but accuracy drops close to zero. This is qualitatively similar to the Qwen two-pass direction on bias reduction, but PaliGemma pays a much larger accuracy cost in the blank-image answer pass.
