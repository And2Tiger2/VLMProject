# PaliGemma2 vs Qwen Notes

## Two-Pass Unified

Both models show the same directional split: allowing the answer pass to see the image again preserves more image-conditioned behavior and produces more bias-aligned answers, while removing the real image in the answer pass reduces bias.

The important difference is the accuracy cost.

Qwen two-pass, seeds `0-4`:

- `direct image_again`: accuracy `0.134`, bias-aligned fraction `0.391`
- `direct description_only`: accuracy `0.134`, bias-aligned fraction `0.151`
- `original description_only`: accuracy `0.154`, bias-aligned fraction `0.187`
- `structured description_only`: accuracy `0.149`, bias-aligned fraction `0.180`

PaliGemma2 two-pass, seeds `0-4`:

- `direct image_again`: accuracy `0.143`, bias-aligned fraction `0.319`
- `direct description_only`: accuracy `0.047`, bias-aligned fraction `0.072`
- `original description_only`: accuracy `0.018`, bias-aligned fraction `0.022`
- `structured description_only`: accuracy `0.018`, bias-aligned fraction `0.022`

Interpretation:

PaliGemma2 can be pushed away from bias-aligned answers by replacing the answer-pass image with a blank image, but it mostly stops solving the task. Qwen retained much more task accuracy in its true no-image answer pass. The comparison is not perfectly symmetric because Qwen can run without an image, while PaliGemma requires an image and uses a blank-image control.

## Attention Alpha Sweep

The checked-in Qwen report currently reflects the earlier seed-0 sweep, while the expanded Qwen run is still running. The PaliGemma2 sweep is complete over seeds `0-9`.

PaliGemma2 attention sweep:

- Baseline VLMBias: accuracy `0.141`, bias-aligned fraction `0.267`
- Moderate alpha values mostly leave VLMBias accuracy and NaturalBench accuracy stable.
- Image attention mass rises strongly with alpha, confirming the mechanism is active.
- Extreme all-layer boosts collapse performance:
  - `alpha10_all` VLMBias accuracy `0.007`, bias-aligned fraction `0.028`
  - `alpha10_all` NaturalBench call accuracy `0.152`, group accuracy `0.000`

Interpretation:

For PaliGemma2, simply forcing more image-token attention is not a clean bias fix. Moderate boosts do not substantially improve VLMBias bias metrics, and extreme boosts destroy general performance. The useful conclusion is mechanistic: the hook works, but saturating attention to image tokens overpowers normal decoding rather than selectively correcting bias.
