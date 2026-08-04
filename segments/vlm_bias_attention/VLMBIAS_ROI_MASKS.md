# VLMBias ROI mask preparation

The downstream Qwen3 token-mask and localized attention experiment is documented
in [QWEN3_ROI_ATTENTION.md](QWEN3_ROI_ATTENTION.md).

This pilot pairs the existing 400-row VLMBias slice with rows from the dataset's
`original` split and creates binary masks from image differences. It is a data
preparation stage for localized pre-softmax attention interventions; it does not
yet run Qwen3-VL.

## Pairing policy

The script works on unique edited images rather than treating duplicate prompts
as independent images. Pairing is restricted to families for which the original
asset can be inferred without looking at the model answer:

- animals: normalized animal name;
- flags: normalized country/organization flag name;
- chess and Xiangqi pieces: the corresponding standard board image.

Sportswear logos do not have scene-matched originals, patterned grids have no
rows in the `original` split, optical-illusion originals are not clean
counterfactual counterparts, and game-board originals use globally different
layouts. Car-logo filenames can be paired, but their images retain systematic
registration and resampling artifacts, so they are excluded from the default
pilot as well. Pass `--include-car-logos` only for alignment-method development.
These families are excluded rather than producing misleading masks.

## Mask construction

1. Resize the original image to the edited image dimensions.
2. Gaussian-smooth both images by two pixels to suppress JPEG/resampling edges.
3. At each pixel, take the maximum absolute RGB-channel difference and threshold
   it at 30/255.
4. Apply a one-pixel morphological opening to remove isolated difference speckles.
5. Dilate the mask by 0.6% of the shorter image side so small edits survive
   conversion to Qwen visual tokens.
6. Reject masks covering less than 0.05% or more than 55% of the image.
7. Reject pairs whose median pixel difference exceeds 8/255, which indicates a
   global mismatch rather than a localized edit.

Every automatically accepted mask still requires manual inspection of its
overlay before it can be used as an oracle mask.

## Reproduce

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 UV_CACHE_DIR=.cache/uv \
  uv run python scripts/prepare_vlmbias_roi_masks.py
```

The output directory is ignored by Git and contains:

- `accepted.jsonl` and `rejected.jsonl`;
- resized originals, grayscale differences, binary masks, and overlays;
- paginated accepted and rejected contact sheets;
- `summary.json` with exact parameters and counts.

Validate every artifact and confirm that every saved mask is binary:

```bash
UV_CACHE_DIR=.cache/uv uv run python scripts/validate_vlmbias_roi_masks.py
```
