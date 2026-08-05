# Optional later real-Waldo transfer

This phase is locked and is not executed by setup or any smoke command.

```bash
kaggle datasets download -d mohaneddz/wheres-waldo -p /path/out --unzip
```

Before using it, the later loader must report the dataset license, audit class
labels, recover original page IDs, split by page, and refuse random crop-level
splitting. Exact boxes are used for head scoring; a 10x10 cell is behavioral
output only.

After manually checking the download and adapting the COCO annotation path and
page-ID regex in `../configs/real_waldo_transfer.json`, run:

```bash
uv run python scripts/prepare_real_waldo_transfer.py \
  --config segments/mechanistic_heads_qwen3_8b/configs/real_waldo_transfer.json \
  --input-dir /path/out \
  --output-dir segments/mechanistic_heads_qwen3_8b/data/external/real_waldo_transfer \
  --seed 260525427
```

The loader accepts COCO annotations only, requires a license file, reports
classes and license hashes/excerpts, requires recoverable page IDs, splits only
by original page, and materializes the six requested zoom/crop conditions. It
does not download anything.
