# Data and frozen sample selections

The code expects ImageNet-1K images in Hugging Face parquet format.  ImageNet is
not included in this repository; obtain it under its original license and set:

```bash
export IMAGENET_PARQUET_DIR=/absolute/path/to/imagenet-1k/data
```

Typical shard names are:

- `test-*.parquet` for the fixed test-image analyses; and
- `val_images-*.parquet` or `validation-*.parquet` for labeled validation
  analyses.

Each row must expose an image value compatible with the Hugging Face ImageNet
schema (encoded bytes and, optionally, a path) and a class label for validation
data.  The runners' `--parquet-dir` option replaces the original machine-local
directory in a frozen index while preserving every shard basename and row
number.

## Included indices

| File | Selection | Used by |
| --- | --- | --- |
| `indices/imagenet_test_10000.json` | 10,000 fixed ImageNet test images | additive reconstruction, order spectrum, residual scaling, and Top-K reconstruction |
| `indices/imagenet_val_balanced_10000.json` | 10 images per class over 1,000 ImageNet validation classes | input-dependent expert sets and difficulty/complexity analyses |
| `indices/resnet34_test_splits_seed3403.json` | frozen permutation and named ResNet-34 test splits | first four ResNet-34 analyses |

ResNet-18 uses all 10,000 entries.  The first four ResNet-34 analyses use the
frozen 1,000-image extension selection recorded by their runner/manifest; the
last two use all 10,000 class-balanced validation images.  The model-specific
READMEs state the exact sample count for each reported table.

## Validate an existing index

An index is JSON with an `entries` list.  Every entry records at least a parquet
shard, row offset, and label or stable sample identity.  Before a long run,
verify that all referenced basenames are present in the remapped data directory:

```bash
python - <<'PY'
import json, os
from pathlib import Path

index = Path("data/indices/imagenet_val_balanced_10000.json")
root = Path(os.environ["IMAGENET_PARQUET_DIR"])
payload = json.loads(index.read_text())
missing = sorted({Path(x["shard"]).name for x in payload["entries"]
                  if not (root / Path(x["shard"]).name).is_file()})
print(f"samples={len(payload['entries'])}, missing_shards={len(missing)}")
if missing:
    print("\n".join(missing))
PY
```

## Create a new selection

The committed indices should be used to reproduce the reported tables.  For a
new uniform test subset:

```bash
python scripts/prepare_sample_index.py \
  --parquet-dir "$IMAGENET_PARQUET_DIR" \
  --pattern 'test-*.parquet' \
  --mode uniform \
  --num-images 10000 \
  --seed 0 \
  --output data/indices/my_test_10000.json
```

For a class-balanced validation subset:

```bash
python scripts/prepare_sample_index.py \
  --parquet-dir "$IMAGENET_PARQUET_DIR" \
  --pattern 'val_images-*.parquet' \
  --mode per-class \
  --num-classes 1000 \
  --per-class 10 \
  --seed 0 \
  --output data/indices/my_val_balanced_10000.json
```

Changing the sample selection changes the experimental population and will not
exactly reproduce the committed tables.
