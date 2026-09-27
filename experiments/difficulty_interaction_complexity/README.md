# Sample Difficulty and Residual Interaction Complexity

## Question

Do difficult inputs recruit more interactions, higher-order interactions, or a
less compressible interaction representation?  The analysis relates label-aware
and label-free difficulty scores to support-size, order, and reconstruction
complexity measures using Spearman correlations, hard-versus-easy quartile
gaps, wrong-versus-correct gaps, and class-controlled slopes.

Across both models, these associations are weak: the identity of dominant
interactions adapts to the input, but the amount and order of interaction
machinery are nearly constant across the difficulty spectrum.

## Reported setting

Both ResNet-18 (`M=255`) and ResNet-34 (`M=65,535`) use the class-balanced
ImageNet-1K validation subset with 10 images per class (`N=10,000`).

## Run

The evaluator writes per-image difficulty and complexity measurements; the
analysis stage produces correlations and group comparisons.

```bash
python experiments/difficulty_interaction_complexity/run_resnet18.py \
  --sample-index data/indices/imagenet_val_balanced_10000.json \
  --parquet-dir "$IMAGENET_PARQUET_DIR" \
  --output-dir runs/difficulty_interaction_complexity/resnet18

python experiments/difficulty_interaction_complexity/analyze_resnet18.py \
  --per-image runs/difficulty_interaction_complexity/resnet18/per_image.pt \
  --output-dir runs/difficulty_interaction_complexity/resnet18
```

```bash
python experiments/difficulty_interaction_complexity/run_resnet34.py \
  --parquet-dir "$IMAGENET_PARQUET_DIR" \
  --output-dir runs/difficulty_interaction_complexity/resnet34

python experiments/difficulty_interaction_complexity/analyze_resnet34.py \
  --per-image runs/difficulty_interaction_complexity/resnet34/per_image.pt \
  --output-dir runs/difficulty_interaction_complexity/resnet34
```

The ResNet-34 exact-mask pass is shared with the input-dependence analysis and
can be resumed after interruption.  The synthetic smoke test checks the
pipeline only; it is not statistically meaningful for difficulty analysis.

## Outputs

Analysis writes `analysis.json` and `table_5.md`.  Large per-image tensors are
regenerable and therefore excluded from version control.  Published tables and
summary metadata are:

- [`results/resnet18.md`](results/resnet18.md)
- [`results/resnet34.md`](results/resnet34.md)
- [`results/resnet18_summary.json`](results/resnet18_summary.json)
- [`results/resnet34_summary.json`](results/resnet34_summary.json)
- [`results/resnet34_verification.json`](results/resnet34_verification.json)

No plotting code or generated figures are included in this release.
