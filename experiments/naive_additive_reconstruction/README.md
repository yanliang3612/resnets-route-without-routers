# Naive Additive Reconstruction of Masked Responses

## Question

Can outputs of masked ResNet subnetworks be added as if they were independent
path contributions?  This analysis compares three naive aggregations of masked
logits—raw sum, mean, and a shortcut-centered sum—with the exact sum of Möbius
coefficients.  The naive constructions retain overlapping nonlinear effects;
only Möbius inversion recovers the full-network logits exactly.

## Reported setting

| Model | Residual blocks | Masks | Images | Selection |
| --- | ---: | ---: | ---: | --- |
| ResNet-18 | 8 | 256 | 10,000 | frozen ImageNet test subset |
| ResNet-34 | 16 | 65,536 | 1,000 | frozen `extension` subset of the same test index |

For each reconstruction, the evaluator reports output-norm ratio, relative
logit error, optimally scaled error, cosine similarity, and agreement with the
full model's top-1 prediction.

## Run

From the repository root:

```bash
python experiments/naive_additive_reconstruction/run_resnet18.py \
  --sample-index data/indices/imagenet_test_10000.json \
  --parquet-dir "$IMAGENET_PARQUET_DIR" \
  --output-dir runs/naive_additive_reconstruction/resnet18
```

The exact ResNet-34 sweep is shared with the interaction-order analysis so that
the 65,536 masked responses are evaluated only once:

```bash
python experiments/naive_additive_reconstruction/run_resnet34.py \
  --parquet-dir "$IMAGENET_PARQUET_DIR" \
  --split extension \
  --work-dir runs/resnet34/reconstruction_and_spectrum \
  --exp1-output-dir runs/naive_additive_reconstruction/resnet34 \
  --exp2-output-dir runs/interaction_order_spectrum/resnet34
```

Use `--resume` to continue an interrupted ResNet-34 run.  A quick installation
check is available as `bash experiments/naive_additive_reconstruction/smoke_test.sh`.

## Outputs

Fresh runs write `summary.json` and `table_1.md`; the ResNet-34 backend also
writes per-image parquet summaries, verification metadata, and checksums.  The
published numerical tables are committed as:

- [`results/resnet18.md`](results/resnet18.md)
- [`results/resnet34.md`](results/resnet34.md)

No plotting code or generated figures are included in this release.
