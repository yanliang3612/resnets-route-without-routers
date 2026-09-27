# Residual Interaction-Order Spectrum

## Question

Are trained ResNets functionally dominated by low-order residual interactions?
For scalar predicted-class and vector-logit readouts, this analysis groups the
squared Möbius-coefficient mass by interaction order.  It reports normalized
order energy, cumulative and tail mass, mean energy per interaction, and the
effective order `kappa`.

The reported spectra peak at order 5 for ResNet-18 and order 10 for ResNet-34.
The mean per-interaction statistic separates this observation from the purely
combinatorial fact that middle orders contain more subsets.

## Reported setting

| Model | Non-empty interactions | Images | Selection |
| --- | ---: | ---: | --- |
| ResNet-18 | 255 | 10,000 | frozen ImageNet test subset |
| ResNet-34 | 65,535 | 1,000 | frozen `extension` subset |

## Run

```bash
python experiments/interaction_order_spectrum/run_resnet18.py \
  --sample-index data/indices/imagenet_test_10000.json \
  --parquet-dir "$IMAGENET_PARQUET_DIR" \
  --output-dir runs/interaction_order_spectrum/resnet18
```

ResNet-34 shares one exact-mask sweep with naive additive reconstruction:

```bash
python experiments/interaction_order_spectrum/run_resnet34.py \
  --parquet-dir "$IMAGENET_PARQUET_DIR" \
  --split extension \
  --work-dir runs/resnet34/reconstruction_and_spectrum \
  --exp1-output-dir runs/naive_additive_reconstruction/resnet34 \
  --exp2-output-dir runs/interaction_order_spectrum/resnet34
```

Use `--resume` for an interrupted ResNet-34 run.  For a synthetic check, run
`bash experiments/interaction_order_spectrum/smoke_test.sh`.

## Outputs

Fresh evaluations write `summary.json` and `table_2.md`; ResNet-34 additionally
stores per-image summaries and verification metadata.  The paper tables are:

- [`results/resnet18.md`](results/resnet18.md)
- [`results/resnet34.md`](results/resnet34.md)

No plotting code or generated figures are included in this release.
