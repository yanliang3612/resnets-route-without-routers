# Residual-Scaling Sweep

## Question

The theory predicts coefficient-wise attenuation
`|Delta_S(lambda)| = O(lambda^|S|)` in the small-residual regime.  This
inference-time intervention multiplies every residual branch by a common
`lambda`, recomputes the complete interaction spectrum without retraining, and
tests whether higher-order mass decreases faster.  Top-1 agreement with the
unscaled (`lambda=1`) model is reported alongside the spectrum to expose
prediction drift.

The default sweep is `1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.5, 0.25, 0.1`.

## Reported setting

| Model | Masks per image and scale | Images | Selection |
| --- | ---: | ---: | --- |
| ResNet-18 | 256 | 10,000 | frozen ImageNet test subset |
| ResNet-34 | 65,536 | 1,000 | frozen `extension` subset |

## Run

```bash
python experiments/residual_scaling_sweep/run_resnet18.py \
  --sample-index data/indices/imagenet_test_10000.json \
  --parquet-dir "$IMAGENET_PARQUET_DIR" \
  --output-dir runs/residual_scaling_sweep/resnet18

python experiments/residual_scaling_sweep/run_resnet34.py \
  --parquet-dir "$IMAGENET_PARQUET_DIR" \
  --split extension \
  --output-dir runs/residual_scaling_sweep/resnet34
```

Override the scale grid with `--lambdas 1.0,0.5,0.1`.  The ResNet-34 runner is
checkpointable; pass `--resume` after interruption.  The synthetic smoke test
is `bash experiments/residual_scaling_sweep/smoke_test.sh`.

## Outputs

Fresh runs write a machine-readable summary and a Markdown table of effective
order, low-order mass, high-order tail mass, and top-1 agreement for both
readouts.  Reference tables and run verification are committed in:

- [`results/resnet18.md`](results/resnet18.md)
- [`results/resnet34.md`](results/resnet34.md)
- [`results/resnet34_verification.json`](results/resnet34_verification.json)

No plotting code or generated figures are included in this release.
