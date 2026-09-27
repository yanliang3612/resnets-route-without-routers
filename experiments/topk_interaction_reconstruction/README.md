# Top-K Residual-Interaction Reconstruction

## Question

How many input-specific interaction terms are needed to preserve coefficient
mass, logits, and the full model's prediction?  Interactions are ranked by
scalar or vector coefficient magnitude and accumulated into a truncated Möbius
reconstruction.  Random-K, order-matched, and low-order selections provide
controls.  The evaluation distinguishes coefficient-mass concentration from
functional reconstruction: large retained mass need not imply a correct logit
vector or top-1 decision.

## Reported setting

| Model | Candidate interactions `M` | Images | Selection |
| --- | ---: | ---: | --- |
| ResNet-18 | 255 | 10,000 | frozen ImageNet test subset |
| ResNet-34 | 65,535 | 1,000 | frozen `extension` subset |

## Run

```bash
python experiments/topk_interaction_reconstruction/run_resnet18.py \
  --sample-index data/indices/imagenet_test_10000.json \
  --parquet-dir "$IMAGENET_PARQUET_DIR" \
  --output-dir runs/topk_interaction_reconstruction/resnet18

python experiments/topk_interaction_reconstruction/run_resnet34.py \
  --parquet-dir "$IMAGENET_PARQUET_DIR" \
  --split extension \
  --output-dir runs/topk_interaction_reconstruction/resnet34
```

ResNet-34 streams the interaction dictionary and supports `--resume`; do not
assume that the full coefficient tensor fits in memory.  Run
`bash experiments/topk_interaction_reconstruction/smoke_test.sh` for a small
synthetic check.

## Outputs

The evaluator records captured mass, relative logit error, cosine similarity,
top-1 agreement, classification-margin diagnostics, order composition, and the
effective K required by mass/error thresholds.  Fresh ResNet-34 runs also save
curves, per-image summaries, configuration, verification metadata, and
checksums.  Published tables are:

- [`results/resnet18.md`](results/resnet18.md)
- [`results/resnet34.md`](results/resnet34.md)
- [`results/resnet34_verification.json`](results/resnet34_verification.json)

No plotting code or generated figures are included in this release.
