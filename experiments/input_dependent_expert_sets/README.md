# Input-Dependent Residual Expert Sets

## Question

Do the dominant residual interactions change with the input, and is that change
class structured?  Each image is represented by binary and magnitude-weighted
Top-K interaction signatures.  The analysis compares same-class and
different-class Jaccard overlap, overlap with a global Top-K set, a random-set
baseline, shuffled-label controls, and nearest-neighbor classification in
signature space.

The results support a shared global interaction core with systematic,
readout-dependent input variation.  Predicted-class signatures show much
stronger class structure than full-logit signatures.

## Reported setting

Both ResNet-18 (`M=255`) and ResNet-34 (`M=65,535`) use the same class-balanced
ImageNet-1K validation selection: 10 images from each of 1,000 classes
(`N=10,000`).

## Run

Evaluation first writes per-image signatures; analysis then computes pairwise
statistics and the reported table.

```bash
python experiments/input_dependent_expert_sets/run_resnet18.py \
  --sample-index data/indices/imagenet_val_balanced_10000.json \
  --parquet-dir "$IMAGENET_PARQUET_DIR" \
  --output-dir runs/input_dependent_expert_sets/resnet18

python experiments/input_dependent_expert_sets/analyze_resnet18.py \
  --signatures runs/input_dependent_expert_sets/resnet18/signatures.pt \
  --output-dir runs/input_dependent_expert_sets/resnet18
```

```bash
python experiments/input_dependent_expert_sets/run_resnet34.py \
  --parquet-dir "$IMAGENET_PARQUET_DIR" \
  --output-dir runs/input_dependent_expert_sets/resnet34

python experiments/input_dependent_expert_sets/analyze_resnet34.py \
  --signature-dir runs/input_dependent_expert_sets/resnet34 \
  --output-dir runs/input_dependent_expert_sets/resnet34
```

The ResNet-34 exact-mask pass is shared with the difficulty analysis and
supports checkpoint/resume.  Use
`bash experiments/input_dependent_expert_sets/smoke_test.sh` for an execution
check; meaningful class-pair statistics require the balanced real-data subset.

## Outputs

Analysis produces `overlap.json` and `table_4.md` (plus compact diagnostics for
ResNet-34).  Per-image signature tensors are large generated artifacts and are
not committed.  Reference outputs are:

- [`results/resnet18.md`](results/resnet18.md)
- [`results/resnet34.md`](results/resnet34.md)
- [`results/resnet18_summary.json`](results/resnet18_summary.json)
- [`results/resnet34_verification.json`](results/resnet34_verification.json)

No plotting code or generated figures are included in this release.
