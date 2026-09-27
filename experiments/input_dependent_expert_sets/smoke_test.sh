#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
smoke_dir="$(mktemp -d)"
cd "${repo_root}"
PYTHONPATH=src python experiments/input_dependent_expert_sets/run_resnet18.py \
  --synthetic --num-images 4 --batch-size 2 --mask-chunk 256 \
  --num-workers 0 --no-pretrained \
  --output-dir "${smoke_dir}/resnet18"
echo "Smoke-test output: ${smoke_dir}"
