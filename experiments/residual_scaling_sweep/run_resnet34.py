"""Run the inference-time residual-scaling sweep on ResNet-34."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from resnet_routes.resnet34.scaling_sweep import main


ROOT = Path(__file__).resolve().parents[2]


def _defaults(argv: list[str]) -> list[str]:
    defaults = {
        "--sample-index": ROOT / "data/indices/imagenet_test_10000.json",
        "--parquet-dir": Path(os.environ.get("IMAGENET_PARQUET_DIR", "imagenet-1k/data")),
        "--work-dir": ROOT / "runs/resnet34/residual_scaling_sweep",
        "--output-dir": ROOT / "experiments/residual_scaling_sweep/generated/resnet34",
        "--lambda-one-summary": ROOT / "experiments/interaction_order_spectrum/generated/resnet34/summary.json",
    }
    prefix: list[str] = []
    if "--selection-manifest" not in argv and "--synthetic-images" not in argv and "--merge-inputs" not in argv:
        prefix.extend(("--selection-manifest", str(ROOT / "data/indices/resnet34_test_splits_seed3403.json")))
    for flag, value in defaults.items():
        if flag not in argv:
            prefix.extend((flag, str(value)))
    return prefix + argv


if __name__ == "__main__":
    raise SystemExit(main(_defaults(sys.argv[1:])))
