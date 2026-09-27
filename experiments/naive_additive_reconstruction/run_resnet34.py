"""Run the shared exact ResNet-34 mask sweep for Observations 1 and 2."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from resnet_routes.resnet34.reconstruction_and_spectrum import main


ROOT = Path(__file__).resolve().parents[2]


def _defaults(argv: list[str]) -> list[str]:
    defaults = {
        "--sample-index": ROOT / "data/indices/imagenet_test_10000.json",
        "--selection-manifest": ROOT / "data/indices/resnet34_test_splits_seed3403.json",
        "--parquet-dir": Path(os.environ.get("IMAGENET_PARQUET_DIR", "imagenet-1k/data")),
        "--work-dir": ROOT / "runs/resnet34/reconstruction_and_spectrum",
        "--exp1-output-dir": ROOT / "experiments/naive_additive_reconstruction/generated/resnet34",
        "--exp2-output-dir": ROOT / "experiments/interaction_order_spectrum/generated/resnet34",
    }
    prefix: list[str] = []
    for flag, value in defaults.items():
        if flag not in argv:
            prefix.extend((flag, str(value)))
    return prefix + argv


if __name__ == "__main__":
    raise SystemExit(main(_defaults(sys.argv[1:])))
