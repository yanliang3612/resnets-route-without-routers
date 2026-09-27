"""Run exact per-input Top-K interaction reconstruction on ResNet-34."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from resnet_routes.resnet34.topk_reconstruction import main


ROOT = Path(__file__).resolve().parents[2]


def _defaults(argv: list[str]) -> list[str]:
    defaults = {
        "--sample-index": ROOT / "data/indices/imagenet_test_10000.json",
        "--selection-manifest": ROOT / "data/indices/resnet34_test_splits_seed3403.json",
        "--parquet-dir": Path(os.environ.get("IMAGENET_PARQUET_DIR", "imagenet-1k/data")),
        "--output-dir": ROOT / "experiments/topk_interaction_reconstruction/generated/resnet34",
    }
    prefix: list[str] = []
    if "--split" not in argv:
        prefix.extend(("--split", "extension"))
    for flag, value in defaults.items():
        if flag not in argv:
            prefix.extend((flag, str(value)))
    return prefix + argv


if __name__ == "__main__":
    main(_defaults(sys.argv[1:]))
