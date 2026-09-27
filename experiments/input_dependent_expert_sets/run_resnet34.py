"""Extract input-dependent residual-interaction signatures for ResNet-34."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from resnet_routes.resnet34.dependence_and_difficulty import main


ROOT = Path(__file__).resolve().parents[2]


def _defaults(argv: list[str], primary: str = "exp4") -> list[str]:
    argv = list(argv)
    if "--output-dir" in argv:
        position = argv.index("--output-dir")
        argv[position] = "--exp4-output-dir" if primary == "exp4" else "--exp5-output-dir"
    defaults = {
        "--sample-index": ROOT / "data/indices/imagenet_val_balanced_10000.json",
        "--parquet-dir": Path(os.environ.get("IMAGENET_PARQUET_DIR", "imagenet-1k/data")),
        "--work-dir": ROOT / "runs/resnet34/dependence_and_difficulty",
        "--exp4-output-dir": ROOT / "experiments/input_dependent_expert_sets/generated/resnet34",
        "--exp5-output-dir": ROOT / "experiments/difficulty_interaction_complexity/generated/resnet34",
    }
    prefix: list[str] = []
    for flag, value in defaults.items():
        if flag not in argv:
            prefix.extend((flag, str(value)))
    return prefix + argv


if __name__ == "__main__":
    raise SystemExit(main(_defaults(sys.argv[1:], primary="exp4")))
