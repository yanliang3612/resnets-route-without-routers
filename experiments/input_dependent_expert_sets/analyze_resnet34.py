"""Analyze class structure in the ResNet-34 interaction signatures."""

from __future__ import annotations

import sys
from pathlib import Path

from resnet_routes.resnet34.analyze_dependence import main


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--signature-dir" not in argv:
        argv = [
            "--signature-dir",
            str(ROOT / "experiments/input_dependent_expert_sets/generated/resnet34"),
            *argv,
        ]
    raise SystemExit(main(argv))
