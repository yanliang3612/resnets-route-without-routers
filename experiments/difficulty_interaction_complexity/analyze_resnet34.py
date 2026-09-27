"""Analyze ResNet-34 per-image difficulty and interaction complexity."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.difficulty_interaction_complexity.analyze_resnet18 import main


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--per-image" not in argv:
        argv = [
            "--per-image",
            str(ROOT / "experiments/difficulty_interaction_complexity/generated/resnet34/per_image.pt"),
            *argv,
        ]
    if "--output-dir" not in argv:
        argv = [
            "--output-dir",
            str(ROOT / "experiments/difficulty_interaction_complexity/generated/resnet34"),
            *argv,
        ]
    main(argv)
