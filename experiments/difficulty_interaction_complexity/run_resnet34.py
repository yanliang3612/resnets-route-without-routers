"""Run the shared ResNet-34 sweep for Observations 5 and 6."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.input_dependent_expert_sets.run_resnet34 import _defaults
from resnet_routes.resnet34.dependence_and_difficulty import main


if __name__ == "__main__":
    raise SystemExit(main(_defaults(sys.argv[1:], primary="exp5")))
