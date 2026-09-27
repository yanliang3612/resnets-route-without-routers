"""Run the shared exact ResNet-34 mask sweep for Observations 1 and 2."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.naive_additive_reconstruction.run_resnet34 import _defaults
from resnet_routes.resnet34.reconstruction_and_spectrum import main


if __name__ == "__main__":
    raise SystemExit(main(_defaults(sys.argv[1:])))
