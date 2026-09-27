#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/setup_environment.sh [options]

Create an isolated Python environment and install this repository.

Options:
  --python PATH       Python interpreter used to create the environment
                      (default: python3.11, falling back to python3).
  --venv PATH         Virtual-environment directory (default: .venv).
  --cpu               Install CPU-only PyTorch wheels before other packages.
  --runtime-only      Do not install the pytest development extra.
  -h, --help          Show this message.

The script never downloads datasets or pretrained model weights.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
else
  PYTHON_BIN="python3"
fi
VENV_DIR="${REPO_ROOT}/.venv"
CPU_ONLY=0
INSTALL_DEV=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || { echo "--python requires a value" >&2; exit 2; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      [[ $# -ge 2 ]] || { echo "--venv requires a value" >&2; exit 2; }
      VENV_DIR="$2"
      shift 2
      ;;
    --cpu)
      CPU_ONLY=1
      shift
      ;;
    --runtime-only)
      INSTALL_DEV=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
VENV_PYTHON="${VENV_DIR}/bin/python"

"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

if [[ ${CPU_ONLY} -eq 1 ]]; then
  "${VENV_PYTHON}" -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.8,<2.9" "torchvision>=0.23,<0.24"
fi

"${VENV_PYTHON}" -m pip install -r requirements.txt

if [[ ${INSTALL_DEV} -eq 1 ]]; then
  "${VENV_PYTHON}" -m pip install -e ".[dev]"
else
  "${VENV_PYTHON}" -m pip install --no-deps -e .
fi

echo
echo "Environment ready: ${VENV_DIR}"
echo "Activate it with: source \"${VENV_DIR}/bin/activate\""
if [[ ${INSTALL_DEV} -eq 1 ]]; then
  echo "Run offline unit tests with: pytest"
fi
