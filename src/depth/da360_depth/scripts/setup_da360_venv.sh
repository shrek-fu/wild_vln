#!/usr/bin/env bash
# Create .venv with uv (no conda). Prefer a recent CUDA wheel for new GPUs (e.g. RTX 50xx).
#
# Mirrors (China):
#   USE_CN_MIRROR=1  (default)  torch from SJTU pytorch-wheels; other pkgs from Tsinghua PyPI
#   USE_CN_MIRROR=0             official download.pytorch.org + PyPI
# Douban is not used: it only mirrors plain PyPI and often lags / lacks CUDA wheels.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

UV="${UV:-$(command -v uv || true)}"
[[ -n "${UV}" ]] || { echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_CN_MIRROR="${USE_CN_MIRROR:-1}"
# Pin cu128 builds: unpinned "torch" on some mirrors resolves to cu130, which
# needs a newer NVIDIA driver than CUDA 12.8 (driver 570.x).
TORCH_SPEC="${TORCH_SPEC:-torch==2.9.1 torchvision==0.24.1}"

if [[ "${USE_CN_MIRROR}" == "1" ]]; then
  # Full cu128 CUDA wheel index (has torch-2.9.1+cu128 for cp310).
  TORCH_INDEX="${TORCH_INDEX:-https://mirror.sjtu.edu.cn/pytorch-wheels/cu128}"
  PYPI_INDEX="${PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
  PYPI_HOST="${PYPI_HOST:-pypi.tuna.tsinghua.edu.cn}"
else
  TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
  PYPI_INDEX="${PYPI_INDEX:-https://pypi.org/simple}"
  PYPI_HOST="${PYPI_HOST:-pypi.org}"
fi

export PYTHONNOUSERSITE=1

echo "Creating ${ROOT}/.venv with ${PYTHON_BIN}"
echo "  USE_CN_MIRROR=${USE_CN_MIRROR}"
echo "  TORCH_INDEX=${TORCH_INDEX}"
echo "  PYPI_INDEX=${PYPI_INDEX}"
"${UV}" venv --python "${PYTHON_BIN}" --clear "${ROOT}/.venv"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"

echo "Installing ${TORCH_SPEC} from ${TORCH_INDEX}"
"${UV}" pip install --python "${ROOT}/.venv/bin/python" \
  --index-url "${TORCH_INDEX}" \
  --extra-index-url "${PYPI_INDEX}" \
  ${TORCH_SPEC}

echo "Installing project requirements (without pinned torch/xformers from requirements.txt)"
# opencv is only listed as "opencv" in environment.yaml (conda); pip needs opencv-python-headless.
grep -vE '^(--extra-index-url|torch==|torchvision==|xformers==)' requirements.txt \
  | "${UV}" pip install --python "${ROOT}/.venv/bin/python" \
      --index-url "${PYPI_INDEX}" \
      --trusted-host "${PYPI_HOST}" \
      -r -

"${UV}" pip install --python "${ROOT}/.venv/bin/python" \
  --index-url "${PYPI_INDEX}" --trusted-host "${PYPI_HOST}" xformers \
  || echo "warning: xformers not installed (optional; DA360 still runs)"

"${ROOT}/.venv/bin/python" - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PY

cat <<EOF

Done. Activate with:
  source ${ROOT}/.venv/bin/activate
Then for live ROS:
  source /opt/ros/humble/setup.bash
  python live_ros.py ...
Download weights:
  bash scripts/download_models.sh

Official torch (no CN mirror):
  USE_CN_MIRROR=0 bash scripts/setup_da360_venv.sh
EOF
