#!/usr/bin/env bash
# Download DA360 checkpoints from Hugging Face (hljiang/DA360).
#
# China: huggingface.co is often unreachable. Default to hf-mirror.com.
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/download_models.sh   # default
#   HF_ENDPOINT=https://huggingface.co bash scripts/download_models.sh  # official
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

REPO_ID="${DA360_HF_REPO:-hljiang/DA360}"
PYTHON="${PYTHON:-python}"
# Prefer venv interpreter when present.
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
fi

# Default to China HF mirror; override with HF_ENDPOINT=...
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p checkpoints

require_hf_hub() {
  if ! "${PYTHON}" -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "Error: huggingface_hub is not installed in this Python env." >&2
    echo "Run: bash scripts/setup_da360_venv.sh" >&2
    exit 1
  fi
}

download_one() {
  local filename="$1"
  local dest="checkpoints/${filename}"
  if [[ -s "${dest}" ]]; then
    echo "skip (exists): ${dest}"
    return 0
  fi
  echo "download -> ${dest}  (HF_ENDPOINT=${HF_ENDPOINT})"
  require_hf_hub
  "${PYTHON}" - <<PY
import os
from huggingface_hub import hf_hub_download

print("HF_ENDPOINT=", os.environ.get("HF_ENDPOINT"))
hf_hub_download(
    repo_id="${REPO_ID}",
    filename="${filename}",
    local_dir="checkpoints",
)
PY
}

for name in DA360_small.pth DA360_base.pth DA360_large.pth; do
  download_one "${name}"
done

echo "Done. Checkpoints in ${ROOT}/checkpoints/"
ls -lh checkpoints/*.pth 2>/dev/null || true
