#!/usr/bin/env bash
# Weights for text-prompt YOLOE (+ its text encoder) into <ws>/models/yoloe/, and the
# metric mono depth model into <ws>/models/depth_anything_v2/. STVLN_MODELS overrides <ws>/models.
# Uses curl/wget. HF_ENDPOINT overrides the host (default: huggingface.co, then hf-mirror.com).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODELS="${STVLN_MODELS:-${ROOT}/models}"
DEST="${MODELS}/yoloe"
DEPTH_DIR="${MODELS}/depth_anything_v2"
mkdir -p "${DEST}" "${DEPTH_DIR}"

MODEL="${1:-yoloe-26s-seg.pt}"

download() {
  local url="$1" out="$2" tmp="${2}.partial"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 --retry-delay 2 -o "${tmp}" "${url}" || { rm -f "${tmp}"; return 1; }
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${tmp}" "${url}" || { rm -f "${tmp}"; return 1; }
  else
    echo "curl or wget is required" >&2
    return 1
  fi
  mv "${tmp}" "${out}"
}

hf_download() {
  local repo="$1" file="$2" out="$3"
  local -a endpoints=()
  if [[ -n "${HF_ENDPOINT:-}" ]]; then
    endpoints+=("${HF_ENDPOINT%/}")
  else
    endpoints=("https://huggingface.co" "https://hf-mirror.com")
  fi
  local ep url
  for ep in "${endpoints[@]}"; do
    url="${ep}/${repo}/resolve/main/${file}?download=true"
    echo "GET ${url}"
    if download "${url}" "${out}"; then
      return 0
    fi
    echo "failed: ${url}" >&2
    rm -f "${out}"
  done
  return 1
}

fetch_if_missing() {
  local out="$1"
  shift
  if [[ -s "${out}" ]]; then
    echo "exists ${out}"
    return 0
  fi
  "$@"
  local bytes
  bytes="$(stat -c%s "${out}")"
  if (( bytes < 1000000 )); then
    echo "download too small (${bytes} bytes): ${out}" >&2
    rm -f "${out}"
    return 1
  fi
  echo "saved ${out} (${bytes} bytes)"
}

if [[ "${MODEL}" == yoloe-26* ]]; then
  fetch_if_missing "${DEST}/${MODEL}" \
    download "https://github.com/ultralytics/assets/releases/download/v8.4.0/${MODEL}" "${DEST}/${MODEL}"
  CLIP="${DEST}/mobileclip2_b.ts"
  fetch_if_missing "${CLIP}" \
    download "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts" "${CLIP}"
else
  fetch_if_missing "${DEST}/${MODEL}" \
    hf_download "jameslahm/yoloe" "${MODEL}" "${DEST}/${MODEL}"
  CLIP="${DEST}/mobileclip_blt.pt"
  fetch_if_missing "${CLIP}" \
    download "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt" "${CLIP}"
fi
echo "text encoder -> ${CLIP}"

# Indoor metric monocular depth for Insta360-only Hydra (Depth-Anything-V2 Small).
DEPTH="${DEPTH_DIR}/depth_anything_v2_metric_hypersim_vits.pth"
fetch_if_missing "${DEPTH}" \
  hf_download "depth-anything/Depth-Anything-V2-Metric-Hypersim-Small" \
  "depth_anything_v2_metric_hypersim_vits.pth" "${DEPTH}"
echo "metric depth -> ${DEPTH}"
