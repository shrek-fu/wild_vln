#!/usr/bin/env bash
# Download Prompt-Depth-Anything weights into checkpoints/ (default: ViT-S for 8 GB GPUs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="$ROOT/checkpoints"
mkdir -p "$CKPT"
ENC="${1:-vits}"
case "$ENC" in
  vits)
    URL="https://huggingface.co/depth-anything/prompt-depth-anything-vits/resolve/main/model.ckpt"
    OUT="$CKPT/prompt-depth-anything-vits.ckpt"
    ;;
  vitl)
    URL="https://huggingface.co/depth-anything/prompt-depth-anything-vitl/resolve/main/model.ckpt"
    OUT="$CKPT/prompt-depth-anything-vitl.ckpt"
    ;;
  *)
    echo "usage: $0 [vits|vitl]"; exit 1
    ;;
esac
if [[ -f "$OUT" ]]; then
  echo "[ok] $OUT already exists"
  exit 0
fi
echo "[download] $URL -> $OUT"
curl -L --fail --retry 3 -o "$OUT" "$URL"
ls -lh "$OUT"
