#!/usr/bin/env bash
# Offline VIO on an already-prepared OpenVIO session directory.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${1:?usage: $0 <session_dir> [config_yaml]}"
CONFIG="${2:-$ROOT/config/x4_air_equirect_1440.yaml}"
OPENVIO_BIN="${OPENVIO_BIN:-$ROOT/build/openvio/bin/vio}"
OUT="$ROOT/data/processed/$(basename "$SESSION")/traj.tum"

export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
python3 -m x4_equirect_vio.cli run \
  --session-dir "$SESSION" \
  --config "$CONFIG" \
  --output "$OUT" \
  --openvio-bin "$OPENVIO_BIN"
echo "Wrote $OUT"