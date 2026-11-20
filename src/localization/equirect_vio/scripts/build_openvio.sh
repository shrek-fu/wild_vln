#!/usr/bin/env bash
# Build the vendored OpenVIO (offline `vio` binary) inside this repo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENVIO="$ROOT/third_party/openvio"
BUILD="$ROOT/build/openvio"

if [[ ! -f "$OPENVIO/CMakeLists.txt" ]]; then
  echo "Missing vendored OpenVIO at $OPENVIO"
  exit 1
fi

mkdir -p "$BUILD"
cmake -S "$OPENVIO" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc)"

BIN="$BUILD/bin/vio"
if [[ ! -x "$BIN" ]]; then
  echo "Build finished but $BIN not found"
  exit 1
fi
echo "OK: $BIN"
"$BIN" 2>&1 | head -8 || true
