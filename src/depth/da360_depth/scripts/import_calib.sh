#!/usr/bin/env bash
# Copy calibrated extrinsics/intrinsics from the MID-360S + Insta360 package (single or dual lidar).
# Prefer ../../insta360_mid360s_calibration/calibrate.sh publish, which also updates data/calib.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DEFAULT="$(cd "${ROOT}/../../insta360_mid360s_calibration/results/export" && pwd)"
SRC="${1:-${SRC_DEFAULT}}"
DST="${ROOT}/calib"
mkdir -p "${DST}"
for f in insta360_intrinsics.json insta360_extrinsic.json lidar_extrinsic.json lidar_extrinsic.initial.json; do
  if [[ ! -f "${SRC}/${f}" ]]; then
    [[ "${f}" == lidar_extrinsic.initial.json ]] && continue
    echo "missing ${SRC}/${f}" >&2; exit 1
  fi
  cp -a "${SRC}/${f}" "${DST}/${f}"
  echo "copied ${f}"
done
python3 - <<PY
import sys
sys.path.insert(0, "${ROOT}")
from insta_live.calib_io import load_rig_calibration, write_merged_summary
c = load_rig_calibration("${DST}")
write_merged_summary(c, "${DST}/rig_calib.json")
print(c.summary())
print("wrote ${DST}/rig_calib.json")
PY
