#!/usr/bin/env bash
# Build tools/insta360_views against the shared SDKs at <ws>/insta360_driver/.
#   insta360_sdk/        Camera SDK 2.1.8 (include/, lib/, lib_aarch64/)
#   insta360_media_sdk/  Media SDK 3.1.5 (x86_64)
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WS_ROOT="$(cd -- "${ROOT}/../.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
CAMERA_SDK_DIR="${INSTA360_SDK_ROOT:-${WS_ROOT}/insta360_driver/insta360_sdk}"
MEDIA_SDK_DIR="${INSTA360_MEDIA_SDK_ROOT:-${WS_ROOT}/insta360_driver/insta360_media_sdk}"
[[ -f "${ROS_SETUP}" ]] || { echo "missing ${ROS_SETUP}" >&2; exit 2; }
set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
set -u
cmake -S "${ROOT}/tools/insta360_views" -B "${ROOT}/build/insta360_views" \
  -DCAMERA_SDK_DIR="${CAMERA_SDK_DIR}" -DMEDIA_SDK_DIR="${MEDIA_SDK_DIR}"
cmake --build "${ROOT}/build/insta360_views" -j"$(nproc)"
echo "built ${ROOT}/build/insta360_views/insta360_views"
