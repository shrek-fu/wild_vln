#!/usr/bin/env bash
# Build the Livox driver copied into this project. Uses the vendored SDK.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
[[ -f "${ROS_SETUP}" ]] || { echo "missing ${ROS_SETUP}" >&2; exit 2; }
set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
set -u
cd "${ROOT}"
colcon build --base-paths drivers --packages-select livox_ros_driver2 --symlink-install
echo "sourced later by ./calibrate.sh from ${ROOT}/install/setup.bash"
