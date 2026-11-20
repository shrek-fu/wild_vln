#!/usr/bin/env bash
# Build ROS 2 packages: equirect_vio (+ optional insta360_ros_driver from driver/).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENVIO="$ROOT/third_party/openvio"
WS="$ROOT/ros2_ws"

if [[ ! -f "$OPENVIO/CMakeLists.txt" ]]; then
  echo "Missing vendored OpenVIO at $OPENVIO"
  exit 1
fi
if [[ ! -f "$ROOT/build/openvio/libvio.a" ]]; then
  echo "Prebuilt OpenVIO missing. Building it first..."
  "$ROOT/scripts/build_openvio.sh"
fi

# Keep ros2_ws/src symlinks pointed at driver/ (idempotent).
mkdir -p "$WS/src"
if [[ -d "$ROOT/driver/insta360_ros_driver" ]]; then
  ln -sfn ../../driver/insta360_ros_driver "$WS/src/insta360_ros_driver"
fi
if [[ -d "$ROOT/driver/Linux_CameraSDK-2.1.8" ]]; then
  ln -sfn ../../driver/Linux_CameraSDK-2.1.8 "$WS/src/Linux_CameraSDK-2.1.8"
fi
if [[ -e "$ROOT/driver/insta360_media_sdk" ]]; then
  ln -sfn ../../driver/insta360_media_sdk "$WS/src/insta360_media_sdk"
fi

MEDIA_SDK_CMAKE=()
if [[ -e "$ROOT/driver/insta360_media_sdk/include/ins_realtime_stitcher.h" ]]; then
  MEDIA_SDK_CMAKE+=(-DINSTA360_MEDIA_SDK_ROOT="$ROOT/driver/insta360_media_sdk")
fi

if [[ -z "${ROS_DISTRO:-}" ]]; then
  for d in jazzy humble iron; do
    if [[ -f "/opt/ros/$d/setup.bash" ]]; then
      # shellcheck disable=SC1090
      source "/opt/ros/$d/setup.bash"
      break
    fi
  done
fi
if [[ -z "${ROS_DISTRO:-}" ]]; then
  echo "ROS 2 not found. source /opt/ros/<distro>/setup.bash first."
  exit 1
fi

PACKAGES=(equirect_vio)
if [[ -e "$WS/src/insta360_ros_driver/package.xml" ]]; then
  PACKAGES+=(insta360_ros_driver)
fi

cd "$WS"
colcon build --packages-select "${PACKAGES[@]}" --cmake-args \
  -DOPENVIO_ROOT="$OPENVIO" \
  -DCMAKE_BUILD_TYPE=Release \
  "${MEDIA_SDK_CMAKE[@]}"

echo
echo "Built: ${PACKAGES[*]}"
echo "  source $WS/install/setup.bash"
if [[ " ${PACKAGES[*]} " == *" insta360_ros_driver "* ]]; then
  echo "  ros2 launch insta360_ros_driver x4_air_equirect.launch.xml"
fi
echo "  ros2 launch equirect_vio x4_air_vio.launch.py"
