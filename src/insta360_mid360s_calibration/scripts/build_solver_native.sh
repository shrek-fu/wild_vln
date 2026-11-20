#!/usr/bin/env bash
# Build direct_visual_lidar_calibration natively (e.g. Jetson aarch64, where the Docker image
# koide3/direct_visual_lidar_calibration is amd64 only). GTSAM and Iridescence go to ${PREFIX}
# (no sudo); the ROS package goes to ${PREFIX}/solver_ws. ./calibrate.sh uses it when
# ${PREFIX}/solver_ws/install/setup.bash exists.
#
# System packages (apt): libsuitesparse-dev libglfw3-dev libglm-dev libomp-dev libfmt-dev libspdlog-dev
#   libboost-all-dev libtbb-dev libmetis-dev libpcl-dev libgoogle-glog-dev libassimp-dev
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${SOLVER_PREFIX:-${ROOT}/opt}"
JOBS="${JOBS:-$(($(nproc) > 6 ? 6 : $(nproc)))}"
SRC="${PREFIX}/src"
mkdir -p "${SRC}"

clone() {  # url dir ref
  [[ -d "${SRC}/$2/.git" ]] || git clone --recursive "$1" "${SRC}/$2"
  git -C "${SRC}/$2" checkout -q "$3"
  git -C "${SRC}/$2" submodule update --init --recursive -q
}

cmake_install() {  # dir [cmake args...]
  local dir="$1"; shift
  cmake -S "${SRC}/${dir}" -B "${SRC}/${dir}/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" -DCMAKE_PREFIX_PATH="${PREFIX}" "$@"
  cmake --build "${SRC}/${dir}/build" -j "${JOBS}"
  cmake --install "${SRC}/${dir}/build"
}

if [[ ! -f "${PREFIX}/lib/cmake/GTSAM/GTSAMConfig.cmake" ]]; then
  clone https://github.com/borglab/gtsam.git gtsam 4.2a9
  cmake_install gtsam -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_TESTS=OFF \
    -DGTSAM_WITH_TBB=OFF -DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF
fi

# Sophus needs ceres/manifold.h (Ceres >= 2.1); Ubuntu 22.04 ships 2.0.
CERES=ceres-solver-2.1.0
if [[ ! -f "${PREFIX}/lib/cmake/Ceres/CeresConfig.cmake" ]]; then
  [[ -d "${SRC}/${CERES}" ]] || \
    curl -fsSL "http://ceres-solver.org/${CERES}.tar.gz" | tar -xz -C "${SRC}"
  cmake_install "${CERES}" -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
    -DUSE_CUDA=OFF
fi

if [[ ! -f "${PREFIX}/lib/cmake/Iridescence/IridescenceConfig.cmake" ]]; then
  clone https://github.com/koide3/iridescence.git iridescence master
  cmake_install iridescence -DBUILD_EXAMPLES=OFF -DBUILD_PYTHON_BINDINGS=OFF
fi

WS="${PREFIX}/solver_ws"
mkdir -p "${WS}/src"
[[ -d "${WS}/src/direct_visual_lidar_calibration/.git" ]] || \
  git clone --recursive https://github.com/koide3/direct_visual_lidar_calibration.git \
    "${WS}/src/direct_visual_lidar_calibration"
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u
export CMAKE_PREFIX_PATH="${PREFIX}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${WS}"
MAKEFLAGS="-j${JOBS}" colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${PREFIX}" \
  --parallel-workers 1
echo "built ${WS}/install; ./calibrate.sh runs the solver natively when it exists"
