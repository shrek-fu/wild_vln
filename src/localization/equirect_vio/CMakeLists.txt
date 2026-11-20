# SPDX-License-Identifier: MIT
# Self-contained EquirectangularVIO: vendors OpenVIO under third_party/openvio.
cmake_minimum_required(VERSION 3.16)
project(equirectangular_vio LANGUAGES C CXX)

option(EQUIRECT_VIO_BUILD_GUI "Build OpenVIO GUI (needs GLFW)" OFF)
option(EQUIRECT_VIO_BUILD_ROS2 "Hint only: build ROS2 via scripts/build_ros2.sh" ON)

set(OPENVIO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openvio" CACHE PATH "Vendored OpenVIO")
if(NOT EXISTS "${OPENVIO_ROOT}/CMakeLists.txt")
  message(FATAL_ERROR "Missing vendored OpenVIO at ${OPENVIO_ROOT}")
endif()

# Build the OpenVIO library + `vio` CLI into this project's build tree.
add_subdirectory("${OPENVIO_ROOT}" "${CMAKE_BINARY_DIR}/openvio")

message(STATUS "EquirectangularVIO: OpenVIO from ${OPENVIO_ROOT}")
message(STATUS "  Offline binary: ${CMAKE_BINARY_DIR}/openvio/bin/vio (after build)")
message(STATUS "  ROS2: run scripts/build_ros2.sh")
