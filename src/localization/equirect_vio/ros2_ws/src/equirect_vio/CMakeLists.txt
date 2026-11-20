cmake_minimum_required(VERSION 3.16)
project(equirect_vio)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PNG REQUIRED)
find_package(Threads REQUIRED)
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

# Vendored OpenVIO inside EquirectangularVIO (self-contained).
get_filename_component(EQUIRECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(OPENVIO_ROOT "${EQUIRECT_ROOT}/third_party/openvio")
set(OPENVIO_BUILD "${EQUIRECT_ROOT}/build/openvio")

if(NOT EXISTS "${OPENVIO_ROOT}/CMakeLists.txt")
  message(FATAL_ERROR "Vendored OpenVIO missing: ${OPENVIO_ROOT}")
endif()
if(NOT EXISTS "${OPENVIO_BUILD}/libvio.a")
  message(FATAL_ERROR
    "Prebuilt OpenVIO library not found: ${OPENVIO_BUILD}/libvio.a\n"
    "Run EquirectangularVIO/scripts/build_openvio.sh first.")
endif()
set(CERES_LIB "${OPENVIO_BUILD}/thirdparty/ceres-solver/lib/libceres.a")
if(NOT EXISTS "${CERES_LIB}")
  set(CERES_LIB "${OPENVIO_BUILD}/lib/libceres.a")
endif()
if(NOT EXISTS "${CERES_LIB}")
  message(FATAL_ERROR "Prebuilt Ceres not found under ${OPENVIO_BUILD}")
endif()

# Do NOT add vendored spdlog include paths — ROS rcl_logging_spdlog links
# system libspdlog.so; mixing formatters caused free(): invalid pointer at init.
include_directories(
  "${OPENVIO_ROOT}/src"
  "${OPENVIO_BUILD}/thirdparty/ceres-solver/config"
  "${OPENVIO_ROOT}/thirdparty/ceres-solver/include"
  "${EIGEN3_INCLUDE_DIR}"
)

add_executable(equirect_vio_node src/equirect_vio_node.cpp)
ament_target_dependencies(equirect_vio_node
  rclcpp sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros cv_bridge
)
# Must match OpenVIO Release flags: -march changes Eigen's max alignment (16 vs 32),
# and mixing with libvio.a caused free()/malloc heap corruption at node startup.
target_compile_definitions(equirect_vio_node PRIVATE EIGEN_NO_CPUID)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(equirect_vio_node PRIVATE -march=x86-64-v3 -ffp-contract=off)
else()
  target_compile_options(equirect_vio_node PRIVATE -ffp-contract=off)
endif()
target_link_libraries(equirect_vio_node
  "${OPENVIO_BUILD}/libvio.a"
  "${CERES_LIB}"
  ${OpenCV_LIBS}
  PNG::PNG
  Threads::Threads
  ${LAPACK_LIBRARIES}
  ${BLAS_LIBRARIES}
)

# glog/gflags are public deps of the static ceres we built
find_package(glog QUIET)
find_package(gflags QUIET)
if(TARGET glog::glog)
  target_link_libraries(equirect_vio_node glog::glog)
endif()
if(TARGET gflags)
  target_link_libraries(equirect_vio_node gflags)
elseif(TARGET gflags::gflags)
  target_link_libraries(equirect_vio_node gflags::gflags)
endif()

install(TARGETS equirect_vio_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()