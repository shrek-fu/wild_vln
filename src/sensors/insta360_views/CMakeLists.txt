cmake_minimum_required(VERSION 3.16)
project(insta360_views)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(JPEG REQUIRED)

# Shared SDKs live at <ws>/insta360_driver/ (outside src/). On x86-64 libCameraSDK.so is next to
# libMediaSDK.so; Media SDK has no aarch64 build, so Jetson uses Camera SDK lib_aarch64 + CUDA stitch.
get_filename_component(_WS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(MEDIA_SDK_DIR "$ENV{INSTA360_MEDIA_SDK_ROOT}")
if(NOT MEDIA_SDK_DIR)
  set(MEDIA_SDK_DIR "${_WS_ROOT}/insta360_driver/insta360_media_sdk")
endif()
set(CAMERA_SDK_DIR "$ENV{INSTA360_SDK_ROOT}")
if(NOT CAMERA_SDK_DIR)
  set(CAMERA_SDK_DIR "${_WS_ROOT}/insta360_driver/insta360_sdk")
endif()
if(NOT EXISTS "${CAMERA_SDK_DIR}/include/camera/camera.h")
  message(FATAL_ERROR "Camera SDK header missing: ${CAMERA_SDK_DIR}/include/camera/camera.h")
endif()

option(USE_MEDIA_SDK "Build the Media SDK stitch backend when the library is present" ON)
option(USE_CUDA_STITCH "Build the CUDA stitch backend when a CUDA compiler is found" ON)

set(HAVE_MEDIA_SDK OFF)
if(USE_MEDIA_SDK AND EXISTS "${MEDIA_SDK_DIR}/lib/libMediaSDK.so" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set(HAVE_MEDIA_SDK ON)
  set(CAMERA_SDK_LIB_DIR "${MEDIA_SDK_DIR}/lib")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(CAMERA_SDK_LIB_DIR "${CAMERA_SDK_DIR}/lib_aarch64")
else()
  set(CAMERA_SDK_LIB_DIR "${CAMERA_SDK_DIR}/lib")
endif()
if(NOT EXISTS "${CAMERA_SDK_LIB_DIR}/libCameraSDK.so")
  message(FATAL_ERROR "Camera SDK library missing: ${CAMERA_SDK_LIB_DIR}/libCameraSDK.so")
endif()

set(HAVE_CUDA_STITCH OFF)
if(USE_CUDA_STITCH)
  if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(GST_PROBE QUIET gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
  endif()
  if(CMAKE_CUDA_COMPILER AND NOT GST_PROBE_FOUND)
    message(STATUS "CUDA stitch backend skipped: GStreamer dev packages missing (libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev)")
  elseif(CMAKE_CUDA_COMPILER)
    add_subdirectory(../insta360_cuda_stitch "${CMAKE_CURRENT_BINARY_DIR}/insta360_cuda_stitch")
    set(HAVE_CUDA_STITCH ON)
  endif()
endif()
if(NOT HAVE_MEDIA_SDK AND NOT HAVE_CUDA_STITCH)
  message(FATAL_ERROR "No stitch backend: need ${MEDIA_SDK_DIR}/lib/libMediaSDK.so (x86-64) or a CUDA compiler")
endif()
message(STATUS "insta360_views stitch backends: mediasdk=${HAVE_MEDIA_SDK} cuda=${HAVE_CUDA_STITCH}, Camera SDK ${CAMERA_SDK_LIB_DIR}")

add_executable(insta360_views src/insta360_views.cpp)
target_include_directories(insta360_views PRIVATE
  "${CAMERA_SDK_DIR}/include"
  ${JPEG_INCLUDE_DIRS})
target_link_libraries(insta360_views
  "${CAMERA_SDK_LIB_DIR}/libCameraSDK.so"
  ${JPEG_LIBRARIES}
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS})
if(HAVE_MEDIA_SDK)
  target_compile_definitions(insta360_views PRIVATE HAVE_MEDIA_SDK)
  target_include_directories(insta360_views PRIVATE "${MEDIA_SDK_DIR}/include")
  target_link_libraries(insta360_views "${MEDIA_SDK_DIR}/lib/libMediaSDK.so")
endif()
if(HAVE_CUDA_STITCH)
  target_compile_definitions(insta360_views PRIVATE HAVE_CUDA_STITCH)
  target_link_libraries(insta360_views insta360_cuda_stitch)
endif()
set_target_properties(insta360_views PROPERTIES
  BUILD_RPATH "${CAMERA_SDK_LIB_DIR}"
  INSTALL_RPATH "${CAMERA_SDK_LIB_DIR}")

install(TARGETS insta360_views DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
