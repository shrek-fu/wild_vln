cmake_minimum_required(VERSION 3.18)
project(insta360_cuda_stitch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Orin (87); desktop GPUs newer than the toolkit run the compute_86 PTX.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86-real 87-real 86-virtual)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)

add_library(insta360_cuda_stitch STATIC
  src/stitch_model.cpp
  src/cuda_remap.cu
  src/live_stitcher.cpp)
target_include_directories(insta360_cuda_stitch PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(insta360_cuda_stitch PUBLIC PkgConfig::GST PRIVATE CUDA::cudart_static)

# Jetson: read decoder NVMM surfaces in place through EGL instead of copying them to system memory.
option(USE_NVMM "Zero-copy NVMM input on Jetson" ON)
find_path(NVBUFSURFACE_INCLUDE_DIR nvbufsurface.h PATHS /usr/src/jetson_multimedia_api/include)
find_library(NVBUFSURFACE_LIBRARY nvbufsurface PATHS /usr/lib/aarch64-linux-gnu/nvidia)
find_library(EGL_LIBRARY EGL)
if(USE_NVMM AND NVBUFSURFACE_INCLUDE_DIR AND NVBUFSURFACE_LIBRARY AND EGL_LIBRARY)
  target_compile_definitions(insta360_cuda_stitch PRIVATE HAVE_NVMM)
  target_include_directories(insta360_cuda_stitch PRIVATE "${NVBUFSURFACE_INCLUDE_DIR}")
  target_link_libraries(insta360_cuda_stitch PRIVATE "${NVBUFSURFACE_LIBRARY}" "${EGL_LIBRARY}")
  message(STATUS "insta360_cuda_stitch: NVMM zero-copy input enabled")
endif()

if(PROJECT_IS_TOP_LEVEL)
  add_executable(stitch_segment tools/stitch_segment.cpp)
  target_link_libraries(stitch_segment PRIVATE insta360_cuda_stitch)
endif()
