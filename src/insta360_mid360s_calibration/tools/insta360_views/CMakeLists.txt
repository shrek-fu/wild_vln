cmake_minimum_required(VERSION 3.16)
project(insta360_views CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CAMERA_SDK_DIR "" CACHE PATH "Insta360 Camera SDK (include/)")
set(MEDIA_SDK_DIR "" CACHE PATH "Insta360 Media SDK (include/ and lib/)")
if(NOT EXISTS "${CAMERA_SDK_DIR}/include/camera/camera.h")
  message(FATAL_ERROR "CAMERA_SDK_DIR has no include/camera/camera.h: ${CAMERA_SDK_DIR}")
endif()
if(NOT EXISTS "${MEDIA_SDK_DIR}/lib/libMediaSDK.so")
  message(FATAL_ERROR "MEDIA_SDK_DIR has no lib/libMediaSDK.so: ${MEDIA_SDK_DIR}")
endif()

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(JPEG REQUIRED)

add_executable(insta360_views insta360_views.cpp)
target_include_directories(insta360_views PRIVATE
  "${CAMERA_SDK_DIR}/include"
  "${MEDIA_SDK_DIR}/include"
  ${JPEG_INCLUDE_DIRS})
# libCameraSDK.so is taken from the Media SDK lib/ so both SDKs come from one build.
target_link_libraries(insta360_views
  "${MEDIA_SDK_DIR}/lib/libMediaSDK.so"
  "${MEDIA_SDK_DIR}/lib/libCameraSDK.so"
  ${JPEG_LIBRARIES}
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS})
set_target_properties(insta360_views PROPERTIES
  BUILD_RPATH "${MEDIA_SDK_DIR}/lib"
  INSTALL_RPATH "${MEDIA_SDK_DIR}/lib")
