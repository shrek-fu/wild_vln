cmake_minimum_required(VERSION 3.5)
project(insta360_ros_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rosidl_default_runtime REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
# Resolve REALPATH so ros2_ws/src symlinks still find sibling SDK trees under driver/.
get_filename_component(_INSTA360_PKG_REAL "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
# MediaSDK: EquirectangularVIO/driver/insta360_media_sdk (symlink to calib tree) or /opt.
set(INSTA360_MEDIA_SDK_ROOT "" CACHE PATH "Insta360 MediaSDK installation prefix")
file(GLOB MEDIA_SDK_ROOTS
  "${_INSTA360_PKG_REAL}/../insta360_media_sdk"
  "${CMAKE_CURRENT_SOURCE_DIR}/../insta360_media_sdk"
  "/opt/MediaSDK-*-linux")
find_path(MEDIA_SDK_INCLUDE_DIR ins_realtime_stitcher.h
  HINTS
    "${INSTA360_MEDIA_SDK_ROOT}/include"
    "${_INSTA360_PKG_REAL}/../insta360_media_sdk/include"
    "${PROJECT_SOURCE_DIR}/include/media_sdk"
    ${MEDIA_SDK_ROOTS}
  PATH_SUFFIXES include)
find_library(MEDIA_SDK_LIBRARY MediaSDK
  HINTS ${INSTA360_MEDIA_SDK_ROOT} ${_INSTA360_PKG_REAL}/../insta360_media_sdk ${MEDIA_SDK_ROOTS}
  PATH_SUFFIXES lib)
# Native x86_64 CameraSDK under driver/Linux_CameraSDK-*.
file(GLOB INSTA360_CAMERA_SDK_ROOTS
  "${_INSTA360_PKG_REAL}/../Linux_CameraSDK-*/CameraSDK-*-linux-x86_64"
  "${CMAKE_CURRENT_SOURCE_DIR}/../Linux_CameraSDK-*/CameraSDK-*-linux-x86_64")
find_path(CAMERA_SDK_INCLUDE_DIR camera/camera.h
  HINTS "${PROJECT_SOURCE_DIR}/include" ${INSTA360_CAMERA_SDK_ROOTS} "${INSTA360_MEDIA_SDK_ROOT}/include"
  PATH_SUFFIXES include)
find_library(CAMERA_SDK_LIBRARY CameraSDK
  HINTS "${PROJECT_SOURCE_DIR}/lib" ${INSTA360_CAMERA_SDK_ROOTS} ${INSTA360_MEDIA_SDK_ROOT} ${MEDIA_SDK_ROOTS}
  PATH_SUFFIXES lib)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CameraPreviewInfo.msg"
  "msg/FactoryCalibration.msg"
  DEPENDENCIES std_msgs
)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_search_module(SWSCALE REQUIRED libswscale)
# Optional: only gstreamer_encoder / gstreamer_decoder need these.
pkg_check_modules(GSTREAMER gstreamer-1.0 gstreamer-app-1.0)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${AVCODEC_INCLUDE_DIRS}
  ${AVFORMAT_INCLUDE_DIRS}
  ${AVUTIL_INCLUDE_DIRS}
  ${SWSCALE_INCLUDE_DIRS}
  ${GSTREAMER_INCLUDE_DIRS}
)

link_directories(
  ${PROJECT_SOURCE_DIR}/lib
  ${PROJECT_SOURCE_DIR}/include
  ${GSTREAMER_LIBRARY_DIRS}
)

install(
  DIRECTORY config launch lib
  DESTINATION share/${PROJECT_NAME}
  PATTERN "*.deb" EXCLUDE
)

install(
  DIRECTORY include/
  DESTINATION include
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

if(CAMERA_SDK_INCLUDE_DIR AND CAMERA_SDK_LIBRARY)
  add_executable(${PROJECT_NAME}_node src/main.cpp)
  set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
  target_include_directories(${PROJECT_NAME}_node PRIVATE ${CAMERA_SDK_INCLUDE_DIR})
  ament_target_dependencies(${PROJECT_NAME}_node
    rclcpp
    rosidl_default_runtime
    camera_info_manager
    cv_bridge
    image_transport
    std_msgs
    std_srvs
    OpenCV
  )
  target_link_libraries(${PROJECT_NAME}_node
    ${CAMERA_SDK_LIBRARY}
    ${OpenCV_LIBRARIES}
    ${AVCODEC_LIBRARIES}
    ${AVFORMAT_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    ${SWSCALE_LIBRARIES}
    "${cpp_typesupport_target}"
  )
  get_filename_component(CAMERA_SDK_LIB_DIR "${CAMERA_SDK_LIBRARY}" DIRECTORY)
  get_filename_component(CAMERA_SDK_ROOT "${CAMERA_SDK_LIB_DIR}" DIRECTORY)
  # libCameraSDK.so loads ./jsons/camera_conf_Insta360_X4.json from the process cwd.
  target_compile_definitions(${PROJECT_NAME}_node PRIVATE
    CAMERA_SDK_BIN_DIR="${CAMERA_SDK_ROOT}/bin")
  set_target_properties(${PROJECT_NAME}_node PROPERTIES
    BUILD_RPATH "${CAMERA_SDK_LIB_DIR}"
    INSTALL_RPATH "${CAMERA_SDK_LIB_DIR}")
  install(TARGETS ${PROJECT_NAME}_node DESTINATION lib/${PROJECT_NAME})
else()
  message(STATUS "Insta360 CameraSDK not found: ${PROJECT_NAME} node will not be built")
endif()

add_executable(ffmpeg_decoder src/decoder.cpp)
ament_target_dependencies(ffmpeg_decoder
  rclcpp
  camera_info_manager
  cv_bridge
  image_transport
  std_msgs
  std_srvs
  OpenCV
)
target_link_libraries(ffmpeg_decoder ${OpenCV_LIBRARIES} ${AVCODEC_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVUTIL_LIBRARIES} ${SWSCALE_LIBRARIES})

install(TARGETS ffmpeg_decoder
  DESTINATION lib/${PROJECT_NAME}
)

if(GSTREAMER_FOUND)
  add_executable(gstreamer_encoder src/gstreamer_encoder.cpp src/gstreamer_ros_sink.cpp)
  ament_target_dependencies(gstreamer_encoder rclcpp cv_bridge sensor_msgs std_msgs OpenCV)
  target_link_libraries(gstreamer_encoder ${OpenCV_LIBRARIES} ${GSTREAMER_LIBRARIES})

  add_executable(gstreamer_decoder src/gstreamer_decoder.cpp)
  ament_target_dependencies(gstreamer_decoder rclcpp cv_bridge sensor_msgs std_msgs OpenCV)
  target_link_libraries(gstreamer_decoder ${GSTREAMER_LIBRARIES})
else()
  message(STATUS "GStreamer not found: gstreamer_encoder/decoder will not be built")
endif()

if(MEDIA_SDK_INCLUDE_DIR AND MEDIA_SDK_LIBRARY)
  get_filename_component(MEDIA_SDK_LIB_DIR "${MEDIA_SDK_LIBRARY}" DIRECTORY)
  add_executable(media_sdk_stitcher src/media_sdk_stitcher.cpp)
  ament_target_dependencies(media_sdk_stitcher rclcpp sensor_msgs rosidl_default_runtime)
  target_include_directories(media_sdk_stitcher PRIVATE ${MEDIA_SDK_INCLUDE_DIR})
  target_compile_definitions(media_sdk_stitcher PRIVATE
    MEDIA_SDK_MODEL_DIR="${MEDIA_SDK_LIB_DIR}/../bin/models/")
  target_link_libraries(media_sdk_stitcher ${MEDIA_SDK_LIBRARY} pthread "${cpp_typesupport_target}")
  set_target_properties(media_sdk_stitcher PROPERTIES INSTALL_RPATH "${MEDIA_SDK_LIB_DIR}")
  install(TARGETS media_sdk_stitcher DESTINATION lib/${PROJECT_NAME})
else()
  message(STATUS "Insta360 MediaSDK not found: media_sdk_stitcher will not be built")
endif()

add_executable(split_fisheye_rectify src/split_fisheye_rectify.cpp)
ament_target_dependencies(split_fisheye_rectify rclcpp sensor_msgs cv_bridge OpenCV)
target_link_libraries(split_fisheye_rectify ${OpenCV_LIBRARIES} "${cpp_typesupport_target}")

add_executable(webcam_equirectangular src/webcam_equirectangular.cpp)
ament_target_dependencies(webcam_equirectangular rclcpp sensor_msgs cv_bridge OpenCV)
target_link_libraries(webcam_equirectangular ${OpenCV_LIBRARIES})

# Add C++ image conversion nodes
add_executable(dual_fisheye2equirectangular_cpp src/dual_fisheye2equirectangular.cpp)
ament_target_dependencies(dual_fisheye2equirectangular_cpp
  rclcpp
  rcl_interfaces
  cv_bridge
  sensor_msgs
  std_msgs
  OpenCV
)
add_executable(dual_fisheye2perspective_cpp src/dual_fisheye2perspective.cpp)
ament_target_dependencies(dual_fisheye2perspective_cpp
  rclcpp
  rcl_interfaces
  cv_bridge
  sensor_msgs
  std_msgs
  OpenCV
  tf2_geometry_msgs
  geometry_msgs
)
add_executable(equirectangular2perspective_cpp src/equirectangular2perspective.cpp)
ament_target_dependencies(equirectangular2perspective_cpp
  rclcpp
  rcl_interfaces
  cv_bridge
  sensor_msgs
  std_msgs
  OpenCV
  tf2_geometry_msgs
  geometry_msgs
)
target_link_libraries(dual_fisheye2equirectangular_cpp ${OpenCV_LIBRARIES})
target_link_libraries(dual_fisheye2perspective_cpp ${OpenCV_LIBRARIES} Eigen3::Eigen)
target_link_libraries(equirectangular2perspective_cpp ${OpenCV_LIBRARIES} Eigen3::Eigen)

set(_INSTA360_INSTALL_TARGETS
  dual_fisheye2equirectangular_cpp
  dual_fisheye2perspective_cpp
  equirectangular2perspective_cpp
  webcam_equirectangular
  split_fisheye_rectify
)
if(GSTREAMER_FOUND)
  list(APPEND _INSTA360_INSTALL_TARGETS gstreamer_encoder gstreamer_decoder)
endif()
install(TARGETS ${_INSTA360_INSTALL_TARGETS}
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/equirectangular.py
  scripts/calibrate.py
  scripts/calibrate_cpp.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
