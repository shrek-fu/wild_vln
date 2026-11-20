cmake_minimum_required(VERSION 3.14)
project(cloud_preprocess)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(livox_ros_driver2 REQUIRED)

set(DEPS
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  livox_ros_driver2
)

add_executable(cloud_transformer src/cloud_transformer.cpp)
ament_target_dependencies(cloud_transformer ${DEPS})

add_executable(ground_trans src/ground_trans.cpp)
ament_target_dependencies(ground_trans ${DEPS})

install(TARGETS
  cloud_transformer
  ground_trans
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})

ament_package()
