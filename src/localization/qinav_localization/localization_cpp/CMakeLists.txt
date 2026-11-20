cmake_minimum_required(VERSION 3.14)
project(localization_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters registration kdtree)
find_package(Eigen3 REQUIRED)

set(DEPS
  rclcpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  tf2_sensor_msgs
  pcl_ros
  pcl_conversions
)

# ── global_localization ───────────────────────────────────────────────────────
add_executable(global_localization src/global_localization.cpp)
ament_target_dependencies(global_localization ${DEPS})
target_include_directories(global_localization PUBLIC
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(global_localization ${PCL_LIBRARIES})

# ── transform_fusion ──────────────────────────────────────────────────────────
add_executable(transform_fusion src/transform_fusion.cpp)
ament_target_dependencies(transform_fusion ${DEPS})
target_include_directories(transform_fusion PUBLIC
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(transform_fusion ${PCL_LIBRARIES})

# ── install ───────────────────────────────────────────────────────────────────
install(TARGETS
  global_localization
  transform_fusion
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
