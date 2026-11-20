cmake_minimum_required(VERSION 3.5)
project(qi_move_base)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(angles REQUIRED)
find_package(pluginlib REQUIRED)
find_package(pcl_conversions REQUIRED)
# find_package(recovery_behaviors REQUIRED)
find_package(dddmr_sys_core REQUIRED)

find_package(PCL REQUIRED)

set(dependencies
  rclcpp
  rclcpp_action
  rclcpp_lifecycle
  rclcpp_components
  std_msgs
  std_srvs
  sensor_msgs
  visualization_msgs
  nav_msgs
  geometry_msgs
  builtin_interfaces
  message_filters
  tf2_ros
  tf2
  tf2_geometry_msgs
  tf2_eigen
  pluginlib
  pcl_conversions
  angles
  # recovery_behaviors
  dddmr_sys_core
)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${dddmr_sys_core_INCLUDE_DIRS}
)

# p2p_fsm
add_library(qi_p2p_fsm SHARED
  src/p2p_fsm.cpp
)
target_link_libraries(qi_p2p_fsm
  ${PCL_LIBRARIES}
)
ament_target_dependencies(qi_p2p_fsm
  ${dependencies}
)

# p2p_global_plan_manager
add_library(qi_p2p_global_plan_manager SHARED
  src/p2p_global_plan_manager.cpp
)
target_link_libraries(qi_p2p_global_plan_manager
  ${PCL_LIBRARIES}
)
ament_target_dependencies(qi_p2p_global_plan_manager
  ${dependencies}
)

# qi_move_base
add_library(qi_p2p_mov2_base SHARED
  src/p2p_move_base.cpp
)
target_link_libraries(qi_p2p_mov2_base
  qi_p2p_fsm
  qi_p2p_global_plan_manager
)
ament_target_dependencies(qi_p2p_mov2_base
  ${dependencies}
)

# qi_move_base_node
add_executable(qi_move_base_node
  src/p2p_move_base_node.cpp
)
target_link_libraries(qi_move_base_node
  qi_p2p_mov2_base
)
ament_target_dependencies(qi_move_base_node
  ${dependencies}
)

install(TARGETS
  qi_p2p_fsm
  qi_p2p_global_plan_manager
  qi_p2p_mov2_base
  qi_move_base_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY
  launch
  config
  rviz
  scripts
  DESTINATION share/${PROJECT_NAME}/
)

# Install Python scripts as executables
install(PROGRAMS
  scripts/clicked2goal.py
  scripts/nav_btree/bt_creater.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  scripts/nav_btree
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(qi_p2p_fsm qi_p2p_global_plan_manager qi_p2p_mov2_base)
ament_export_dependencies(${dependencies})
ament_package()
