cmake_minimum_required(VERSION 3.8)
project(go2w_keyboard_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(Threads REQUIRED)

set(UNITREE_SDK2_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../unitree_sdk2)
set(UNITREE_ARCH ${CMAKE_SYSTEM_PROCESSOR})

include_directories(
  include
  ${UNITREE_SDK2_ROOT}/include
  ${UNITREE_SDK2_ROOT}/thirdparty/include
  ${UNITREE_SDK2_ROOT}/thirdparty/include/ddscxx
)

link_directories(
  ${UNITREE_SDK2_ROOT}/lib/${UNITREE_ARCH}
  ${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_ARCH}
)

add_executable(go2w_keyboard_node src/go2w_keyboard_node.cpp)
ament_target_dependencies(go2w_keyboard_node
  geometry_msgs
  rclcpp
  std_msgs
)

add_executable(go2_cmd_vel_node src/go2_cmd_vel_node.cpp)
ament_target_dependencies(go2_cmd_vel_node
  geometry_msgs
  rclcpp
  std_msgs
)

add_executable(go2w_sport_client_demo src/go2w_sport_client_demo.cpp)

add_executable(go2w_battery_node src/go2w_battery_node.cpp)
ament_target_dependencies(go2w_battery_node
  rclcpp
  sensor_msgs
  diagnostic_msgs
)

target_link_libraries(go2w_keyboard_node
  unitree_sdk2
  ddsc
  ddscxx
  Threads::Threads
)

target_link_libraries(go2_cmd_vel_node
  unitree_sdk2
  ddsc
  ddscxx
  Threads::Threads
)

target_link_libraries(go2w_battery_node
  unitree_sdk2
  ddsc
  ddscxx
  Threads::Threads
)

target_link_libraries(go2w_sport_client_demo
  unitree_sdk2
  ddsc
  ddscxx
  Threads::Threads
)

set(_UNITREE_RPATH "-Wl,-rpath,${UNITREE_SDK2_ROOT}/lib/${UNITREE_ARCH}:${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_ARCH}")
target_link_options(go2w_keyboard_node PRIVATE ${_UNITREE_RPATH})
target_link_options(go2_cmd_vel_node PRIVATE ${_UNITREE_RPATH})
target_link_options(go2w_sport_client_demo PRIVATE ${_UNITREE_RPATH})
target_link_options(go2w_battery_node PRIVATE ${_UNITREE_RPATH})

install(TARGETS go2w_keyboard_node go2_cmd_vel_node go2w_sport_client_demo go2w_battery_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
