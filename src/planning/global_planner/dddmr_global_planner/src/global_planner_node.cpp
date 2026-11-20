/*
* BSD 3-Clause License

* Copyright (c) 2024, DDDMobileRobot

* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:

* 1. Redistributions of source code must retain the above copyright notice, this
*    list of conditions and the following disclaimer.

* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.

* 3. Neither the name of the copyright holder nor the names of its
*    contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.

* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <global_planner/dynamic_window_aware_global_planner.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <chrono>

int main(int argc, char **argv)
{

  rclcpp::init(argc, argv);
  auto perception_3d = std::make_shared<perception_3d::Perception3D_ROS>("perception_3d_global");
  auto global_planner = std::make_shared<global_planner::GlobalPlanner>("global_planner");
  auto dwa_global_planner = std::make_shared<global_planner::DWA_GlobalPlanner>("dynamic_window_aware_global_planner");
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(perception_3d);
  perception_3d->initial();
  executor.add_node(global_planner);
  global_planner->initial(perception_3d);
  executor.add_node(dwa_global_planner);
  dwa_global_planner->initial(perception_3d, global_planner);

  // Optional in-process bootstrap:
  // If map_file / ground_file are provided, publish mapcloud/mapground once so
  // static layer can build graph even when external pcl_publisher is absent.
  global_planner->declare_parameter("map_file", std::string(""));
  global_planner->declare_parameter("ground_file", std::string(""));
  global_planner->declare_parameter("global_frame", std::string("map"));
  std::string map_file = global_planner->get_parameter("map_file").as_string();
  std::string ground_file = global_planner->get_parameter("ground_file").as_string();
  std::string global_frame = global_planner->get_parameter("global_frame").as_string();

  auto map_pub = global_planner->create_publisher<sensor_msgs::msg::PointCloud2>(
    "mapcloud", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());
  auto ground_pub = global_planner->create_publisher<sensor_msgs::msg::PointCloud2>(
    "mapground", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());

  if(!ground_file.empty()){
    auto publish_timer = global_planner->create_wall_timer(
      std::chrono::milliseconds(300),
      [global_planner, map_pub, ground_pub, map_file, ground_file, global_frame]() {
        static bool published = false;
        if(published){
          return;
        }
        pcl::PointCloud<pcl::PointXYZI> ground_cloud;
        const int ground_ret = pcl::io::loadPCDFile<pcl::PointXYZI>(ground_file, ground_cloud);
        if(ground_ret != 0){
          RCLCPP_WARN(
            global_planner->get_logger(),
            "Failed to preload ground_file for graph bootstrap: ground=%s (ret=%d)",
            ground_file.c_str(), ground_ret);
          return;
        }

        sensor_msgs::msg::PointCloud2 ground_msg;
        pcl::toROSMsg(ground_cloud, ground_msg);
        ground_msg.header.frame_id = global_frame;
        ground_msg.header.stamp = global_planner->now();
        ground_pub->publish(ground_msg);

        if(!map_file.empty()){
          pcl::PointCloud<pcl::PointXYZI> map_cloud;
          const int map_ret = pcl::io::loadPCDFile<pcl::PointXYZI>(map_file, map_cloud);
          if(map_ret == 0){
            sensor_msgs::msg::PointCloud2 map_msg;
            pcl::toROSMsg(map_cloud, map_msg);
            map_msg.header.frame_id = global_frame;
            map_msg.header.stamp = global_planner->now();
            map_pub->publish(map_msg);
          } else {
            RCLCPP_WARN(
              global_planner->get_logger(),
              "Failed to preload map_file for graph bootstrap: map=%s (ret=%d)",
              map_file.c_str(), map_ret);
          }
        }
        published = true;
        RCLCPP_INFO(
          global_planner->get_logger(),
          "Published in-process mapground%s for graph bootstrap.",
          map_file.empty() ? "" : "/mapcloud");
      });
    (void)publish_timer;
  } else {
    RCLCPP_INFO(
      global_planner->get_logger(),
      "Skip in-process map bootstrap because ground_file is empty.");
  }

  executor.spin();

  rclcpp::shutdown();

  return 0;
}