#pragma once

#include "dlio/dlio.h"

#include "rclcpp/rclcpp.hpp"
#include "direct_lidar_inertial_odometry/srv/save_pcd.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <mutex>
#include <vector>

#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/string.hpp>
#include <Eigen/Dense>

#include "dlio/voxel_map.h"

namespace dlio {

class MultiMapNode : public rclcpp::Node {
public:
  MultiMapNode();
  ~MultiMapNode();

private:
  void getParams();

  void callbackMap(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud);
  void callbackGround(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud);
  void callbackNonGround(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud);
  void callbackOdom(const nav_msgs::msg::Odometry::ConstSharedPtr& odom);
  void callbackLocStatus(const std_msgs::msg::String::ConstSharedPtr& msg);

  bool mappingEnabled();
  bool loadPriorCloud(const std::string& path, VoxelMap<PointType>& voxel_map, bool& has_prior);

  // filter non_ground cloud: remove points inside boxes at each recorded pose
  pcl::PointCloud<PointType>::Ptr filterRobotBoxes(
      pcl::PointCloud<PointType>::Ptr cloud) const;

  void publishMaps();

  void savePCD(std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,
               std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res);

  // subscriptions
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr ground_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr non_ground_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr loc_status_sub_;

  // recorded robot poses (odom frame)
  std::vector<Eigen::Vector3d> pose_list_;
  std::mutex pose_mutex_;
  double odom_interval_;   // min distance (m) between recorded poses
  Eigen::Vector3d last_pose_;
  bool has_last_pose_{false};

  // box filter dimensions (half-extents in robot frame, metres)
  double box_half_x_;   // forward/backward
  double box_half_y_;   // left/right
  double box_half_z_;   // up/down

  // independent callback groups: each topic runs in its own thread
  rclcpp::CallbackGroup::SharedPtr cb_group_map_;
  rclcpp::CallbackGroup::SharedPtr cb_group_ground_;
  rclcpp::CallbackGroup::SharedPtr cb_group_non_ground_;
  rclcpp::CallbackGroup::SharedPtr cb_group_odom_;
  rclcpp::CallbackGroup::SharedPtr pub_cb_group_;
  rclcpp::CallbackGroup::SharedPtr srv_cb_group_;

  // publishers
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr non_ground_pub_;
  rclcpp::TimerBase::SharedPtr pub_timer_;

  // save service
  rclcpp::Service<direct_lidar_inertial_odometry::srv::SavePCD>::SharedPtr save_srv_;

  // voxel maps (thread-safe via mutex)
  VoxelMap<PointType> voxel_map_;
  VoxelMap<PointType> voxel_ground_;
  VoxelMap<PointType> voxel_non_ground_;

  std::mutex map_mutex_;
  std::mutex ground_mutex_;
  std::mutex non_ground_mutex_;

  // last-message timestamps for gap monitoring
  rclcpp::Time last_stamp_map_;
  rclcpp::Time last_stamp_ground_;
  rclcpp::Time last_stamp_non_ground_;
  rclcpp::Time last_stamp_odom_;
  bool has_stamp_map_{false};
  bool has_stamp_ground_{false};
  bool has_stamp_non_ground_{false};
  bool has_stamp_odom_{false};
  bool has_prior_map_{false};
  bool has_prior_ground_{false};
  bool has_prior_nonground_{false};
  bool wait_for_localization_{false};
  bool insert_enabled_{true};
  std::mutex insert_mutex_;
  double stamp_gap_thresh_{0.5};  // seconds

  pcl::VoxelGrid<PointType> voxelgrid_;

  std::string odom_frame_;
  std::string odom_topic_;
  double leaf_size_;
  double map_pub_rate_;
};

} // namespace dlio
