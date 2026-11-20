/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

#include "dlio/map.h"
#include "dlio/utils.h"

dlio::MapNode::MapNode(): Node("dlio_map_node") {

  this->getParams();

  this->keyframe_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto keyframe_sub_opt = rclcpp::SubscriptionOptions();
  keyframe_sub_opt.callback_group = this->keyframe_cb_group;
  this->keyframe_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("keyframes", 10,
      std::bind(&dlio::MapNode::callbackCloud, this, std::placeholders::_1), keyframe_sub_opt);

  this->map_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("map", rclcpp::QoS(100).best_effort());
  this->map_pub_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  this->map_pub_timer = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / this->map_pub_rate_),
      std::bind(&dlio::MapNode::publishMap, this),
      this->map_pub_cb_group);

  this->save_pcd_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  this->save_pcd_srv = this->create_service<direct_lidar_inertial_odometry::srv::SavePCD>("save_pcd",
      std::bind(&dlio::MapNode::savePCD2, this, std::placeholders::_1, std::placeholders::_2), rmw_qos_profile_services_default, this->save_pcd_cb_group);

  this->dlio_map = std::make_shared<pcl::PointCloud<PointType>>();

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

}

dlio::MapNode::~MapNode() {}

void dlio::MapNode::getParams() {

  this->declare_parameter<std::string>("odom/odom_frame", "odom");
  this->declare_parameter<double>("map/sparse/leafSize", 0.5);
  this->declare_parameter<double>("map/voxel/leafSize", 0.1);
  this->declare_parameter<double>("map/publishRate", 1.0);

  this->get_parameter("odom/odom_frame", this->odom_frame);
  this->get_parameter("map/sparse/leafSize", this->leaf_size_);
  this->get_parameter("map/publishRate", this->map_pub_rate_);
  
  double voxel_leaf_size;
  this->get_parameter("map/voxel/leafSize", voxel_leaf_size);
  this->voxel_map = VoxelMap<PointType>(voxel_leaf_size);
}

void dlio::MapNode::start() {
}

void dlio::MapNode::callbackKeyframe(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& keyframe) {

  // convert scan to pcl format
  pcl::PointCloud<PointType>::Ptr keyframe_pcl = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*keyframe, *keyframe_pcl);

  // voxel filter
  if (this->leaf_size_ > 1e-6) {
    this->voxelgrid.setLeafSize(this->leaf_size_, this->leaf_size_, this->leaf_size_);
    this->voxelgrid.setInputCloud(keyframe_pcl);
    this->voxelgrid.filter(*keyframe_pcl);
  }

  // save filtered keyframe to map for rviz
  *this->dlio_map += *keyframe_pcl;

  // publish full map
  if (this->dlio_map->points.size() == this->dlio_map->width * this->dlio_map->height) {
    sensor_msgs::msg::PointCloud2 map_ros;
    pcl::toROSMsg(*this->dlio_map, map_ros);
    map_ros.header.stamp = this->now();
    map_ros.header.frame_id = this->odom_frame;
    this->map_pub->publish(map_ros);
  } 
}

void dlio::MapNode::callbackCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud) {

  // convert scan to pcl format
  pcl::PointCloud<PointType>::Ptr cloud_pcl = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*cloud, *cloud_pcl);

  // add cloud to voxel map
  this->voxel_map.addCloud(cloud_pcl);
}

void dlio::MapNode::publishMap() {

  sensor_msgs::msg::PointCloud2 map_ros;
  pcl::PointCloud<PointType>::Ptr voxel_map_cloud = this->voxel_map.getCloud();
  // voxel filter
  if (this->leaf_size_ > 1e-6) {
    this->voxelgrid.setLeafSize(this->leaf_size_, this->leaf_size_, this->leaf_size_);
    this->voxelgrid.setInputCloud(voxel_map_cloud);
    this->voxelgrid.filter(*voxel_map_cloud);
  }
  pcl::toROSMsg(*voxel_map_cloud, map_ros);
  map_ros.header.stamp = this->now();
  map_ros.header.frame_id = this->odom_frame;
  this->map_pub->publish(map_ros);
}

void dlio::MapNode::savePCD(std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,
                            std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res) {

  pcl::PointCloud<PointType>::Ptr m = std::make_shared<pcl::PointCloud<PointType>>(*this->dlio_map);

  float leaf_size = req->leaf_size;
  std::string p = req->save_path;

  std::cout << std::setprecision(2) << "Saving map to " << p + "/dlio_map.pcd"
    << " with leaf size " << to_string_with_precision(leaf_size, 2) << "... "; std::cout.flush();

  // voxelize map
  pcl::VoxelGrid<PointType> vg;
  vg.setLeafSize(leaf_size, leaf_size, leaf_size);
  vg.setInputCloud(m);
  vg.filter(*m);

  // save map
  int ret = pcl::io::savePCDFileBinary(p + "/dlio_map.pcd", *m);
  res->success = ret == 0;

  if (res->success) {
    std::cout << "done" << std::endl;
  } else {
    std::cout << "failed" << std::endl;
  }
}

void dlio::MapNode::savePCD2(std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,
                             std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res) {

  pcl::PointCloud<PointType>::Ptr voxel_map_cloud = this->voxel_map.getCloud();

  std::string p = req->save_path;
  std::cout << std::setprecision(2) << "Saving map to " << p + "/dlio_map.pcd" << "... "; std::cout.flush();

  // save map
  int ret = pcl::io::savePCDFileBinary(p + "/dlio_map.pcd", *voxel_map_cloud);
  res->success = ret == 0;

  if (res->success) {
    std::cout << "done" << std::endl;
  } else {
    std::cout << "failed" << std::endl;
  }
}
