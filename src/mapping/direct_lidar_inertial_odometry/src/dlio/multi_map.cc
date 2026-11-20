#include "dlio/multi_map.h"

#include <stdexcept>

dlio::MultiMapNode::MultiMapNode() : Node("dlio_multi_map_node") {

  this->getParams();

  // independent callback groups: each subscription runs in its own thread
  cb_group_map_        = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  cb_group_ground_     = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  cb_group_non_ground_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  cb_group_odom_       = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  pub_cb_group_        = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  srv_cb_group_        = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  auto opt_map        = rclcpp::SubscriptionOptions(); opt_map.callback_group        = cb_group_map_;
  auto opt_ground     = rclcpp::SubscriptionOptions(); opt_ground.callback_group     = cb_group_ground_;
  auto opt_non_ground = rclcpp::SubscriptionOptions(); opt_non_ground.callback_group = cb_group_non_ground_;
  auto opt_odom       = rclcpp::SubscriptionOptions(); opt_odom.callback_group       = cb_group_odom_;

  // subscriptions
  map_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "cloud_map", 10,
      std::bind(&dlio::MultiMapNode::callbackMap, this, std::placeholders::_1), opt_map);
  ground_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "cloud_ground", 10,
      std::bind(&dlio::MultiMapNode::callbackGround, this, std::placeholders::_1), opt_ground);
  non_ground_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "cloud_non_ground", 10,
      std::bind(&dlio::MultiMapNode::callbackNonGround, this, std::placeholders::_1), opt_non_ground);
  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "odom", 100,
      std::bind(&dlio::MultiMapNode::callbackOdom, this, std::placeholders::_1), opt_odom);

  if (wait_for_localization_) {
    auto opt_loc = rclcpp::SubscriptionOptions();
    opt_loc.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    loc_status_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/localize/status", 10,
        std::bind(&dlio::MultiMapNode::callbackLocStatus, this, std::placeholders::_1),
        opt_loc);
    RCLCPP_INFO(this->get_logger(),
        "Waiting for localization_success before inserting new scans");
  }

  // publishers
  map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "map", rclcpp::QoS(10).best_effort());
  ground_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "ground_map", rclcpp::QoS(10).best_effort());
  non_ground_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "non_ground_map", rclcpp::QoS(10).best_effort());

  pub_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / map_pub_rate_),
      std::bind(&dlio::MultiMapNode::publishMaps, this),
      pub_cb_group_);

  // save service
  save_srv_ = this->create_service<direct_lidar_inertial_odometry::srv::SavePCD>(
      "save_pcd",
      std::bind(&dlio::MultiMapNode::savePCD, this,
                std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, srv_cb_group_);

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  RCLCPP_INFO(this->get_logger(),
    "dlio_multi_map_node started. Topics: cloud_map / cloud_ground / cloud_non_ground");
}

dlio::MultiMapNode::~MultiMapNode() {}

void dlio::MultiMapNode::getParams() {
  this->declare_parameter<std::string>("odom/odom_frame", "odom");
  this->declare_parameter<double>("map/sparse/leafSize", 0.5);
  this->declare_parameter<double>("map/voxel/leafSize", 0.1);
  this->declare_parameter<double>("map/publishRate", 1.0);
  this->declare_parameter<double>("map/odom_interval", 0.5);   // m
  this->declare_parameter<double>("map/robot_box/x", 0.6);     // half-length (m)
  this->declare_parameter<double>("map/robot_box/y", 0.6);     // half-width  (m)
  this->declare_parameter<double>("map/robot_box/z", 1.0);     // half-height (m)
  this->declare_parameter<double>("map/stamp_gap_thresh", 1.0); // seconds
  this->declare_parameter<std::string>("map/prior_map", "");
  this->declare_parameter<std::string>("map/prior_ground", "");
  this->declare_parameter<std::string>("map/prior_nonground", "");
  this->declare_parameter<bool>("map/wait_for_localization", false);

  this->get_parameter("odom/odom_frame", odom_frame_);
  this->get_parameter("map/sparse/leafSize", leaf_size_);
  this->get_parameter("map/publishRate", map_pub_rate_);
  this->get_parameter("map/odom_interval", odom_interval_);
  this->get_parameter("map/robot_box/x", box_half_x_);
  this->get_parameter("map/robot_box/y", box_half_y_);
  this->get_parameter("map/robot_box/z", box_half_z_);
  this->get_parameter("map/stamp_gap_thresh", stamp_gap_thresh_);

  double voxel_leaf_size;
  this->get_parameter("map/voxel/leafSize", voxel_leaf_size);
  voxel_map_        = VoxelMap<PointType>(voxel_leaf_size);
  voxel_ground_     = VoxelMap<PointType>(voxel_leaf_size);
  voxel_non_ground_ = VoxelMap<PointType>(voxel_leaf_size);

  std::string prior_map;
  std::string prior_ground;
  std::string prior_nonground;
  this->get_parameter("map/prior_map", prior_map);
  this->get_parameter("map/prior_ground", prior_ground);
  this->get_parameter("map/prior_nonground", prior_nonground);
  this->get_parameter("map/wait_for_localization", wait_for_localization_);
  insert_enabled_ = !wait_for_localization_;

  const bool priors_ok =
      loadPriorCloud(prior_map, voxel_map_, has_prior_map_) &&
      loadPriorCloud(prior_ground, voxel_ground_, has_prior_ground_) &&
      loadPriorCloud(prior_nonground, voxel_non_ground_, has_prior_nonground_);
  if (!priors_ok) {
    throw std::runtime_error("dlio_multi_map_node failed to load prior maps");
  }
}

bool dlio::MultiMapNode::mappingEnabled() {
  std::lock_guard<std::mutex> lk(insert_mutex_);
  return insert_enabled_;
}

bool dlio::MultiMapNode::loadPriorCloud(
    const std::string& path, VoxelMap<PointType>& voxel_map, bool& has_prior) {
  if (path.empty()) {
    if (wait_for_localization_) {
      RCLCPP_ERROR(this->get_logger(), "Incremental mapping requires a prior cloud path");
      return false;
    }
    return true;
  }
  auto cloud = std::make_shared<pcl::PointCloud<PointType>>();
  if (pcl::io::loadPCDFile<PointType>(path, *cloud) < 0 || cloud->empty()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load prior cloud: %s", path.c_str());
    return false;
  }
  voxel_map.addCloud(cloud);
  has_prior = true;
  RCLCPP_INFO(this->get_logger(), "Loaded prior cloud %s (%zu pts)", path.c_str(), cloud->size());
  return true;
}

void dlio::MultiMapNode::callbackLocStatus(const std_msgs::msg::String::ConstSharedPtr& msg) {
  if (!msg || msg->data != "localization_success") {
    return;
  }
  std::lock_guard<std::mutex> lk(insert_mutex_);
  if (insert_enabled_) {
    return;
  }
  insert_enabled_ = true;
  RCLCPP_INFO(this->get_logger(), "Localization succeeded; incremental mapping started");
}

void dlio::MultiMapNode::callbackMap(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud) {
  if (!mappingEnabled()) {
    return;
  }
  rclcpp::Time stamp(cloud->header.stamp);

  auto pcl_cloud = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*cloud, *pcl_cloud);

  std::lock_guard<std::mutex> lk(map_mutex_);
  if (has_stamp_map_) {
    double gap = (stamp - last_stamp_map_).seconds();
    if (gap > stamp_gap_thresh_)
      RCLCPP_WARN(this->get_logger(),
          "[cloud_map] stamp gap %.3f s > %.1f s", gap, stamp_gap_thresh_);
  }
  last_stamp_map_ = stamp;
  has_stamp_map_  = true;
  voxel_map_.addCloud(pcl_cloud);
}

void dlio::MultiMapNode::callbackGround(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud) {
  if (!mappingEnabled()) {
    return;
  }
  rclcpp::Time stamp(cloud->header.stamp);

  auto pcl_cloud = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*cloud, *pcl_cloud);

  std::lock_guard<std::mutex> lk(ground_mutex_);
  if (has_stamp_ground_) {
    double gap = (stamp - last_stamp_ground_).seconds();
    if (gap > stamp_gap_thresh_)
      RCLCPP_WARN(this->get_logger(),
          "[cloud_ground] stamp gap %.3f s > %.1f s", gap, stamp_gap_thresh_);
  }
  last_stamp_ground_ = stamp;
  has_stamp_ground_  = true;
  voxel_ground_.addCloud(pcl_cloud);
}

void dlio::MultiMapNode::callbackNonGround(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud) {
  if (!mappingEnabled()) {
    return;
  }
  rclcpp::Time stamp(cloud->header.stamp);

  auto pcl_cloud = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*cloud, *pcl_cloud);

  std::lock_guard<std::mutex> lk(non_ground_mutex_);
  if (has_stamp_non_ground_) {
    double gap = (stamp - last_stamp_non_ground_).seconds();
    if (gap > stamp_gap_thresh_)
      RCLCPP_WARN(this->get_logger(),
          "[cloud_non_ground] stamp gap %.3f s > %.1f s", gap, stamp_gap_thresh_);
  }
  last_stamp_non_ground_ = stamp;
  has_stamp_non_ground_  = true;
  voxel_non_ground_.addCloud(pcl_cloud);
}

void dlio::MultiMapNode::callbackOdom(
    const nav_msgs::msg::Odometry::ConstSharedPtr& odom) {
  if (!mappingEnabled()) {
    return;
  }
  rclcpp::Time stamp(odom->header.stamp);
  if (has_stamp_odom_) {
    double gap = (stamp - last_stamp_odom_).seconds();
    if (gap > stamp_gap_thresh_)
      RCLCPP_WARN(this->get_logger(),
          "[odom] stamp gap %.3f s > %.1f s", gap, stamp_gap_thresh_);
  }
  last_stamp_odom_ = stamp;
  has_stamp_odom_  = true;

  Eigen::Vector3d p(
      odom->pose.pose.position.x,
      odom->pose.pose.position.y,
      odom->pose.pose.position.z);
  std::lock_guard<std::mutex> lk(pose_mutex_);
  if (!has_last_pose_ || (p - last_pose_).norm() >= odom_interval_) {
    pose_list_.push_back(p);
    last_pose_   = p;
    has_last_pose_ = true;
  }
}

pcl::PointCloud<PointType>::Ptr dlio::MultiMapNode::filterRobotBoxes(
    pcl::PointCloud<PointType>::Ptr cloud) const
{
  // For each recorded pose, remove points whose world-frame position falls
  // inside [pose ± box_half_x, pose ± box_half_y, pose ± box_half_z].
  // We treat the box as axis-aligned in the odom frame for efficiency.
  auto out = std::make_shared<pcl::PointCloud<PointType>>();
  out->reserve(cloud->size());

  std::vector<Eigen::Vector3d> poses;
  {
    std::lock_guard<std::mutex> lk(const_cast<std::mutex&>(pose_mutex_));
    poses = pose_list_;
  }

  for (const auto& pt : cloud->points) {
    bool inside = false;
    for (const auto& p : poses) {
      if (std::abs(pt.x - p.x()) <= box_half_x_ &&
          std::abs(pt.y - p.y()) <= box_half_y_ &&
          std::abs(pt.z - p.z()) <= box_half_z_) {
        inside = true;
        break;
      }
    }
    if (!inside) out->points.push_back(pt);
  }
  out->width  = out->points.size();
  out->height = 1;
  out->is_dense = true;
  return out;
}

// helper: copy + optional sparse voxel filter + toROS
static sensor_msgs::msg::PointCloud2 toMsg(
    pcl::PointCloud<PointType>::Ptr cloud,
    pcl::VoxelGrid<PointType>& vg,
    double leaf_size,
    const std::string& frame,
    const rclcpp::Time& stamp)
{
  if (leaf_size > 1e-6) {
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    vg.setInputCloud(cloud);
    vg.filter(*cloud);
  }
  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(*cloud, msg);
  msg.header.stamp    = stamp;
  msg.header.frame_id = frame;
  return msg;
}

void dlio::MultiMapNode::publishMaps() {
  {
    std::lock_guard<std::mutex> lk(map_mutex_);
    auto c = voxel_map_.getCloud();
    if (map_pub_->get_subscription_count() > 0 && !c->empty() && (has_stamp_map_ || has_prior_map_))
      map_pub_->publish(toMsg(
          c, voxelgrid_, leaf_size_, odom_frame_,
          has_stamp_map_ ? last_stamp_map_ : this->now()));
  }
  {
    std::lock_guard<std::mutex> lk(ground_mutex_);
    auto c = voxel_ground_.getCloud();
    if (ground_pub_->get_subscription_count() > 0 && !c->empty() && (has_stamp_ground_ || has_prior_ground_))
      ground_pub_->publish(toMsg(
          c, voxelgrid_, leaf_size_, odom_frame_,
          has_stamp_ground_ ? last_stamp_ground_ : this->now()));
  }
  {
    std::lock_guard<std::mutex> lk(non_ground_mutex_);
    auto c = voxel_non_ground_.getCloud();
    if (non_ground_pub_->get_subscription_count() > 0 && !c->empty() &&
        (has_stamp_non_ground_ || has_prior_nonground_))
      non_ground_pub_->publish(toMsg(
          c, voxelgrid_, leaf_size_, odom_frame_,
          has_stamp_non_ground_ ? last_stamp_non_ground_ : this->now()));
  }
}

void dlio::MultiMapNode::savePCD(
    std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,
    std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res)
{
  std::string p = req->save_path;
  RCLCPP_INFO(this->get_logger(), "Saving three maps to %s ...", p.c_str());

  bool ok = true;

  // helper: save any PointCloud<PointType> by pointer
  auto saveCloud = [&](pcl::PointCloud<PointType>::Ptr cloud, const std::string& filename) {
    if (cloud->empty()) {
      RCLCPP_WARN(this->get_logger(), "%s is empty, skipping.", filename.c_str());
      return;
    }
    std::string path = p + "/" + filename;
    int ret = pcl::io::savePCDFileBinary(path, *cloud);
    if (ret == 0) {
      RCLCPP_INFO(this->get_logger(), "Saved %s (%zu pts)", path.c_str(), cloud->size());
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to save %s", path.c_str());
      ok = false;
    }
  };

  // helper: lock voxel map, get cloud, delegate to saveCloud
  auto save = [&](VoxelMap<PointType>& vm, std::mutex& mtx, const std::string& filename) {
    std::lock_guard<std::mutex> lk(mtx);
    saveCloud(vm.getCloud(), filename);
  };

  save(voxel_map_,    map_mutex_,    "map.pcd");
  save(voxel_ground_, ground_mutex_, "ground_map.pcd");
//   save(voxel_non_ground_, non_ground_mutex_, "non_ground_map_raw.pcd");

  // non_ground: save raw, apply robot-box filter, save filtered
  {
    std::lock_guard<std::mutex> lk(non_ground_mutex_);
    auto raw = voxel_non_ground_.getCloud();
    saveCloud(raw, "non_ground_map_raw.pcd");
    if (!raw->empty()) {
      RCLCPP_INFO(this->get_logger(),
          "Filtering non_ground_map (%zu pts, %zu poses) ...",
          raw->size(), pose_list_.size());
      auto filtered = filterRobotBoxes(raw);
      RCLCPP_INFO(this->get_logger(),
          "After box filter: %zu pts removed, %zu pts remaining",
          raw->size() - filtered->size(), filtered->size());
      saveCloud(filtered, "non_ground_map.pcd");
    }
  }

  // save odometry poses as PCD (x,y,z per pose)
  {
    std::vector<Eigen::Vector3d> poses;
    {
      std::lock_guard<std::mutex> lk(pose_mutex_);
      poses = pose_list_;
    }
    if (poses.empty()) {
      RCLCPP_WARN(this->get_logger(), "No odometry poses recorded, skipping pose pcd.");
    } else {
      auto pose_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      pose_cloud->reserve(poses.size());
      for (const auto& pos : poses)
        pose_cloud->push_back(pcl::PointXYZ(
            static_cast<float>(pos.x()),
            static_cast<float>(pos.y()),
            static_cast<float>(pos.z())));
      pose_cloud->width   = pose_cloud->size();
      pose_cloud->height  = 1;
      pose_cloud->is_dense = true;
      if (pose_cloud->empty()) {
        RCLCPP_WARN(this->get_logger(), "odom_poses.pcd is empty, skipping.");
      } else {
        std::string path = p + "/odom_poses.pcd";
        int ret = pcl::io::savePCDFileBinary(path, *pose_cloud);
        if (ret == 0)
          RCLCPP_INFO(this->get_logger(), "Saved %s (%zu poses)", path.c_str(), pose_cloud->size());
        else {
          RCLCPP_ERROR(this->get_logger(), "Failed to save %s", path.c_str());
          ok = false;
        }
      }
    }
  }

  res->success = ok;
}
