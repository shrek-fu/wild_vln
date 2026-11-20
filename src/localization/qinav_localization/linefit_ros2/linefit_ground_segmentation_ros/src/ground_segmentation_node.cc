#include <memory>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
// #include <pcl_ros/point_cloud.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "ground_segmentation/ground_segmentation.h"

class SegmentationNode : public rclcpp::Node {
public:
  SegmentationNode(const rclcpp::NodeOptions &node_options);
  void scanCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obstacle_pub_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  GroundSegmentationParams params_;
  std::shared_ptr<GroundSegmentation> segmenter_;
  double voxel_leaf_size_;  // 体素滤波的叶子大小参数
  double max_distance_;     // 最大距离过滤参数
};

SegmentationNode::SegmentationNode(const rclcpp::NodeOptions &node_options)
    : Node("ground_segmentation", node_options) {
  voxel_leaf_size_ = this->declare_parameter("voxel_leaf_size", 0.05);
  max_distance_ = this->declare_parameter("max_distance", 10.0);
  params_.n_bins = this->declare_parameter("n_bins", params_.n_bins);
  params_.n_segments =
      this->declare_parameter("n_segments", params_.n_segments);
  params_.max_dist_to_line =
      this->declare_parameter("max_dist_to_line", params_.max_dist_to_line);
  params_.max_slope = this->declare_parameter("max_slope", params_.max_slope);
  params_.min_slope = this->declare_parameter("min_slope", params_.min_slope);
  params_.long_threshold =
      this->declare_parameter("long_threshold", params_.long_threshold);
  params_.max_long_height =
      this->declare_parameter("max_long_height", params_.max_long_height);
  params_.max_start_height =
      this->declare_parameter("max_start_height", params_.max_start_height);
  params_.sensor_height =
      this->declare_parameter("sensor_height", params_.sensor_height);
  params_.line_search_angle =
      this->declare_parameter("line_search_angle", params_.line_search_angle);
  params_.n_threads = this->declare_parameter("n_threads", params_.n_threads);
  params_.r_min = this->declare_parameter("r_min", params_.r_min);
  params_.r_max = this->declare_parameter("r_max", params_.r_max);
  params_.max_fit_error =
      this->declare_parameter("max_fit_error", params_.max_fit_error);
  // Params that need to be squared.
  double r_min, r_max, max_fit_error;
  if (this->get_parameter("r_min", r_min)) {
    params_.r_min_square = r_min * r_min;
  }
  if (this->get_parameter("r_max", r_max)) {
    params_.r_max_square = r_max * r_max;
  }
  if (this->get_parameter("max_fit_error", max_fit_error)) {
    params_.max_error_square = max_fit_error * max_fit_error;
  }

  //打印参数
  RCLCPP_INFO(this->get_logger(), "params_.n_bins: %d", params_.n_bins);
  RCLCPP_INFO(this->get_logger(), "params_.n_segments: %d", params_.n_segments);
  RCLCPP_INFO(this->get_logger(), "params_.max_dist_to_line: %f", params_.max_dist_to_line);
  RCLCPP_INFO(this->get_logger(), "params_.max_slope: %f", params_.max_slope);
  RCLCPP_INFO(this->get_logger(), "params_.min_slope: %f", params_.min_slope);
  RCLCPP_INFO(this->get_logger(), "params_.long_threshold: %f", params_.long_threshold);
  RCLCPP_INFO(this->get_logger(), "params_.max_long_height: %f", params_.max_long_height);
  RCLCPP_INFO(this->get_logger(), "params_.max_start_height: %f", params_.max_start_height);
  RCLCPP_INFO(this->get_logger(), "params_.sensor_height: %f", params_.sensor_height);
  RCLCPP_INFO(this->get_logger(), "params_.line_search_angle: %f", params_.line_search_angle);
  RCLCPP_INFO(this->get_logger(), "params_.n_threads: %d", params_.n_threads);
  RCLCPP_INFO(this->get_logger(), "params_.r_min: %f", params_.r_min);
  RCLCPP_INFO(this->get_logger(), "params_.r_max: %f", params_.r_max);
  RCLCPP_INFO(this->get_logger(), "params_.max_fit_error: %f", params_.max_fit_error);
  RCLCPP_INFO(this->get_logger(), "voxel_leaf_size: %f", voxel_leaf_size_);
  RCLCPP_INFO(this->get_logger(), "max_distance: %f", max_distance_);



  segmenter_ = std::make_shared<GroundSegmentation>(params_);

  std::string ground_topic, obstacle_topic, input_topic;
  ground_topic = this->declare_parameter("ground_output_topic", "ground_cloud");
  obstacle_topic =
      this->declare_parameter("obstacle_output_topic", "obstacle_cloud");
  input_topic = this->declare_parameter("input_topic", "input_cloud");
  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic, rclcpp::SensorDataQoS(),
      std::bind(&SegmentationNode::scanCallback, this, std::placeholders::_1));
  ground_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      ground_topic, rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
  obstacle_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      obstacle_topic, rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  RCLCPP_INFO(this->get_logger(), "Segmentation node initialized");
  RCLCPP_INFO(this->get_logger(), "params_.r_min_square: %f",
              params_.r_min_square);
}

void SegmentationNode::scanCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  // 添加点云大小检查
  if (msg->width * msg->height == 0) {
    RCLCPP_WARN(this->get_logger(), "Received empty point cloud, skipping processing");
    return;
  }
  
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::fromROSMsg(*msg, cloud);
  
  // 添加点云转换后的检查
  if (cloud.empty()) {
    RCLCPP_WARN(this->get_logger(), "Converted point cloud is empty, skipping processing");
    return;
  }
  auto start_time0 = this->get_clock()->now();
  // 距离过滤处理 - 过滤掉超过最大距离的点
  pcl::PointCloud<pcl::PointXYZ> cloud_distance_filtered;
  if (max_distance_ > 0.0) {
    // 预分配内存，避免频繁push_back
    cloud_distance_filtered.reserve(cloud.size());
    
    // 计算最大距离的平方，避免在循环中计算平方根
    double max_distance_square = max_distance_ * max_distance_;
    
    for (const auto& point : cloud) {
      // 使用平方距离比较，避免计算平方根
      double distance_square = point.x * point.x + point.y * point.y + point.z * point.z;
      if (distance_square <= max_distance_square) {
          cloud_distance_filtered.push_back(point);
      }
    }
  } else {
    cloud_distance_filtered = cloud;  // 如果最大距离为0，则不进行距离过滤
  }

  // 体素滤波处理
  pcl::PointCloud<pcl::PointXYZ> cloud_filtered;
  if (voxel_leaf_size_ > 0.0) {
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(cloud_distance_filtered.makeShared());
    voxel_filter.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
    voxel_filter.filter(cloud_filtered);
  } else {
    cloud_filtered = cloud_distance_filtered;  // 如果叶子大小为0，则不进行滤波
    RCLCPP_INFO(this->get_logger(), "Voxel filtering disabled (leaf_size = 0)");
  }
  
  pcl::PointCloud<pcl::PointXYZ> cloud_transformed;
  
  std::vector<int> labels;

  bool is_original_pc = true;

  // //将点云转到base_link
  pcl::PointCloud<pcl::PointXYZ> cloud_base_link;
  std::string base_link_frame = "base_link";  
  
  try {
    geometry_msgs::msg::TransformStamped tf_base_link;
    tf_base_link = tf_buffer_->lookupTransform(
    base_link_frame, 
    msg->header.frame_id, 
    tf2::TimePointZero,
    tf2::durationFromSec(0.1));
    
    Eigen::Affine3d tf_base;
    tf_base.translation() = Eigen::Vector3d(
        tf_base_link.transform.translation.x,
        tf_base_link.transform.translation.y, 
        tf_base_link.transform.translation.z);
    tf_base.linear() = Eigen::Quaterniond(
        tf_base_link.transform.rotation.w,
        tf_base_link.transform.rotation.x,
        tf_base_link.transform.rotation.y,
        tf_base_link.transform.rotation.z).toRotationMatrix();
    
    pcl::transformPointCloud(cloud_filtered, cloud_base_link, tf_base);
    
    RCLCPP_INFO(this->get_logger(), "Transformed point cloud to base_link frame");
  } catch (tf2::TransformException &ex) {
    RCLCPP_WARN(this->get_logger(),
                "Failed to transform point cloud to base_link frame: %s",
                ex.what());
    // 如果转换失败，使用原始点云
    cloud_base_link = cloud_filtered;
  }

  // Trick to avoid PC copy if we do not transform.
  const pcl::PointCloud<pcl::PointXYZ> &cloud_proc =
      is_original_pc ? cloud_base_link : cloud_transformed;

  auto start_time1 = this->get_clock()->now();
  RCLCPP_INFO(this->get_logger(), "voxel filtering: %zu points (reduction: %.1f%%)", 
                cloud_proc.size(), 
                (1.0 - static_cast<double>(cloud_proc.size()) / cloud_distance_filtered.size()) * 100.0);
  segmenter_->segment(cloud_proc, &labels);
  auto end_time = this->get_clock()->now();
  auto duration0 = start_time1 - start_time0;
  auto duration = end_time - start_time1;
  
  RCLCPP_INFO(this->get_logger(), "Segmentation completed in %.2f ms, voxel filtering: %.2f ms)", 
              duration.seconds() * 1000.0, duration0.seconds() * 1000.0);
  
  pcl::PointCloud<pcl::PointXYZ> ground_cloud, obstacle_cloud;
  for (size_t i = 0; i < cloud_proc.size(); ++i) {
    if (labels[i] == 1)
      ground_cloud.push_back(cloud_proc[i]);
    else
      obstacle_cloud.push_back(cloud_proc[i]);
  }
  
  RCLCPP_INFO(this->get_logger(), "Ground points: %zu, Obstacle points: %zu", 
              ground_cloud.size(), obstacle_cloud.size());
              
  auto ground_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
  auto obstacle_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
  pcl::toROSMsg(ground_cloud, *ground_msg);
  pcl::toROSMsg(obstacle_cloud, *obstacle_msg);
  ground_msg->header = msg->header;
  obstacle_msg->header = msg->header;

  ground_msg->header.frame_id = base_link_frame;
  obstacle_msg->header.frame_id = base_link_frame;

  ground_pub_->publish(*ground_msg);
  obstacle_pub_->publish(*obstacle_msg);
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto node = std::make_shared<SegmentationNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}