/*
 * Copyright (c) 2016-2020, the mcl_3dl authors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */


#include "rclcpp/rclcpp.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/io/pcd_io.h>

#include <pcl_conversions/pcl_conversions.h>

#include <pcl/common/transforms.h>
#include <pcl/common/io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
/*voxel*/
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>

static bool loadAndPublishCloud(
    rclcpp::Logger logger,
    const std::string& file,
    const std::string& kind,
    const std::string& global_frame,
    double rot_x, double rot_y, double rot_z,
    double tx, double ty, double tz,
    double down_sample,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub)
{
  if (file.empty()) {
    RCLCPP_INFO(logger, "Skip %s pcd: path is empty.", kind.c_str());
    return true;
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  if (pcl::io::loadPCDFile<pcl::PointXYZI>(file, *cloud) == -1 || cloud->empty()) {
    pcl::PointCloud<pcl::PointXYZ> xyz;
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(file, xyz) == -1) {
      RCLCPP_ERROR(logger, "Couldn't read %s pcd: %s", kind.c_str(), file.c_str());
      return false;
    }
    pcl::copyPointCloud(xyz, *cloud);
  }

  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  transform.translation() << static_cast<float>(tx), static_cast<float>(ty), static_cast<float>(tz);
  if (fabs(rot_x) > 0.01)
    transform.rotate(Eigen::AngleAxisf(static_cast<float>(rot_x), Eigen::Vector3f::UnitX()));
  if (fabs(rot_y) > 0.01)
    transform.rotate(Eigen::AngleAxisf(static_cast<float>(rot_y), Eigen::Vector3f::UnitY()));
  if (fabs(rot_z) > 0.01)
    transform.rotate(Eigen::AngleAxisf(static_cast<float>(rot_z), Eigen::Vector3f::UnitZ()));
  pcl::transformPointCloud(*cloud, *cloud, transform);
  RCLCPP_INFO(logger, "%s pointcloud size: %lu", kind.c_str(), cloud->points.size());

  if (down_sample > 1e-6) {
    pcl::VoxelGrid<pcl::PointXYZI> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(static_cast<float>(down_sample), static_cast<float>(down_sample),
                    static_cast<float>(down_sample));
    sor.filter(*cloud);
    cloud->is_dense = false;
    std::vector<int> ind;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, ind);
    RCLCPP_INFO(logger, "%s pointcloud size after down size: %lu", kind.c_str(), cloud->points.size());
  } else {
    RCLCPP_INFO(logger, "%s downsample skipped (leaf<=0), preserving intensity.", kind.c_str());
  }

  cloud->header.frame_id = global_frame;
  sensor_msgs::msg::PointCloud2 ros_msg;
  pcl::toROSMsg(*cloud, ros_msg);
  ros_msg.header.frame_id = global_frame;
  pub->publish(ros_msg);
  return true;
}

class PCLPublisherNode : public rclcpp::Node {

  public:
    PCLPublisherNode(std::string name);

  private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground_;

    double map_rotate_around_x_, map_rotate_around_y_, map_rotate_around_z_;
    double map_translate_x_, map_translate_y_, map_translate_z_;
    double ground_rotate_around_x_, ground_rotate_around_y_, ground_rotate_around_z_;
    double ground_translate_x_, ground_translate_y_, ground_translate_z_;
    
    double map_down_sample_, ground_down_sample_;

    std::string package_share_directory_, map_file_, ground_file_, global_frame_;
};

PCLPublisherNode::PCLPublisherNode(std::string name): Node(name){

  // package_share_directory_ = ament_index_cpp::get_package_share_directory("mcl_3dl");

  this->declare_parameter("map_file", rclcpp::ParameterValue(""));
  rclcpp::Parameter map_file = this->get_parameter("map_file");
  map_file_ = map_file.as_string();
  RCLCPP_INFO(this->get_logger(), "map_file: %s", map_file_.c_str());

  this->declare_parameter("ground_file", rclcpp::ParameterValue(""));
  rclcpp::Parameter ground_file = this->get_parameter("ground_file");
  ground_file_ = ground_file.as_string();
  RCLCPP_INFO(this->get_logger(), "ground_file: %s", ground_file_.c_str());

  this->declare_parameter("global_frame", rclcpp::ParameterValue(""));
  rclcpp::Parameter global_frame = this->get_parameter("global_frame");
  global_frame_ = global_frame.as_string();
  RCLCPP_INFO(this->get_logger(), "global_frame: %s", global_frame_.c_str());

  this->declare_parameter("map_rotate_around_x", rclcpp::ParameterValue(0.0));
  rclcpp::Parameter map_rotate_around_x = this->get_parameter("map_rotate_around_x");
  map_rotate_around_x_ = map_rotate_around_x.as_double();
  RCLCPP_INFO(this->get_logger(), "map_rotate_around_x: %.2f", map_rotate_around_x_);

  this->declare_parameter("map_rotate_around_y", rclcpp::ParameterValue(0.0));
  rclcpp::Parameter map_rotate_around_y = this->get_parameter("map_rotate_around_y");
  map_rotate_around_y_ = map_rotate_around_y.as_double();
  RCLCPP_INFO(this->get_logger(), "map_rotate_around_y: %.2f", map_rotate_around_y_);

  this->declare_parameter("map_rotate_around_z", rclcpp::ParameterValue(0.0));
  rclcpp::Parameter map_rotate_around_z = this->get_parameter("map_rotate_around_z");
  map_rotate_around_z_ = map_rotate_around_z.as_double();
  RCLCPP_INFO(this->get_logger(), "map_rotate_around_z: %.2f", map_rotate_around_z_);

  this->declare_parameter("map_translate_x", rclcpp::ParameterValue(0.0));
  rclcpp::Parameter map_translate_x = this->get_parameter("map_translate_x");
  map_translate_x_ = map_translate_x.as_double();
  RCLCPP_INFO(this->get_logger(), "map_translate_x: %.2f", map_translate_x_);

  this->declare_parameter("map_translate_y", rclcpp::ParameterValue(0.0));
  rclcpp::Parameter map_translate_y = this->get_parameter("map_translate_y");
  map_translate_y_ = map_translate_y.as_double();
  RCLCPP_INFO(this->get_logger(), "map_translate_y: %.2f", map_translate_y_);

  this->declare_parameter("map_translate_z", rclcpp::ParameterValue(0.0));
  rclcpp::Parameter map_translate_z = this->get_parameter("map_translate_z");
  map_translate_z_ = map_translate_z.as_double();
  RCLCPP_INFO(this->get_logger(), "map_translate_z: %.2f", map_translate_z_);

  this->declare_parameter("ground_rotate_around_x", rclcpp::ParameterValue(0.0));
  rclcpp::Parameter ground_rotate_around_x = this->get_parameter("ground_rotate_around_x");
  ground_rotate_around_x_ = ground_rotate_around_x.as_double();
  RCLCPP_INFO(this->get_logger(), "ground_rotate_around_x: %.2f", ground_rotate_around_x_);

  this->declare_parameter("ground_rotate_around_y", rclcpp::ParameterValue(0.0));
  rclcpp::Parameter ground_rotate_around_y = this->get_parameter("ground_rotate_around_y");
  ground_rotate_around_y_ = ground_rotate_around_y.as_double();
  RCLCPP_INFO(this->get_logger(), "ground_rotate_around_y: %.2f", ground_rotate_around_y_);

  this->declare_parameter("ground_rotate_around_z", rclcpp::ParameterValue(0.0));
  rclcpp::Parameter ground_rotate_around_z = this->get_parameter("ground_rotate_around_z");
  ground_rotate_around_z_ = ground_rotate_around_z.as_double();
  RCLCPP_INFO(this->get_logger(), "ground_rotate_around_z: %.2f", ground_rotate_around_z_);

  this->declare_parameter("ground_translate_x", rclcpp::ParameterValue(0.0));
  rclcpp::Parameter ground_translate_x = this->get_parameter("ground_translate_x");
  ground_translate_x_ = ground_translate_x.as_double();
  RCLCPP_INFO(this->get_logger(), "ground_translate_x: %.2f", ground_translate_x_);

  this->declare_parameter("ground_translate_y", rclcpp::ParameterValue(0.0));
  rclcpp::Parameter ground_translate_y = this->get_parameter("ground_translate_y");
  ground_translate_y_ = ground_translate_y.as_double();
  RCLCPP_INFO(this->get_logger(), "ground_translate_y: %.2f", ground_translate_y_);

  this->declare_parameter("ground_translate_z", rclcpp::ParameterValue(0.0));
  rclcpp::Parameter ground_translate_z = this->get_parameter("ground_translate_z");
  ground_translate_z_ = ground_translate_z.as_double();
  RCLCPP_INFO(this->get_logger(), "ground_translate_z: %.2f", ground_translate_z_);

  this->declare_parameter("map_down_sample", rclcpp::ParameterValue(0.0));
  rclcpp::Parameter map_down_sample = this->get_parameter("map_down_sample");
  map_down_sample_ = map_down_sample.as_double();
  RCLCPP_INFO(this->get_logger(), "map_down_sample: %.2f", map_down_sample_);

  this->declare_parameter("ground_down_sample", rclcpp::ParameterValue(0.0));
  rclcpp::Parameter ground_down_sample = this->get_parameter("ground_down_sample");
  ground_down_sample_ = ground_down_sample.as_double();
  RCLCPP_INFO(this->get_logger(), "ground_down_sample: %.2f", ground_down_sample_);
  
  //@ Latched topic, Create a publisher using the QoS settings to emulate a ROS1 latched topic
  pub_map_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("mapcloud",
              rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());  

  pub_ground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("mapground",
                rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());

  loadAndPublishCloud(
      this->get_logger(), map_file_, "Map", global_frame_,
      map_rotate_around_x_, map_rotate_around_y_, map_rotate_around_z_,
      map_translate_x_, map_translate_y_, map_translate_z_,
      map_down_sample_, pub_map_);
  loadAndPublishCloud(
      this->get_logger(), ground_file_, "Ground", global_frame_,
      ground_rotate_around_x_, ground_rotate_around_y_, ground_rotate_around_z_,
      ground_translate_x_, ground_translate_y_, ground_translate_z_,
      ground_down_sample_, pub_ground_);
}

int main(int argc, char** argv) {

  rclcpp::init(argc, argv);
  auto PP = std::make_shared<PCLPublisherNode>("pcl_publisher");
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(PP);
  executor.spin();
  rclcpp::shutdown();  
  return 0;
}
