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
#include <perception_3d/no_entry_layer.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <algorithm>
#include <cstdlib>
#include <limits>

namespace
{
struct Point2D
{
  double x;
  double y;
};

double cross2d(const Point2D & o, const Point2D & a, const Point2D & b)
{
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

std::vector<Point2D> convexHull2D(std::vector<Point2D> pts)
{
  if (pts.size() <= 1) {
    return pts;
  }

  std::sort(pts.begin(), pts.end(), [](const Point2D & a, const Point2D & b) {
    if (a.x == b.x) {
      return a.y < b.y;
    }
    return a.x < b.x;
  });

  std::vector<Point2D> hull;
  hull.reserve(pts.size() * 2);

  for (const auto & p : pts) {
    while (hull.size() >= 2 && cross2d(hull[hull.size() - 2], hull[hull.size() - 1], p) <= 0.0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  const size_t lower_size = hull.size();
  for (int i = static_cast<int>(pts.size()) - 2; i >= 0; --i) {
    const auto & p = pts[static_cast<size_t>(i)];
    while (hull.size() > lower_size && cross2d(hull[hull.size() - 2], hull[hull.size() - 1], p) <= 0.0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  if (!hull.empty()) {
    hull.pop_back();
  }
  return hull;
}

std::string resolveNoEntryZoneYamlPath(
  const rclcpp::Logger & logger,
  const std::string & configured_path)
{
  namespace fs = std::filesystem;
  std::error_code ec;

  const std::string candidate = configured_path.empty() ? "config/no_entry_layer.yaml" : configured_path;
  fs::path input_path(candidate);

  if (input_path.is_absolute() && fs::exists(input_path, ec) && !ec) {
    return input_path.string();
  }
  ec.clear();

  if (fs::exists(input_path, ec) && !ec) {
    return fs::absolute(input_path).string();
  }
  ec.clear();

  try {
    const fs::path pkg_share = ament_index_cpp::get_package_share_directory("perception_3d");

    const fs::path pkg_relative = pkg_share / input_path;
    if (fs::exists(pkg_relative, ec) && !ec) {
      return pkg_relative.string();
    }
    ec.clear();

    if (input_path.parent_path().empty()) {
      const fs::path config_relative = pkg_share / "config" / input_path;
      if (fs::exists(config_relative, ec) && !ec) {
        return config_relative.string();
      }
    }
  } catch (const std::exception & e) {
    RCLCPP_WARN(logger, "Failed to query package share directory for perception_3d: %s", e.what());
  }

  RCLCPP_WARN(logger, "Cannot resolve no_entry_zone_pcd_file_dir: %s", candidate.c_str());
  return candidate;
}

std::string resolveZonePcdPath(const std::string & yaml_path, const std::string & configured_pcd)
{
  namespace fs = std::filesystem;
  std::error_code ec;

  if (configured_pcd.empty()) {
    return configured_pcd;
  }

  // Expand "~".
  std::string expanded = configured_pcd;
  if (!expanded.empty() && expanded[0] == '~') {
    const char * home = std::getenv("HOME");
    if (home != nullptr) {
      expanded = std::string(home) + expanded.substr(1);
    }
  }

  fs::path pcd_path(expanded);
  if (pcd_path.is_absolute() && fs::exists(pcd_path, ec) && !ec) {
    return pcd_path.string();
  }
  ec.clear();

  const fs::path yaml_dir = fs::path(yaml_path).parent_path();
  if (!yaml_dir.empty()) {
    const fs::path beside_yaml = yaml_dir / pcd_path.filename();
    if (fs::exists(beside_yaml, ec) && !ec) {
      return beside_yaml.string();
    }
    ec.clear();

    if (!pcd_path.is_absolute()) {
      const fs::path relative_to_yaml = yaml_dir / pcd_path;
      if (fs::exists(relative_to_yaml, ec) && !ec) {
        return relative_to_yaml.string();
      }
    }
  }

  return expanded;
}
}  // namespace

PLUGINLIB_EXPORT_CLASS(perception_3d::NoEntryLayer, perception_3d::Sensor)

namespace perception_3d
{

NoEntryLayer::NoEntryLayer(){

}

NoEntryLayer::~NoEntryLayer(){

}

void NoEntryLayer::onInitialize()
{ 
  
  current_lethal_.reset(new pcl::PointCloud<pcl::PointXYZI>);

  rclcpp::QoS map_qos(10);  // initialize to default
  map_qos.transient_local();
  map_qos.reliable();
  map_qos.keep_last(1);
  zone_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>("no_entry_zones", map_qos);
  no_entry_pc_zone_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("no_entry_pc_zones", map_qos);
  obstacle_pc_zone_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("no_entry_obstacle_pc", map_qos);

  cbs_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = cbs_group_;
  
  /*
  pcl_map_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
    "mapcloud", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable(), 
    std::bind(&NoEntryLayer::cbMap, this, std::placeholders::_1), sub_options);

  pcl_ground_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
    "mapground", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable(), 
    std::bind(&NoEntryLayer::cbGround, this, std::placeholders::_1), sub_options);
  */

  node_->declare_parameter(name_ + "." + "no_entry_zone_pcd_file_dir", rclcpp::ParameterValue(""));
  node_->get_parameter(name_ + "." + "no_entry_zone_pcd_file_dir", no_entry_zone_pcd_file_dir_);
  no_entry_zone_pcd_file_dir_ = resolveNoEntryZoneYamlPath(
    node_->get_logger().get_child(name_), no_entry_zone_pcd_file_dir_);
  RCLCPP_INFO(node_->get_logger().get_child(name_), "no_entry_zone_pcd_file_dir: %s" , no_entry_zone_pcd_file_dir_.c_str());
  
  parseYAML(no_entry_zone_pcd_file_dir_);

  node_->declare_parameter(name_ + "." + "inflation_distance", rclcpp::ParameterValue(1.0));
  node_->get_parameter(name_ + "." + "inflation_distance", inflation_distance_);
  RCLCPP_INFO(node_->get_logger().get_child(name_), "inflation_distance: %.2f" , inflation_distance_);

  current_zone_ = "";
  is_zone_initialized_ = false;
}

void NoEntryLayer::parseYAML(std::string path){

  //@Start zone yaml stuff
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  rcl_params_t * params_hdl = rcl_yaml_node_struct_init(allocator);

  char* cpath = const_cast<char*>(path.c_str());

  if(rcl_parse_yaml_file(cpath, params_hdl)){
    RCLCPP_INFO(node_->get_logger().get_child(name_), "Found: %s", path.c_str());
  }
  
  rclcpp::ParameterMap PM = rclcpp::parameter_map_from(params_hdl);
  if(PM.find("/no_entry_layer") == PM.end()){
    RCLCPP_INFO(node_->get_logger().get_child(name_), "/no_entry_layer not found, disable the layer.");
    return;
  }
  else{
  }

  auto zones = PM.at("/no_entry_layer");

  bool zone_read = false;
  bool enable_read = false;
  std::string zone_name = "";
  std::string zone_dir = "";
  bool is_enabled = false;

  //@ Following loop:
  // zone1.pcd
  // zone1.speed
  // zone2.pcd
  // zone2.speed
  for(auto i=zones.begin();i!=zones.end();i++){
    
    if(strstr((*i).get_name().c_str(), std::string(".pcd").c_str())){
      std::string unit_zone_string = (*i).value_to_string();
      zone_read = true;
      //split by /
      std::stringstream unit_zone_string_stream(unit_zone_string);
      std::vector<std::string> unit_zone_string_list;
      std::string unit_split;
      while(std::getline(unit_zone_string_stream, unit_split, '/'))
      {
        unit_zone_string_list.push_back(unit_split);
      }
      zone_name = (*i).get_name(); //zone1.pcd
      zone_name += " : " + unit_zone_string_list.back();
      zone_dir = resolveZonePcdPath(path, unit_zone_string);
    }
    else if(strstr((*i).get_name().c_str(), std::string(".is_enabled").c_str())){
      is_enabled = (*i).as_bool();
      enable_read = true;
    }

    if(zone_read && enable_read){
      RCLCPP_INFO(
        node_->get_logger().get_child(name_),
        "Read no entry zone: %s with enable flag: %d (pcd=%s)",
        zone_name.c_str(), is_enabled, zone_dir.c_str());
      if (zone_dir.empty()) {
        RCLCPP_WARN(node_->get_logger().get_child(name_), "Skip zone with empty pcd path: %s", zone_name.c_str());
        zone_read = false;
        enable_read = false;
        zone_name = "";
        zone_dir = "";
        is_enabled = false;
        continue;
      }
      perception_3d::NoEntryZone a_zone(name_, node_->get_node_logging_interface(), zone_name, zone_dir, is_enabled);
      getBoundingBox(a_zone);
      no_entry_zones_.insert({zone_name, a_zone});
      zone_marker_array_.markers.push_back(a_zone.bb_marker_);
      zone_intensity_pc_ += (*a_zone.zone_intensity_pc_);
      getObstaclePointCloud(a_zone);
      obstacle_pc_ += (*a_zone.obstacle_pc_);

      //reset
      zone_read = false;
      enable_read = false;
      zone_name = "";
      zone_dir = "";
    }
    
  }
  zone_pub_->publish(zone_marker_array_);

  sensor_msgs::msg::PointCloud2 ros_pc2_msg;
  pcl::toROSMsg(zone_intensity_pc_, ros_pc2_msg);
  ros_pc2_msg.header.frame_id = gbl_utils_->getGblFrame();     //gbl_utils_->getRobotFrame();
  no_entry_pc_zone_pub_->publish(ros_pc2_msg);

  sensor_msgs::msg::PointCloud2 obs_pc2_msg;
  pcl::toROSMsg(obstacle_pc_, obs_pc2_msg);
  obs_pc2_msg.header.frame_id = gbl_utils_->getGblFrame();
  obstacle_pc_zone_pub_->publish(obs_pc2_msg);
}

// void NoEntryLayer::getBoundingBox(perception_3d::NoEntryZone& a_zone){
  
//   // Generate an oriented bounding box around the selected points in RVIZ
//   // Compute principal direction
//   Eigen::Vector4f centroid;
//   pcl::compute3DCentroid(*a_zone.zone_pc_, centroid);
//   Eigen::Matrix3f covariance;
//   pcl::computeCovarianceMatrixNormalized(*a_zone.zone_pc_, centroid, covariance);
//   Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
//   Eigen::Matrix3f eigDx = eigen_solver.eigenvectors();
//   eigDx.col(2) = eigDx.col(0).cross(eigDx.col(1));

//   // Move the points to the reference frame
//   Eigen::Matrix4f p2w(Eigen::Matrix4f::Identity());
//   p2w.block<3,3>(0,0) = eigDx.transpose();
//   p2w.block<3,1>(0,3) = -1.f * (p2w.block<3,3>(0,0) * centroid.head<3>());
//   pcl::PointCloud<pcl::PointXYZ> cPoints;
//   pcl::transformPointCloud(*a_zone.zone_pc_, cPoints, p2w);

//   pcl::PointXYZ min_pt, max_pt;
//   pcl::getMinMax3D(cPoints, min_pt, max_pt);
//   const Eigen::Vector3f mean_diag = 0.5f*(max_pt.getVector3fMap() + min_pt.getVector3fMap());
//   // Final transform and bounding box size
//   const Eigen::Quaternionf qfinal(eigDx);
//   const Eigen::Vector3f tfinal = eigDx*mean_diag + centroid.head<3>();
//   double bb_size_x = max_pt.x - min_pt.x;
//   double bb_size_y = max_pt.y - min_pt.y;
//   double bb_size_z = max_pt.z - min_pt.z;

//   // Set the frame ID and timestamp.  See the TF tutorials for information on these.
//   a_zone.bb_marker_.header.frame_id = gbl_utils_->getGblFrame();
//   a_zone.bb_marker_.ns = a_zone.zone_name_;
//   a_zone.bb_marker_.id = 0;
//   a_zone.bb_marker_.type = visualization_msgs::msg::Marker::CUBE;
//   a_zone.bb_marker_.action = visualization_msgs::msg::Marker::ADD;
//   a_zone.bb_marker_.pose.position.x = tfinal.x();
//   a_zone.bb_marker_.pose.position.y = tfinal.y();
//   a_zone.bb_marker_.pose.position.z = tfinal.z();
//   a_zone.bb_marker_.pose.orientation.x = qfinal.x();
//   a_zone.bb_marker_.pose.orientation.y = qfinal.y();
//   a_zone.bb_marker_.pose.orientation.z = qfinal.z();
//   a_zone.bb_marker_.pose.orientation.w = qfinal.w();
//   a_zone.bb_marker_.scale.x = bb_size_x;
//   a_zone.bb_marker_.scale.y = bb_size_y;
//   a_zone.bb_marker_.scale.z = bb_size_z;
//   a_zone.bb_marker_.color.r = 1.0f;
//   a_zone.bb_marker_.color.g = 1.0f;
//   a_zone.bb_marker_.color.b = 0.0f;
//   a_zone.bb_marker_.color.a = 0.5;

// }

void NoEntryLayer::getBoundingBox(perception_3d::NoEntryZone& a_zone){
  if (!a_zone.zone_pc_ || a_zone.zone_pc_->empty()) {
    return;
  }

  std::vector<Point2D> pts_xy;
  pts_xy.reserve(a_zone.zone_pc_->size());

  double min_z = std::numeric_limits<double>::max();
  double max_z = std::numeric_limits<double>::lowest();

  for (const auto & p : a_zone.zone_pc_->points) {
    pts_xy.push_back({static_cast<double>(p.x), static_cast<double>(p.y)});
    min_z = std::min(min_z, static_cast<double>(p.z));
    max_z = std::max(max_z, static_cast<double>(p.z));
  }

  const auto hull = convexHull2D(std::move(pts_xy));
  if (hull.empty()) {
    return;
  }

  double best_area = std::numeric_limits<double>::max();
  double best_umin = 0.0, best_umax = 0.0, best_vmin = 0.0, best_vmax = 0.0;
  Eigen::Vector2d best_u(1.0, 0.0);
  Eigen::Vector2d best_v(0.0, 1.0);

  auto evaluate_axis = [&](const Eigen::Vector2d & u, const Eigen::Vector2d & v) {
    double umin = std::numeric_limits<double>::max();
    double umax = std::numeric_limits<double>::lowest();
    double vmin = std::numeric_limits<double>::max();
    double vmax = std::numeric_limits<double>::lowest();

    for (const auto & hp : hull) {
      const Eigen::Vector2d p(hp.x, hp.y);
      const double pu = p.dot(u);
      const double pv = p.dot(v);
      umin = std::min(umin, pu);
      umax = std::max(umax, pu);
      vmin = std::min(vmin, pv);
      vmax = std::max(vmax, pv);
    }

    const double area = (umax - umin) * (vmax - vmin);
    if (area < best_area) {
      best_area = area;
      best_umin = umin;
      best_umax = umax;
      best_vmin = vmin;
      best_vmax = vmax;
      best_u = u;
      best_v = v;
    }
  };

  if (hull.size() == 1) {
    evaluate_axis(Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(0.0, 1.0));
  } else {
    for (size_t i = 0; i < hull.size(); ++i) {
      const auto & p0 = hull[i];
      const auto & p1 = hull[(i + 1) % hull.size()];
      Eigen::Vector2d edge(p1.x - p0.x, p1.y - p0.y);
      const double len = edge.norm();
      if (len < 1e-9) {
        continue;
      }
      edge /= len;
      Eigen::Vector2d perp(-edge.y(), edge.x());
      evaluate_axis(edge, perp);
    }
  }

  const Eigen::Vector2d center_xy = best_u * (0.5 * (best_umin + best_umax))
                                   + best_v * (0.5 * (best_vmin + best_vmax));
  const double bb_size_x = std::max(1e-3, best_umax - best_umin);
  const double bb_size_y = std::max(1e-3, best_vmax - best_vmin);
  const double bb_size_z = std::max(1e-3, max_z - min_z);
  const double center_z = 0.5 * (min_z + max_z);
  const double yaw = std::atan2(best_u.y(), best_u.x());

  // Set the frame ID and timestamp.  See the TF tutorials for information on these.
  a_zone.bb_marker_.header.frame_id = gbl_utils_->getGblFrame();
  a_zone.bb_marker_.ns = a_zone.zone_name_;
  a_zone.bb_marker_.id = 0;
  a_zone.bb_marker_.type = visualization_msgs::msg::Marker::CUBE;
  a_zone.bb_marker_.action = visualization_msgs::msg::Marker::ADD;
  a_zone.bb_marker_.pose.position.x = center_xy.x();
  a_zone.bb_marker_.pose.position.y = center_xy.y();
  a_zone.bb_marker_.pose.position.z = center_z;
  a_zone.bb_marker_.pose.orientation.x = 0.0;
  a_zone.bb_marker_.pose.orientation.y = 0.0;
  a_zone.bb_marker_.pose.orientation.z = std::sin(0.5 * yaw);
  a_zone.bb_marker_.pose.orientation.w = std::cos(0.5 * yaw);
  a_zone.bb_marker_.scale.x = bb_size_x;
  a_zone.bb_marker_.scale.y = bb_size_y;
  a_zone.bb_marker_.scale.z = bb_size_z;
  a_zone.bb_marker_.color.r = 1.0f;
  a_zone.bb_marker_.color.g = 1.0f;
  a_zone.bb_marker_.color.b = 0.0f;
  a_zone.bb_marker_.color.a = 0.5;

}

void NoEntryLayer::getObstaclePointCloud(perception_3d::NoEntryZone& a_zone){
    if(!a_zone.obstacle_pc_) {
        a_zone.obstacle_pc_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    }
    a_zone.obstacle_pc_->clear();

    Eigen::Vector3f center(a_zone.bb_marker_.pose.position.x,
                           a_zone.bb_marker_.pose.position.y,
                           a_zone.bb_marker_.pose.position.z);
    Eigen::Quaternionf q(
        a_zone.bb_marker_.pose.orientation.w,
        a_zone.bb_marker_.pose.orientation.x,
        a_zone.bb_marker_.pose.orientation.y,
        a_zone.bb_marker_.pose.orientation.z
    );
    Eigen::Matrix3f rot = q.toRotationMatrix();

    double sx = a_zone.bb_marker_.scale.x;
    double sy = a_zone.bb_marker_.scale.y;
    double sz = a_zone.bb_marker_.scale.z;

    // ************************************************************************************
    // 如果 Marker 的主方向（即旋转矩阵 rot 的某个轴）与地图坐标系的 Z 方向接近，就在对应的尺寸上延伸。
    // Eigen::Vector3f map_z_axis(0,0,1);
    // Eigen::Vector3f marker_x = rot.col(0); // Marker的X方向
    // Eigen::Vector3f marker_y = rot.col(1); // Marker的Y方向
    // Eigen::Vector3f marker_z = rot.col(2); // Marker的Z方向

    // double dot_x = fabs(marker_x.dot(map_z_axis));
    // double dot_y = fabs(marker_y.dot(map_z_axis));
    // double dot_z = fabs(marker_z.dot(map_z_axis));

    // if(dot_x > dot_y && dot_x > dot_z){
    //     sx += 2.0; // 上下各延伸1m
    // } else if(dot_y > dot_x && dot_y > dot_z){
    //     sy += 2.0;
    // } else {
    //     sz += 2.0;
    // }
    // ************************************************************************************

    sz += 2.0;
    double step = 0.2;
    double eps = step/2.0;

    for(double x=-sx/2; x<=sx/2+eps; x+=step){
        for(double y=-sy/2; y<=sy/2+eps; y+=step){
            for(double z=-sz/2; z<=sz/2+eps; z+=step){
                bool is_surface = (fabs(x+sx/2) < eps || fabs(x-sx/2) < eps ||
                                   fabs(y+sy/2) < eps || fabs(y-sy/2) < eps ||
                                   fabs(z+sz/2) < eps || fabs(z-sz/2) < eps);
                if(!is_surface) continue;

                Eigen::Vector3f local(x,y,z);
                Eigen::Vector3f global = rot*local + center;

                pcl::PointXYZ pt(global.x(), global.y(), global.z());
                a_zone.obstacle_pc_->points.push_back(pt);
            }
        }
    }

    a_zone.obstacle_pc_->width = a_zone.obstacle_pc_->points.size();
    a_zone.obstacle_pc_->height = 1;
    a_zone.obstacle_pc_->is_dense = true;
}


void NoEntryLayer::updateLethalPointCloud(){

  //@ current_lethal_ is updated in marking

}

void NoEntryLayer::selfMark(){
  
  std::unique_lock<std::recursive_mutex> lock(shared_data_->ground_kdtree_cb_mutex_);

  if(!shared_data_->is_static_layer_ready_){
    return;
  }

  if(shared_data_->dgraph_update_request_[name_]){
    //@ need to regenerate dynamic graph
    resetdGraph();
    shared_data_->dgraph_update_request_[name_] = false;
  }
  
  //@ use dGraph for no entry zone, because we want to turn on/off of the zones in run time
  //@ for example, turn zone A on in morning, turn zone B on in evening

  //@ Collect zone status
  std::vector<bool> zone_enabled_status_current;
  for(auto it=no_entry_zones_.begin(); it!=no_entry_zones_.end(); it++){
    if(!is_zone_initialized_){
      zone_enabled_status_.push_back((*it).second.is_enabled_);
    }
    else{
      zone_enabled_status_current.push_back((*it).second.is_enabled_);
    }
  }

  //@ Check whether we need to reinflate or not
  bool need_reinflate = false;
  if (zone_enabled_status_current==zone_enabled_status_){
    //@ the status of array is the same, we dont need to update dGraph
  }
  else{
    RCLCPP_INFO(node_->get_logger(), "No entry zone status changed, reset dGraph.");
    zone_enabled_status_ = zone_enabled_status_current;
    resetdGraph();
    need_reinflate = true;
  }
  
  if(need_reinflate){
    for(auto it=no_entry_zones_.begin(); it!=no_entry_zones_.end(); it++){
      
      if(!(*it).second.is_enabled_)
        continue;
      //@ loop all point in no entry zone and set dGraph value to 0.0, also consider inflation
      for(auto pt=(*it).second.zone_pc_->points.begin(); pt!=(*it).second.zone_pc_->points.end(); pt++){
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        pcl::PointXYZI ipt;
        ipt.x = (*pt).x; ipt.y = (*pt).y; ipt.z = (*pt).z; 
        if(shared_data_->kdtree_ground_->radiusSearch(ipt, inflation_distance_, pointIdxRadiusSearch, pointRadiusSquaredDistance)>0){
          for(int i=0;i<pointIdxRadiusSearch.size();i++){
            dGraph_.setValue(pointIdxRadiusSearch[i], sqrt(pointRadiusSquaredDistance[i]));
            current_lethal_->push_back(ipt);
          }
        }
      }
    }
  }

  is_zone_initialized_ = true;
}

void NoEntryLayer::selfClear(){

}

void NoEntryLayer::resetdGraph(){
  std::unique_lock<std::recursive_mutex> lock(shared_data_->ground_kdtree_cb_mutex_);
  RCLCPP_INFO(node_->get_logger().get_child(name_), "%s starts to reset dynamic graph.", name_.c_str());
  dGraph_.clear();
  dGraph_.initial(shared_data_->static_ground_size_, gbl_utils_->getMaxObstacleDistance());
}

double NoEntryLayer::get_dGraphValue(const unsigned int index){
  if (dGraph_.graph_.find(index) == dGraph_.graph_.end())
    return 0.0;
  return dGraph_.getValue(index);
}

bool NoEntryLayer::isCurrent(){
  
  current_ = true;

  return current_;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr NoEntryLayer::getObservation(){
  return sensor_current_observation_;
}
pcl::PointCloud<pcl::PointXYZI>::Ptr NoEntryLayer::getLethal(){
  return current_lethal_;
}
}//end of name space
