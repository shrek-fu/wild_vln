#ifndef PERCEPTION_3D_TRAVERSABLE_CLOUD_H_
#define PERCEPTION_3D_TRAVERSABLE_CLOUD_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <string>

namespace perception_3d
{

struct TraversableCloudOptions {
  float ground_leaf_size = 0.4f;
  float map_leaf_size = 0.3f;
  float imposing_radius = 1.5f;
  float lethal_distance = 0.25f;
  float max_obstacle_distance = 9999.0f;
  float z_min_offset = 0.1f;
  float z_max_offset = 1.0f;
  float xy_half = 0.5f;
  int min_obstacle_points = 10;
};

bool loadCloudXYZI(const std::string& path, pcl::PointCloud<pcl::PointXYZI>& cloud);

void downsampleCloudXYZI(pcl::PointCloud<pcl::PointXYZI>& cloud, float leaf_size);

bool hasVerticalObstacle(
    pcl::KdTreeFLANN<pcl::PointXYZI>& kdtree_map,
    const pcl::PointCloud<pcl::PointXYZI>& pcl_map,
    const pcl::PointXYZI& ground_pt,
    const TraversableCloudOptions& opt);

bool computeTraversableCloud(
    const pcl::PointCloud<pcl::PointXYZI>& ground,
    const pcl::PointCloud<pcl::PointXYZI>& map,
    pcl::PointCloud<pcl::PointXYZI>& out,
    const TraversableCloudOptions& opt);

}

#endif
