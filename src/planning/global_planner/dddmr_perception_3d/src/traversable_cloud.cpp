#include <perception_3d/traversable_cloud.h>

#include <pcl/conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <cstdio>

namespace perception_3d
{

bool loadCloudXYZI(const std::string& path, pcl::PointCloud<pcl::PointXYZI>& cloud)
{
  pcl::PCLPointCloud2 blob;
  if (pcl::io::loadPCDFile(path, blob) != 0) {
    return false;
  }
  pcl::fromPCLPointCloud2(blob, cloud);
  return !cloud.empty();
}

void downsampleCloudXYZI(pcl::PointCloud<pcl::PointXYZI>& cloud, float leaf_size)
{
  if (leaf_size <= 1e-6f || cloud.empty()) {
    return;
  }
  auto ptr = cloud.makeShared();
  pcl::VoxelGrid<pcl::PointXYZI> vg;
  vg.setInputCloud(ptr);
  vg.setLeafSize(leaf_size, leaf_size, leaf_size);
  pcl::PointCloud<pcl::PointXYZI> filtered;
  vg.filter(filtered);
  cloud.swap(filtered);
  std::vector<int> index;
  pcl::removeNaNFromPointCloud(cloud, cloud, index);
}

bool hasVerticalObstacle(
    pcl::KdTreeFLANN<pcl::PointXYZI>& kdtree_map,
    const pcl::PointCloud<pcl::PointXYZI>& pcl_map,
    const pcl::PointXYZI& ground_pt,
    const TraversableCloudOptions& opt)
{
  if (pcl_map.empty()) {
    return false;
  }
  std::vector<int> ids;
  std::vector<float> sqdist;
  if (kdtree_map.radiusSearch(ground_pt, opt.imposing_radius, ids, sqdist) <= 0) {
    return false;
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_z_axes(new pcl::PointCloud<pcl::PointXYZI>);
  pcl_z_axes->reserve(ids.size());
  for (int id : ids) {
    if (id >= 0 && static_cast<size_t>(id) < pcl_map.size()) {
      pcl_z_axes->push_back(pcl_map.points[id]);
    }
  }

  pcl::PassThrough<pcl::PointXYZI> pass;
  pass.setInputCloud(pcl_z_axes);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(ground_pt.z + opt.z_min_offset, ground_pt.z + opt.z_max_offset);
  pass.filter(*pcl_z_axes);
  pass.setInputCloud(pcl_z_axes);
  pass.setFilterFieldName("x");
  pass.setFilterLimits(ground_pt.x - opt.xy_half, ground_pt.x + opt.xy_half);
  pass.filter(*pcl_z_axes);
  pass.setInputCloud(pcl_z_axes);
  pass.setFilterFieldName("y");
  pass.setFilterLimits(ground_pt.y - opt.xy_half, ground_pt.y + opt.xy_half);
  pass.filter(*pcl_z_axes);
  return static_cast<int>(pcl_z_axes->points.size()) > opt.min_obstacle_points;
}

bool computeTraversableCloud(
    const pcl::PointCloud<pcl::PointXYZI>& ground,
    const pcl::PointCloud<pcl::PointXYZI>& map,
    pcl::PointCloud<pcl::PointXYZI>& out,
    const TraversableCloudOptions& opt)
{
  out.clear();
  if (ground.empty()) {
    return false;
  }

  pcl::KdTreeFLANN<pcl::PointXYZI> kdtree_map;
  pcl::PointCloud<pcl::PointXYZI>::Ptr map_ptr;
  if (!map.empty()) {
    map_ptr = map.makeShared();
    kdtree_map.setInputCloud(map_ptr);
  }

  out.reserve(ground.size());
  size_t lethal_cnt = 0;
  for (size_t i = 0; i < ground.size(); ++i) {
    pcl::PointXYZI pt = ground.points[i];
    const bool lethal = map_ptr &&
        hasVerticalObstacle(kdtree_map, map, pt, opt);
    pt.intensity = lethal ? opt.lethal_distance : opt.max_obstacle_distance;
    if (lethal) {
      ++lethal_cnt;
    }
    out.push_back(pt);
    if ((i + 1) % 20000 == 0) {
      std::fprintf(stderr, "export_traversable: %zu / %zu ground points\n",
                   i + 1, ground.size());
    }
  }
  out.width = out.size();
  out.height = 1;
  out.is_dense = false;
  std::fprintf(stderr, "export_traversable: done, %zu pts, %zu lethal\n",
               out.size(), lethal_cnt);
  return true;
}

}
