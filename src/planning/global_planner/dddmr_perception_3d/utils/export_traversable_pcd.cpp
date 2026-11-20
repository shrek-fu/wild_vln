#include <perception_3d/traversable_cloud.h>

#include <pcl/io/pcd_io.h>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv)
{
  if (argc < 4) {
    std::fprintf(stderr,
      "Usage: export_traversable_pcd <ground.pcd> <nonground.pcd> <output.pcd> "
      "[ground_leaf=0.4] [map_leaf=0.3] [imposing_radius=1.5] [xy_half=0.5] [z_min_offset=0.1] "
      "[min_obstacle_points=10]\n"
      "  a ground point is lethal when more than min_obstacle_points non-ground points lie in the box\n"
      "  +-xy_half around it, z_min_offset..1.0 m above it\n");
    return 1;
  }

  const std::string ground_path = argv[1];
  const std::string map_path = argv[2];
  const std::string out_path = argv[3];

  perception_3d::TraversableCloudOptions opt;
  if (argc > 4) {
    opt.ground_leaf_size = static_cast<float>(std::atof(argv[4]));
  }
  if (argc > 5) {
    opt.map_leaf_size = static_cast<float>(std::atof(argv[5]));
  }
  if (argc > 6) {
    opt.imposing_radius = static_cast<float>(std::atof(argv[6]));
  }
  if (argc > 7) {
    opt.xy_half = static_cast<float>(std::atof(argv[7]));
  }
  if (argc > 8) {
    opt.z_min_offset = static_cast<float>(std::atof(argv[8]));
  }
  if (argc > 9) {
    opt.min_obstacle_points = std::atoi(argv[9]);
  }

  pcl::PointCloud<pcl::PointXYZI> ground;
  pcl::PointCloud<pcl::PointXYZI> map;
  if (!perception_3d::loadCloudXYZI(ground_path, ground)) {
    std::fprintf(stderr, "Failed to load ground: %s\n", ground_path.c_str());
    return 1;
  }
  if (!perception_3d::loadCloudXYZI(map_path, map)) {
    std::fprintf(stderr, "Failed to load nonground: %s\n", map_path.c_str());
    return 1;
  }

  std::fprintf(stderr, "Loaded ground=%zu nonground=%zu\n", ground.size(), map.size());
  perception_3d::downsampleCloudXYZI(ground, opt.ground_leaf_size);
  perception_3d::downsampleCloudXYZI(map, opt.map_leaf_size);
  std::fprintf(stderr, "Downsampled ground=%zu nonground=%zu (leaf %.2f / %.2f, r=%.2f, box +-%.2f, "
               "z >= %.2f, > %d pts)\n",
               ground.size(), map.size(), opt.ground_leaf_size, opt.map_leaf_size,
               opt.imposing_radius, opt.xy_half, opt.z_min_offset, opt.min_obstacle_points);

  pcl::PointCloud<pcl::PointXYZI> out;
  if (!perception_3d::computeTraversableCloud(ground, map, out, opt)) {
    std::fprintf(stderr, "Failed to compute traversable cloud.\n");
    return 1;
  }

  if (pcl::io::savePCDFileBinary(out_path, out) != 0) {
    std::fprintf(stderr, "Failed to save %s\n", out_path.c_str());
    return 1;
  }
  std::fprintf(stderr, "Saved %s (%zu pts)\n", out_path.c_str(), out.size());
  return 0;
}
