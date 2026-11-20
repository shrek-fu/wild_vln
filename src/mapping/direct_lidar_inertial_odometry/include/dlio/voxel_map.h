#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <unordered_map>
#include <cmath>

struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

namespace std {
    template <>
    struct hash<VoxelKey> {
        size_t operator()(const VoxelKey& k) const {
            return ((hash<int>()(k.x) ^ (hash<int>()(k.y) << 1)) >> 1) ^ (hash<int>()(k.z) << 1);
        }
    };
}

template <typename PointT>
class VoxelMap {
public:
    VoxelMap(float leaf_size = 0.2f) : leaf_size_(leaf_size), inv_leaf_size_(1.0f / leaf_size) {}

    void addCloud(const typename pcl::PointCloud<PointT>::Ptr& cloud) {
        for (const auto& pt : cloud->points) {
            VoxelKey key = getVoxelKey(pt);
            if (map_.find(key) == map_.end()) {
                map_[key] = pt;
            }
        }
    }

    typename pcl::PointCloud<PointT>::Ptr getCloud() {
        typename pcl::PointCloud<PointT>::Ptr out(new pcl::PointCloud<PointT>());
        for (const auto& kv : map_) {
            out->points.push_back(kv.second);
        }
        return out;
    }

private:
    VoxelKey getVoxelKey(const PointT& pt) {
        return {
            static_cast<int>(std::floor(pt.x * inv_leaf_size_)),
            static_cast<int>(std::floor(pt.y * inv_leaf_size_)),
            static_cast<int>(std::floor(pt.z * inv_leaf_size_))
        };
    }

    float leaf_size_, inv_leaf_size_;
    std::unordered_map<VoxelKey, PointT> map_;
};