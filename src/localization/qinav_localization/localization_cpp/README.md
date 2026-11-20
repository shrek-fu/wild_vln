# Localization C++ Package

C++ 移植版本的定位框架，基于 [FAST_LIO_LOCALIZATION](https://github.com/HViktorTsoi/FAST_LIO_LOCALIZATION)，在已建地图中进行重定位。

与 Python 版本算法逻辑完全一致，使用 PCL 替代 Open3D 进行 ICP 配准，性能更高。

## 依赖

### ROS 2 系统依赖
- `rclcpp` - ROS 2 C++ 客户端库
- `geometry_msgs` - 几何消息类型
- `nav_msgs` - 导航消息类型
- `sensor_msgs` - 传感器消息类型
- `std_msgs` - 标准消息类型
- `tf2` / `tf2_ros` / `tf2_eigen` / `tf2_geometry_msgs` / `tf2_sensor_msgs` - 坐标变换库
- `pcl_ros` / `pcl_conversions` - PCL ROS 集成
- `Eigen3` - 矩阵运算库

### 系统依赖
- `PCL` (≥ 1.10) - 点云处理库，包含 `common` / `io` / `filters` / `registration` / `kdtree` 模块

## 使用方法

### 1. 编译包

```bash
cd <workspace>
colcon build --packages-select localization_cpp --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

### 2. 启动定位系统

```bash
ros2 launch localization_cpp localization.launch.py map:=/path/to/map.pcd
```

### 3. 启动参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `map` | PCD 地图文件路径（必需） | `""` |
| `pcd_map_topic` | 全局地图点云发布话题 | `/localize/map` |
| `rviz` | 是否启动 RViz（true/false） | `false` |
| `rviz_cfg` | RViz 配置文件路径 | 包内 `localization.rviz` |

## 话题与 TF

### 订阅
| 话题 | 类型 | 说明 |
|------|------|------|
| `/Odometry` | `nav_msgs/Odometry` | 里程计（重映射自 `/ql/nav/odometry`） |
| `/cloud_registered` | `sensor_msgs/PointCloud2` | 已配准点云 |
| `/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | 初始位姿（RViz 2D Pose Estimate） |
| `/localize/map_to_odom` | `nav_msgs/Odometry` | ICP 输出的 map→odom 变换 |
| `/localize/icp_score` | `std_msgs/Float32MultiArray` | ICP fitness + RMSE |

### 发布
| 话题 | 类型 | 说明 |
|------|------|------|
| `/localize/map_to_odom` | `nav_msgs/Odometry` | map→odom 变换 |
| `/localization` | `nav_msgs/Odometry` | 融合后的 map→base_link 位姿（50 Hz） |
| `/ql/nav/localization` | `nav_msgs/Odometry` | 可视化用低频定位结果（10 Hz） |
| `/localize/amcl_pose` | `geometry_msgs/PoseWithCovarianceStamped` | 定位结果（AMCL 格式） |
| `/localize/submap` | `sensor_msgs/PointCloud2` | 当前 FOV 内的地图子集 |
| `/localize/status` | `std_msgs/String` | 定位状态（`localization_success` / `localization_failed`） |
| `/ql/nav/cloud_in_map` | `sensor_msgs/PointCloud2` | 变换到 map 系的点云 |
| `/ql/nav/cloud_in_body` | `sensor_msgs/PointCloud2` | 变换到 base_link 系的点云 |

### TF
```
map
  └── odom          ← transform_fusion 广播（50 Hz）
        └── base_link
```



## 与 Python 版本的差异

| 项目 | Python 版本 | C++ 版本 |
|------|-------------|----------|
| ICP 库 | Open3D `registration_icp` | PCL `IterativeClosestPoint` |
| 地图加载 | `open3d.io.read_point_cloud` | `pcl::io::loadPCDFile` |
| 体素下采样 | Open3D `voxel_down_sample` | `pcl::VoxelGrid` |
| fitness 计算 | Open3D 原生（inlier 比例） | 手动计算（KdTree 搜索 inlier 比例，语义一致） |
| 点云变换 | `tf_transformations` + numpy | `tf2_sensor_msgs::doTransform` |
| 矩阵运算 | numpy | Eigen |
