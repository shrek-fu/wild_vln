# LineFit Ground Segmentation (ROS 2) for https://github.com/baiyeweiguang/linefit_ground_segmentation_ros2

一个高性能、可在 ROS 2 中实时运行的基于线拟合（LineFit）的点云地面分割库与节点，实现对原始 LiDAR/深度相机点云的地面与障碍物分离。核心算法参考并实现论文《Fast Segmentation of 3D Point Clouds for Ground Vehicles》，在保留精度的同时兼顾工程实用性与易用性。

## 特性
- 高效 C++17 实现，支持多线程并行处理
- 线拟合地面估计，稳健识别局部地面线段
- ROS 2 节点开箱即用，包含完整参数化与 Launch 文件
- 内置体素滤波与距离裁剪，便于工程部署与性能调优
- 与 TF 集成，可将点云统一转换到 `base_link` 坐标系

## 仓库结构
- `linefit_ground_segmentation/`：核心算法库（C++，PCL/Eigen）
  - `include/ground_segmentation/`：算法头文件（`ground_segmentation.h` 等）
  - `src/`：算法实现（`ground_segmentation.cc`、`segment.cc`、`bin.cc`）
  - `CMakeLists.txt`、`package.xml`：ROS 2 ament 构建配置
- `linefit_ground_segmentation_ros/`：ROS 2 封装节点
  - `src/ground_segmentation_node.cc`：实时分割节点
  - `launch/segmentation.launch.py`：启动分割节点
  - `launch/segmentation_params.yaml`：参数配置示例
- `doc/`：相关论文 PDF（算法参考）

## 环境与依赖
- 操作系统：Ubuntu（建议 20.04/22.04），ROS 2（例如 Foxy/Humble/Iron）
- 编译工具：`colcon`、`ament_cmake`
- 依赖库：`PCL`、`Eigen`、`rclcpp`、`tf2_ros`、`sensor_msgs`、`geometry_msgs`

示例安装命令（请在 Ubuntu + ROS 2 环境中执行）：
```bash
sudo apt update
sudo apt install -y ros-$ROS_DISTRO-rclcpp ros-$ROS_DISTRO-tf2-ros \
  ros-$ROS_DISTRO-sensor-msgs ros-$ROS_DISTRO-geometry-msgs \
  ros-$ROS_DISTRO-pcl-conversions libpcl-dev libeigen3-dev
```

## 构建与运行
1) 将本仓库放入你的 ROS 2 工作空间 `src/` 目录下：
```bash
mkdir -p ~/ws_linefit/src
cd ~/ws_linefit/src
# 将本项目克隆至此
```
2) 构建：
```bash
cd ~/ws_linefit
colcon build --symlink-install
source install/setup.bash
```
3) 启动节点：
```bash
ros2 launch linefit_ground_segmentation_ros segmentation.launch.py
```
默认订阅 `segmentation_params.yaml` 中声明的输入话题（默认：`/depth_camera/points`），并发布：
- 障碍物点云：`segmentation/obstacle`
- 地面点云：`segmentation/ground`

在 RViz2 中可添加上述两路 PointCloud2 话题进行可视化。

## 参数说明（核心）
以下参数既可在 Launch 的 `segmentation_params.yaml` 中配置，也可通过 ROS 2 参数服务动态设置：
- `n_threads`：并行线程数
- `r_min`、`r_max`：处理的最小/最大半径（米），内部以平方参与计算
- `n_bins`：径向分桶数量（越大越细）
- `n_segments`：角度分段数量（越大越细）
- `max_dist_to_line`：到地面线的最大垂直距离（判定地面阈值）
- `sensor_height`：传感器相对地面的高度（米）
- `min_slope`、`max_slope`：地面线的斜率范围约束
- `max_fit_error`：线拟合允许的最大点误差（平方后参与比较）
- `long_threshold`：相邻点被视为“远点”的径向距离阈值
- `max_long_height`：长线段中相邻点允许的最大高度变化
- `max_start_height`：从当前估计地面高度开始新线段的允许偏差
- `line_search_angle`：在角度邻域内搜寻更优线的范围（弧度）
- `voxel_leaf_size`：体素滤波叶子大小（米，设置为 0 关闭）
- `max_distance`：三维空间距离裁剪阈值（米，设置为 0 关闭）
- `input_topic`：输入点云话题名
- `ground_output_topic`、`obstacle_output_topic`：输出话题名

参数示例见 `linefit_ground_segmentation_ros/launch/segmentation_params.yaml`。

## 节点话题接口
- 订阅：`sensor_msgs/PointCloud2`（`input_topic`）
- 发布：
  - `segmentation/ground`（地面点云，`sensor_msgs/PointCloud2`）
  - `segmentation/obstacle`（障碍物点云，`sensor_msgs/PointCloud2`）
- 坐标系：节点尝试通过 TF 将点云转换到 `base_link`，若转换失败则使用原始坐标系（见 `ground_segmentation_node.cc`）。

## 库用法示例（C++）
如下示例展示如何在非 ROS 环境中直接调用核心库进行分割：
```cpp
#include "ground_segmentation/ground_segmentation.h"
#include <pcl/io/ply_io.h>

int main() {
  // 读取点云
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::io::loadPLYFile("example.ply", cloud);

  // 配置参数（关键参数附注释）
  GroundSegmentationParams params;
  params.n_bins = 120;            // 径向分桶数量
  params.n_segments = 360;        // 角度分段数量
  params.r_min_square = 0.2 * 0.2; // 处理的最小距离平方
  params.r_max_square = 10.0 * 10.0; // 处理的最大距离平方
  params.max_dist_to_line = 0.05; // 判定地面的最大垂直距离
  params.sensor_height = 0.8;     // 传感器高度（米）
  params.n_threads = 4;           // 并行线程数

  // 创建分割器并执行分割
  GroundSegmentation segmenter(params);
  std::vector<int> labels;        // 1=地面，0=障碍
  segmenter.segment(cloud, &labels);

  // 根据标签拆分点云
  pcl::PointCloud<pcl::PointXYZ> ground, obstacle;
  for (size_t i = 0; i < cloud.size(); ++i) {
    if (labels[i] == 1) ground.push_back(cloud[i]);
    else obstacle.push_back(cloud[i]);
  }

  // TODO: 保存/可视化 ground 与 obstacle
  return 0;
}
```

## 性能与调优建议
- 首先通过 `voxel_leaf_size` 做下采样，显著降低计算量
- 合理设置 `r_min/r_max` 限定有效工作半径
- 调大 `n_bins/n_segments` 能提升拟合精细度，但会增加计算
- 根据 CPU 核心数设置 `n_threads`，避免过多线程带来调度开销
- 传感器高度 `sensor_height` 与坐标系（TF）一致性非常关键

## 常见问题
- RViz 中看不到输出：确认 `input_topic`、`frame_id` 与 TF 是否匹配
- 地面误判较多：增大/减小 `max_dist_to_line`、检查 `sensor_height`
- 算法速度慢：开启体素滤波、调整 `n_bins/n_segments/n_threads`

## 算法参考与致谢
- Bogoslavskyi, I., & Stachniss, C. (2017). Fast Segmentation of 3D Point Clouds for Ground Vehicles.
- 本仓库 `doc/` 目录提供论文 PDF 以便查阅。

## 许可证
- 本项目采用 BSD 许可证（详见各 `package.xml`）。如需在仓库根目录提供 `LICENSE` 文件，可选择 BSD-3-Clause 模板。

## 贡献指南
- 欢迎通过 Pull Request 提交优化与新特性
- 代码规范：C++17，保持与当前工程一致的风格与依赖
- 提交前请在本地完成构建与基本运行验证（`colcon build` + `ros2 launch`）
