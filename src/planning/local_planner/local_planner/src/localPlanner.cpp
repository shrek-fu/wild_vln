/**
 * ============================================================================
 * 本地路径规划器 (Local Planner)
 * ============================================================================
 * 功能说明：
 *   - 从预生成的343条路径中实时选择最优路径
 *   - 综合考虑：障碍物检测、方向权重、成本函数等因素
 *   - 适应性调整：根据速度和目标位置动态调整路径搜索参数
 *   - 支持：自主导航模式、遥控模式、双向驾驶等功能
 * 
 * 核心算法：
 *   1. 全局路径评分系统：对所有可行路径进行多维度评分
 *   2. 多旋转方向搜索：36个方向（每10°一个）评估所有路径
 *   3. 自适应缩放策略：根据障碍物情况调整路径尺度和范围
 *   4. 成本函数融合：结合方向偏差、旋转方向偏好、成本高度等因素
 * ============================================================================
 */

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp/clock.hpp"
#include "builtin_interfaces/msg/time.hpp"

#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <sensor_msgs/msg/joy.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <nav_msgs/msg/path.hpp>

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/imu.h>

#include "tf2/transform_datatypes.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "rmw/types.h"
#include "rmw/qos_profiles.h"

using namespace std;

const double PI = 3.1415926;

#define PLOTPATHSET 1  // 是否发布可视化的路径集合

// ============================================================================
// 车辆和传感器参数
// ============================================================================
string pathFolder;              // 预生成路径文件存放目录
double vehicleLength = 0.6;     // 车辆长度
double vehicleWidth = 0.6;      // 车辆宽度
double footprintSafetyMargin = 0.05; // 双圆 footprint 额外安全余量
int footprintCheckSkip = 3;     // footprint 沿候选路径的采样间隔
double sensorOffsetX = 0;       // 传感器相对于车体中心的X偏移
double sensorOffsetY = 0;       // 传感器相对于车体中心的Y偏移
bool twoWayDrive = true;        // 是否支持双向驾驶

// ============================================================================
// 点云处理参数
// ============================================================================
double laserVoxelSize = 0.05;   // 激光点云体素大小(降采样)
double terrainVoxelSize = 0.2;  // 地形点云体素大小
bool useTerrainAnalysis = false;// 是否使用地形分析
bool checkObstacle = true;      // 是否检查障碍物
bool checkRotObstacle = false;  // 是否检查旋转障碍物

// ============================================================================
// 障碍物检测参数
// ============================================================================
double adjacentRange = 3.5;           // 相邻范围(感知范围)
double obstacleHeightThre = 0.2;      // 障碍物高度阈值
double groundHeightThre = 0.1;        // 地面高度阈值
double costHeightThre = 0.1;          // 成本高度阈值
double costScore = 0.02;              // 成本评分权重
bool useCost = false;                 // 是否使用成本函数

// ============================================================================
// 路径选择参数
// ============================================================================
const int laserCloudStackNum = 1;  // 点云堆栈数量
int laserCloudCount = 0;
int pointPerPathThre = 2;          // 判定路径为阻挡的最少点数
double minRelZ = -0.5;             // Z相对范围最小值
double maxRelZ = 1.2;              // Z相对范围最大值
double maxSpeed = 1.0;             // 最大速度
double dirWeight = 0.02;           // 方向权重系数
double dirThre = 90.0;             // 方向阈值(度)
bool dirToVehicle = false;         // 是否以车体方向为参考
double maxForwardPathAngle = 75.0;  // 非双向模式下路径初始切线最大偏角(度)
double pathSwitchHysteresis = 0.9;  // 上一帧路径达到最优分数该比例时保持原路径
int lastSelectedCandidateID = -1;   // 上一帧选中的“旋转方向×具体路径”索引
int lastSelectedPathID = -1;        // 上一帧选中的“旋转方向×路径分组”索引

// ============================================================================
// 路径缩放与范围参数 (自适应调整)
// ============================================================================
double pathScale = 1.0;            // 路径缩放因子
double minPathScale = 0.75;        // 最小路径缩放
double pathScaleStep = 0.25;       // 缩放步长
bool pathScaleBySpeed = true;      // 是否根据速度调整缩放
double minPathRange = 1.0;         // 最小搜索范围
double pathRangeStep = 0.5;        // 范围步长
bool pathRangeBySpeed = true;      // 是否根据速度调整范围
bool pathCropByGoal = true;        // 是否根据目标点裁剪路径

// ============================================================================
// 自主导航参数
// ============================================================================
bool autonomyMode = false;         // 自主模式标志
double autonomySpeed = 1.0;        // 自主导航速度
double joyToSpeedDelay = 2.0;      // 摇杆到速度转换延迟
double joyToCheckObstacleDelay = 5.0; // 检查障碍物延迟
double goalClearRange = 0.5;       // 目标点周围清晰范围
double goalX = 0;                  // 目标X坐标
double goalY = 0;                  // 目标Y坐标

// ============================================================================
// 全局路径跟踪参数 (来自外部全局规划器 /global_path)
// ============================================================================
nav_msgs::msg::Path globalPath;         // 全局路径缓存
bool globalPathReceived = false;        // 是否收到全局路径
bool wayPointReceived = false;          // 是否收到有效的 way_point 目标
int globalPathPointID = 0;              // 当前追踪的全局路径点索引
int globalPathLookAheadID = 0;          // 前视目标点索引（供地形裁剪使用）
double globalPathLookAheadDis = 3.0;   // 从全局路径提取中间目标的前视距离(m)
int globalPathSearchPointNum = 100;    // 从当前进度向前搜索最近路径段的最大点数
double goalReachedDis = 0.5;           // 认为到达 way_point / 全局路径终点的距离阈值(m)
double goalRemainDis = -1.0;          // 到最终目标的剩余距离(m)，<0 表示无效

// 路径正下方地面点裁剪参数
double pathClearRadius = 0.5;          // 路径下方裁剪水平半径(m)
double pathClearHeight = 0.2;          // 裁剪圆柱体高度(m)

// 终点朝向对齐 (来自 /goal_pose)
bool enableHeadingAlign = false;       // 是否启用到达终点后的朝向对齐功能
double goalPoseYaw = 0.0;              // 目标朝向 (从 PoseStamped 提取的 yaw, 弧度)
bool goalPoseReceived = false;         // 是否收到 goal_pose
bool aligningHeading = false;          // 是否处于朝向对齐阶段
double headingAlignThre = 0.05;        // 朝向对齐阈值 (弧度, ~3°)
double headingAlignSpeed = 0.3;        // 朝向对齐时的角速度 (rad/s)

// ============================================================================
// 摇杆控制变量
// ============================================================================
float joySpeed = 0;               // 摇杆速度指令(0-1)
float joySpeedRaw = 0;            // 原始摇杆速度
float joyDir = 0;                 // 摇杆方向(度)

// ============================================================================
// 路径规划参数
// ============================================================================
const int pathNum = 343;          // 预生成的总路径数
const int groupNum = 7;           // 路径分组数(7个初始方向)
float gridVoxelSize = 0.02;       // 网格体素大小
float searchRadius = 0.45;        // 搜索半径
float gridVoxelOffsetX = 3.2;     // 网格体素X方向偏移
float gridVoxelOffsetY = 4.5;     // 网格体素Y方向偏移
const int gridVoxelNumX = 161;    // X方向网格数量
const int gridVoxelNumY = 451;    // Y方向网格数量
const int gridVoxelNum = gridVoxelNumX * gridVoxelNumY;  // 总网格数

// ============================================================================
// 点云数据结构 (PCL智能指针)
// ============================================================================
// 激光点云相关
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCrop(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudDwz(new pcl::PointCloud<pcl::PointXYZI>());

// 地形点云相关
pcl::PointCloud<pcl::PointXYZI>::Ptr terrainCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr terrainCloudCrop(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr terrainCloudDwz(new pcl::PointCloud<pcl::PointXYZI>());

// 点云堆栈(用于融合多帧数据)
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudStack[laserCloudStackNum];

// 规划器使用的点云(融合所有障碍物)
pcl::PointCloud<pcl::PointXYZI>::Ptr plannerCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr plannerCloudCrop(new pcl::PointCloud<pcl::PointXYZI>());

// 边界和额外障碍物
pcl::PointCloud<pcl::PointXYZI>::Ptr boundaryCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr addedObstacles(new pcl::PointCloud<pcl::PointXYZI>());

// 起点路径(7组,每组是一条路径的起点序列)
pcl::PointCloud<pcl::PointXYZ>::Ptr startPaths[groupNum];

// 343条完整候选路径（评分、footprint检查和最终发布使用同一条路径）
pcl::PointCloud<pcl::PointXYZI>::Ptr paths[pathNum];
float initialDirPathList[pathNum] = {0};            // 每条路径的初始切线方向(度)

// 完整路径可视化
#if PLOTPATHSET == 1
pcl::PointCloud<pcl::PointXYZI>::Ptr freePaths(new pcl::PointCloud<pcl::PointXYZI>()); // 可行路径
#endif

// ============================================================================
// 路径评分数据结构
// ============================================================================
int pathList[pathNum] = {0};                        // 路径所属分组映射
float endDirPathList[pathNum] = {0};                // 各路径的终点方向
int clearPathList[36 * pathNum] = {0};              // 清晰路径标志(36个方向×343条路径)
float pathPenaltyList[36 * pathNum] = {0};          // 路径成本惩罚
float candidatePathScore[36 * pathNum] = {0};       // 每条旋转后具体路径的评分
float clearPathPerGroupScore[36 * groupNum] = {0};  // 每组路径的评分(36个方向×7组)

// 网格对应表:存储每个网格体素对应的路径列表
std::vector<int> correspondences[gridVoxelNum];

// ============================================================================
// 数据更新标志
// ============================================================================
bool newLaserCloud = false;      // 是否收到新的激光点云
bool newTerrainCloud = false;    // 是否收到新的地形点云

// ============================================================================
// 时间戳和状态变量
// ============================================================================
double odomTime = 0;              // 里程计消息时间戳
double joyTime = 0;               // 摇杆消息时间戳

// ============================================================================
// 车辆位姿状态 (在车体坐标系中)
// ============================================================================
float vehicleRoll = 0, vehiclePitch = 0, vehicleYaw = 0;  // 姿态角
float vehicleX = 0, vehicleY = 0, vehicleZ = 0;          // 位置坐标

// ============================================================================
// 点云处理工具
// ============================================================================
pcl::VoxelGrid<pcl::PointXYZI> laserDwzFilter, terrainDwzFilter;  // 体素栅格滤波器

// ROS 2 节点指针
rclcpp::Node::SharedPtr nh;

// 路径发布器 (全局，以便在目标到达时发布空路径)
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;
rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pubGoalDistance;
rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pubCmdVelAlign;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubTerrainCloud;
rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr pubFootprint;

double normalizeAngleDeg(double angle)
{
  while (angle > 180.0) angle -= 360.0;
  while (angle < -180.0) angle += 360.0;
  return angle;
}

/**
 * 使用方向相关的双圆 footprint 检查候选路径。
 *
 * 与 planner/robot_radius.md 中的方法一致：
 *   radius = vehicleWidth / 2 + safety margin
 *   offset = max(0, vehicleLength / 2 - vehicleWidth / 2)
 * 前、后圆心沿候选路径的局部切线方向放置。每个障碍点最多计数一次，
 * 避免同一个点因命中多个相邻路径采样而被重复累计。
 */
template <typename PointT>
bool isFootprintBlockedForPathCloud(
    const pcl::PointCloud<PointT> &pathCloud, double rotAng,
    double currentPathScale, double maxPathDistance)
{
  if (!checkObstacle || pathCloud.empty()) {
    return false;
  }

  const double baseRadius = std::max(0.0, vehicleWidth * 0.5);
  const double radius = baseRadius + std::max(0.0, footprintSafetyMargin);
  const double offset = std::max(0.0, vehicleLength * 0.5 - baseRadius);
  const double radiusSq = radius * radius;
  const int sampleSkip = std::max(1, footprintCheckSkip);
  const int pathSize = static_cast<int>(pathCloud.points.size());
  int obstacleHits = 0;

  for (const auto & obstacle : plannerCloudCrop->points) {
    if (useTerrainAnalysis && obstacle.intensity <= obstacleHeightThre) {
      continue;
    }

    bool hitsThisObstacle = false;
    for (int i = 0; i < pathSize; i += sampleSkip) {
      const auto & pathPoint = pathCloud.points[i];
      double pathDistance = hypot(pathPoint.x, pathPoint.y) * currentPathScale;
      if (pathDistance > maxPathDistance) break;

      int nextID = std::min(i + sampleSkip, pathSize - 1);
      int prevID = std::max(0, i - sampleSkip);
      const auto & prevPoint = pathCloud.points[prevID];
      const auto & nextPoint = pathCloud.points[nextID];
      double tangentYaw = atan2(nextPoint.y - prevPoint.y,
                                nextPoint.x - prevPoint.x) + rotAng;

      double centerX = currentPathScale *
          (cos(rotAng) * pathPoint.x - sin(rotAng) * pathPoint.y);
      double centerY = currentPathScale *
          (sin(rotAng) * pathPoint.x + cos(rotAng) * pathPoint.y);
      double headingX = cos(tangentYaw);
      double headingY = sin(tangentYaw);

      double frontDx = obstacle.x - (centerX + offset * headingX);
      double frontDy = obstacle.y - (centerY + offset * headingY);
      double rearDx = obstacle.x - (centerX - offset * headingX);
      double rearDy = obstacle.y - (centerY - offset * headingY);
      if (frontDx * frontDx + frontDy * frontDy <= radiusSq ||
          rearDx * rearDx + rearDy * rearDy <= radiusSq) {
        hitsThisObstacle = true;
        break;
      }
    }

    if (hitsThisObstacle && ++obstacleHits >= pointPerPathThre) {
      return true;
    }
  }
  return false;
}

bool isPathFootprintBlocked(int pathID, double rotAng, double currentPathScale,
                            double maxPathDistance)
{
  if (pathID < 0 || pathID >= pathNum || paths[pathID] == nullptr ||
      paths[pathID]->empty()) {
    return false;
  }

  return isFootprintBlockedForPathCloud(*paths[pathID], rotAng,
                                         currentPathScale, maxPathDistance);
}

bool isGroupFootprintBlocked(int groupID, double rotAng, double currentPathScale,
                             double maxPathDistance)
{
  if (groupID < 0 || groupID >= groupNum) {
    return false;
  }

  if (startPaths[groupID] == nullptr || startPaths[groupID]->empty()) {
    return false;
  }

  // 用分组对应的代表路径进行 footprint 检查，避免某条偏靠墙壁的成员路径
  // 把整个分组误判为不可通过，导致局部规划在狭窄通道前产生左右绕行。
  return isFootprintBlockedForPathCloud(*startPaths[groupID], rotAng,
                                         currentPathScale, maxPathDistance);
}

/**
 * 发布车身 footprint（base_link 坐标系）
 * sensorOffsetX > 0 表示传感器在车体中心前方，车体中心在传感器后方
 * 所以：车头 x = +halfL - sensorOffsetX，车尾 x = -halfL - sensorOffsetX
        左侧 y = +(halfW - sensorOffsetY)，右侧 y = -(halfW + sensorOffsetY)
 * 四角坐标固定，只在首次调用时计算，之后仅更新 stamp。
 */
void publishFootprint()
{
  static geometry_msgs::msg::PolygonStamped fp;
  static bool initialized = false;

  if (!initialized) {
    float halfL = static_cast<float>(vehicleLength / 2.0);
    float halfW = static_cast<float>(vehicleWidth  / 2.0);
    // 传感器为 base_link 原点，车体中心相对传感器的偏移
    float cx = -static_cast<float>(sensorOffsetX);
    float cy = -static_cast<float>(sensorOffsetY);

    fp.header.frame_id = "base_link";
    auto makePoint = [](float x, float y) {
      geometry_msgs::msg::Point32 p;
      p.x = x; p.y = y; p.z = 0.0f;
      return p;
    };
    fp.polygon.points.push_back(makePoint(cx + halfL,  cy + halfW));  // 前左
    fp.polygon.points.push_back(makePoint(cx + halfL,  cy - halfW));  // 前右
    fp.polygon.points.push_back(makePoint(cx - halfL,  cy - halfW));  // 后右
    fp.polygon.points.push_back(makePoint(cx - halfL,  cy + halfW));  // 后左
    fp.polygon.points.push_back(makePoint(cx + halfL,  cy + halfW));  // 闭合
    initialized = true;
  }

  fp.header.stamp = nh->now();
  pubFootprint->publish(fp);
}

// 发布空路径，通知 pathFollower 停止追踪
void publishGoalRemain(double remain)
{
  goalRemainDis = remain;
  if (!pubGoalDistance) return;
  std_msgs::msg::Float32 remainMsg;
  remainMsg.data = static_cast<float>(remain);
  pubGoalDistance->publish(remainMsg);
}

void publishEmptyPath()
{
  nav_msgs::msg::Path emptyPath;
  emptyPath.header.stamp = nh->now();
  emptyPath.header.frame_id = "base_link";
  pubPath->publish(emptyPath);
  // 目标已结束，通知 pathFollower 取消终点减速限制。
  publishGoalRemain(-1.0);
}

/**
 * 路径正下方地面点裁剪
 * 功能：将 globalPathPointID ~ globalPathLookAheadID 段路径点
 *         正下方圆柱区域内的地面点从 terrainCloudDwz 中移除
 * 参数：通过全局变量 pathClearRadius / groundHeightThre / pathClearHeight 控制
 */
void filterTerrainByPathCylinder()
{
  if (!globalPathReceived || globalPathLookAheadID <= globalPathPointID) return;

  int pathSize = static_cast<int>(globalPath.poses.size());
  int startID  = std::max(0, globalPathPointID);
  int endID    = std::min(globalPathLookAheadID + 10, pathSize - 1);

  float r2    = static_cast<float>(pathClearRadius * pathClearRadius);
  float zLow  = static_cast<float>(groundHeightThre - pathClearHeight * 0.5);
  float zHigh = static_cast<float>(groundHeightThre + pathClearHeight * 0.5);

  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>());
  filtered->reserve(terrainCloudDwz->points.size());

  for (const auto& pt : terrainCloudDwz->points) {
    bool inPathCylinder = false;
    for (int id = startID; id <= endID && !inPathCylinder; id += 2) {
      float px = static_cast<float>(globalPath.poses[id].pose.position.x);
      float py = static_cast<float>(globalPath.poses[id].pose.position.y);
      float pz = static_cast<float>(globalPath.poses[id].pose.position.z);

      float dx = pt.x - px;
      float dy = pt.y - py;
      float dz = pt.z - pz;

      if ((dx * dx + dy * dy) < r2 && dz > zLow && dz < zHigh) {
        inPathCylinder = true;
      }
    }
    if (!inPathCylinder) {
      filtered->push_back(pt);
    }
  }

  terrainCloudDwz = filtered;
}

/**
 * 里程计消息回调函数
 * 功能：
 *   - 提取车辆当前位姿(位置和姿态)
 *   - 根据传感器偏移调整车体坐标
 *   - 更新时间戳
 */
void odometryHandler(const nav_msgs::msg::Odometry::ConstSharedPtr odom)
{
  odomTime = rclcpp::Time(odom->header.stamp).seconds();
  double roll, pitch, yaw;
  geometry_msgs::msg::Quaternion geoQuat = odom->pose.pose.orientation;
  // 四元数转欧拉角(RPY)
  tf2::Matrix3x3(tf2::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);

  vehicleRoll = roll;
  vehiclePitch = pitch;
  vehicleYaw = yaw;
  // 根据传感器偏移调整车体中心位置
  vehicleX = odom->pose.pose.position.x - cos(yaw) * sensorOffsetX + sin(yaw) * sensorOffsetY;
  vehicleY = odom->pose.pose.position.y - sin(yaw) * sensorOffsetX - cos(yaw) * sensorOffsetY;
  vehicleZ = odom->pose.pose.position.z;
}

/**
 * 激光点云处理回调函数
 * 功能：
 *   1. 将PointCloud2格式转为PCL点云
 *   2. 按距离裁剪：只保留相邻范围内的点
 *   3. 体素栅格降采样：减少计算量
 *   4. 设置标志供主循环处理
 */
void laserCloudHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr laserCloud2)
{
  if (!useTerrainAnalysis) {
    laserCloud->clear();
    bool hasIntensity = false;
    for (const auto& field : laserCloud2->fields) {
      if (field.name == "intensity") { hasIntensity = true; break; }
    }
    if (hasIntensity) {
      pcl::fromROSMsg(*laserCloud2, *laserCloud);
    } else {
      pcl::PointCloud<pcl::PointXYZ> laserCloudXYZ;
      pcl::fromROSMsg(*laserCloud2, laserCloudXYZ);
      pcl::copyPointCloud(laserCloudXYZ, *laserCloud);  // intensity 字段自动填 0
    }

    pcl::PointXYZI point;
    laserCloudCrop->clear();
    int laserCloudSize = laserCloud->points.size();
    
    // 按距离裁剪:保留相邻范围内的点
    for (int i = 0; i < laserCloudSize; i++) {
      point = laserCloud->points[i];

      float pointX = point.x;
      float pointY = point.y;
      float pointZ = point.z;

      // 计算点到车辆的距离
      float dis = sqrt((pointX - vehicleX) * (pointX - vehicleX) + (pointY - vehicleY) * (pointY - vehicleY));
      if (dis < adjacentRange) {
        point.x = pointX;
        point.y = pointY;
        point.z = pointZ;
        laserCloudCrop->push_back(point);
      }
    }

    // 体素栅格滤波:降采样处理
    laserCloudDwz->clear();
    laserDwzFilter.setInputCloud(laserCloudCrop);
    laserDwzFilter.filter(*laserCloudDwz);

    newLaserCloud = true;
  }
}

/**
 * 地形点云处理回调函数
 * 功能：
 *   - 处理地形分析数据(高程图)
 *   - 按距离和高度阈值(障碍物识别)进行裁剪
 *   - 体素栅格降采样
 */
void terrainCloudHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr terrainCloud2)
{
  if (useTerrainAnalysis) {
    terrainCloud->clear();
    bool hasIntensity = false;
    for (const auto& field : terrainCloud2->fields) {
      if (field.name == "intensity") { hasIntensity = true; break; }
    }
    if (hasIntensity) {
      pcl::fromROSMsg(*terrainCloud2, *terrainCloud);
    } else {
      pcl::PointCloud<pcl::PointXYZ> terrainCloudXYZ;
      pcl::fromROSMsg(*terrainCloud2, terrainCloudXYZ);
      pcl::copyPointCloud(terrainCloudXYZ, *terrainCloud);  // intensity 字段自动填 0
    }

    pcl::PointXYZI point;
    terrainCloudCrop->clear();
    int terrainCloudSize = terrainCloud->points.size();
    for (int i = 0; i < terrainCloudSize; i++) {
      point = terrainCloud->points[i];

      float pointX = point.x;
      float pointY = point.y;
      float pointZ = point.z;

      float dis = sqrt((pointX - vehicleX) * (pointX - vehicleX) + (pointY - vehicleY) * (pointY - vehicleY));
      // 相对车体高度写入 intensity，供路径评分和 footprint 判障碍
      float pt_h = pointZ - vehicleZ;
      if (dis < adjacentRange && ((pt_h > obstacleHeightThre && pt_h < maxRelZ) || useCost)) {
        point.x = pointX;
        point.y = pointY;
        point.z = pointZ;
        point.intensity = pt_h;
        terrainCloudCrop->push_back(point);
      }
    }

    terrainCloudDwz->clear();
    terrainDwzFilter.setInputCloud(terrainCloudCrop);
    terrainDwzFilter.filter(*terrainCloudDwz);

    // ── 路径正下方地面点裁剪 ──────────────────────────────────────────────────
    filterTerrainByPathCylinder();
    // ───────────────────────────────────────────────────────────────────────

    // 发布裁剪后的地形点云
    sensor_msgs::msg::PointCloud2 terrainCloudDwzMsg;
    pcl::toROSMsg(*terrainCloudDwz, terrainCloudDwzMsg);
    terrainCloudDwzMsg.header.stamp    = terrainCloud2->header.stamp;
    terrainCloudDwzMsg.header.frame_id = terrainCloud2->header.frame_id;
    pubTerrainCloud->publish(terrainCloudDwzMsg);

    newTerrainCloud = true;
  }
}

/**
 * 摇杆输入处理回调函数
 * 功能：
 *   - 提取摇杆速度指令 (axes[3]:Y轴, axes[4]:X轴)
 *   - 计算运动方向和速度幅度
 *   - 处理自主模式切换
 *   - 处理障碍物检查使能
 */
void joystickHandler(const sensor_msgs::msg::Joy::ConstSharedPtr joy)
{
  joyTime = nh->now().seconds();
  // 计算摇杆速度幅度 (使用X-Y轴的矢量模)
  joySpeedRaw = sqrt(joy->axes[3] * joy->axes[3] + joy->axes[4] * joy->axes[4]);
  joySpeed = joySpeedRaw;
  if (joySpeed > 1.0) joySpeed = 1.0;
  if (joy->axes[4] == 0) joySpeed = 0;

  // 计算运动方向 (相对于摇杆坐标系)
  if (joySpeed > 0) {
    joyDir = atan2(joy->axes[3], joy->axes[4]) * 180 / PI;
    if (joy->axes[4] < 0) joyDir *= -1;  // 后退时方向取反
  }

  // 是否支持后退
  if (joy->axes[4] < 0 && !twoWayDrive) joySpeed = 0;

  // 摇杆轴[2]切换自主模式 (>-0.1为手动模式, ≤-0.1为自主模式)
  if (joy->axes[2] > -0.1) {
    autonomyMode = false;
  } else {
    autonomyMode = true;
  }

  // 摇杆轴[5]控制是否检查障碍物
  if (joy->axes[5] > -0.1) {
    checkObstacle = true;
  } else {
    checkObstacle = false;
  }
}

/**
 * 目标点处理回调函数
 * 功能：在自主导航模式下更新目标点
 */
void goalHandler(const geometry_msgs::msg::PointStamped::ConstSharedPtr goal)
{
  goalX = goal->point.x;
  goalY = goal->point.y;
  wayPointReceived = true;
  globalPathReceived = false;  // way_point 与全局路径互斥
  RCLCPP_INFO(nh->get_logger(), "Received waypoint goal: (%.2f, %.2f)", goalX, goalY);
}

/**
 * 速度指令处理回调函数 (用于自主模式)
 * 功能：
 *   - 在自主导航模式下接收预规划速度
 *   - 具有延迟检查防止手动切换造成的误干扰
 */
void speedHandler(const std_msgs::msg::Float32::ConstSharedPtr speed)
{
  double speedTime = nh->now().seconds();
  // 只在自主模式且足够延迟后才接受速度指令
  if (autonomyMode && speedTime - joyTime > joyToSpeedDelay && joySpeedRaw == 0) {
    joySpeed = speed->data / maxSpeed;

    if (joySpeed < 0) joySpeed = 0;
    else if (joySpeed > 1.0) joySpeed = 1.0;
  }
}

/**
 * 导航边界处理回调函数
 * 功能：
 *   - 将多边形边界转换为点云表示
 *   - 对边界进行离散化处理(按terrainVoxelSize间隔采样)
 *   - 用于路径评估时的碰撞检测
 */
void boundaryHandler(const geometry_msgs::msg::PolygonStamped::ConstSharedPtr boundary)
{
  boundaryCloud->clear();
  pcl::PointXYZI point, point1, point2;
  int boundarySize = boundary->polygon.points.size();

  if (boundarySize >= 1) {
    point2.x = boundary->polygon.points[0].x;
    point2.y = boundary->polygon.points[0].y;
    point2.z = boundary->polygon.points[0].z;
  }

  // 遍历多边形的每条边
  for (int i = 0; i < boundarySize; i++) {
    point1 = point2;

    point2.x = boundary->polygon.points[i].x;
    point2.y = boundary->polygon.points[i].y;
    point2.z = boundary->polygon.points[i].z;

    // 仅处理Z坐标相同的边(水平面)
    if (point1.z == point2.z) {
      float disX = point1.x - point2.x;
      float disY = point1.y - point2.y;
      float dis = sqrt(disX * disX + disY * disY);

      // 按terrainVoxelSize间隔离散化边
      int pointNum = int(dis / terrainVoxelSize) + 1;
      for (int pointID = 0; pointID < pointNum; pointID++) {
        // 线性插值生成边上的点
        point.x = float(pointID) / float(pointNum) * point1.x + (1.0 - float(pointID) / float(pointNum)) * point2.x;
        point.y = float(pointID) / float(pointNum) * point1.y + (1.0 - float(pointID) / float(pointNum)) * point2.y;
        point.z = 0;
        point.intensity = 100.0;  // 边界标记为100

        // 每个点重复添加pointPerPathThre次(增加权重)
        for (int j = 0; j < pointPerPathThre; j++) {
          boundaryCloud->push_back(point);
        }
      }
    }
  }
}

/**
 * 手动添加障碍物处理回调函数
 * 功能：
 *   - 接收用户手动添加的额外障碍物
 *   - 设置强度值为200以区别于其他点云
 */
void addedObstaclesHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr addedObstacles2)
{
  addedObstacles->clear();
  pcl::fromROSMsg(*addedObstacles2, *addedObstacles);

  int addedObstaclesSize = addedObstacles->points.size();
  for (int i = 0; i < addedObstaclesSize; i++) {
    addedObstacles->points[i].intensity = 200.0;  // 标记为添加的障碍物
  }
}

/**
 * 障碍物检查使能处理回调函数 (自主模式)
 * 功能：在自主模式下接收是否启用障碍物检查的指令
 */
void checkObstacleHandler(const std_msgs::msg::Bool::ConstSharedPtr checkObs)
{
  double checkObsTime = nh->now().seconds();
  // 仅在自主模式且足够延迟后才接受指令
  if (autonomyMode && checkObsTime - joyTime > joyToCheckObstacleDelay) {
    checkObstacle = checkObs->data;
  }
}

/**
 * 全局路径回调函数
 * 功能：
 *   - 接收外部全局规划器发布的三维路径 /global_path
 *   - 缓存全局路径并重置追踪索引
 * 注意：
 *   - 全局路径坐标系为世界坐标系(world frame)
 *   - localPlanner 的局部避障在此基础上实时覆盖最终方向
 */
void globalPathHandler(const nav_msgs::msg::Path::ConstSharedPtr globalPathIn)
{
  int pathSize = globalPathIn->poses.size();
  if (pathSize == 0) {
    globalPathReceived = false;
    publishEmptyPath();
    RCLCPP_INFO(nh->get_logger(), "Received empty global path, stopping navigation.");
    return;
  }

  globalPath = *globalPathIn;
  globalPathPointID = 0;
  lastSelectedPathID = -1;
  globalPathReceived = true;
  wayPointReceived = false;  // 全局路径与 way_point 互斥
  aligningHeading = false;   // 新路径收到时重置对齐状态

  RCLCPP_INFO(nh->get_logger(), "Received global path with %d waypoints.", pathSize);
}

/**
 * 终点朝向目标处理器
 * 订阅 /goal_pose (geometry_msgs/PoseStamped)
 * 提取目标偏航角，到达路径终点位置后用于朝向对齐
 */
void goalPoseHandler(const geometry_msgs::msg::PoseStamped::ConstSharedPtr poseIn)
{
  // 从四元数提取 yaw
  tf2::Quaternion q(
    poseIn->pose.orientation.x,
    poseIn->pose.orientation.y,
    poseIn->pose.orientation.z,
    poseIn->pose.orientation.w);
  tf2::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);
  goalPoseYaw = yaw;
  goalPoseReceived = true;
  RCLCPP_INFO(nh->get_logger(), "Received goal_pose, target yaw = %.3f rad.", goalPoseYaw);
}

/**
 * 从全局路径中提取当前中间目标点
 * 功能：
 *   - 在当前进度前方搜索车辆到全局路径的最近投影点
 *   - 从投影点沿路径弧长前视 globalPathLookAheadDis 距离
 *   - 更新 goalX/goalY 供 localPlanner 方向评分使用
 *   - 路径进度只向前推进，允许局部规划绕障时暂时偏离全局路径
 */
void updateGoalFromGlobalPath()
{
  if (!globalPathReceived) return;

  int pathSize = globalPath.poses.size();
  if (pathSize == 0) return;

  if (pathSize == 1) {
    goalX = globalPath.poses[0].pose.position.x;
    goalY = globalPath.poses[0].pose.position.y;
    globalPathLookAheadID = 0;
    goalRemainDis = hypot(goalX - vehicleX, goalY - vehicleY);
  } else {
    // 在当前进度的前方窗口内搜索最近路径段。采用线段投影而不是要求
    // 车辆进入离散路径点半径，因此绕障跨过某些点后不会再被旧点拉回。
    int searchBegin = std::min(globalPathPointID, pathSize - 2);
    int searchEnd = std::min(pathSize - 2,
                             searchBegin + std::max(1, globalPathSearchPointNum));
    int nearestSegment = searchBegin;
    double nearestT = 0.0;
    double minDistSq = std::numeric_limits<double>::max();

    for (int i = searchBegin; i <= searchEnd; ++i) {
      double x0 = globalPath.poses[i].pose.position.x;
      double y0 = globalPath.poses[i].pose.position.y;
      double x1 = globalPath.poses[i + 1].pose.position.x;
      double y1 = globalPath.poses[i + 1].pose.position.y;
      double vx = x1 - x0;
      double vy = y1 - y0;
      double lengthSq = vx * vx + vy * vy;
      double t = 0.0;
      if (lengthSq > 1e-8) {
        t = ((vehicleX - x0) * vx + (vehicleY - y0) * vy) / lengthSq;
        if (t < 0.0) t = 0.0;
        else if (t > 1.0) t = 1.0;
      }
      double projX = x0 + t * vx;
      double projY = y0 + t * vy;
      double dx = vehicleX - projX;
      double dy = vehicleY - projY;
      double distSq = dx * dx + dy * dy;
      if (distSq < minDistSq) {
        minDistSq = distSq;
        nearestSegment = i;
        nearestT = t;
      }
    }

    // 单调更新进度；投影越过线段中点后即可认为对应的下一个点已通过。
    int projectedProgress = nearestSegment + (nearestT >= 0.5 ? 1 : 0);
    globalPathPointID = std::max(globalPathPointID, projectedProgress);

    // 到最终目标的剩余距离。弯道上弧长大于直线距离，用较大值避免提前刹车。
    // 进度已经到终点后改用直线距离，冲过目标时仍按实际偏离慢速回到目标。
    double arcRemain = 0.0;
    {
      const auto& p0 = globalPath.poses[nearestSegment].pose.position;
      const auto& p1 = globalPath.poses[nearestSegment + 1].pose.position;
      arcRemain = hypot(p1.x - p0.x, p1.y - p0.y) * (1.0 - nearestT);
      for (int seg = nearestSegment + 1; seg < pathSize - 1; ++seg) {
        const auto& a = globalPath.poses[seg].pose.position;
        const auto& b = globalPath.poses[seg + 1].pose.position;
        arcRemain += hypot(b.x - a.x, b.y - a.y);
      }
    }
    double finalX = globalPath.poses[pathSize - 1].pose.position.x;
    double finalY = globalPath.poses[pathSize - 1].pose.position.y;
    double euclid = hypot(finalX - vehicleX, finalY - vehicleY);
    if (globalPathPointID >= pathSize - 1) goalRemainDis = euclid;
    else goalRemainDis = std::max(arcRemain, euclid);

    // 从最近投影点开始按路径弧长前视，而不是用车辆到路径点的直线距离。
    double x0 = globalPath.poses[nearestSegment].pose.position.x;
    double y0 = globalPath.poses[nearestSegment].pose.position.y;
    double x1 = globalPath.poses[nearestSegment + 1].pose.position.x;
    double y1 = globalPath.poses[nearestSegment + 1].pose.position.y;
    double targetX = x0 + nearestT * (x1 - x0);
    double targetY = y0 + nearestT * (y1 - y0);
    double remaining = globalPathLookAheadDis;
    int segment = nearestSegment;
    double segmentT = nearestT;

    while (segment < pathSize - 1) {
      x0 = globalPath.poses[segment].pose.position.x;
      y0 = globalPath.poses[segment].pose.position.y;
      x1 = globalPath.poses[segment + 1].pose.position.x;
      y1 = globalPath.poses[segment + 1].pose.position.y;
      double segmentLength = hypot(x1 - x0, y1 - y0);
      double available = segmentLength * (1.0 - segmentT);
      if (segmentLength > 1e-8 && remaining <= available) {
        double targetT = segmentT + remaining / segmentLength;
        targetX = x0 + targetT * (x1 - x0);
        targetY = y0 + targetT * (y1 - y0);
        break;
      }
      remaining -= available;
      targetX = x1;
      targetY = y1;
      ++segment;
      segmentT = 0.0;
    }

    globalPathLookAheadID = std::min(segment + 1, pathSize - 1);
    goalX = targetX;
    goalY = targetY;
  }

  // 到达全局路径终点检测
  if (globalPathPointID == pathSize - 1) {
    float dx = globalPath.poses[pathSize - 1].pose.position.x - vehicleX;
    float dy = globalPath.poses[pathSize - 1].pose.position.y - vehicleY;
    if (sqrt(dx * dx + dy * dy) < goalReachedDis) {
      globalPathReceived = false;
      publishEmptyPath();
      if (enableHeadingAlign && goalPoseReceived) {
        // 进入朝向对齐阶段
        aligningHeading = true;
        RCLCPP_INFO(nh->get_logger(), "Reached end of global path, starting heading alignment to yaw=%.3f.", goalPoseYaw);
      } else {
        RCLCPP_INFO(nh->get_logger(), "Reached end of global path, stopping navigation.");
      }
    }
  }
}

/**
 * PLY文件头读取函数
 * 功能：
 *   - 解析PLY文件格式头
 *   - 返回文件中的点数
 */
int readPlyHeader(FILE *filePtr)
{
  char str[50];
  int val, pointNum;
  string strCur, strLast;
  
  // 逐行读取直到找到"end_header"
  while (strCur != "end_header") {
    val = fscanf(filePtr, "%s", str);
    if (val != 1) {
      RCLCPP_INFO(nh->get_logger(), "Error reading input files, exit.");
      exit(1);
    }

    strLast = strCur;
    strCur = string(str);

    // 找到"element vertex"后获取点数
    if (strCur == "vertex" && strLast == "element") {
      val = fscanf(filePtr, "%d", &pointNum);
      if (val != 1) {
        RCLCPP_INFO(nh->get_logger(), "Error reading input files, exit.");
        exit(1);
      }
    }
  }

  return pointNum;
}

/**
 * 读取起点路径文件 (startPaths.ply)
 * 功能：
 *   - 加载7条起点路径(每条对应一个初始方向)
 *   - 每条路径是一个点序列
 *   - 用作路径规划的基础
 */
void readStartPaths()
{
  string fileName = pathFolder + "/startPaths.ply";

  FILE *filePtr = fopen(fileName.c_str(), "r");
  if (filePtr == NULL) {
    RCLCPP_INFO(nh->get_logger(), "Cannot read input files, exit.");
    exit(1);
  }

  int pointNum = readPlyHeader(filePtr);

  pcl::PointXYZ point;
  int val1, val2, val3, val4, groupID;
  for (int i = 0; i < pointNum; i++) {
    val1 = fscanf(filePtr, "%f", &point.x);
    val2 = fscanf(filePtr, "%f", &point.y);
    val3 = fscanf(filePtr, "%f", &point.z);
    val4 = fscanf(filePtr, "%d", &groupID);

    if (val1 != 1 || val2 != 1 || val3 != 1 || val4 != 1) {
      RCLCPP_INFO(nh->get_logger(), "Error reading input files, exit.");
        exit(1);
    }

    // 将点添加到对应的分组
    if (groupID >= 0 && groupID < groupNum) {
      startPaths[groupID]->push_back(point);
    }
  }

  fclose(filePtr);
}

/**
 * 读取完整路径集合 (paths.ply) - 用于可视化
 * 功能：
 *   - 加载343条完整路径
 *   - 为了减少数据量,对路径进行采样(pointSkipNum=30)
 *   - 用于在RViz中可视化所有可行路径
 */
#if PLOTPATHSET == 1
void readPaths()
{
  string fileName = pathFolder + "/paths.ply";

  FILE *filePtr = fopen(fileName.c_str(), "r");
  if (filePtr == NULL) {
    RCLCPP_INFO(nh->get_logger(), "Cannot read input files, exit.");
    exit(1);
  }

  int pointNum = readPlyHeader(filePtr);

  pcl::PointXYZI point;
  int pointSkipNum = 30;  // 路径采样间隔
  int pointSkipCount = 0;
  int val1, val2, val3, val4, val5, pathID;
  for (int i = 0; i < pointNum; i++) {
    val1 = fscanf(filePtr, "%f", &point.x);
    val2 = fscanf(filePtr, "%f", &point.y);
    val3 = fscanf(filePtr, "%f", &point.z);
    val4 = fscanf(filePtr, "%d", &pathID);
    val5 = fscanf(filePtr, "%f", &point.intensity);

    if (val1 != 1 || val2 != 1 || val3 != 1 || val4 != 1 || val5 != 1) {
      RCLCPP_INFO(nh->get_logger(), "Error reading input files, exit.");
        exit(1);
    }

    // 采样添加点到对应路径
    if (pathID >= 0 && pathID < pathNum) {
      pointSkipCount++;
      if (pointSkipCount > pointSkipNum) {
        paths[pathID]->push_back(point);
        pointSkipCount = 0;
      }
    }
  }

  fclose(filePtr);
}
#endif

/**
 * 读取路径列表 (pathList.ply)
 * 功能：
 *   - 建立路径ID到分组ID的映射
 *   - 读取每条路径的终点方向(endDirPathList)
 *   - 这些信息用于路径评分
 */
void readPathList()
{
  string fileName = pathFolder + "/pathList.ply";

  FILE *filePtr = fopen(fileName.c_str(), "r");
  if (filePtr == NULL) {
    RCLCPP_INFO(nh->get_logger(), "Cannot read input files, exit.");
    exit(1);
  }

  // 验证路径数量
  if (pathNum != readPlyHeader(filePtr)) {
    RCLCPP_INFO(nh->get_logger(), "Incorrect path number, exit.");
    exit(1);
  }

  int val1, val2, val3, val4, val5, pathID, groupID;
  float endX, endY, endZ;
  for (int i = 0; i < pathNum; i++) {
    val1 = fscanf(filePtr, "%f", &endX);
    val2 = fscanf(filePtr, "%f", &endY);
    val3 = fscanf(filePtr, "%f", &endZ);
    val4 = fscanf(filePtr, "%d", &pathID);
    val5 = fscanf(filePtr, "%d", &groupID);

    if (val1 != 1 || val2 != 1 || val3 != 1 || val4 != 1 || val5 != 1) {
      RCLCPP_INFO(nh->get_logger(), "Error reading input files, exit.");
        exit(1);
    }

    if (pathID >= 0 && pathID < pathNum && groupID >= 0 && groupID < groupNum) {
      pathList[pathID] = groupID;  // 路径所属分组
      // 计算终点方向(相对于起点)
      endDirPathList[pathID] = 2.0 * atan2(endY, endX) * 180 / PI;
    }
  }

  fclose(filePtr);
}

/**
 * 读取网格对应表 (correspondences.txt)
 * 功能：
 *   - 建立网格体素到路径的对应关系
 *   - 用于快速查询哪些路径经过某个网格区域
 *   - 加速障碍物碰撞检测
 * 文件格式：
 *   - 每行格式: gridVoxelID pathID1 pathID2 ... -1
 *   - -1表示该网格对应的路径列表结束
 */
void readCorrespondences()
{
  string fileName = pathFolder + "/correspondences.txt";

  FILE *filePtr = fopen(fileName.c_str(), "r");
  if (filePtr == NULL) {
    RCLCPP_INFO(nh->get_logger(), "Cannot read input files, exit.");
    exit(1);
  }

  int val1, gridVoxelID, pathID;
  for (int i = 0; i < gridVoxelNum; i++) {
    val1 = fscanf(filePtr, "%d", &gridVoxelID);
    if (val1 != 1) {
      RCLCPP_INFO(nh->get_logger(), "Error reading input files, exit.");
        exit(1);
    }

    // 读取该网格对应的所有路径,直到遇到-1
    while (1) {
      val1 = fscanf(filePtr, "%d", &pathID);
      if (val1 != 1) {
        RCLCPP_INFO(nh->get_logger(), "Error reading input files, exit.");
          exit(1);
      }

      if (pathID != -1) {
        // 有效的路径ID
        if (gridVoxelID >= 0 && gridVoxelID < gridVoxelNum && pathID >= 0 && pathID < pathNum) {
          correspondences[gridVoxelID].push_back(pathID);
        }
      } else {
        // -1表示该网格的路径列表结束
        break;
      }
    }
  }

  fclose(filePtr);
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  nh = rclcpp::Node::make_shared("localPlanner");

  nh->declare_parameter<std::string>("pathFolder", pathFolder);
  nh->declare_parameter<double>("vehicleLength", vehicleLength);
  nh->declare_parameter<double>("vehicleWidth", vehicleWidth);
  nh->declare_parameter<double>("footprintSafetyMargin", footprintSafetyMargin);
  nh->declare_parameter<int>("footprintCheckSkip", footprintCheckSkip);
  nh->declare_parameter<double>("sensorOffsetX", sensorOffsetX);
  nh->declare_parameter<double>("sensorOffsetY", sensorOffsetY);
  nh->declare_parameter<bool>("twoWayDrive", twoWayDrive);
  nh->declare_parameter<double>("laserVoxelSize", laserVoxelSize);
  nh->declare_parameter<double>("terrainVoxelSize", terrainVoxelSize);
  nh->declare_parameter<bool>("useTerrainAnalysis", useTerrainAnalysis);
  nh->declare_parameter<bool>("checkObstacle", checkObstacle);
  nh->declare_parameter<bool>("checkRotObstacle", checkRotObstacle);
  nh->declare_parameter<double>("adjacentRange", adjacentRange);
  nh->declare_parameter<double>("obstacleHeightThre", obstacleHeightThre);
  nh->declare_parameter<double>("groundHeightThre", groundHeightThre);
  nh->declare_parameter<double>("costHeightThre", costHeightThre);
  nh->declare_parameter<double>("costScore", costScore);
  nh->declare_parameter<bool>("useCost", useCost);
  nh->declare_parameter<int>("pointPerPathThre", pointPerPathThre);
  nh->declare_parameter<double>("minRelZ", minRelZ);
  nh->declare_parameter<double>("maxRelZ", maxRelZ);
  nh->declare_parameter<double>("maxSpeed", maxSpeed);
  nh->declare_parameter<double>("dirWeight", dirWeight);
  nh->declare_parameter<double>("dirThre", dirThre);
  nh->declare_parameter<bool>("dirToVehicle", dirToVehicle);
  nh->declare_parameter<double>("pathSwitchHysteresis", pathSwitchHysteresis);
  nh->declare_parameter<double>("pathScale", pathScale);
  nh->declare_parameter<double>("minPathScale", minPathScale);
  nh->declare_parameter<double>("pathScaleStep", pathScaleStep);
  nh->declare_parameter<bool>("pathScaleBySpeed", pathScaleBySpeed);
  nh->declare_parameter<double>("minPathRange", minPathRange);
  nh->declare_parameter<double>("pathRangeStep", pathRangeStep);
  nh->declare_parameter<bool>("pathRangeBySpeed", pathRangeBySpeed);
  nh->declare_parameter<bool>("pathCropByGoal", pathCropByGoal);
  nh->declare_parameter<bool>("autonomyMode", autonomyMode);
  nh->declare_parameter<double>("autonomySpeed", autonomySpeed);
  nh->declare_parameter<double>("joyToSpeedDelay", joyToSpeedDelay);
  nh->declare_parameter<double>("joyToCheckObstacleDelay", joyToCheckObstacleDelay);
  nh->declare_parameter<double>("goalClearRange", goalClearRange);
  nh->declare_parameter<double>("goalX", goalX);
  nh->declare_parameter<double>("goalY", goalY);
  nh->declare_parameter<double>("globalPathLookAheadDis", globalPathLookAheadDis);
  nh->declare_parameter<int>("globalPathSearchPointNum", globalPathSearchPointNum);
  nh->declare_parameter<double>("goalReachedDis", goalReachedDis);
  nh->declare_parameter<bool>("enableHeadingAlign", enableHeadingAlign);
  nh->declare_parameter<double>("headingAlignThre", headingAlignThre);
  nh->declare_parameter<double>("headingAlignSpeed", headingAlignSpeed);
  nh->declare_parameter<double>("pathClearRadius", pathClearRadius);
  nh->declare_parameter<double>("pathClearHeight", pathClearHeight);

  nh->get_parameter("pathFolder", pathFolder);
  nh->get_parameter("vehicleLength", vehicleLength);
  nh->get_parameter("vehicleWidth", vehicleWidth);
  nh->get_parameter("footprintSafetyMargin", footprintSafetyMargin);
  nh->get_parameter("footprintCheckSkip", footprintCheckSkip);
  nh->get_parameter("sensorOffsetX", sensorOffsetX);
  nh->get_parameter("sensorOffsetY", sensorOffsetY);
  nh->get_parameter("twoWayDrive", twoWayDrive);
  nh->get_parameter("laserVoxelSize", laserVoxelSize);
  nh->get_parameter("terrainVoxelSize", terrainVoxelSize);
  nh->get_parameter("useTerrainAnalysis", useTerrainAnalysis);
  nh->get_parameter("checkObstacle", checkObstacle);
  nh->get_parameter("checkRotObstacle", checkRotObstacle);
  nh->get_parameter("adjacentRange", adjacentRange);
  nh->get_parameter("obstacleHeightThre", obstacleHeightThre);
  nh->get_parameter("groundHeightThre", groundHeightThre);
  nh->get_parameter("costHeightThre", costHeightThre);
  nh->get_parameter("costScore", costScore);
  nh->get_parameter("useCost", useCost);
  nh->get_parameter("pointPerPathThre", pointPerPathThre);
  nh->get_parameter("minRelZ", minRelZ);
  nh->get_parameter("maxRelZ", maxRelZ);
  nh->get_parameter("maxSpeed", maxSpeed);
  nh->get_parameter("dirWeight", dirWeight);
  nh->get_parameter("dirThre", dirThre);
  nh->get_parameter("dirToVehicle", dirToVehicle);
  nh->get_parameter("pathSwitchHysteresis", pathSwitchHysteresis);
  nh->get_parameter("pathScale", pathScale);
  nh->get_parameter("minPathScale", minPathScale);
  nh->get_parameter("pathScaleStep", pathScaleStep);
  nh->get_parameter("pathScaleBySpeed", pathScaleBySpeed);
  nh->get_parameter("minPathRange", minPathRange);
  nh->get_parameter("pathRangeStep", pathRangeStep);
  nh->get_parameter("pathRangeBySpeed", pathRangeBySpeed);
  nh->get_parameter("pathCropByGoal", pathCropByGoal);
  nh->get_parameter("autonomyMode", autonomyMode);
  nh->get_parameter("autonomySpeed", autonomySpeed);
  nh->get_parameter("joyToSpeedDelay", joyToSpeedDelay);
  nh->get_parameter("joyToCheckObstacleDelay", joyToCheckObstacleDelay);
  nh->get_parameter("goalClearRange", goalClearRange);
  nh->get_parameter("goalX", goalX);
  nh->get_parameter("goalY", goalY);
  nh->get_parameter("globalPathLookAheadDis", globalPathLookAheadDis);
  nh->get_parameter("globalPathSearchPointNum", globalPathSearchPointNum);
  nh->get_parameter("goalReachedDis", goalReachedDis);
  nh->get_parameter("enableHeadingAlign", enableHeadingAlign);
  nh->get_parameter("pathClearRadius", pathClearRadius);
  nh->get_parameter("pathClearHeight", pathClearHeight);
  nh->get_parameter("headingAlignThre", headingAlignThre);
  nh->get_parameter("headingAlignSpeed", headingAlignSpeed);

  // 里程计信息:车辆位姿
  auto subOdometry = nh->create_subscription<nav_msgs::msg::Odometry>("/state_estimation", 5, odometryHandler);

  // 激光点云:建图传感器数据
  auto subLaserCloud = nh->create_subscription<sensor_msgs::msg::PointCloud2>("/registered_scan", 5, laserCloudHandler);

  // 地形点云:高程图
  auto subTerrainCloud = nh->create_subscription<sensor_msgs::msg::PointCloud2>("/terrain_map", 5, terrainCloudHandler);

  // 摇杆输入:用户手动控制
  auto subJoystick = nh->create_subscription<sensor_msgs::msg::Joy>("/local_planner/joy", 5, joystickHandler);

  // 目标点:自主导航目标 (单点,与全局路径二选一)
  auto subGoal = nh->create_subscription<geometry_msgs::msg::PointStamped> ("/way_point", 5, goalHandler);

  // 全局路径:外部全局规划器发布的三维路径
  auto subGlobalPath = nh->create_subscription<nav_msgs::msg::Path>("/global_path", 5, globalPathHandler);

  // 终点朝向目标:到达路径终点后对齐机器人朝向
  auto subGoalPose = nh->create_subscription<geometry_msgs::msg::PoseStamped>("/goal_pose", 5, goalPoseHandler);

  // 速度指令:自主导航速度
  auto subSpeed = nh->create_subscription<std_msgs::msg::Float32>("/speed", 5, speedHandler);

  // 导航边界:禁飞区域
  auto subBoundary = nh->create_subscription<geometry_msgs::msg::PolygonStamped>("/navigation_boundary", 5, boundaryHandler);

  // 添加的障碍物:手动添加的额外障碍物
  rclcpp::QoS obs_qos(10);
  obs_qos.transient_local();
  obs_qos.reliable();
  obs_qos.keep_last(1);
  auto subAddedObstacles = nh->create_subscription<sensor_msgs::msg::PointCloud2>("/added_obstacles", obs_qos, addedObstaclesHandler);

  // 障碍物检查使能:自主模式下的检查指令
  auto subCheckObstacle = nh->create_subscription<std_msgs::msg::Bool>("/check_obstacle", 5, checkObstacleHandler);

  // 发布规划的路径
  pubPath = nh->create_publisher<nav_msgs::msg::Path>("/local_planner/path", 5);
  pubGoalDistance = nh->create_publisher<std_msgs::msg::Float32>("/local_planner/goal_distance", 5);

  // 发布朝向对齐时的旋转速度指令 (直接写入 /cmd_vel)
  pubCmdVelAlign = nh->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 5);

  // 发布裁剪后的地形点云
  pubTerrainCloud = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/terrain_cloud", 5);

  // 发布车辆 footprint（矩形，frame=base_link）
  pubFootprint = nh->create_publisher<geometry_msgs::msg::PolygonStamped>("/local_planner/footprint", 5);
  nav_msgs::msg::Path path;

  // 发布可行的路径集合(用于可视化)
  #if PLOTPATHSET == 1
  auto pubFreePaths = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/local_planner/free_paths", 2);
  #endif

  //auto pubLaserCloud = nh->create_publisher<sensor_msgs::msg::PointCloud2> ("/stacked_scans", 2);

  RCLCPP_INFO(nh->get_logger(), "Reading path files.");

  // ============================================================================
  // 初始化自主导航速度
  // ============================================================================
  if (autonomyMode) {
    joySpeed = autonomySpeed / maxSpeed;

    if (joySpeed < 0) joySpeed = 0;
    else if (joySpeed > 1.0) joySpeed = 1.0;
  }

  // ============================================================================
  // 初始化数据结构
  // ============================================================================
  // 初始化点云堆栈
  for (int i = 0; i < laserCloudStackNum; i++) {
    laserCloudStack[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
  }
  // 初始化起点路径数组
  for (int i = 0; i < groupNum; i++) {
    startPaths[i].reset(new pcl::PointCloud<pcl::PointXYZ>());
  }
  // 初始化完整路径数组(用于可视化)
  #if PLOTPATHSET == 1
  for (int i = 0; i < pathNum; i++) {
    paths[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
  }
  #endif
  // 初始化网格对应关系
  for (int i = 0; i < gridVoxelNum; i++) {
    correspondences[i].resize(0);
  }

  // ============================================================================
  // 设置点云过滤参数
  // ============================================================================
  laserDwzFilter.setLeafSize(laserVoxelSize, laserVoxelSize, laserVoxelSize);
  terrainDwzFilter.setLeafSize(terrainVoxelSize, terrainVoxelSize, terrainVoxelSize);

  // ============================================================================
  // 加载预生成的路径和配置信息
  // ============================================================================
  readStartPaths();
  #if PLOTPATHSET == 1
  readPaths();
  #endif
  readPathList();
  readCorrespondences();

  RCLCPP_INFO(nh->get_logger(), "Initialization complete.");

  // ============================================================================
  // 主事件循环 (100 Hz控制循环)
  // ============================================================================
  rclcpp::Rate rate(100);
  bool status = rclcpp::ok();

  // footprint 定时发布（10 Hz，不依赖目标状态）
  double footprintLastSec = 0.0;

  while (status) {
    rclcpp::spin_some(nh);  // 处理待处理的消息回调

    // ── 发布 footprint（10 Hz）────────────────────────────────────────────────
    double nowSec = nh->now().seconds();
    if (nowSec - footprintLastSec >= 0.1) {
      footprintLastSec = nowSec;
      publishFootprint();
    }

    // 无有效目标且无手动摇杆输入时，跳过全部规划计算
    bool goalActive = wayPointReceived || globalPathReceived;
    if (!goalActive && joySpeedRaw == 0 && !aligningHeading) {
      status = rclcpp::ok();
      rate.sleep();
      continue;
    }

    // ========================================================================
    // 朝向对齐阶段：到达全局路径终点位置后，旋转对齐 goal_pose 方向
    // ========================================================================
    if (aligningHeading && goalPoseReceived) {
      double yawError = goalPoseYaw - vehicleYaw;
      // 将误差归一化到 [-π, π]
      while (yawError > M_PI)  yawError -= 2.0 * M_PI;
      while (yawError < -M_PI) yawError += 2.0 * M_PI;

      if (fabs(yawError) < headingAlignThre) {
        // 对齐完成，发布零速度，清除状态
        geometry_msgs::msg::Twist stopCmd;
        pubCmdVelAlign->publish(stopCmd);
        aligningHeading = false;
        goalPoseReceived = false;
        RCLCPP_INFO(nh->get_logger(), "Heading alignment complete. Task finished.");
      } else {
        // 发布原地旋转速度
        geometry_msgs::msg::Twist rotCmd;
        rotCmd.angular.z = (yawError > 0 ? 1.0 : -1.0) * headingAlignSpeed;
        pubCmdVelAlign->publish(rotCmd);
      }
      status = rclcpp::ok();
      rate.sleep();
      continue;
    }

    // 检查是否收到了新的点云数据
    if (newLaserCloud || newTerrainCloud) {
      // ========================================================================
      // 第一阶段:点云融合与坐标变换
      // ========================================================================
      if (newLaserCloud) {
        newLaserCloud = false;

        // 更新点云堆栈(循环缓冲区)
        laserCloudStack[laserCloudCount]->clear();
        *laserCloudStack[laserCloudCount] = *laserCloudDwz;
        laserCloudCount = (laserCloudCount + 1) % laserCloudStackNum;

        // 融合所有堆栈中的点云
        plannerCloud->clear();
        for (int i = 0; i < laserCloudStackNum; i++) {
          *plannerCloud += *laserCloudStack[i];
        }
      }

      // 使用地形分析时的点云替换
      if (newTerrainCloud) {
        newTerrainCloud = false;

        plannerCloud->clear();
        *plannerCloud = *terrainCloudDwz;
      }

      // ========================================================================
      // 第二阶段:坐标转换(世界坐标系→车体坐标系)
      // ========================================================================
      // 计算车体姿态的旋转矩阵参数
      float sinVehicleYaw = sin(vehicleYaw);
      float cosVehicleYaw = cos(vehicleYaw);

      // 将点云从世界坐标系变换到车体坐标系,并按距离和高度进行裁剪
      pcl::PointXYZI point;
      plannerCloudCrop->clear();
      int plannerCloudSize = plannerCloud->points.size();
      for (int i = 0; i < plannerCloudSize; i++) {
        // 相对于车体中心的坐标
        float pointX1 = plannerCloud->points[i].x - vehicleX;
        float pointY1 = plannerCloud->points[i].y - vehicleY;
        float pointZ1 = plannerCloud->points[i].z - vehicleZ;

        // 旋转变换到车体坐标系(X轴指向前方,Y轴指向左侧)
        point.x = pointX1 * cosVehicleYaw + pointY1 * sinVehicleYaw;
        point.y = -pointX1 * sinVehicleYaw + pointY1 * cosVehicleYaw;
        point.z = pointZ1;
        point.intensity = plannerCloud->points[i].intensity;

        // 距离和高度阈值裁剪
        float dis = sqrt(point.x * point.x + point.y * point.y);
        if (dis < adjacentRange && ((point.z > minRelZ && point.z < maxRelZ) || useTerrainAnalysis)) {
          plannerCloudCrop->push_back(point);
        }
      }

      // 处理导航边界
      int boundaryCloudSize = boundaryCloud->points.size();
      for (int i = 0; i < boundaryCloudSize; i++) {
        point.x = ((boundaryCloud->points[i].x - vehicleX) * cosVehicleYaw 
                + (boundaryCloud->points[i].y - vehicleY) * sinVehicleYaw);
        point.y = (-(boundaryCloud->points[i].x - vehicleX) * sinVehicleYaw 
                + (boundaryCloud->points[i].y - vehicleY) * cosVehicleYaw);
        point.z = boundaryCloud->points[i].z;
        point.intensity = boundaryCloud->points[i].intensity;

        float dis = sqrt(point.x * point.x + point.y * point.y);
        if (dis < adjacentRange) {
          plannerCloudCrop->push_back(point);
        }
      }

      // 处理手动添加的障碍物
      int addedObstaclesSize = addedObstacles->points.size();
      for (int i = 0; i < addedObstaclesSize; i++) {
        point.x = ((addedObstacles->points[i].x - vehicleX) * cosVehicleYaw 
                + (addedObstacles->points[i].y - vehicleY) * sinVehicleYaw);
        point.y = (-(addedObstacles->points[i].x - vehicleX) * sinVehicleYaw 
                + (addedObstacles->points[i].y - vehicleY) * cosVehicleYaw);
        point.z = addedObstacles->points[i].z;
        point.intensity = addedObstacles->points[i].intensity;

        float dis = sqrt(point.x * point.x + point.y * point.y);
        if (dis < adjacentRange) {
          plannerCloudCrop->push_back(point);
        }
      }

      float pathRange = adjacentRange;
      if (pathRangeBySpeed) pathRange = adjacentRange * joySpeed;
      if (pathRange < minPathRange) pathRange = minPathRange;
      float relativeGoalDis = adjacentRange;

      // ========================================================================
      // 目标导航：仅在收到 /way_point 或 /global_path 后才进行目标导向规划
      // ========================================================================
      if (goalActive) {
        // 更新全局路径的中间目标点（内部可能将 globalPathReceived 置 false 并发布空路径）
        if (globalPathReceived) {
          updateGoalFromGlobalPath();
        }

        // way_point 到达检测
        if (wayPointReceived) {
          float dx = goalX - vehicleX;
          float dy = goalY - vehicleY;
          if (sqrt(dx * dx + dy * dy) < goalReachedDis) {
            wayPointReceived = false;
            publishEmptyPath();
            RCLCPP_INFO(nh->get_logger(), "Reached waypoint goal, stopping navigation.");
          }
        }

        // 刷新 goalActive：两个到达检测均可能将标志位清除
        goalActive = wayPointReceived || globalPathReceived;
      }

      // 到达终点后本帧不再进行路径规划，避免紧接着发出非空路径
      if (!goalActive && joySpeedRaw == 0) {
        status = rclcpp::ok();
        rate.sleep();
        continue;
      }

      if (wayPointReceived) {
        goalRemainDis = hypot(goalX - vehicleX, goalY - vehicleY);
      }
      if (goalActive && goalRemainDis >= 0.0) {
        publishGoalRemain(goalRemainDis);
      }

      if (autonomyMode) {
        float relativeGoalX = ((goalX - vehicleX) * cosVehicleYaw + (goalY - vehicleY) * sinVehicleYaw);
        float relativeGoalY = (-(goalX - vehicleX) * sinVehicleYaw + (goalY - vehicleY) * cosVehicleYaw);

        relativeGoalDis = sqrt(relativeGoalX * relativeGoalX + relativeGoalY * relativeGoalY);
        joyDir = atan2(relativeGoalY, relativeGoalX) * 180 / PI;

        if (!twoWayDrive) {
          if (joyDir > 90.0) joyDir = 90.0;
          else if (joyDir < -90.0) joyDir = -90.0;
        }
      }

      bool pathFound = false;
      float defPathScale = pathScale;
      if (pathScaleBySpeed) pathScale = defPathScale * joySpeed;
      if (pathScale < minPathScale) pathScale = minPathScale;

      while (pathScale >= minPathScale && pathRange >= minPathRange) {
        for (int i = 0; i < 36 * pathNum; i++) {
          clearPathList[i] = 0;
          pathPenaltyList[i] = 0;
        }
        for (int i = 0; i < 36 * groupNum; i++) {
          clearPathPerGroupScore[i] = 0;
        }
        float minObsAngCW = -180.0;
        float minObsAngCCW = 180.0;
        float diameter = sqrt(vehicleLength / 2.0 * vehicleLength / 2.0 + vehicleWidth / 2.0 * vehicleWidth / 2.0);
        float angOffset = atan2(vehicleWidth, vehicleLength) * 180.0 / PI;
        int plannerCloudCropSize = plannerCloudCrop->points.size();
        for (int i = 0; i < plannerCloudCropSize; i++) {
          float x = plannerCloudCrop->points[i].x / pathScale;
          float y = plannerCloudCrop->points[i].y / pathScale;
          float h = plannerCloudCrop->points[i].intensity;
          float dis = sqrt(x * x + y * y);

          if (dis < pathRange / pathScale && (dis <= (relativeGoalDis + goalClearRange) / pathScale || !pathCropByGoal) && checkObstacle) {
            for (int rotDir = 0; rotDir < 36; rotDir++) {
              float rotAng = (10.0 * rotDir - 180.0) * PI / 180;
              float angDiff = fabs(joyDir - (10.0 * rotDir - 180.0));
              if (angDiff > 180.0) {
                angDiff = 360.0 - angDiff;
              }
              if ((angDiff > dirThre && !dirToVehicle) || (fabs(10.0 * rotDir - 180.0) > dirThre && fabs(joyDir) <= 90.0 && dirToVehicle) ||
                  ((10.0 * rotDir > dirThre && 360.0 - 10.0 * rotDir > dirThre) && fabs(joyDir) > 90.0 && dirToVehicle)) {
                continue;
              }

              float x2 = cos(rotAng) * x + sin(rotAng) * y;
              float y2 = -sin(rotAng) * x + cos(rotAng) * y;

              float scaleY = x2 / gridVoxelOffsetX + searchRadius / gridVoxelOffsetY 
                             * (gridVoxelOffsetX - x2) / gridVoxelOffsetX;

              int indX = int((gridVoxelOffsetX + gridVoxelSize / 2 - x2) / gridVoxelSize);
              int indY = int((gridVoxelOffsetY + gridVoxelSize / 2 - y2 / scaleY) / gridVoxelSize);
              if (indX >= 0 && indX < gridVoxelNumX && indY >= 0 && indY < gridVoxelNumY) {
                int ind = gridVoxelNumY * indX + indY;
                int blockedPathByVoxelNum = correspondences[ind].size();
                for (int j = 0; j < blockedPathByVoxelNum; j++) {
                  if (h > obstacleHeightThre || !useTerrainAnalysis) {
                    clearPathList[pathNum * rotDir + correspondences[ind][j]]++;
                  } else {
                    if (pathPenaltyList[pathNum * rotDir + correspondences[ind][j]] < h && h > groundHeightThre) {
                      pathPenaltyList[pathNum * rotDir + correspondences[ind][j]] = h;
                    }
                  }
                }
              }
            }
          }

          if (dis < diameter / pathScale && (fabs(x) > vehicleLength / pathScale / 2.0 || fabs(y) > vehicleWidth / pathScale / 2.0) && 
              (h > obstacleHeightThre || !useTerrainAnalysis) && checkRotObstacle) {
            float angObs = atan2(y, x) * 180.0 / PI;
            if (angObs > 0) {
              if (minObsAngCCW > angObs - angOffset) minObsAngCCW = angObs - angOffset;
              if (minObsAngCW < angObs + angOffset - 180.0) minObsAngCW = angObs + angOffset - 180.0;
            } else {
              if (minObsAngCW < angObs + angOffset) minObsAngCW = angObs + angOffset;
              if (minObsAngCCW > 180.0 + angObs - angOffset) minObsAngCCW = 180.0 + angObs - angOffset;
            }
          }
        }

        if (minObsAngCW > 0) minObsAngCW = 0;
        if (minObsAngCCW < 0) minObsAngCCW = 0;

        for (int i = 0; i < 36 * pathNum; i++) {
          int rotDir = int(i / pathNum);
          float angDiff = fabs(joyDir - (10.0 * rotDir - 180.0));
          if (angDiff > 180.0) {
            angDiff = 360.0 - angDiff;
          }
          if ((angDiff > dirThre && !dirToVehicle) || (fabs(10.0 * rotDir - 180.0) > dirThre && fabs(joyDir) <= 90.0 && dirToVehicle) ||
              ((10.0 * rotDir > dirThre && 360.0 - 10.0 * rotDir > dirThre) && fabs(joyDir) > 90.0 && dirToVehicle)) {
            continue;
          }

          if (clearPathList[i] < pointPerPathThre) {
            float penaltyScore = 1.0 - pathPenaltyList[i] / costHeightThre;
            if (penaltyScore < costScore) penaltyScore = costScore;

            float dirDiff = fabs(joyDir - endDirPathList[i % pathNum] - (10.0 * rotDir - 180.0));
            if (dirDiff > 360.0) {
              dirDiff -= 360.0;
            }
            if (dirDiff > 180.0) {
              dirDiff = 360.0 - dirDiff;
            }

            float rotDirW;
            if (rotDir < 18) rotDirW = fabs(fabs(rotDir - 9) + 1);
            else rotDirW = fabs(fabs(rotDir - 27) + 1);
            float score = (1 - sqrt(sqrt(dirWeight * dirDiff))) * rotDirW * rotDirW * rotDirW * rotDirW * penaltyScore;
            if (score > 0) {
              clearPathPerGroupScore[groupNum * rotDir + pathList[i % pathNum]] += score;
            }
          }
        }

        float maxScore = 0;
        int selectedGroupID = -1;
        int footprintBlocked[36 * groupNum];
        std::fill_n(footprintBlocked, 36 * groupNum, -1);
        double footprintPathDistance = std::min(pathRange, relativeGoalDis);
        for (int i = 0; i < 36 * groupNum; i++) {
          int rotDir = int(i / groupNum);
          float rotAng = (10.0 * rotDir - 180.0) * PI / 180;
          float rotDeg = 10.0 * rotDir;
          if (rotDeg > 180.0) rotDeg -= 360.0;
          bool rotationAllowed =
              (rotAng * 180.0 / PI > minObsAngCW &&
               rotAng * 180.0 / PI < minObsAngCCW) ||
              (rotDeg > minObsAngCW && rotDeg < minObsAngCCW && twoWayDrive) ||
              !checkRotObstacle;
          if (maxScore < clearPathPerGroupScore[i] && rotationAllowed) {
            footprintBlocked[i] = isGroupFootprintBlocked(
                i % groupNum, rotAng, pathScale, footprintPathDistance) ? 1 : 0;
          }
          if (maxScore < clearPathPerGroupScore[i] && rotationAllowed &&
              footprintBlocked[i] == 0) {
            maxScore = clearPathPerGroupScore[i];
            selectedGroupID = i;
          }
        }

        // 路径选择滞回：当上一帧路径仍可行且分数与本帧最优值接近时，
        // 保持上一帧结果，避免左右两侧候选路径评分接近时来回跳变。
        if (lastSelectedPathID >= 0 && lastSelectedPathID < 36 * groupNum) {
          int lastRotDir = lastSelectedPathID / groupNum;
          float lastRotAng = (10.0 * lastRotDir - 180.0) * PI / 180.0;
          float lastRotDeg = 10.0 * lastRotDir;
          if (lastRotDeg > 180.0) lastRotDeg -= 360.0;
          bool lastRotationAllowed =
              (lastRotAng * 180.0 / PI > minObsAngCW &&
               lastRotAng * 180.0 / PI < minObsAngCCW) ||
              (lastRotDeg > minObsAngCW && lastRotDeg < minObsAngCCW && twoWayDrive) ||
              !checkRotObstacle;
          if (footprintBlocked[lastSelectedPathID] < 0 && lastRotationAllowed) {
            footprintBlocked[lastSelectedPathID] = isGroupFootprintBlocked(
                lastSelectedPathID % groupNum, lastRotAng, pathScale,
                footprintPathDistance) ? 1 : 0;
          }
          float lastScore = clearPathPerGroupScore[lastSelectedPathID];
          if (lastRotationAllowed && footprintBlocked[lastSelectedPathID] == 0 &&
              lastScore > 0.0 &&
              lastScore >= maxScore * pathSwitchHysteresis) {
            selectedGroupID = lastSelectedPathID;
          }
        }

        if (selectedGroupID >= 0) {
          lastSelectedPathID = selectedGroupID;
          int rotDir = int(selectedGroupID / groupNum);
          float rotAng = (10.0 * rotDir - 180.0) * PI / 180;

          selectedGroupID = selectedGroupID % groupNum;
          int selectedPathLength = startPaths[selectedGroupID]->points.size();
          path.poses.resize(selectedPathLength);
          for (int i = 0; i < selectedPathLength; i++) {
            float x = startPaths[selectedGroupID]->points[i].x;
            float y = startPaths[selectedGroupID]->points[i].y;
            float z = startPaths[selectedGroupID]->points[i].z;
            float dis = sqrt(x * x + y * y);

            if (dis <= pathRange / pathScale && dis <= relativeGoalDis / pathScale) {
              path.poses[i].pose.position.x = pathScale * (cos(rotAng) * x - sin(rotAng) * y);
              path.poses[i].pose.position.y = pathScale * (sin(rotAng) * x + cos(rotAng) * y);
              path.poses[i].pose.position.z = pathScale * z;
            } else {
              path.poses.resize(i);
              break;
            }
          }

          path.header.stamp = rclcpp::Time(static_cast<uint64_t>(odomTime * 1e9));
          path.header.frame_id = "base_link";
          pubPath->publish(path);

          #if PLOTPATHSET == 1
          freePaths->clear();
          for (int i = 0; i < 36 * pathNum; i++) {
            int rotDir = int(i / pathNum);
            float rotAng = (10.0 * rotDir - 180.0) * PI / 180;
            float rotDeg = 10.0 * rotDir;
            if (rotDeg > 180.0) rotDeg -= 360.0;
            float angDiff = fabs(joyDir - (10.0 * rotDir - 180.0));
            if (angDiff > 180.0) {
              angDiff = 360.0 - angDiff;
            }
            if ((angDiff > dirThre && !dirToVehicle) || (fabs(10.0 * rotDir - 180.0) > dirThre && fabs(joyDir) <= 90.0 && dirToVehicle) ||
                ((10.0 * rotDir > dirThre && 360.0 - 10.0 * rotDir > dirThre) && fabs(joyDir) > 90.0 && dirToVehicle) || 
                !((rotAng * 180.0 / PI > minObsAngCW && rotAng * 180.0 / PI < minObsAngCCW) || 
                (rotDeg > minObsAngCW && rotDeg < minObsAngCCW && twoWayDrive) || !checkRotObstacle)) {
              continue;
            }

            if (clearPathList[i] < pointPerPathThre) {
              int freePathLength = paths[i % pathNum]->points.size();
              for (int j = 0; j < freePathLength; j++) {
                point = paths[i % pathNum]->points[j];

                float x = point.x;
                float y = point.y;
                float z = point.z;

                float dis = sqrt(x * x + y * y);
                if (dis <= pathRange / pathScale && (dis <= (relativeGoalDis + goalClearRange) / pathScale || !pathCropByGoal)) {
                  point.x = pathScale * (cos(rotAng) * x - sin(rotAng) * y);
                  point.y = pathScale * (sin(rotAng) * x + cos(rotAng) * y);
                  point.z = pathScale * z;
                  point.intensity = 1.0;

                  freePaths->push_back(point);
                }
              }
            }
          }

          sensor_msgs::msg::PointCloud2 freePaths2;
          pcl::toROSMsg(*freePaths, freePaths2);
          freePaths2.header.stamp = rclcpp::Time(static_cast<uint64_t>(odomTime * 1e9));
          freePaths2.header.frame_id = "base_link";
          pubFreePaths->publish(freePaths2);
          #endif
        }

        if (selectedGroupID < 0) {
          if (pathScale >= minPathScale + pathScaleStep) {
            pathScale -= pathScaleStep;
            pathRange = adjacentRange * pathScale / defPathScale;
          } else {
            pathRange -= pathRangeStep;
          }
        } else {
          pathFound = true;
          break;
        }
      }
      pathScale = defPathScale;

      if (!pathFound) {
        path.poses.resize(1);
        path.poses[0].pose.position.x = 0;
        path.poses[0].pose.position.y = 0;
        path.poses[0].pose.position.z = 0;

        path.header.stamp = rclcpp::Time(static_cast<uint64_t>(odomTime * 1e9));
        path.header.frame_id = "base_link";
        pubPath->publish(path);

        #if PLOTPATHSET == 1
        freePaths->clear();
        sensor_msgs::msg::PointCloud2 freePaths2;
        pcl::toROSMsg(*freePaths, freePaths2);
        freePaths2.header.stamp = rclcpp::Time(static_cast<uint64_t>(odomTime * 1e9));
        freePaths2.header.frame_id = "base_link";
        pubFreePaths->publish(freePaths2);
        #endif
      }

      /*sensor_msgs::msg::PointCloud2 plannerCloud2;
      pcl::toROSMsg(*plannerCloudCrop, plannerCloud2);
      plannerCloud2.header.stamp = rclcpp::Time(static_cast<uint64_t>(odomTime * 1e9));
      plannerCloud2.header.frame_id = "base_link";
      pubLaserCloud->publish(plannerCloud2);*/
    }

    status = rclcpp::ok();
    rate.sleep();
  }

  return 0;
}
