/**
 * ============================================================================
 * 路径追踪器 (Path Follower)
 * ============================================================================
 * 功能说明：
 *   - 根据规划器发送的路径输出速度和转角控制命令
 *   - 采用前向视界(look-ahead)方法进行路径追踪
 *   - 动态调整速度:根据目标距离、倾斜度、加速度限制等因素
 *   - 支持双向驾驶,可根据需要自动切换前进/后退
 *   - 安全机制:倾斜度检测(防止翻覆)、安全停止指令、旋转限制等
 * 
 * 核心控制模式：
 *   1. 路径追踪：根据偏差计算转角和速度
 *   2. 速度控制：平滑的加速/减速,防止突变
 *   3. 安全保护：倾斜度监测、自动减速、强制停止
 *   4. 方向切换：双向驾驶时自动判断前进/后退
 * ============================================================================
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp/clock.hpp"
#include "builtin_interfaces/msg/time.hpp"

#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <sensor_msgs/msg/joy.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int8.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
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

// ============================================================================
// 传感器参数
// ============================================================================
double sensorOffsetX = 0;      // 传感器相对车体中心的X偏移
double sensorOffsetY = 0;      // 传感器相对车体中心的Y偏移

// ============================================================================
// 控制输出参数
// ============================================================================
int pubSkipNum = 1;            // 发布频率减频系数(每pubSkipNum个周期发布一次)
int pubSkipCount = 0;          // 发布计数器
bool twoWayDrive = true;       // 是否支持双向驾驶

// ============================================================================
// 路径追踪参数
// ============================================================================
double lookAheadDis = 0.5;     // 前向视界距离(选择目标点的距离)
double lookAheadTime = 0.8;    // 按速度增加前视距离的时间系数(s)
double maxLookAheadDis = 1.5;  // 最大前视距离(m)

// ============================================================================
// 转向控制参数 (PD控制器的转向增益)
// ============================================================================
double yawRateGain = 7.5;      // 正常运动时的转向增益
double stopYawRateGain = 7.5;  // 停止时的转向增益(可能更高以加快对准)
double maxYawRate = 45.0;      // 最大转向角速度(度/秒)
double yawRateFilterAlpha = 0.2; // 角速度一阶低通系数(0~1)
double maxYawAccel = 1.0;      // 最大角加速度(rad/s^2)

// ============================================================================
// 速度控制参数
// ============================================================================
double maxSpeed = 1.0;         // 最大速度(m/s)
double maxAccel = 1.0;         // 最大加速度(作为增量,单位:m/s/100ms)

// ============================================================================
// 双向驾驶参数
// ============================================================================
double switchTimeThre = 1.0;   // 前进/后退切换时间阈值(防止频繁切换)
double dirDiffThre = 0.1;      // 方向偏差阈值,低于该值才进行加速

// ============================================================================
// 停止相关参数
// ============================================================================
double stopDisThre = 0.2;      // 停止距离阈值(到达目标时的距离)
double slowDwnDisThre = 1.0;   // 开始减速的距离阈值
double brakeAccel = 0.22;      // 终点刹车用的保守减速度(m/s^2)，越小减速距离越长
double brakeLagTime = 0.55;    // 速度跟踪滞后补偿(s)

// ============================================================================
// 倾斜度检测与减速参数
// ============================================================================
bool useInclRateToSlow = false;    // 是否使用倾斜角速率触发减速
double inclRateThre = 120.0;       // 倾斜角速率阈值(度/秒)
double slowRate1 = 0.25;           // 第一阶段减速率
double slowRate2 = 0.5;            // 第二阶段减速率
double slowTime1 = 2.0;            // 第一阶段持续时间(秒)
double slowTime2 = 2.0;            // 第二阶段持续时间(秒)

// ============================================================================
// 倾斜度停止参数
// ============================================================================
bool useInclToStop = false;        // 是否使用倾斜度触发停止
double inclThre = 45.0;            // 倾斜度阈值(度)
double stopTime = 5.0;             // 停止持续时间(秒)

// ============================================================================
// 旋转限制参数
// ============================================================================
bool noRotAtStop = false;          // 在停止时是否禁止旋转
bool noRotAtGoal = true;           // 在到达目标时是否禁止旋转

// ============================================================================
// 自主导航参数
// ============================================================================
bool autonomyMode = false;         // 自主模式标志
double autonomySpeed = 1.0;        // 自主导航速度
double joyToSpeedDelay = 2.0;      // 从摇杆到速度指令的延迟

// ============================================================================
// 用户输入变量
// ============================================================================
float joySpeed = 0;                // 摇杆速度指令(0-1)
float joySpeedRaw = 0;             // 原始摇杆速度
float joyYaw = 0;                  // 摇杆转向指令
int safetyStop = 0;                // 安全停止标志(0=正常, 1=减速, 2=停止转向)

// ============================================================================
// 车辆位姿状态 (当前)
// ============================================================================
float vehicleX = 0, vehicleY = 0, vehicleZ = 0;          // 位置
float vehicleRoll = 0, vehiclePitch = 0, vehicleYaw = 0; // 姿态

// ============================================================================
// 车辆位姿记录 (路径收到时的位姿)
// ============================================================================
// 这些变量用于建立路径参考坐标系(相对坐标)
float vehicleXRec = 0, vehicleYRec = 0, vehicleZRec = 0;
float vehicleRollRec = 0, vehiclePitchRec = 0, vehicleYawRec = 0;

// ============================================================================
// 控制命令变量
// ============================================================================
float vehicleYawRate = 0;          // 转向角速度指令(rad/s)
float vehicleSpeed = 0;            // 速度指令(m/s)

// ============================================================================
// 时间戳和状态变量
// ============================================================================
double odomTime = 0;              // 最后收到的里程计时间戳
double joyTime = 0;               // 最后收到的摇杆时间戳
double slowInitTime = 0;          // 减速启动时间(倾斜度触发)
double stopInitTime = false;      // 停止启动时间(倾斜度触发)
int pathPointID = 0;              // 当前追踪的路径点索引
bool pathInit = false;            // 是否收到有效的路径
bool navFwd = true;               // 当前导航方向(true=前进, false=后退)
double switchTime = 0;            // 上次方向切换的时间
double goalRemainDis = -1.0;     // 到最终目标的剩余距离(m)，<0 表示无效
double goalRemainTime = -1.0;    // 最近一次收到剩余距离的时间(s)

nav_msgs::msg::Path path;      // 当前追踪的路径
rclcpp::Node::SharedPtr nh;    // ROS 2 节点指针
rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pubSpeed;  // 速度指令发布器(全局，空路径时发布停止)

/**
 * 里程计消息回调函数
 * 功能：
 *   - 提取车辆当前位姿(位置和姿态)
 *   - 根据传感器偏移调整到车体中心
 *   - 检测倾斜度(用于安全保护)
 *   - 检测倾斜角速率(用于减速保护)
 */
void odomHandler(const nav_msgs::msg::Odometry::ConstSharedPtr odomIn)
{
  odomTime = rclcpp::Time(odomIn->header.stamp).seconds();
  double roll, pitch, yaw;
  geometry_msgs::msg::Quaternion geoQuat = odomIn->pose.pose.orientation;
  // 四元数转欧拉角(RPY)
  tf2::Matrix3x3(tf2::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);

  vehicleRoll = roll;
  vehiclePitch = pitch;
  vehicleYaw = yaw;
  // 根据传感器偏移调整车体中心位置
  vehicleX = odomIn->pose.pose.position.x - cos(yaw) * sensorOffsetX + sin(yaw) * sensorOffsetY;
  vehicleY = odomIn->pose.pose.position.y - sin(yaw) * sensorOffsetX - cos(yaw) * sensorOffsetY;
  vehicleZ = odomIn->pose.pose.position.z;

  // 检测倾斜度:如果超过阈值,启用停止保护
  if ((fabs(roll) > inclThre * PI / 180.0 || fabs(pitch) > inclThre * PI / 180.0) && useInclToStop) {
    stopInitTime = rclcpp::Time(odomIn->header.stamp).seconds();
  }

  // 检测倾斜角速率:如果超过阈值,启用减速保护
  if ((fabs(odomIn->twist.twist.angular.x) > inclRateThre * PI / 180.0 || fabs(odomIn->twist.twist.angular.y) > inclRateThre * PI / 180.0) && useInclRateToSlow) {
    slowInitTime = rclcpp::Time(odomIn->header.stamp).seconds();
  }
}

/**
 * 路径消息回调函数
 * 功能：
 *   - 接收新的规划路径
 *   - 记录接收路径时的车辆位姿(作为参考)
 *   - 重置路径追踪状态(pathPointID=0, pathInit=true)
 * 注意：
 *   - 路径是在车体坐标系中表示的相对坐标
 *   - 所有后续追踪都相对于接收路径时的车体坐标系
 */
void pathHandler(const nav_msgs::msg::Path::ConstSharedPtr pathIn)
{
  int pathSize = pathIn->poses.size();
  RCLCPP_DEBUG(nh->get_logger(), "Received new path with %d points.", pathSize);

  // 空路径表示目标已到达，停止追踪
  if (pathSize == 0) {
    pathInit = false;
    // 立即发布 0 速度，防止机器人按上一帧速度继续前进
    geometry_msgs::msg::Twist stopCmd;
    pubSpeed->publish(stopCmd);
    RCLCPP_INFO(nh->get_logger(), "Received empty path, publishing zero velocity.");
    return;
  }

  path.poses.resize(pathSize);
  
  // 复制路径点(只需要位置,不需要姿态)
  for (int i = 0; i < pathSize; i++) {
    path.poses[i].pose.position.x = pathIn->poses[i].pose.position.x;
    path.poses[i].pose.position.y = pathIn->poses[i].pose.position.y;
    path.poses[i].pose.position.z = pathIn->poses[i].pose.position.z;
  }

  // 记录此刻的车辆位姿(作为参考坐标系)
  vehicleXRec = vehicleX;
  vehicleYRec = vehicleY;
  vehicleZRec = vehicleZ;
  vehicleRollRec = vehicleRoll;
  vehiclePitchRec = vehiclePitch;
  vehicleYawRec = vehicleYaw;

  // 重置路径追踪状态
  pathPointID = 0;
  pathInit = true;
}

/**
 * 摇杆输入处理回调函数
 * 功能：
 *   - 提取摇杆速度和转向指令
 *   - 处理自主模式切换
 *   - 处理停止时的旋转禁止
 */
void joystickHandler(const sensor_msgs::msg::Joy::ConstSharedPtr joy)
{
  joyTime = nh->now().seconds();
  // 计算摇杆速度幅度(使用X-Y轴的矢量模)
  joySpeedRaw = sqrt(joy->axes[3] * joy->axes[3] + joy->axes[4] * joy->axes[4]);
  joySpeed = joySpeedRaw;
  if (joySpeed > 1.0) joySpeed = 1.0;
  if (joy->axes[4] == 0) joySpeed = 0;
  
  // 提取转向指令(摇杆X轴)
  joyYaw = joy->axes[3];
  // 停止时禁止旋转
  if (joySpeed == 0 && noRotAtStop) joyYaw = 0;

  // 是否支持后退
  if (joy->axes[4] < 0 && !twoWayDrive) {
    joySpeed = 0;
    joyYaw = 0;
  }

  // 摇杆轴[2]切换自主模式(>-0.1为手动模式, ≤-0.1为自主模式)
  if (joy->axes[2] > -0.1) {
    autonomyMode = false;
  } else {
    autonomyMode = true;
  }
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
 * 安全停止指令处理回调函数
 * 功能：
 *   - safetyStop=0: 正常运行
 *   - safetyStop=1: 停止速度(仍可旋转)
 *   - safetyStop≥2: 停止速度和转向(完全停止)
 */
void stopHandler(const std_msgs::msg::Int8::ConstSharedPtr stop)
{
  safetyStop = stop->data;
}

// 规划器发布的到最终目标剩余距离。局部轨迹只有约 1.25 m，
// 且每周期都会从车体原点重发，单靠轨迹末端无法为 1 m/s 提前刹车。
void goalDistanceHandler(const std_msgs::msg::Float32::ConstSharedPtr msg)
{
  goalRemainDis = msg->data;
  goalRemainTime = nh->now().seconds();
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  nh = rclcpp::Node::make_shared("pathFollower");

  nh->declare_parameter<double>("sensorOffsetX", sensorOffsetX);
  nh->declare_parameter<double>("sensorOffsetY", sensorOffsetY);
  nh->declare_parameter<int>("pubSkipNum", pubSkipNum);
  nh->declare_parameter<bool>("twoWayDrive", twoWayDrive);
  nh->declare_parameter<double>("lookAheadDis", lookAheadDis);
  nh->declare_parameter<double>("lookAheadTime", lookAheadTime);
  nh->declare_parameter<double>("maxLookAheadDis", maxLookAheadDis);
  nh->declare_parameter<double>("yawRateGain", yawRateGain);
  nh->declare_parameter<double>("stopYawRateGain", stopYawRateGain);
  nh->declare_parameter<double>("maxYawRate", maxYawRate);
  nh->declare_parameter<double>("yawRateFilterAlpha", yawRateFilterAlpha);
  nh->declare_parameter<double>("maxYawAccel", maxYawAccel);
  nh->declare_parameter<double>("maxSpeed", maxSpeed);
  nh->declare_parameter<double>("maxAccel", maxAccel);
  nh->declare_parameter<double>("switchTimeThre", switchTimeThre);
  nh->declare_parameter<double>("dirDiffThre", dirDiffThre);
  nh->declare_parameter<double>("stopDisThre", stopDisThre);
  nh->declare_parameter<double>("slowDwnDisThre", slowDwnDisThre);
  nh->declare_parameter<double>("brakeAccel", brakeAccel);
  nh->declare_parameter<double>("brakeLagTime", brakeLagTime);
  nh->declare_parameter<bool>("useInclRateToSlow", useInclRateToSlow);
  nh->declare_parameter<double>("inclRateThre", inclRateThre);
  nh->declare_parameter<double>("slowRate1", slowRate1);
  nh->declare_parameter<double>("slowRate2", slowRate2);
  nh->declare_parameter<double>("slowTime1", slowTime1);
  nh->declare_parameter<double>("slowTime2", slowTime2);
  nh->declare_parameter<bool>("useInclToStop", useInclToStop);
  nh->declare_parameter<double>("inclThre", inclThre);
  nh->declare_parameter<double>("stopTime", stopTime);
  nh->declare_parameter<bool>("noRotAtStop", noRotAtStop);
  nh->declare_parameter<bool>("noRotAtGoal", noRotAtGoal);
  nh->declare_parameter<bool>("autonomyMode", autonomyMode);
  nh->declare_parameter<double>("autonomySpeed", autonomySpeed);
  nh->declare_parameter<double>("joyToSpeedDelay", joyToSpeedDelay);

  nh->get_parameter("sensorOffsetX", sensorOffsetX);
  nh->get_parameter("sensorOffsetY", sensorOffsetY);
  nh->get_parameter("pubSkipNum", pubSkipNum);
  nh->get_parameter("twoWayDrive", twoWayDrive);
  nh->get_parameter("lookAheadDis", lookAheadDis);
  nh->get_parameter("lookAheadTime", lookAheadTime);
  nh->get_parameter("maxLookAheadDis", maxLookAheadDis);
  nh->get_parameter("yawRateGain", yawRateGain);
  nh->get_parameter("stopYawRateGain", stopYawRateGain);
  nh->get_parameter("maxYawRate", maxYawRate);
  nh->get_parameter("yawRateFilterAlpha", yawRateFilterAlpha);
  nh->get_parameter("maxYawAccel", maxYawAccel);
  nh->get_parameter("maxSpeed", maxSpeed);
  nh->get_parameter("maxAccel", maxAccel);
  nh->get_parameter("switchTimeThre", switchTimeThre);
  nh->get_parameter("dirDiffThre", dirDiffThre);
  nh->get_parameter("stopDisThre", stopDisThre);
  nh->get_parameter("slowDwnDisThre", slowDwnDisThre);
  nh->get_parameter("brakeAccel", brakeAccel);
  nh->get_parameter("brakeLagTime", brakeLagTime);
  nh->get_parameter("useInclRateToSlow", useInclRateToSlow);
  nh->get_parameter("inclRateThre", inclRateThre);
  nh->get_parameter("slowRate1", slowRate1);
  nh->get_parameter("slowRate2", slowRate2);
  nh->get_parameter("slowTime1", slowTime1);
  nh->get_parameter("slowTime2", slowTime2);
  nh->get_parameter("useInclToStop", useInclToStop);
  nh->get_parameter("inclThre", inclThre);
  nh->get_parameter("stopTime", stopTime);
  nh->get_parameter("noRotAtStop", noRotAtStop);
  nh->get_parameter("noRotAtGoal", noRotAtGoal);
  nh->get_parameter("autonomyMode", autonomyMode);
  nh->get_parameter("autonomySpeed", autonomySpeed);
  nh->get_parameter("joyToSpeedDelay", joyToSpeedDelay);

  // 里程计信息:车辆位姿
  auto subOdom = nh->create_subscription<nav_msgs::msg::Odometry>("/state_estimation", 5, odomHandler);

  // 规划的路径
  auto subPath = nh->create_subscription<nav_msgs::msg::Path>("/local_planner/path", 5, pathHandler);
  auto subGoalDistance = nh->create_subscription<std_msgs::msg::Float32>(
      "/local_planner/goal_distance", 5, goalDistanceHandler);

  // 摇杆输入:用户手动控制
  auto subJoystick = nh->create_subscription<sensor_msgs::msg::Joy>("/local_planner/joy", 5, joystickHandler);

  // 速度指令:自主导航速度
  auto subSpeed = nh->create_subscription<std_msgs::msg::Float32>("/speed", 5, speedHandler);

  // 安全停止指令
  auto subStop = nh->create_subscription<std_msgs::msg::Int8>("/stop", 5, stopHandler);

  // 速度和转角控制命令
  // auto pubSpeed = nh->create_publisher<geometry_msgs::msg::TwistStamped>("/cmd_vel", 5);

  // geometry_msgs::msg::TwistStamped cmd_vel;
  // cmd_vel.header.frame_id = "vehicle";  // 在车体坐标系中发布

  pubSpeed = nh->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 5);
  geometry_msgs::msg::Twist cmd_vel;

  // ============================================================================
  // 初始化自主导航速度
  // ============================================================================
  if (autonomyMode) {
    joySpeed = autonomySpeed / maxSpeed;

    if (joySpeed < 0) joySpeed = 0;
    else if (joySpeed > 1.0) joySpeed = 1.0;
  }

  // ============================================================================
  // 主事件循环 (100 Hz控制循环)
  // ============================================================================
  rclcpp::Rate rate(100);
  bool status = rclcpp::ok();
  while (status) {
    rclcpp::spin_some(nh);  // 处理待处理的消息回调

    // 仅在收到有效路径时进行控制
    if (pathInit) {
      // ========================================================================
      // 第一阶段:坐标变换 - 相对于参考坐标系的位置
      // ========================================================================
      // 将当前车辆位置从世界坐标系变换到接收路径时的车体坐标系
      float vehicleXRel = cos(vehicleYawRec) * (vehicleX - vehicleXRec) 
                        + sin(vehicleYawRec) * (vehicleY - vehicleYRec);
      float vehicleYRel = -sin(vehicleYawRec) * (vehicleX - vehicleXRec) 
                        + cos(vehicleYawRec) * (vehicleY - vehicleYRec);

      // ========================================================================
      // 第二阶段:前向视界 - 选择目标点
      // ========================================================================
      int pathSize = path.poses.size();
      
      // 计算目标终点(路径最后一个点)到当前位置的距离
      float endDisX = path.poses[pathSize - 1].pose.position.x - vehicleXRel;
      float endDisY = path.poses[pathSize - 1].pose.position.y - vehicleYRel;
      float endDis = sqrt(endDisX * endDisX + endDisY * endDisY);

      float disX, disY, dis;
      // 随速度增大前视距离，高速时减少对近处路径点噪声的敏感度。
      float dynamicLookAhead = lookAheadDis + lookAheadTime * fabs(vehicleSpeed);
      if (dynamicLookAhead > maxLookAheadDis) dynamicLookAhead = maxLookAheadDis;

      // 沿着路径前进,直到找到距离当前位置动态前视距离以外的点
      while (pathPointID < pathSize - 1) {
        disX = path.poses[pathPointID].pose.position.x - vehicleXRel;
        disY = path.poses[pathPointID].pose.position.y - vehicleYRel;
        dis = sqrt(disX * disX + disY * disY);
        if (dis < dynamicLookAhead) {
          pathPointID++;  // 继续向前搜索
        } else {
          break;
        }
      }

      // 获取目标点(前向视界点或路径终点)
      disX = path.poses[pathPointID].pose.position.x - vehicleXRel;
      disY = path.poses[pathPointID].pose.position.y - vehicleYRel;
      dis = sqrt(disX * disX + disY * disY);
      // 计算目标点的方向角
      float pathDir = atan2(disY, disX);

      // ========================================================================
      // 第三阶段:计算方向误差 (方向跟踪PD控制)
      // ========================================================================
      float dirDiff = vehicleYaw - vehicleYawRec - pathDir;
      // 将角度误差规范到[-π, π]范围内
      if (dirDiff > PI) dirDiff -= 2 * PI;
      else if (dirDiff < -PI) dirDiff += 2 * PI;
      if (dirDiff > PI) dirDiff -= 2 * PI;
      else if (dirDiff < -PI) dirDiff += 2 * PI;

      // ========================================================================
      // 第四阶段:双向驾驶决策 - 自动选择前进或后退
      // ========================================================================
      if (twoWayDrive) {
        double time = nh->now().seconds();
        // 如果前进时方向误差过大,切换为后退
        if (fabs(dirDiff) > PI / 2 && navFwd && time - switchTime > switchTimeThre) {
          navFwd = false;
          switchTime = time;
        } 
        // 如果后退时方向误差足够小,切换为前进
        else if (fabs(dirDiff) < PI / 2 && !navFwd && time - switchTime > switchTimeThre) {
          navFwd = true;
          switchTime = time;
        }
      }

      // ========================================================================
      // 第五阶段:速度指令处理
      // ========================================================================
      float joySpeed2 = maxSpeed * joySpeed;  // 转换为实际速度
      if (!navFwd) {
        // 后退模式:调整方向误差和速度符号
        dirDiff += PI;
        if (dirDiff > PI) dirDiff -= 2 * PI;
        joySpeed2 *= -1;  // 负速度表示后退
      }

      // ========================================================================
      // 第六阶段:转向控制 (PD控制器)
      // ========================================================================
      // 根据速度大小选择转向增益(低速时使用stopYawRateGain可能更高)
      float targetYawRate;
      if (fabs(vehicleSpeed) < 2.0 * maxAccel / 100.0)
        targetYawRate = -stopYawRateGain * dirDiff;
      else
        targetYawRate = -yawRateGain * dirDiff;

      // 限制最大转向角速度
      if (targetYawRate > maxYawRate * PI / 180.0) targetYawRate = maxYawRate * PI / 180.0;
      else if (targetYawRate < -maxYawRate * PI / 180.0) targetYawRate = -maxYawRate * PI / 180.0;

      // 一阶低通滤波并限制角加速度，抑制路径小幅跳变造成的左右猛打方向。
      float filteredYawRate =
          vehicleYawRate + yawRateFilterAlpha * (targetYawRate - vehicleYawRate);
      float maxYawRateStep = maxYawAccel / 100.0;  // 控制循环为100 Hz
      if (filteredYawRate > vehicleYawRate + maxYawRateStep)
        filteredYawRate = vehicleYawRate + maxYawRateStep;
      else if (filteredYawRate < vehicleYawRate - maxYawRateStep)
        filteredYawRate = vehicleYawRate - maxYawRateStep;
      vehicleYawRate = filteredYawRate;

      // 手动控制优先级
      if (joySpeed2 == 0 && !autonomyMode) {
        // 停止时允许手动旋转
        vehicleYawRate = maxYawRate * joyYaw * PI / 180.0;
      } else if (pathSize <= 1 || (dis < stopDisThre && noRotAtGoal)) {
        // 到达目标或路径无效时停止转向
        vehicleYawRate = 0;
      }

      // ========================================================================
      // 第七阶段:速度调整 - 根据剩余距离减速
      // ========================================================================
      if (pathSize <= 1) {
        // 无有效路径时停止
        joySpeed2 = 0;
      } else if (endDis / slowDwnDisThre < joySpeed) {
        // 接近目标时减速(二次曲线减速)
        joySpeed2 *= endDis / slowDwnDisThre;
      }

      // 按到最终目标的剩余距离限速：v^2 <= 2 * a * (s - 停车余量 - 滞后距离)。
      // 1 m/s 的实际刹车距离大于局部轨迹长度，只靠上面的线性比例会冲过目标。
      const double nowSec = nh->now().seconds();
      if (brakeAccel > 0.0 && goalRemainDis >= 0.0 && goalRemainTime >= 0.0 &&
          nowSec - goalRemainTime < 0.5) {
        float lag = static_cast<float>(std::max(0.0, brakeLagTime) * fabs(vehicleSpeed));
        float usable = static_cast<float>(goalRemainDis - stopDisThre) - lag;
        if (usable < 0.0f) usable = 0.0f;
        float vLimit = static_cast<float>(sqrt(2.0 * brakeAccel * usable));
        if (joySpeed2 > vLimit) joySpeed2 = vLimit;
        else if (joySpeed2 < -vLimit) joySpeed2 = -vLimit;
      }

      // ========================================================================
      // 第八阶段:倾斜度保护 - 根据倾斜度检测结果降速
      // ========================================================================
      float joySpeed3 = joySpeed2;
      if (odomTime < slowInitTime + slowTime1 && slowInitTime > 0) 
        joySpeed3 *= slowRate1;  // 第一阶段减速
      else if (odomTime < slowInitTime + slowTime1 + slowTime2 && slowInitTime > 0) 
        joySpeed3 *= slowRate2;  // 第二阶段减速

      // ========================================================================
      // 第九阶段:加速度限制 - 平滑的加速/减速
      // ========================================================================
      // 仅当方向跟踪精确且不接近停止时,才进行加速
      if (fabs(dirDiff) < dirDiffThre && dis > stopDisThre) {
        if (vehicleSpeed < joySpeed3) vehicleSpeed += maxAccel / 100.0;
        else if (vehicleSpeed > joySpeed3) vehicleSpeed -= maxAccel / 100.0;
      } else {
        // 方向误差大或接近停止时,进行减速
        if (vehicleSpeed > 0) vehicleSpeed -= maxAccel / 100.0;
        else if (vehicleSpeed < 0) vehicleSpeed += maxAccel / 100.0;
      }

      // ========================================================================
      // 第十阶段:倾斜度停止保护 - 如果倾斜度过大,完全停止
      // ========================================================================
      if (odomTime < stopInitTime + stopTime && stopInitTime > 0) {
        vehicleSpeed = 0;
        vehicleYawRate = 0;
      }

      // ========================================================================
      // 第十一阶段:安全停止 - 外部安全指令
      // ========================================================================
      if (safetyStop >= 1) vehicleSpeed = 0;      // 停止速度
      if (safetyStop >= 2) vehicleYawRate = 0;    // 停止转向

      // ========================================================================
      // 第十二阶段:消息发布 (减频发布)
      // ========================================================================
      pubSkipCount--;
      if (pubSkipCount < 0) {
        // cmd_vel.header.stamp = rclcpp::Time(static_cast<uint64_t>(odomTime * 1e9));
        // 设置线速度(防止极小的速度噪声)
        if (fabs(vehicleSpeed) <= maxAccel / 100.0) cmd_vel.linear.x = 0;
        else cmd_vel.linear.x = vehicleSpeed;
        // 设置角速度
        cmd_vel.angular.z = vehicleYawRate;
        pubSpeed->publish(cmd_vel);

        pubSkipCount = pubSkipNum;
      }
    }

    status = rclcpp::ok();
    rate.sleep();
  }

  return 0;
}
