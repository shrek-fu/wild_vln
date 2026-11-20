/*
* BSD 3-Clause License
* Copyright (c) 2024, DDDMobileRobot
*
* qi_move_base: FSM-based point-to-point navigation
* Interfaces with localPlanner + pathFollower via ROS topics instead of
* using the dddmr_local_planner plugin system.
*
* Topic interface:
*   Publishes: /global_path  (nav_msgs/Path)       → localPlanner
*              /cmd_vel      (geometry_msgs/Twist)  → stop + heading alignment
*   Subscribes: /state_estimation (nav_msgs/Odometry) → pose tracking
*/
#pragma once

#include <chrono>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "tf2_ros/transform_listener.h"
#include "tf2_ros/create_timer_ros.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2_ros/buffer.h"
#include "tf2/utils.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_srvs/srv/empty.hpp>

#include "dddmr_sys_core/action/p_to_p_move_base.hpp"
#include "dddmr_sys_core/action/recovery_behaviors.hpp"
#include "qi_move_base/p2p_fsm.h"
#include "qi_move_base/p2p_global_plan_manager.h"

namespace qi_move_base
{

class P2PMoveBase : public rclcpp::Node {

public:
  P2PMoveBase(std::string name);
  ~P2PMoveBase();

  void initial(const std::shared_ptr<qi_move_base::P2PGlobalPlanManager>& gpm);

private:
  // ── Action server callbacks ─────────────────────────────────────────────────
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const dddmr_sys_core::action::PToPMoveBase::Goal> goal);

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<dddmr_sys_core::action::PToPMoveBase>> goal_handle);

  void handle_accepted(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<dddmr_sys_core::action::PToPMoveBase>> goal_handle);

  rclcpp_action::Server<dddmr_sys_core::action::PToPMoveBase>::SharedPtr action_server_p2p_move_base_;
  std::shared_ptr<rclcpp_action::ServerGoalHandle<dddmr_sys_core::action::PToPMoveBase>> current_handle_;

  bool is_active(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<dddmr_sys_core::action::PToPMoveBase>> handle) const
  {
    return handle != nullptr && handle->is_active();
  }

  // ── FSM / GPM ───────────────────────────────────────────────────────────────
  std::shared_ptr<qi_move_base::FSM> FSM_;
  std::shared_ptr<qi_move_base::P2PGlobalPlanManager> GPM_;

  void executeCb(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<dddmr_sys_core::action::PToPMoveBase>> goal_handle);
  bool executeCycle(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<dddmr_sys_core::action::PToPMoveBase>> goal_handle);

  // ── Odometry (replaces LP_->getGlobalPose / isGoalReached / isGoalHeadingAligned) ──
  nav_msgs::msg::Odometry current_odom_;
  bool odom_received_{false};
  std::mutex odom_mutex_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  void cbOdom(const nav_msgs::msg::Odometry::SharedPtr msg);

  /// Convert current odometry to TransformStamped (used by FSM helpers)
  geometry_msgs::msg::TransformStamped getGlobalPose();

  /// XY distance from current odom to goal (m)
  double distanceToGoal(const geometry_msgs::msg::PoseStamped& goal);

  /// Yaw error between current odom and goal orientation (rad, in [-pi, pi])
  double yawErrorToGoal(const geometry_msgs::msg::PoseStamped& goal);

  // Tunable thresholds (read from params in initial())
  double goal_reached_dis_{0.5};        ///< XY tolerance to enter heading-align phase
  double heading_align_thre_{0.05};     ///< Yaw tolerance (rad) to declare success
  double heading_align_speed_{0.3};     ///< In-place rotation speed (rad/s)

  // Stall watchdog: the distance to the goal stopped shrinking (goal occupied, path blocked)
  double stall_progress_{0.2};          ///< m the goal distance must shrink by to count as progress
  double stall_patience_{8.0};          ///< s without progress within near_goal_accept_dis -> accept (<= 0: off)
  double stall_abort_patience_{30.0};   ///< s without progress farther out -> abort (<= 0: off)
  double near_goal_accept_dis_{1.5};    ///< m from the goal at which a blocked goal counts as reached
  double best_goal_dis_{0.0};
  rclcpp::Time last_progress_, last_moved_;
  geometry_msgs::msg::TransformStamped stall_pose_;

  /// Send plan to localPlanner (replaces LP_->setPlan)
  void setPlan(const std::vector<geometry_msgs::msg::PoseStamped>& plan);

  /// Publish empty path → tells localPlanner to stop
  void clearLocalPlan();

  /// Publish in-place rotation command to /cmd_vel for heading alignment
  void publishHeadingAlignVelocity(double yaw_error);

  // ── Velocity output (only zero-vel stop; normal motion is from pathFollower) ─
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr stamped_cmd_vel_pub_;
  void publishZeroVelocity();

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_global_path_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pose_pub_;

  // ── TF ──────────────────────────────────────────────────────────────────────
  rclcpp::CallbackGroup::SharedPtr tf_listener_group_;
  rclcpp::CallbackGroup::SharedPtr action_server_group_;
  rclcpp::CallbackGroup::SharedPtr odom_sub_group_;
  std::shared_ptr<tf2_ros::TransformListener> tfl_;
  std::shared_ptr<tf2_ros::Buffer> tf2Buffer_;

  // ── Recovery behaviors ───────────────────────────────────────────────────────
  rclcpp::CallbackGroup::SharedPtr recovery_behaviors_client_group_;
  rclcpp_action::Client<dddmr_sys_core::action::RecoveryBehaviors>::SharedPtr recovery_behaviors_client_ptr_;
  void startRecoveryBehaviors();
  void recovery_behaviors_client_goal_response_callback(
    const rclcpp_action::ClientGoalHandle<dddmr_sys_core::action::RecoveryBehaviors>::SharedPtr & goal_handle);
  void recovery_behaviors_client_result_callback(
    const rclcpp_action::ClientGoalHandle<dddmr_sys_core::action::RecoveryBehaviors>::WrappedResult & result);
  bool is_recoverying_{false};
  bool is_recoverying_succeed_{false};

  // ── Misc ─────────────────────────────────────────────────────────────────────
  std::string name_;
  rclcpp::Clock::SharedPtr clock_;
  bool isQuaternionValid(const geometry_msgs::msg::Quaternion& q);

  // ── Perception reset ──────────────────────────────────────────────────────────
  /// Calls clear_perception_marking service to reset dGraph to static map before planning.
  void resetPerceptionGraph();
  rclcpp::Client<std_srvs::srv::Empty>::SharedPtr clear_perception_client_;
  rclcpp::CallbackGroup::SharedPtr perception_client_group_;
};

}  // namespace qi_move_base
