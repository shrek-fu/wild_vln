/*
* BSD 3-Clause License
* Copyright (c) 2024, DDDMobileRobot
*
* qi_move_base  –  FSM move-base that communicates with the
* standalone localPlanner / pathFollower nodes via ROS topics.
*
* Differences from dddmr_p2p_move_base:
*   • No dependency on dddmr_local_planner plugin system.
*   • LP_->computeVelocityCommand() is gone; velocity is produced by
*     pathFollower which subscribes to /path published by localPlanner.
*   • d_align_heading state is removed; path following begins immediately
*     after a plan is received (localPlanner handles it internally).
*   • d_align_goal_heading: we publish /goal_pose_3d so localPlanner
*     handles the in-place rotation; FSM monitors yaw to detect completion.
*   • d_controlling: FSM monitors XY distance to goal; oscillation
*     watchdog is still active.
*/
#include "qi_move_base/p2p_move_base.h"

#include <limits>

namespace qi_move_base
{

// ─────────────────────────────────────────────────────────────────────────────
//  Construction / destruction
// ─────────────────────────────────────────────────────────────────────────────

P2PMoveBase::P2PMoveBase(std::string name)
  : Node(name), name_(name)
{
  clock_ = this->get_clock();
}

P2PMoveBase::~P2PMoveBase(){
  FSM_.reset();
  tf2Buffer_.reset();
  tfl_.reset();
  GPM_.reset();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Initialisation
// ─────────────────────────────────────────────────────────────────────────────

void P2PMoveBase::initial(const std::shared_ptr<qi_move_base::P2PGlobalPlanManager>& gpm)
{
  GPM_ = gpm;

  // FSM reads its own parameters from this node's parameter interface
  FSM_ = std::make_shared<qi_move_base::FSM>(
    this->get_node_logging_interface(),
    this->get_node_parameters_interface());

  // Tunable thresholds
  this->declare_parameter("goal_reached_dis", rclcpp::ParameterValue(0.5));
  goal_reached_dis_ = this->get_parameter("goal_reached_dis").as_double();
  RCLCPP_INFO(this->get_logger(), "goal_reached_dis: %.2f m", goal_reached_dis_);

  this->declare_parameter("heading_align_thre", rclcpp::ParameterValue(0.05));
  heading_align_thre_ = this->get_parameter("heading_align_thre").as_double();
  RCLCPP_INFO(this->get_logger(), "heading_align_thre: %.4f rad", heading_align_thre_);

  this->declare_parameter("heading_align_speed", rclcpp::ParameterValue(0.3));
  heading_align_speed_ = this->get_parameter("heading_align_speed").as_double();
  RCLCPP_INFO(this->get_logger(), "heading_align_speed: %.3f rad/s", heading_align_speed_);

  stall_progress_ = this->declare_parameter("stall_progress", stall_progress_);
  stall_patience_ = this->declare_parameter("stall_patience", stall_patience_);
  stall_abort_patience_ = this->declare_parameter("stall_abort_patience", stall_abort_patience_);
  near_goal_accept_dis_ = this->declare_parameter("near_goal_accept_dis", near_goal_accept_dis_);
  RCLCPP_INFO(this->get_logger(), "stall: %.2f m progress, accept within %.2f m after %.1f s, abort after %.1f s",
              stall_progress_, near_goal_accept_dis_, stall_patience_, stall_abort_patience_);

  // Velocity publishers
  if(FSM_->use_twist_stamped_){
    stamped_cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("cmd_vel_stamped", 1);
  } else {
    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 1);
  }

  pub_global_path_ = this->create_publisher<nav_msgs::msg::Path>("/global_path", 1);
  goal_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/p2p_goal_pose", 10);

  // Odometry subscriber
  odom_sub_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::SubscriptionOptions odom_opts;
  odom_opts.callback_group = odom_sub_group_;
  sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
    "/state_estimation", 10,
    std::bind(&P2PMoveBase::cbOdom, this, std::placeholders::_1),
    odom_opts);

  // TF
  tf_listener_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  tf2Buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
    this->get_node_base_interface(),
    this->get_node_timers_interface(),
    tf_listener_group_);
  tf2Buffer_->setCreateTimerInterface(timer_interface);
  tfl_ = std::make_shared<tf2_ros::TransformListener>(*tf2Buffer_);

  // Recovery behaviors action client
  recovery_behaviors_client_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  recovery_behaviors_client_ptr_ = rclcpp_action::create_client<dddmr_sys_core::action::RecoveryBehaviors>(
    this, "recovery_behaviors", recovery_behaviors_client_group_);

  // Action server
  action_server_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  action_server_p2p_move_base_ = rclcpp_action::create_server<dddmr_sys_core::action::PToPMoveBase>(
    this,
    "/p2p_move_base",
    std::bind(&P2PMoveBase::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
    std::bind(&P2PMoveBase::handle_cancel, this, std::placeholders::_1),
    std::bind(&P2PMoveBase::handle_accepted, this, std::placeholders::_1),
    rcl_action_server_get_default_options(),
    action_server_group_);

  RCLCPP_INFO(this->get_logger(), "\033[1;32m---->\033[0m QL move base launched.");

  // Service client: resets dGraph to static map before each new navigation goal
  perception_client_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  clear_perception_client_ = this->create_client<std_srvs::srv::Empty>(
    "clear_perception_marking", rmw_qos_profile_services_default, perception_client_group_);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Odometry callback
// ─────────────────────────────────────────────────────────────────────────────

void P2PMoveBase::cbOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(odom_mutex_);
  current_odom_ = *msg;
  odom_received_ = true;
}

geometry_msgs::msg::TransformStamped P2PMoveBase::getGlobalPose()
{
  std::lock_guard<std::mutex> lock(odom_mutex_);
  geometry_msgs::msg::TransformStamped ts;
  ts.header = current_odom_.header;
  ts.child_frame_id = current_odom_.child_frame_id;
  ts.transform.translation.x = current_odom_.pose.pose.position.x;
  ts.transform.translation.y = current_odom_.pose.pose.position.y;
  ts.transform.translation.z = current_odom_.pose.pose.position.z;
  ts.transform.rotation      = current_odom_.pose.pose.orientation;
  return ts;
}

double P2PMoveBase::distanceToGoal(const geometry_msgs::msg::PoseStamped& goal)
{
  std::lock_guard<std::mutex> lock(odom_mutex_);
  double dx = current_odom_.pose.pose.position.x - goal.pose.position.x;
  double dy = current_odom_.pose.pose.position.y - goal.pose.position.y;
  return std::sqrt(dx*dx + dy*dy);
}

double P2PMoveBase::yawErrorToGoal(const geometry_msgs::msg::PoseStamped& goal)
{
  std::lock_guard<std::mutex> lock(odom_mutex_);
  double current_yaw = tf2::getYaw(current_odom_.pose.pose.orientation);
  double goal_yaw    = tf2::getYaw(goal.pose.orientation);
  double err = goal_yaw - current_yaw;
  while(err >  M_PI) err -= 2.0 * M_PI;
  while(err < -M_PI) err += 2.0 * M_PI;
  return err;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Plan / velocity helpers
// ─────────────────────────────────────────────────────────────────────────────

void P2PMoveBase::setPlan(const std::vector<geometry_msgs::msg::PoseStamped>& plan)
{
  return;
  // nav_msgs::msg::Path path_msg;
  // path_msg.header.stamp    = clock_->now();
  // path_msg.header.frame_id = "map";
  // path_msg.poses = plan;
  // pub_global_path_->publish(path_msg);
}

void P2PMoveBase::clearLocalPlan()
{
  // Publish empty global path → localPlanner stops path following
  nav_msgs::msg::Path empty;
  empty.header.stamp    = clock_->now();
  empty.header.frame_id = "map";
  pub_global_path_->publish(empty);
}

void P2PMoveBase::publishZeroVelocity()
{
  geometry_msgs::msg::Twist cmd_vel;
  cmd_vel.linear.x = 0.0;
  cmd_vel.linear.y = 0.0;
  cmd_vel.angular.z = 0.0;
  if(FSM_->use_twist_stamped_){
    geometry_msgs::msg::TwistStamped stamped;
    stamped.header.frame_id = FSM_->twist_frame_id_;
    stamped.header.stamp    = clock_->now();
    stamped.twist = cmd_vel;
    stamped_cmd_vel_pub_->publish(stamped);
  } else {
    cmd_vel_pub_->publish(cmd_vel);
  }
}

void P2PMoveBase::publishHeadingAlignVelocity(double yaw_error)
{
  // Pure in-place rotation: sign follows yaw_error, clamped to heading_align_speed_
  geometry_msgs::msg::Twist cmd_vel;
  cmd_vel.linear.x  = 0.0;
  cmd_vel.linear.y  = 0.0;
  cmd_vel.angular.z = (yaw_error > 0.0 ? 1.0 : -1.0) * heading_align_speed_;
  if(FSM_->use_twist_stamped_){
    geometry_msgs::msg::TwistStamped stamped;
    stamped.header.frame_id = FSM_->twist_frame_id_;
    stamped.header.stamp    = clock_->now();
    stamped.twist = cmd_vel;
    stamped_cmd_vel_pub_->publish(stamped);
  } else {
    cmd_vel_pub_->publish(cmd_vel);
  }
}

void P2PMoveBase::resetPerceptionGraph()
{
  if(!clear_perception_client_->wait_for_service(std::chrono::seconds(1))){
    RCLCPP_WARN(this->get_logger(), "clear_perception_marking service not available, skipping reset.");
    return;
  }
  auto req = std::make_shared<std_srvs::srv::Empty::Request>();
  auto future = clear_perception_client_->async_send_request(req);
  if(future.wait_for(std::chrono::seconds(3)) == std::future_status::ready){
    RCLCPP_INFO(this->get_logger(), "Perception dGraph reset to static map.");
  } else {
    RCLCPP_WARN(this->get_logger(), "clear_perception_marking call timed out.");
  }
}

// ────────────────────────────────────────────────────────────────────────────────
//  Action server callbacks
// ────────────────────────────────────────────────────────────────────────────────

rclcpp_action::GoalResponse P2PMoveBase::handle_goal(
  const rclcpp_action::GoalUUID & /*uuid*/,
  std::shared_ptr<const dddmr_sys_core::action::PToPMoveBase::Goal> goal)
{
  if(goal_pose_pub_){
    auto goal_msg = goal->target_pose;
    goal_msg.header.stamp = clock_->now();
    goal_pose_pub_->publish(goal_msg);
  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse P2PMoveBase::handle_cancel(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<dddmr_sys_core::action::PToPMoveBase>> /*goal_handle*/)
{
  RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void P2PMoveBase::handle_accepted(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<dddmr_sys_core::action::PToPMoveBase>> goal_handle)
{
  if(is_active(current_handle_)){
    RCLCPP_INFO(this->get_logger(), "An older goal is active, aborting it.");
    auto result = std::make_shared<dddmr_sys_core::action::PToPMoveBase::Result>();
    current_handle_->abort(result);
    return;
  }
  current_handle_ = goal_handle;
  std::thread{std::bind(&P2PMoveBase::executeCb, this, std::placeholders::_1), goal_handle}.detach();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main execution loop
// ─────────────────────────────────────────────────────────────────────────────

bool P2PMoveBase::isQuaternionValid(const geometry_msgs::msg::Quaternion& q)
{
  if(!std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z) || !std::isfinite(q.w)){
    RCLCPP_ERROR(this->get_logger(), "Quaternion has nans or infs... discarding as navigation goal");
    return false;
  }
  tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
  if(tf_q.length2() < 1e-6){
    RCLCPP_ERROR(this->get_logger(), "Quaternion has length close to zero... discarding as navigation goal");
    return false;
  }
  tf_q.normalize();
  tf2::Vector3 up(0, 0, 1);
  double dot = up.dot(up.rotate(tf_q.getAxis(), tf_q.getAngle()));
  if(fabs(dot - 1) > 1e-3){
    RCLCPP_ERROR(this->get_logger(), "Quaternion is invalid (z-axis must be close to vertical).");
    return false;
  }
  return true;
}

void P2PMoveBase::executeCb(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<dddmr_sys_core::action::PToPMoveBase>> goal_handle)
{
  auto result = std::make_shared<dddmr_sys_core::action::PToPMoveBase::Result>();
  auto move_base_goal = goal_handle->get_goal();

  if(!isQuaternionValid(move_base_goal->target_pose.pose.orientation)){
    RCLCPP_WARN(this->get_logger(), "Aborting on goal because it was sent with an invalid quaternion");
    goal_handle->abort(result);
    publishZeroVelocity();
    return;
  }

  // Wait for first odom
  rclcpp::Rate wait_rate(10);
  int wait_count = 0;
  while(rclcpp::ok() && !odom_received_ && wait_count < 50){
    RCLCPP_INFO_THROTTLE(this->get_logger(), *clock_, 1000, "Waiting for /state_estimation odom...");
    wait_rate.sleep();
    wait_count++;
  }
  if(!odom_received_){
    RCLCPP_ERROR(this->get_logger(), "No odometry received on /state_estimation, aborting.");
    goal_handle->abort(result);
    return;
  }

  rclcpp::Rate r(FSM_->controller_frequency_);

  FSM_->initialParams(getGlobalPose(), clock_->now());
  FSM_->current_goal_ = move_base_goal->target_pose;
  best_goal_dis_ = std::numeric_limits<double>::infinity();
  last_progress_ = last_moved_ = clock_->now();
  stall_pose_ = getGlobalPose();

  // Reset dGraph to static map before the first plan so stale lidar markings don't affect routing
  // resetPerceptionGraph();

  GPM_->setGoal(FSM_->current_goal_);
  GPM_->resume();

  while(rclcpp::ok()){

    if(!goal_handle->is_active()){
      if(FSM_->isCurrentDecision("d_recovery_waitdone")){
        recovery_behaviors_client_ptr_->async_cancel_all_goals();
      }
      RCLCPP_INFO(this->get_logger(), "P2P move base preempted.");
      clearLocalPlan();
      publishZeroVelocity();
      GPM_->stop();
      return;
    }

    if(goal_handle->is_canceling()){
      if(FSM_->isCurrentDecision("d_recovery_waitdone")){
        recovery_behaviors_client_ptr_->async_cancel_all_goals();
      }
      goal_handle->canceled(result);
      RCLCPP_INFO(this->get_logger(), "P2P move base cancelled.");
      clearLocalPlan();
      publishZeroVelocity();
      GPM_->stop();
      return;
    }

    // Update global pose and oscillation watchdog
    FSM_->global_pose_ = getGlobalPose();
    if(FSM_->getDistance(FSM_->global_pose_, FSM_->oscillation_pose_) >= FSM_->oscillation_distance_ ||
       FSM_->getAngle(FSM_->global_pose_, FSM_->oscillation_pose_)    >= FSM_->oscillation_angle_)
    {
      FSM_->oscillation_pose_ = FSM_->global_pose_;
      FSM_->last_oscillation_reset_ = clock_->now();
    }

    bool done = executeCycle(goal_handle);

    auto feedback = std::make_shared<dddmr_sys_core::action::PToPMoveBase::Feedback>();
    feedback->base_position    = FSM_->global_pose_;
    feedback->last_decision    = FSM_->getLastDecision();
    feedback->current_decision = FSM_->getCurrentDecision();
    goal_handle->publish_feedback(feedback);

    if(done){
      GPM_->stop();
      return;
    }

    r.sleep();
  }

  GPM_->stop();
}

// ─────────────────────────────────────────────────────────────────────────────
//  FSM cycle
// ─────────────────────────────────────────────────────────────────────────────

bool P2PMoveBase::executeCycle(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<dddmr_sys_core::action::PToPMoveBase>> goal_handle)
{
  // ──────────────────────────────────────────────────────────────────────────
  if(FSM_->isCurrentDecision("d_initial")){
    FSM_->setDecision("d_planning");
    return false;
  }

  // ──────────────────────────────────────────────────────────────────────────
  else if(FSM_->isCurrentDecision("d_planning")){
    GPM_->queryThread();
    FSM_->setDecision("d_planning_waitdone");
    return false;
  }

  // ──────────────────────────────────────────────────────────────────────────
  else if(FSM_->isCurrentDecision("d_planning_waitdone")){

    std::vector<geometry_msgs::msg::PoseStamped> plan;
    if(GPM_->hasPlan()){
      GPM_->copyPlan(plan);
      if(plan.empty()){
        RCLCPP_DEBUG(this->get_logger(), "Global planner returned empty plan, re-planning.");
        FSM_->setDecision("d_planning");
      } else {
        FSM_->last_valid_plan_ = clock_->now();
        // Send plan to localPlanner via /global_path topic
        setPlan(plan);
        // Skip d_align_heading: localPlanner handles initial steering internally
        FSM_->setDecision("d_controlling");
      }
    }

    if((clock_->now() - FSM_->last_valid_plan_).seconds() > FSM_->planner_patience_){
      RCLCPP_WARN(this->get_logger(), "Planner patience exceeded, going to recovery.");
      startRecoveryBehaviors();
      FSM_->setDecision("d_recovery_waitdone");
    }

    return false;
  }

  // ──────────────────────────────────────────────────────────────────────────
  else if(FSM_->isCurrentDecision("d_controlling")){

    // Check XY goal reached → transition to heading alignment
    double dis = distanceToGoal(FSM_->current_goal_);
    if(dis < goal_reached_dis_){
      RCLCPP_INFO(this->get_logger(), "Goal XY tolerance reached (%.3f m), switching to d_align_goal_heading.", dis);
      GPM_->stop();
      clearLocalPlan();          // tell localPlanner to stop forwarding
      publishZeroVelocity();
      FSM_->setDecision("d_align_goal_heading");
      return false;
    }

    // Stall watchdog: the static-map plan stays valid when someone stands on the goal, so without this the
    // local planner would hover at the obstacle forever.
    // Near the goal progress means getting closer; farther out (detours) it only means moving at all.
    if(dis < best_goal_dis_ - stall_progress_){
      best_goal_dis_ = dis;
      last_progress_ = clock_->now();
    }
    if(FSM_->getDistance(FSM_->global_pose_, stall_pose_) >= stall_progress_){
      stall_pose_ = FSM_->global_pose_;
      last_moved_ = clock_->now();
    }
    const double stalled = (clock_->now() - last_progress_).seconds();
    const double still = (clock_->now() - last_moved_).seconds();
    if(stall_patience_ > 0 && dis < near_goal_accept_dis_ && stalled >= stall_patience_){
      RCLCPP_WARN(this->get_logger(), "No progress for %.1f s at %.2f m from the goal (occupied?): accepting it as "
                  "reached, switching to d_align_goal_heading.", stalled, dis);
      GPM_->stop();
      clearLocalPlan();
      publishZeroVelocity();
      FSM_->setDecision("d_align_goal_heading");
      return false;
    }
    if(stall_abort_patience_ > 0 && dis >= near_goal_accept_dis_ && still >= stall_abort_patience_){
      RCLCPP_WARN(this->get_logger(), "Not moving for %.1f s at %.2f m from the goal: aborting.", still, dis);
      GPM_->stop();
      clearLocalPlan();
      publishZeroVelocity();
      auto result = std::make_shared<dddmr_sys_core::action::PToPMoveBase::Result>();
      goal_handle->abort(result);
      return true;
    }

    // Update global plan if GPM has a fresher one
    if(GPM_->hasPlan()){
      std::vector<geometry_msgs::msg::PoseStamped> plan;
      GPM_->copyPlan(plan);
      if(!plan.empty()) setPlan(plan);
    }

    // Oscillation watchdog
    if(FSM_->oscillation_patience_ > 0 &&
       (clock_->now() - FSM_->last_oscillation_reset_).seconds() >= FSM_->oscillation_patience_)
    {
      auto diff = (clock_->now() - FSM_->last_oscillation_reset_).seconds();
      RCLCPP_WARN(this->get_logger(), "Oscillation detected: %.2f secs / %.2f m.", diff,
                  FSM_->getDistance(FSM_->global_pose_, FSM_->oscillation_pose_));
      clearLocalPlan();
      publishZeroVelocity();
      startRecoveryBehaviors();
      FSM_->setDecision("d_recovery_waitdone");
      return false;
    }

    // localPlanner + pathFollower are running autonomously; no further action needed here.
    return false;
  }

  // ──────────────────────────────────────────────────────────────────────────
  else if(FSM_->isCurrentDecision("d_align_goal_heading")){

    double yaw_err = yawErrorToGoal(FSM_->current_goal_);

    if(fabs(yaw_err) < heading_align_thre_){
      RCLCPP_INFO(this->get_logger(),
        "Goal heading aligned (err=%.4f rad). Navigation succeeded!", yaw_err);
      publishZeroVelocity();
      auto result = std::make_shared<dddmr_sys_core::action::PToPMoveBase::Result>();
      goal_handle->succeed(result);
      return true;
    }

    // Oscillation watchdog during heading alignment
    if(FSM_->oscillation_patience_ > 0 &&
       (clock_->now() - FSM_->last_oscillation_reset_).seconds() >= FSM_->oscillation_patience_)
    {
      auto diff = (clock_->now() - FSM_->last_oscillation_reset_).seconds();
      RCLCPP_WARN(this->get_logger(),
        "Oscillation detected in heading alignment: %.2f secs.", diff);
      publishZeroVelocity();
      startRecoveryBehaviors();
      FSM_->setDecision("d_recovery_waitdone");
      return false;
    }

    // Publish in-place rotation command directly to /cmd_vel
    RCLCPP_INFO_THROTTLE(this->get_logger(), *clock_, 500,
      "Aligning heading: yaw_err=%.4f rad, speed=%.3f rad/s", yaw_err, heading_align_speed_);
    publishHeadingAlignVelocity(yaw_err);
    return false;
  }

  // ──────────────────────────────────────────────────────────────────────────
  else if(FSM_->isCurrentDecision("d_recovery_waitdone")){
    // TODO: valid recovery behaviors should be added to the FSM. 
    // This is a temporary hack.
    GPM_->stop();
    auto result = std::make_shared<dddmr_sys_core::action::PToPMoveBase::Result>();
    goal_handle->abort(result);
    publishZeroVelocity();
    return true;

    if(is_recoverying_){
      return false;
    }

    if(FSM_->no_plan_recovery_count_ >= FSM_->no_plan_retry_num_){
      RCLCPP_ERROR(this->get_logger(), "Exceeded recovery retry count (%d). Aborting.",
                   FSM_->no_plan_recovery_count_);
      auto result = std::make_shared<dddmr_sys_core::action::PToPMoveBase::Result>();
      goal_handle->abort(result);
      publishZeroVelocity();
      return true;
    }

    if(is_recoverying_succeed_){
      RCLCPP_INFO(this->get_logger(), "Recovery succeeded, back to planning.");
      FSM_->no_plan_recovery_count_++;
      FSM_->last_valid_plan_ = clock_->now();
      FSM_->setDecision("d_planning");
      return false;
    } else {
      RCLCPP_ERROR(this->get_logger(), "Recovery failed. Aborting.");
      auto result = std::make_shared<dddmr_sys_core::action::PToPMoveBase::Result>();
      goal_handle->abort(result);
      publishZeroVelocity();
      return true;
    }
  }

  return false;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Recovery behaviors
// ─────────────────────────────────────────────────────────────────────────────

void P2PMoveBase::startRecoveryBehaviors()
{
  auto goal_msg = dddmr_sys_core::action::RecoveryBehaviors::Goal();
  goal_msg.behavior_name = "rotate_inplace";

  auto send_goal_options = rclcpp_action::Client<dddmr_sys_core::action::RecoveryBehaviors>::SendGoalOptions();
  send_goal_options.goal_response_callback =
    std::bind(&P2PMoveBase::recovery_behaviors_client_goal_response_callback, this, std::placeholders::_1);
  send_goal_options.result_callback =
    std::bind(&P2PMoveBase::recovery_behaviors_client_result_callback, this, std::placeholders::_1);

  is_recoverying_ = true;
  recovery_behaviors_client_ptr_->async_send_goal(goal_msg, send_goal_options);
}

void P2PMoveBase::recovery_behaviors_client_goal_response_callback(
  const rclcpp_action::ClientGoalHandle<dddmr_sys_core::action::RecoveryBehaviors>::SharedPtr & goal_handle)
{
  if (!goal_handle) {
    RCLCPP_ERROR(this->get_logger(), "Recovery goal was rejected by server");
  } else {
    RCLCPP_INFO(this->get_logger(), "Recovery goal accepted, waiting for result");
  }
}

void P2PMoveBase::recovery_behaviors_client_result_callback(
  const rclcpp_action::ClientGoalHandle<dddmr_sys_core::action::RecoveryBehaviors>::WrappedResult & result)
{
  is_recoverying_succeed_ = false;
  switch (result.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      is_recoverying_succeed_ = true;
      break;
    case rclcpp_action::ResultCode::ABORTED:
      RCLCPP_ERROR(this->get_logger(), "Recovery Behaviors: Goal was aborted");
      break;
    case rclcpp_action::ResultCode::CANCELED:
      RCLCPP_ERROR(this->get_logger(), "Recovery Behaviors: Goal was canceled");
      break;
    default:
      RCLCPP_ERROR(this->get_logger(), "Recovery Behaviors: Unknown result code");
      break;
  }
  is_recoverying_ = false;
}

}  // namespace qi_move_base
