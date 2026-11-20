/*
* BSD 3-Clause License
* Copyright (c) 2024, DDDMobileRobot
*/
#include "qi_move_base/p2p_fsm.h"

namespace qi_move_base
{

FSM::FSM(const rclcpp::node_interfaces::NodeLoggingInterface::SharedPtr& m_logger,
              const rclcpp::node_interfaces::NodeParametersInterface::SharedPtr& m_parameter)
{
  logger_ = m_logger;
  parameter_ = m_parameter;
  current_decision_ = "d_initial";
  last_decision_ = "d_initial";

  parameter_->declare_parameter("planner_patience", rclcpp::ParameterValue(10.0));
  planner_patience_ = parameter_->get_parameter("planner_patience").as_double();
  RCLCPP_INFO(logger_->get_logger(), "planner_patience: %.2f", planner_patience_);

  parameter_->declare_parameter("oscillation_distance", rclcpp::ParameterValue(10.0));
  oscillation_distance_ = parameter_->get_parameter("oscillation_distance").as_double();
  RCLCPP_INFO(logger_->get_logger(), "oscillation_distance: %.2f", oscillation_distance_);

  parameter_->declare_parameter("oscillation_angle", rclcpp::ParameterValue(0.5));
  oscillation_angle_ = parameter_->get_parameter("oscillation_angle").as_double();
  RCLCPP_INFO(logger_->get_logger(), "oscillation_angle: %.2f", oscillation_angle_);

  parameter_->declare_parameter("oscillation_patience", rclcpp::ParameterValue(20.0));
  oscillation_patience_ = parameter_->get_parameter("oscillation_patience").as_double();
  RCLCPP_INFO(logger_->get_logger(), "oscillation_patience: %.2f", oscillation_patience_);

  parameter_->declare_parameter("controller_patience", rclcpp::ParameterValue(10.0));
  controller_patience_ = parameter_->get_parameter("controller_patience").as_double();
  RCLCPP_INFO(logger_->get_logger(), "controller_patience: %.2f", controller_patience_);

  parameter_->declare_parameter("no_plan_retry_num", rclcpp::ParameterValue(10));
  no_plan_retry_num_ = parameter_->get_parameter("no_plan_retry_num").as_int();
  RCLCPP_INFO(logger_->get_logger(), "no_plan_retry_num: %d", no_plan_retry_num_);

  parameter_->declare_parameter("waiting_patience", rclcpp::ParameterValue(10.0));
  waiting_patience_ = parameter_->get_parameter("waiting_patience").as_double();
  RCLCPP_INFO(logger_->get_logger(), "waiting_patience: %.2f", waiting_patience_);

  parameter_->declare_parameter("controller_frequency", rclcpp::ParameterValue(20.0));
  controller_frequency_ = parameter_->get_parameter("controller_frequency").as_double();
  RCLCPP_INFO(logger_->get_logger(), "controller_frequency: %.2f", controller_frequency_);

  parameter_->declare_parameter("use_twist_stamped", rclcpp::ParameterValue(false));
  use_twist_stamped_ = parameter_->get_parameter("use_twist_stamped").as_bool();
  RCLCPP_INFO(logger_->get_logger(), "use_stamped_twist: %d", use_twist_stamped_);

  parameter_->declare_parameter("twist_frame_id", rclcpp::ParameterValue(std::string("base_link")));
  twist_frame_id_ = parameter_->get_parameter("twist_frame_id").as_string();
  RCLCPP_INFO(logger_->get_logger(), "twist_frame_id: %s", twist_frame_id_.c_str());
}

bool FSM::isCurrentDecision(std::string m_decision){
  return m_decision == current_decision_;
}

void FSM::initialParams(geometry_msgs::msg::TransformStamped robot_curent_pose, rclcpp::Time current_time){
  last_decision_ = "d_initial";
  current_decision_ = "d_initial";
  oscillation_pose_ = robot_curent_pose;
  last_oscillation_reset_ = current_time;
  last_valid_plan_ = current_time;
  last_valid_control_ = current_time;
  no_plan_recovery_count_ = 0;
}

double FSM::getDistance(geometry_msgs::msg::TransformStamped& a, geometry_msgs::msg::TransformStamped& b){
  double dx = a.transform.translation.x - b.transform.translation.x;
  double dy = a.transform.translation.y - b.transform.translation.y;
  double dz = a.transform.translation.z - b.transform.translation.z;
  return sqrt(dx*dx + dy*dy + dz*dz);
}

double FSM::getAngle(geometry_msgs::msg::TransformStamped& a, geometry_msgs::msg::TransformStamped& b){
  tf2::Transform pose_a;
  pose_a.setRotation(tf2::Quaternion(a.transform.rotation.x, a.transform.rotation.y, a.transform.rotation.z, a.transform.rotation.w));
  pose_a.setOrigin(tf2::Vector3(a.transform.translation.x, a.transform.translation.y, a.transform.translation.z));
  
  tf2::Transform pose_b;
  pose_b.setRotation(tf2::Quaternion(b.transform.rotation.x, b.transform.rotation.y, b.transform.rotation.z, b.transform.rotation.w));
  pose_b.setOrigin(tf2::Vector3(b.transform.translation.x, b.transform.translation.y, b.transform.translation.z));
  
  auto pose_a_inverse = pose_a.inverse();
  tf2::Transform pose_a2b;
  pose_a2b.mult(pose_a_inverse, pose_b);
  tf2::Matrix3x3 m(pose_a2b.getRotation());
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);
  return fabs(yaw);
}

bool FSM::setDecision(std::string m_decision){
  last_decision_ = current_decision_;
  current_decision_ = m_decision;
  return true;
}

}  // namespace qi_move_base
