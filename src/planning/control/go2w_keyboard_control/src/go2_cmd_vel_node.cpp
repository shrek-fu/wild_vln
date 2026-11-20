#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

class Go2CmdVelNode : public rclcpp::Node
{
public:
  Go2CmdVelNode()
  : Node("go2_cmd_vel_node")
  {
    robot_ip_ = declare_parameter<std::string>("robot_ip", "192.168.123.161");
    network_interface_ = declare_parameter<std::string>("network_interface", "eth0");
    cmd_vel_topic_ = declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    posture_cmd_topic_ = declare_parameter<std::string>("posture_cmd_topic", "/unitree/posture_cmd");
    cmd_vel_timeout_ = declare_parameter<double>("cmd_vel_timeout", 0.5);
    publish_rate_ = declare_parameter<double>("publish_rate", 50.0);
    max_linear_x_ = declare_parameter<double>("max_linear_x", 0.6);
    max_linear_y_ = declare_parameter<double>("max_linear_y", 0.4);
    max_angular_z_ = declare_parameter<double>("max_angular_z", 0.8);
    auto_stand_up_ = declare_parameter<bool>("auto_stand_up", false);
    auto_prepare_motion_ = declare_parameter<bool>("auto_prepare_motion", true);
    prepare_retry_sec_ = declare_parameter<double>("prepare_retry_sec", 1.0);
    // Navigation goals that stand the robot up (once, while it is not prepared); "" = off.
    goal_point_topic_ = declare_parameter<std::string>("goal_point_topic", "");
    goal_pose_topic_ = declare_parameter<std::string>("goal_pose_topic", "");

    cmd_vel_timeout_ = std::max(0.1, cmd_vel_timeout_);
    publish_rate_ = std::clamp(publish_rate_, 10.0, 100.0);
    max_linear_x_ = std::fabs(max_linear_x_);
    max_linear_y_ = std::fabs(max_linear_y_);
    max_angular_z_ = std::fabs(max_angular_z_);

    InitializeSdk();

    subscription_ = create_subscription<geometry_msgs::msg::Twist>(
      cmd_vel_topic_, rclcpp::QoS(10),
      std::bind(&Go2CmdVelNode::CmdVelCallback, this, std::placeholders::_1));
    posture_sub_ = create_subscription<std_msgs::msg::Int32>(
      posture_cmd_topic_, rclcpp::QoS(10),
      std::bind(&Go2CmdVelNode::PostureCallback, this, std::placeholders::_1));
    if (!goal_point_topic_.empty()) {
      goal_point_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
        goal_point_topic_, rclcpp::QoS(10),
        [this](geometry_msgs::msg::PointStamped::ConstSharedPtr) {PrepareMotion(goal_point_topic_);});
    }
    if (!goal_pose_topic_.empty()) {
      goal_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        goal_pose_topic_, rclcpp::QoS(10),
        [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr) {PrepareMotion(goal_pose_topic_);});
    }

    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / publish_rate_),
      std::bind(&Go2CmdVelNode::SendCommand, this));

    RCLCPP_INFO(
      get_logger(),
      "Go2/Go2W cmd_vel controller ready: topic=%s, posture=%s, interface=%s",
      cmd_vel_topic_.c_str(), posture_cmd_topic_.c_str(), network_interface_.c_str());
  }

  ~Go2CmdVelNode() override
  {
    if (sport_client_) {
      (void)sport_client_->StopMove();
    }
  }

private:
  std::string ResolveNetworkInterface(const std::string & requested)
  {
    if (!requested.empty() && if_nametoindex(requested.c_str()) != 0) {
      return requested;
    }
    if (!requested.empty()) {
      RCLCPP_WARN(
        get_logger(), "Configured network_interface '%s' not found, trying fallback.",
        requested.c_str());
    }

    std::vector<std::string> candidates;
    ifaddrs * ifaddr = nullptr;
    if (getifaddrs(&ifaddr) == 0 && ifaddr != nullptr) {
      for (ifaddrs * ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_name || !ifa->ifa_addr) {
          continue;
        }
        if ((ifa->ifa_flags & IFF_UP) == 0 || (ifa->ifa_flags & IFF_LOOPBACK) != 0) {
          continue;
        }
        if (ifa->ifa_addr->sa_family != AF_INET) {
          continue;
        }
        const std::string iface = ifa->ifa_name;
        if (std::find(candidates.begin(), candidates.end(), iface) == candidates.end()) {
          candidates.push_back(iface);
        }
      }
      freeifaddrs(ifaddr);
    }
    for (const auto & iface : candidates) {
      if (if_nametoindex(iface.c_str()) != 0) {
        RCLCPP_INFO(get_logger(), "Using detected network interface: %s", iface.c_str());
        return iface;
      }
    }
    static const std::vector<std::string> kFallback{"eth0", "enp1s0", "eno1", "wlan0"};
    for (const auto & iface : kFallback) {
      if (if_nametoindex(iface.c_str()) != 0) {
        RCLCPP_INFO(get_logger(), "Using network interface: %s", iface.c_str());
        return iface;
      }
    }
    return requested;
  }

  void InitializeSdk()
  {
    network_interface_ = ResolveNetworkInterface(network_interface_);
    std::ostringstream dds_config;
    dds_config << "<CycloneDDS><Domain id=\"any\"><General><Interfaces>"
               << "<NetworkInterface name=\"" << network_interface_ << "\" />"
               << "</Interfaces></General><Discovery><Peers><Peer Address=\""
               << robot_ip_ << "\" /></Peers></Discovery></Domain></CycloneDDS>";
    setenv("CYCLONEDDS_URI", dds_config.str().c_str(), 1);

    unitree::robot::ChannelFactory::Instance()->Init(0, network_interface_);
    sport_client_ = std::make_unique<unitree::robot::go2::SportClient>();
    sport_client_->SetTimeout(10.0f);
    sport_client_->Init();
    sport_client_->WaitLeaseApplied();

    if (auto_stand_up_) {
      const int32_t result = StandForMotion();
      motion_prepared_ = (result >= 0);
      RCLCPP_INFO(get_logger(), "StandUp+BalanceStand returned %d", result);
    }
  }

  static float Clamp(double value, double limit)
  {
    return static_cast<float>(std::clamp(value, -limit, limit));
  }

  bool EnsureMotionPrepared()
  {
    if (motion_prepared_) {
      return true;
    }
    if (!auto_prepare_motion_) {
      return false;
    }
    const rclcpp::Time now_time = now();
    if (last_prepare_attempt_set_ &&
      (now_time - last_prepare_attempt_).seconds() < prepare_retry_sec_)
    {
      return false;
    }
    last_prepare_attempt_ = now_time;
    last_prepare_attempt_set_ = true;
    return PrepareMotion("first motion command");
  }

  bool PrepareMotion(const std::string & reason)
  {
    if (motion_prepared_) {
      return true;
    }
    const int32_t ret_stand = StandForMotion();
    const int32_t ret_speed = sport_client_->SpeedLevel(1);
    RCLCPP_INFO(
      get_logger(), "prepare (%s): StandUp+BalanceStand=%d SpeedLevel=%d", reason.c_str(), ret_stand,
      ret_speed);
    motion_prepared_ = (ret_stand >= 0);
    return motion_prepared_;
  }

  // StandUp() alone leaves the joints locked and Move() is accepted but ignored; BalanceStand() is the
  // locomotion-ready stand (same order as go2w_sport_client_demo).
  int32_t StandForMotion()
  {
    const int32_t ret_up = sport_client_->StandUp();
    if (ret_up < 0) {
      return ret_up;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
    const int32_t ret_balance = sport_client_->BalanceStand();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    return ret_balance;
  }

  void CmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    vx_ = Clamp(msg->linear.x, max_linear_x_);
    vy_ = Clamp(msg->linear.y, max_linear_y_);
    wz_ = Clamp(msg->angular.z, max_angular_z_);
    last_command_time_ = now();
    command_received_ = true;
  }

  void PostureCallback(const std_msgs::msg::Int32::SharedPtr msg)
  {
    const int32_t code = msg->data;
    int32_t result = 0;
    switch (code) {
      case 0:
        result = sport_client_->StopMove();
        moving_ = false;
        RCLCPP_INFO(get_logger(), "posture StopMove -> %d", result);
        break;
      case 1:
        result = StandForMotion();
        motion_prepared_ = (result >= 0);
        RCLCPP_INFO(get_logger(), "posture StandUp+BalanceStand -> %d", result);
        break;
      case 5:
        result = sport_client_->BalanceStand();
        motion_prepared_ = (result >= 0);
        RCLCPP_INFO(get_logger(), "posture BalanceStand -> %d", result);
        break;
      case 2:
        (void)sport_client_->StopMove();
        result = sport_client_->StandDown();
        motion_prepared_ = false;
        moving_ = false;
        RCLCPP_INFO(get_logger(), "posture StandDown -> %d", result);
        break;
      case 3:
        result = sport_client_->Damp();
        motion_prepared_ = false;
        moving_ = false;
        RCLCPP_INFO(get_logger(), "posture Damp -> %d", result);
        break;
      case 4:
        result = sport_client_->Hello();
        RCLCPP_INFO(get_logger(), "posture Hello -> %d", result);
        break;
      case 21:
      case 22:
      case 23:
        result = sport_client_->SpeedLevel(code - 20);
        RCLCPP_INFO(get_logger(), "posture SpeedLevel(%d) -> %d", code - 20, result);
        break;
      default:
        RCLCPP_WARN(get_logger(), "unsupported posture_cmd=%d", code);
        break;
    }
  }

  void SendCommand()
  {
    if (!command_received_) {
      return;
    }

    if ((now() - last_command_time_).seconds() > cmd_vel_timeout_) {
      if (moving_) {
        const int32_t result = sport_client_->StopMove();
        moving_ = false;
        RCLCPP_WARN(get_logger(), "cmd_vel timed out; StopMove() returned %d", result);
      }
      command_received_ = false;
      return;
    }

    constexpr float kZero = 1e-4f;
    const bool zero_command =
      std::fabs(vx_) < kZero && std::fabs(vy_) < kZero && std::fabs(wz_) < kZero;
    if (zero_command) {
      if (moving_) {
        const int32_t result = sport_client_->StopMove();
        moving_ = false;
        RCLCPP_INFO(get_logger(), "Zero cmd_vel; StopMove() returned %d", result);
      }
      return;
    }

    if (!EnsureMotionPrepared()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1500, "motion requested but robot is not prepared");
      return;
    }

    const int32_t result = sport_client_->Move(vx_, vy_, wz_);
    moving_ = result >= 0;
    if (result < 0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000, "Move(%.3f, %.3f, %.3f) failed: %d",
        vx_, vy_, wz_, result);
    }
  }

  std::unique_ptr<unitree::robot::go2::SportClient> sport_client_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr subscription_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr posture_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr goal_point_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pose_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::string robot_ip_;
  std::string network_interface_;
  std::string cmd_vel_topic_;
  std::string posture_cmd_topic_;
  std::string goal_point_topic_;
  std::string goal_pose_topic_;
  double cmd_vel_timeout_;
  double publish_rate_;
  double max_linear_x_;
  double max_linear_y_;
  double max_angular_z_;
  bool auto_stand_up_;
  bool auto_prepare_motion_;
  double prepare_retry_sec_;
  bool motion_prepared_{false};
  bool last_prepare_attempt_set_{false};
  rclcpp::Time last_prepare_attempt_;

  float vx_{0.0F};
  float vy_{0.0F};
  float wz_{0.0F};
  bool command_received_{false};
  bool moving_{false};
  rclcpp::Time last_command_time_;
};

int main(int argc, char ** argv)
{
  setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp", 1);
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<Go2CmdVelNode>());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("go2_cmd_vel_node"), "%s", error.what());
  }
  rclcpp::shutdown();
  return 0;
}
