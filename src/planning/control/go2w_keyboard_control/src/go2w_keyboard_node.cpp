#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

class Go2wKeyboardNode : public rclcpp::Node
{
public:
  Go2wKeyboardNode()
  : Node("go2w_keyboard_node"),
    keyboard_ready_(false),
    motion_active_(false),
    vx_(0.0f),
    vy_(0.0f),
    wz_(0.0f)
  {
    robot_ip_ = declare_parameter<std::string>("robot_ip", "192.168.123.161");
    network_interface_ = declare_parameter<std::string>("network_interface", "eno1");
    cmd_vel_topic_ = declare_parameter<std::string>("cmd_vel_topic", "/ql/nav/robot_cmd_vel");
    posture_cmd_topic_ = declare_parameter<std::string>("posture_cmd_topic", "/go2w/posture_cmd");
    cmd_vel_timeout_sec_ = declare_parameter<double>("cmd_vel_timeout_sec", 0.30);
    zero_cmd_confirm_sec_ = declare_parameter<double>("zero_cmd_confirm_sec", 0.20);
    auto_prepare_motion_ = declare_parameter<bool>("auto_prepare_motion", true);
    prepare_retry_sec_ = declare_parameter<double>("prepare_retry_sec", 1.0);
    linear_speed_ = declare_parameter<double>("linear_speed", 0.35);
    lateral_speed_ = declare_parameter<double>("lateral_speed", 0.20);
    yaw_speed_ = declare_parameter<double>("yaw_speed", 0.60);
    min_linear_cmd_ = declare_parameter<double>("min_linear_cmd", 0.18);
    min_lateral_cmd_ = declare_parameter<double>("min_lateral_cmd", 0.12);
    min_yaw_cmd_ = declare_parameter<double>("min_yaw_cmd", 0.25);
    move_send_hz_ = declare_parameter<double>("move_send_hz", 50.0);
    move_send_hz_ = std::clamp(move_send_hz_, 10.0, 100.0);

    InitKeyboard();
    PrintHelp();
    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      cmd_vel_topic_, 20,
      std::bind(&Go2wKeyboardNode::OnCmdVel, this, std::placeholders::_1));
    posture_cmd_sub_ = create_subscription<std_msgs::msg::Int32>(
      posture_cmd_topic_, 20,
      std::bind(&Go2wKeyboardNode::OnPostureCmd, this, std::placeholders::_1));

    {
      const double period_sec = 1.0 / move_send_hz_;
      timer_ = create_wall_timer(
        std::chrono::duration<double>(period_sec),
        std::bind(&Go2wKeyboardNode::OnTimer, this));
    }
    RCLCPP_INFO(
      get_logger(),
      "Move loop: %.1f Hz — SDK requires repeated Move() while moving (see unitree examples).",
      move_send_hz_);
  }

  ~Go2wKeyboardNode() override
  {
    if (motion_active_ && sport_client_) {
      (void)sport_client_->StopMove();
    }
    RestoreKeyboard();
  }

private:
  bool EnsureMotionPrepared()
  {
    if (motion_prepared_) {
      return true;
    }
    if (!auto_prepare_motion_) {
      return false;
    }
    const rclcpp::Time now_time = now();
    const bool allow_retry =
      !last_prepare_attempt_set_ ||
      (now_time - last_prepare_attempt_).seconds() >= prepare_retry_sec_;
    if (!allow_retry) {
      return false;
    }

    last_prepare_attempt_ = now_time;
    last_prepare_attempt_set_ = true;
    RCLCPP_INFO(get_logger(), "Preparing motion state: StandUp -> SpeedLevel(1)");
    const int32_t ret_stand = sport_client_->StandUp();
    const int32_t ret_speed = sport_client_->SpeedLevel(1);
    RCLCPP_INFO(
      get_logger(),
      "Prepare results: StandUp=%d, SpeedLevel=%d",
      ret_stand, ret_speed);

    // Some firmware may return -1 for SpeedLevel while StandUp + Move still works.
    motion_prepared_ = (ret_stand >= 0);
    return motion_prepared_;
  }

  bool EnsureSdkInitialized()
  {
    if (sdk_initialized_) {
      return true;
    }
    if (sdk_init_attempted_) {
      return false;
    }

    sdk_init_attempted_ = true;
    network_interface_ = ResolveNetworkInterface(network_interface_);
    ConfigureCycloneDDS();
    unitree::robot::ChannelFactory::Instance()->Init(0, network_interface_);
    sport_client_ = std::make_unique<unitree::robot::go2::SportClient>();
    sport_client_->SetTimeout(10.0f);
    sport_client_->Init();
    sport_client_->WaitLeaseApplied();
    sdk_initialized_ = true;
    RCLCPP_INFO(get_logger(), "Unitree SDK initialized.");
    return true;
  }

  std::string ResolveNetworkInterface(const std::string & requested)
  {
    if (!requested.empty() && if_nametoindex(requested.c_str()) != 0) {
      return requested;
    }

    if (!requested.empty()) {
      RCLCPP_WARN(
        get_logger(),
        "Configured network_interface '%s' not found, trying fallback interfaces.",
        requested.c_str());
    }

    // Prefer interfaces that are up and have IPv4 address.
    std::vector<std::string> dynamic_candidates;
    ifaddrs * ifaddr = nullptr;
    if (getifaddrs(&ifaddr) == 0 && ifaddr != nullptr) {
      for (ifaddrs * ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_name || !ifa->ifa_addr) {
          continue;
        }
        if ((ifa->ifa_flags & IFF_UP) == 0) {
          continue;
        }
        if (ifa->ifa_flags & IFF_LOOPBACK) {
          continue;
        }
        if (ifa->ifa_addr->sa_family != AF_INET) {
          continue;
        }
        const std::string iface = ifa->ifa_name;
        if (std::find(dynamic_candidates.begin(), dynamic_candidates.end(), iface) == dynamic_candidates.end()) {
          dynamic_candidates.push_back(iface);
        }
      }
      freeifaddrs(ifaddr);
    }
    for (const auto & iface : dynamic_candidates) {
      if (if_nametoindex(iface.c_str()) != 0) {
        RCLCPP_INFO(get_logger(), "Using detected network interface: %s", iface.c_str());
        return iface;
      }
    }

    static const std::vector<std::string> kCandidates{
      "eth0", "enp1s0", "enp2s0", "enp3s0", "eno1", "wlan0"};
    for (const auto & iface : kCandidates) {
      if (if_nametoindex(iface.c_str()) != 0) {
        RCLCPP_INFO(get_logger(), "Using network interface: %s", iface.c_str());
        return iface;
      }
    }

    RCLCPP_WARN(get_logger(), "No usable network interface found, fallback to requested='%s'.", requested.c_str());
    return requested;
  }

  void ConfigureCycloneDDS()
  {
    std::ostringstream xml;
    xml << "<CycloneDDS>"
        << "<Domain id=\"any\">"
        << "<General><Interfaces><NetworkInterface name=\"" << network_interface_
        << "\" /></Interfaces></General>"
        << "<Discovery><Peers><Peer Address=\"" << robot_ip_
        << "\" /></Peers></Discovery>"
        << "</Domain>"
        << "</CycloneDDS>";

    setenv("CYCLONEDDS_URI", xml.str().c_str(), 1);
    RCLCPP_INFO(
      get_logger(),
      "Go2W target ip: %s, wired interface: %s",
      robot_ip_.c_str(), network_interface_.c_str());
  }

  void InitKeyboard()
  {
    if (!isatty(STDIN_FILENO)) {
      RCLCPP_WARN(
        get_logger(),
        "STDIN is not a TTY, keyboard input disabled. Node will run in cmd_vel-only mode.");
      return;
    }

    if (tcgetattr(STDIN_FILENO, &original_termios_) != 0) {
      RCLCPP_WARN(get_logger(), "Failed to read terminal attributes, keyboard input disabled.");
      return;
    }

    termios raw = original_termios_;
    raw.c_lflag &= static_cast<unsigned long>(~(ICANON | ECHO));
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);

    const int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    keyboard_ready_ = true;
  }

  void RestoreKeyboard()
  {
    if (keyboard_ready_) {
      tcsetattr(STDIN_FILENO, TCSANOW, &original_termios_);
      keyboard_ready_ = false;
    }
  }

  void PrintHelp() const
  {
    if (keyboard_ready_) {
      RCLCPP_INFO(get_logger(), "Keyboard control ready.");
    } else {
      RCLCPP_INFO(get_logger(), "Keyboard input disabled, cmd_vel control ready.");
    }
    RCLCPP_INFO(get_logger(), "Also subscribing cmd_vel topic: %s", cmd_vel_topic_.c_str());
    RCLCPP_INFO(get_logger(), "Also subscribing posture topic: %s", posture_cmd_topic_.c_str());
    RCLCPP_INFO(get_logger(), "1: stand up, 2: stand down");
    RCLCPP_INFO(get_logger(), "w/s: forward/backward, q/e: left/right shift");
    RCLCPP_INFO(get_logger(), "a/d: turn left/right, space: stop velocity");
    RCLCPP_INFO(get_logger(), "x: stop_move, h: show help, Ctrl+C: exit");
  }

  static float Clamp(float value, float limit)
  {
    const float abs_limit = std::fabs(limit);
    if (value > abs_limit) {
      return abs_limit;
    }
    if (value < -abs_limit) {
      return -abs_limit;
    }
    return value;
  }

  static float ApplyMinMagnitude(float value, float min_abs)
  {
    const float abs_value = std::fabs(value);
    const float min_value = std::fabs(min_abs);
    if (abs_value < 1e-3f) {
      return 0.0f;
    }
    if (abs_value < min_value) {
      return std::copysign(min_value, value);
    }
    return value;
  }

  void OnCmdVel(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    float new_vx = Clamp(static_cast<float>(msg->linear.x), static_cast<float>(linear_speed_));
    float new_vy = Clamp(static_cast<float>(msg->linear.y), static_cast<float>(lateral_speed_));
    float new_wz = Clamp(static_cast<float>(msg->angular.z), static_cast<float>(yaw_speed_));
    new_vx = ApplyMinMagnitude(new_vx, static_cast<float>(min_linear_cmd_));
    new_vy = ApplyMinMagnitude(new_vy, static_cast<float>(min_lateral_cmd_));
    new_wz = ApplyMinMagnitude(new_wz, static_cast<float>(min_yaw_cmd_));

    constexpr float kZeroEps = 1e-3f;
    const bool is_zero_cmd =
      std::fabs(new_vx) < kZeroEps &&
      std::fabs(new_vy) < kZeroEps &&
      std::fabs(new_wz) < kZeroEps;

    if (!is_zero_cmd) {
      vx_ = new_vx;
      vy_ = new_vy;
      wz_ = new_wz;
      zero_cmd_first_seen_set_ = false;
    } else {
      const rclcpp::Time now_time = now();
      if (!zero_cmd_first_seen_set_) {
        zero_cmd_first_seen_ = now_time;
        zero_cmd_first_seen_set_ = true;
      }
      const double zero_hold = (now_time - zero_cmd_first_seen_).seconds();
      if (zero_hold >= zero_cmd_confirm_sec_) {
        vx_ = 0.0f;
        vy_ = 0.0f;
        wz_ = 0.0f;
      }
    }
    const double hold_sec = std::max(0.8, cmd_vel_timeout_sec_);
    motion_deadline_ = now() + rclcpp::Duration::from_seconds(hold_sec);
    motion_deadline_set_ = true;

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 500,
      "cmd_vel recv: linear(x=%.3f, y=%.3f, z=%.3f), angular(x=%.3f, y=%.3f, z=%.3f) -> "
      "clamped(vx=%.3f, vy=%.3f, wz=%.3f)",
      msg->linear.x, msg->linear.y, msg->linear.z,
      msg->angular.x, msg->angular.y, msg->angular.z,
      vx_, vy_, wz_);
  }

  void OnPostureCmd(const std_msgs::msg::Int32::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    RCLCPP_INFO(
      get_logger(),
      "[posture_cmd] recv topic=%s data=%d (1=StandUp, 2=StandDown)",
      posture_cmd_topic_.c_str(),
      msg->data);

    if (!EnsureSdkInitialized()) {
      RCLCPP_WARN(get_logger(), "[posture_cmd] SDK not initialized, ignore command=%d.", msg->data);
      return;
    }

    switch (msg->data) {
      case 1:
      {
        RCLCPP_INFO(get_logger(), "[posture_cmd] forwarding to SDK: StandUp()");
        const int32_t ret = sport_client_->StandUp();
        motion_prepared_ = (ret >= 0);
        RCLCPP_INFO(get_logger(), "[posture_cmd] SDK result StandUp() -> %d", ret);
        break;
      }
      case 2:
      {
        RCLCPP_INFO(get_logger(), "[posture_cmd] forwarding to SDK: StandDown()");
        const int32_t ret = sport_client_->StandDown();
        motion_prepared_ = false;
        RCLCPP_INFO(get_logger(), "[posture_cmd] SDK result StandDown() -> %d", ret);
        break;
      }
      default:
        RCLCPP_WARN(
          get_logger(),
          "[posture_cmd] unsupported command: %d (expect 1=StandUp, 2=StandDown)",
          msg->data);
        break;
    }
  }

  void OnTimer()
  {
    ReadKeyboard();
    PublishMotion();
  }

  void ReadKeyboard()
  {
    if (!keyboard_ready_) {
      return;
    }

    char c = 0;
    while (read(STDIN_FILENO, &c, 1) > 0) {
      HandleKey(c);
    }
  }

  void HandleKey(char c)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    switch (c) {
      case '1':
      {
        if (!EnsureSdkInitialized()) {
          RCLCPP_WARN(get_logger(), "SDK not initialized, ignore stand up.");
          break;
        }
        const int32_t ret = sport_client_->StandUp();
        motion_prepared_ = (ret >= 0);
        RCLCPP_INFO(get_logger(), "StandUp -> %d", ret);
        break;
      }
      case '2':
      {
        if (!EnsureSdkInitialized()) {
          RCLCPP_WARN(get_logger(), "SDK not initialized, ignore stand down.");
          break;
        }
        const int32_t ret = sport_client_->StandDown();
        motion_prepared_ = false;
        RCLCPP_INFO(get_logger(), "StandDown -> %d", ret);
        break;
      }
      case 'w':
        vx_ = static_cast<float>(linear_speed_);
        vy_ = 0.0f;
        wz_ = 0.0f;
        ExtendMotionDeadline();
        break;
      case 's':
        vx_ = static_cast<float>(-linear_speed_);
        vy_ = 0.0f;
        wz_ = 0.0f;
        ExtendMotionDeadline();
        break;
      case 'q':
        vx_ = 0.0f;
        vy_ = static_cast<float>(lateral_speed_);
        wz_ = 0.0f;
        ExtendMotionDeadline();
        break;
      case 'e':
        vx_ = 0.0f;
        vy_ = static_cast<float>(-lateral_speed_);
        wz_ = 0.0f;
        ExtendMotionDeadline();
        break;
      case 'a':
        vx_ = 0.0f;
        vy_ = 0.0f;
        wz_ = static_cast<float>(yaw_speed_);
        ExtendMotionDeadline();
        break;
      case 'd':
        vx_ = 0.0f;
        vy_ = 0.0f;
        wz_ = static_cast<float>(-yaw_speed_);
        ExtendMotionDeadline();
        break;
      case ' ':
      {
        if (!EnsureSdkInitialized()) {
          RCLCPP_WARN(get_logger(), "SDK not initialized, ignore stop.");
          break;
        }
        vx_ = 0.0f;
        vy_ = 0.0f;
        wz_ = 0.0f;
        const int32_t ret = sport_client_->StopMove();
        motion_active_ = false;
        motion_prepared_ = false;
        RCLCPP_INFO(get_logger(), "StopMove(space) -> %d", ret);
        break;
      }
      case 'x':
      {
        if (!EnsureSdkInitialized()) {
          RCLCPP_WARN(get_logger(), "SDK not initialized, ignore stop_move.");
          break;
        }
        const int32_t ret = sport_client_->StopMove();
        motion_active_ = false;
        motion_prepared_ = false;
        RCLCPP_INFO(get_logger(), "StopMove(x) -> %d", ret);
        break;
      }
      case 'h':
        PrintHelp();
        break;
      default:
        break;
    }
  }

  void ExtendMotionDeadline()
  {
    const double hold_sec = std::max(0.8, cmd_vel_timeout_sec_);
    motion_deadline_ = now() + rclcpp::Duration::from_seconds(hold_sec);
    motion_deadline_set_ = true;
  }

  // Unitree SportClient::Move must be invoked periodically (~50 Hz in SDK demos); if it stops,
  // the controller times out. This loop re-sends the last cmd_vel-derived velocity every tick.
  void PublishMotion()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!motion_deadline_set_) {
      return;
    }

    if (now() < motion_deadline_) {
      if (!EnsureSdkInitialized()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000, "SDK not initialized, skip Move command.");
        return;
      }
      if ((std::fabs(vx_) > 1e-3f || std::fabs(vy_) > 1e-3f || std::fabs(wz_) > 1e-3f) &&
        !EnsureMotionPrepared())
      {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1500,
          "Motion requested but robot is not prepared, retrying...");
        return;
      }
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "sending Move(vx=%.3f, vy=%.3f, wz=%.3f)",
        vx_, vy_, wz_);
      const int32_t ret = sport_client_->Move(vx_, vy_, wz_);
      motion_active_ = (ret >= 0);
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Move return code: %d", ret);
      if (ret < 0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1500, "Move failed: %d", ret);
      }
      return;
    }

    if (motion_active_) {
      if (!EnsureSdkInitialized()) {
        return;
      }
      const int32_t ret = sport_client_->StopMove();
      motion_active_ = false;
      if (ret < 0) {
        RCLCPP_WARN(get_logger(), "StopMove failed: %d", ret);
      }
    }
  }

private:
  std::unique_ptr<unitree::robot::go2::SportClient> sport_client_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr posture_cmd_sub_;

  std::string robot_ip_;
  std::string network_interface_;
  std::string cmd_vel_topic_;
  std::string posture_cmd_topic_;
  double cmd_vel_timeout_sec_;
  double zero_cmd_confirm_sec_;
  bool auto_prepare_motion_;
  double prepare_retry_sec_;
  double linear_speed_;
  double lateral_speed_;
  double yaw_speed_;
  double min_linear_cmd_;
  double min_lateral_cmd_;
  double min_yaw_cmd_;
  double move_send_hz_;

  termios original_termios_{};
  bool keyboard_ready_;

  std::mutex mutex_;
  bool motion_active_;
  bool sdk_initialized_{false};
  bool sdk_init_attempted_{false};
  bool motion_prepared_{false};
  bool last_prepare_attempt_set_{false};
  rclcpp::Time last_prepare_attempt_;
  float vx_;
  float vy_;
  float wz_;
  bool zero_cmd_first_seen_set_{false};
  rclcpp::Time zero_cmd_first_seen_;
  bool motion_deadline_set_{false};
  rclcpp::Time motion_deadline_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Go2wKeyboardNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
