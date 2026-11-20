// Go2-W battery from the robot's rt/lowstate (unitree_sdk2, DDS domain 0 on the robot network) to ROS:
//   /go2w/battery      sensor_msgs/BatteryState  (percentage 0..1, voltage, current, temperature, cells)
//   /go2w/diagnostics  diagnostic_msgs/DiagnosticArray  OK / WARN (<= warn_percent) / ERROR (<= error_percent),
//                      STALE when rt/lowstate stops
// Read-only: no SportClient, no lease, so it runs next to QiNav's or our go2_cmd_vel_node.

#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/battery_state.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

using LowState = unitree_go::msg::dds_::LowState_;
using diagnostic_msgs::msg::DiagnosticStatus;

namespace
{

std::string ResolveNetworkInterface(const std::string & requested, const rclcpp::Logger & log)
{
  if (!requested.empty() && if_nametoindex(requested.c_str()) != 0) {
    return requested;
  }
  RCLCPP_WARN(log, "network_interface '%s' not found, trying fallback", requested.c_str());
  ifaddrs * ifaddr = nullptr;
  std::string found;
  if (getifaddrs(&ifaddr) == 0) {
    for (ifaddrs * ifa = ifaddr; ifa != nullptr && found.empty(); ifa = ifa->ifa_next) {
      if (ifa->ifa_name && ifa->ifa_addr && (ifa->ifa_flags & IFF_UP) && !(ifa->ifa_flags & IFF_LOOPBACK) &&
        ifa->ifa_addr->sa_family == AF_INET)
      {
        found = ifa->ifa_name;
      }
    }
    freeifaddrs(ifaddr);
  }
  return found.empty() ? requested : found;
}

struct Sample
{
  uint8_t soc;
  float voltage;
  int32_t current_ma;
  uint16_t cycle;
  uint8_t status;
  std::vector<float> cells;
  float temperature;
  rclcpp::Time stamp;
};

}  // namespace

class Go2wBatteryNode : public rclcpp::Node
{
public:
  Go2wBatteryNode()
  : Node("go2w_battery")
  {
    network_interface_ = declare_parameter<std::string>("network_interface", "eth0");
    robot_ip_ = declare_parameter<std::string>("robot_ip", "192.168.123.161");
    warn_percent_ = declare_parameter<int>("warn_percent", 30);
    error_percent_ = declare_parameter<int>("error_percent", 15);
    stale_s_ = declare_parameter<double>("stale_s", 5.0);
    const double rate = declare_parameter<double>("publish_rate_hz", 1.0);

    pub_battery_ = create_publisher<sensor_msgs::msg::BatteryState>("/go2w/battery", 10);
    pub_diag_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/go2w/diagnostics", 10);

    network_interface_ = ResolveNetworkInterface(network_interface_, get_logger());
    std::ostringstream dds;
    dds << "<CycloneDDS><Domain id=\"any\"><General><Interfaces><NetworkInterface name=\"" << network_interface_
        << "\" /></Interfaces></General><Discovery><Peers><Peer Address=\"" << robot_ip_
        << "\" /></Peers></Discovery></Domain></CycloneDDS>";
    setenv("CYCLONEDDS_URI", dds.str().c_str(), 1);
    unitree::robot::ChannelFactory::Instance()->Init(0, network_interface_);
    sub_ = std::make_shared<unitree::robot::ChannelSubscriber<LowState>>("rt/lowstate");
    sub_->InitChannel([this](const void * msg) {OnLowState(*static_cast<const LowState *>(msg));}, 1);

    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / std::max(rate, 0.1)), [this]() {Publish();});
    RCLCPP_INFO(
      get_logger(), "rt/lowstate on %s (robot %s) -> /go2w/battery, /go2w/diagnostics (warn <= %d%%, error <= %d%%)",
      network_interface_.c_str(), robot_ip_.c_str(), warn_percent_, error_percent_);
  }

private:
  void OnLowState(const LowState & msg)
  {
    const auto & bms = msg.bms_state();
    Sample s{bms.soc(), msg.power_v(), bms.current(), bms.cycle(), bms.status(), {}, 0.0f, now()};
    for (uint16_t mv : bms.cell_vol()) {
      if (mv > 0) {
        s.cells.push_back(mv / 1000.0f);
      }
    }
    s.temperature = static_cast<float>(std::max({bms.bq_ntc()[0], bms.bq_ntc()[1], bms.mcu_ntc()[0], bms.mcu_ntc()[1]}));
    std::lock_guard<std::mutex> lock(mutex_);
    latest_ = std::move(s);
  }

  void Publish()
  {
    std::optional<Sample> s;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      s = latest_;
    }
    const rclcpp::Time t = now();
    DiagnosticStatus status;
    status.name = "go2w: battery";
    status.hardware_id = robot_ip_;
    if (!s || (t - s->stamp).seconds() > stale_s_) {
      status.level = DiagnosticStatus::STALE;
      status.message = "no rt/lowstate from the robot on " + network_interface_;
      Report(status);
      return;
    }

    sensor_msgs::msg::BatteryState b;
    b.header.stamp = s->stamp;
    b.header.frame_id = "base_link";
    b.percentage = s->soc / 100.0f;
    b.voltage = s->voltage;
    b.current = s->current_ma / 1000.0f;
    b.temperature = s->temperature;
    b.cell_voltage = s->cells;
    b.cell_temperature = {};
    b.charge = b.capacity = b.design_capacity = std::numeric_limits<float>::quiet_NaN();
    b.power_supply_status = s->current_ma > 0 ? sensor_msgs::msg::BatteryState::POWER_SUPPLY_STATUS_CHARGING :
      sensor_msgs::msg::BatteryState::POWER_SUPPLY_STATUS_DISCHARGING;
    b.power_supply_technology = sensor_msgs::msg::BatteryState::POWER_SUPPLY_TECHNOLOGY_LION;
    b.present = true;
    b.serial_number = "cycles " + std::to_string(s->cycle);
    pub_battery_->publish(b);

    const int soc = s->soc;
    status.level = soc <= error_percent_ ? DiagnosticStatus::ERROR :
      soc <= warn_percent_ ? DiagnosticStatus::WARN : DiagnosticStatus::OK;
    std::ostringstream text;
    text.precision(3);
    text << soc << "% " << s->voltage << " V " << b.current << " A";
    if (status.level == DiagnosticStatus::ERROR) {
      text << " - LOW BATTERY, charge now";
    } else if (status.level == DiagnosticStatus::WARN) {
      text << " - battery low";
    }
    status.message = text.str();
    auto kv = [&status](const std::string & k, const std::string & v) {
        diagnostic_msgs::msg::KeyValue p;
        p.key = k;
        p.value = v;
        status.values.push_back(p);
      };
    kv("percent", std::to_string(soc));
    kv("voltage_v", std::to_string(s->voltage));
    kv("current_a", std::to_string(b.current));
    kv("temperature_c", std::to_string(s->temperature));
    kv("cycles", std::to_string(s->cycle));
    kv("bms_status", std::to_string(s->status));
    Report(status);
  }

  void Report(const DiagnosticStatus & status)
  {
    diagnostic_msgs::msg::DiagnosticArray arr;
    arr.header.stamp = now();
    arr.status.push_back(status);
    pub_diag_->publish(arr);
    if (status.level != last_level_) {
      if (status.level == DiagnosticStatus::OK) {
        RCLCPP_INFO(get_logger(), "battery %s", status.message.c_str());
      } else if (status.level == DiagnosticStatus::WARN || status.level == DiagnosticStatus::STALE) {
        RCLCPP_WARN(get_logger(), "battery %s", status.message.c_str());
      } else {
        RCLCPP_ERROR(get_logger(), "battery %s", status.message.c_str());
      }
      last_level_ = status.level;
    }
  }

  std::string network_interface_, robot_ip_;
  int warn_percent_, error_percent_;
  double stale_s_;
  rclcpp::Publisher<sensor_msgs::msg::BatteryState>::SharedPtr pub_battery_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr pub_diag_;
  std::shared_ptr<unitree::robot::ChannelSubscriber<LowState>> sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::mutex mutex_;
  std::optional<Sample> latest_;
  uint8_t last_level_ = 255;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Go2wBatteryNode>());
  rclcpp::shutdown();
  return 0;
}
