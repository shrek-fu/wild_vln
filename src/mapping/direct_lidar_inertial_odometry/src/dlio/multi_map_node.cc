#include "dlio/multi_map.h"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<dlio::MultiMapNode>();
  // 6 groups: map / ground / non_ground / odom / pub_timer / save_srv
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 6);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
