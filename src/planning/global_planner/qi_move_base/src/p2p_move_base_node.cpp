/*
* BSD 3-Clause License
* Copyright (c) 2024, DDDMobileRobot
*
* qi_move_base_node
*
* Spins only three nodes:
*   1. P2PGlobalPlanManager  – queries global_planner action server
*   2. P2PMoveBase            – FSM that coordinates navigation
*
* The local motion (path following, obstacle avoidance, heading alignment)
* is handled by the standalone localPlanner + pathFollower nodes which are
* expected to be launched separately (see launch/qi_move_base.launch.py).
*
* Recovery behaviors are provided by the recovery_behaviors_ros node which
* is also launched separately and communicates via the /recovery_behaviors
* action server.
*/

#include <memory>
#include "qi_move_base/p2p_global_plan_manager.h"
#include "qi_move_base/p2p_move_base.h"

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor;

  auto node_gpm = std::make_shared<qi_move_base::P2PGlobalPlanManager>("global_plan_manager");
  auto node_p2p = std::make_shared<qi_move_base::P2PMoveBase>("p2p_move_base");

  executor.add_node(node_gpm);
  executor.add_node(node_p2p);

  node_gpm->initial();
  node_p2p->initial(node_gpm);

  executor.spin();

  rclcpp::shutdown();
  return 0;
}
