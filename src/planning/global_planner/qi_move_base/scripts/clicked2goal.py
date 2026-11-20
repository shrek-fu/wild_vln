#!/usr/bin/env python3
"""
clicked2goal.py  –  qi_move_base navigation trigger

Bridges a PoseStamped or PointStamped topic to the /p2p_move_base action server.

Subscribed topics:
  /goal_pose    (geometry_msgs/PoseStamped)  – full pose goal (position + heading)
  /clicked_point (geometry_msgs/PointStamped) – position-only goal (heading = identity)

Action client:
  /p2p_move_base  (dddmr_sys_core/PToPMoveBase)
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from dddmr_sys_core.action import PToPMoveBase
from geometry_msgs.msg import PoseStamped, PointStamped


class Clicked2Goal(Node):

    def __init__(self):
        super().__init__('clicked2goal')

        cb_group = MutuallyExclusiveCallbackGroup()

        # Subscribe to RViz "2D Nav Goal" / custom topic
        self.sub_pose = self.create_subscription(
            PoseStamped,
            'goal_pose_3d',
            self._goal_pose_cb,
            10,
            callback_group=cb_group)

        # Subscribe to RViz "Publish Point" (heading = identity quaternion)
        self.sub_point = self.create_subscription(
            PointStamped,
            'clicked_point',
            self._clicked_point_cb,
            10,
            callback_group=cb_group)

        self._action_client = ActionClient(self, PToPMoveBase, 'p2p_move_base')

        self._pending_goal = None
        self._goal_handle  = None

        # 10 Hz dispatch timer
        self._timer = self.create_timer(0.1, self._dispatch_timer_cb, callback_group=cb_group)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _goal_pose_cb(self, msg: PoseStamped):
        self.get_logger().info(
            f'New goal_pose: ({msg.pose.position.x:.2f}, '
            f'{msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})')
        goal = PToPMoveBase.Goal()
        goal.target_pose = msg
        self._pending_goal = goal

    def _clicked_point_cb(self, msg: PointStamped):
        self.get_logger().info(
            f'Clicked point: ({msg.point.x:.2f}, {msg.point.y:.2f}, {msg.point.z:.2f})')
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.x = msg.point.x
        pose.pose.position.y = msg.point.y
        pose.pose.position.z = msg.point.z
        pose.pose.orientation.w = 1.0   # identity (no heading constraint)
        goal = PToPMoveBase.Goal()
        goal.target_pose = pose
        self._pending_goal = goal

    # ── Timer dispatch ────────────────────────────────────────────────────────

    def _dispatch_timer_cb(self):
        if self._pending_goal is None:
            return

        goal = self._pending_goal
        self._pending_goal = None

        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('/p2p_move_base action server not available, dropping goal.')
            return

        self.get_logger().info('Sending goal to /p2p_move_base ...')
        send_future = self._action_client.send_goal_async(
            goal,
            feedback_callback=self._feedback_cb)
        send_future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected by /p2p_move_base.')
            return
        self.get_logger().info('Goal accepted.')
        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f'[{fb.current_decision}]  '
            f'pos=({fb.base_position.transform.translation.x:.2f}, '
            f'{fb.base_position.transform.translation.y:.2f})',
            throttle_duration_sec=1.0)

    def _result_cb(self, future):
        result = future.result()
        self.get_logger().info(f'Navigation finished. Code: {result.status}')
        self._goal_handle = None


def main(args=None):
    rclpy.init(args=args)
    node = Clicked2Goal()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
