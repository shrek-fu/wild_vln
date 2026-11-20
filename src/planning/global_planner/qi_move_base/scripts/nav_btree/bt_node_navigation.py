#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from dddmr_sys_core.action import PToPMoveBase
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import py_trees
import time

STATUS_SUCCEEDED = 4
STATUS_CANCELED = 5
STATUS_ABORTED = 6


class NavigateToPoseBt(py_trees.behaviour.Behaviour):
    def __init__(self, name, pose, node: Node, status_pub: rclpy.publisher.Publisher):
        super().__init__(name)
        self.pose = pose
        self.node = node
        self._client = None
        self._send_goal_future = None
        self._result_future = None
        self._goal_handle = None
        self.cancel_requested = False
        self.failure_reason = ""
        self.pub = status_pub
        self.pose_pub = self.node.create_publisher(PoseStamped, "/target_pose_for_vis", 10)

        # 创建服务接口
        self.node.create_service(Trigger, 'cancel_navigation', self.cancel_callback)
        self.node.get_logger().info("Cancel navigation service 'cancel_navigation' created.")

    def setup(self, **kwargs):
        self._client = ActionClient(self.node, PToPMoveBase, 'p2p_move_base')
        return True

    def initialise(self):
        self._send_goal_future = None
        self._result_future = None
        self._goal_handle = None
        self.cancel_requested = False
        self.failure_reason = ""
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.node.get_logger().error('PToPMoveBase action server not ready')
            return
        self.publish_target_pose()
        self.send_goal()
        self.publish_status(f"navigating:{self.name}")

    def update(self):
        if self.cancel_requested:
            self.failure_reason = "canceled"
            self.node.get_logger().warn("Navigation cancelled via service.")
            return py_trees.common.Status.FAILURE

        if self._send_goal_future is None:
            self.node.get_logger().error("Goal was not sent properly.")
            return py_trees.common.Status.FAILURE

        if self._send_goal_future.done():
            self._goal_handle = self._send_goal_future.result()
            if not self._goal_handle.accepted:
                self.node.get_logger().warn("Goal rejected by action server..")
                time.sleep(0.2)
                self.send_goal()
                return py_trees.common.Status.RUNNING

            if self._result_future is None:
                self._result_future = self._goal_handle.get_result_async()

            if self._result_future.done():
                status = self._result_future.result().status
                if status == STATUS_SUCCEEDED:
                    self.node.get_logger().info("Navigation succeeded.")
                    self.publish_status(f"reached:{self.name}")
                    return py_trees.common.Status.SUCCESS
                if status == STATUS_ABORTED:
                    self.failure_reason = "aborted"
                elif status == STATUS_CANCELED:
                    self.failure_reason = "canceled"
                else:
                    self.failure_reason = ""
                self.node.get_logger().error(f"Navigation failed with status: {status}")
                return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._goal_handle and not self._result_future:
            self._goal_handle.cancel_goal_async()

    def send_goal(self):
        goal = PToPMoveBase.Goal()
        goal.target_pose = self.pose
        self._send_goal_future = self._client.send_goal_async(goal, feedback_callback=self._feedback_cb)
        self.node.get_logger().info(f"Sending navigation goal: {self.pose.pose.position.x}, {self.pose.pose.position.y}, {self.pose.pose.position.z}")
        yaw = 2 * math.atan2(self.pose.pose.orientation.z, self.pose.pose.orientation.w)
        self.node.get_logger().info(f"Goal orientation (yaw): {math.degrees(yaw):.2f} degrees")

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.node.get_logger().info(
            f"[{fb.current_decision}] pos=({fb.base_position.transform.translation.x:.2f}, "
            f"{fb.base_position.transform.translation.y:.2f}, {fb.base_position.transform.translation.z:.2f})",
            throttle_duration_sec=1.0,
        )

    def cancel_callback(self, request, response):
        self.cancel_requested = True
        self.node.get_logger().info("Cancel navigation request received.")
        if self._goal_handle:
            self._goal_handle.cancel_goal_async()
            response.success = True
            response.message = "Navigation goal cancelled."
        else:
            response.success = False
            response.message = "No active goal to cancel."
        return response

    def publish_status(self, status: str):
        msg = String(data=status)
        self.pub.publish(msg)

    def publish_target_pose(self):
        if self.pose_pub is None:
            return
        if isinstance(self.pose, PoseStamped):
            self.pose.header.stamp = self.node.get_clock().now().to_msg()
            self.pose_pub.publish(self.pose)