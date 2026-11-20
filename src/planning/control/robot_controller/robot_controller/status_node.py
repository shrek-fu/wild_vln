"""UDP status listener for YSC Lite3: bind 43897, publish /joint_states."""

from __future__ import annotations

import socket
from typing import List, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32

from robot_controller.lite3_protocol import (
    DEFAULT_JOINT_NAMES,
    JOINT_COUNT,
    parse_basic_state,
    parse_joint_angles,
    parse_joint_velocities,
)

BEST_EFFORT = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)


class Lite3StatusNode(Node):
    def __init__(self):
        super().__init__('lite3_status')
        self.declare_parameter('status_port', 43897)
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('basic_state_topic', '/lite3/basic_state')
        self.declare_parameter('joint_names', list(DEFAULT_JOINT_NAMES))
        self.declare_parameter('poll_hz', 10.0)

        self.status_port = int(self.get_parameter('status_port').value)
        joint_topic = str(self.get_parameter('joint_states_topic').value)
        basic_topic = str(self.get_parameter('basic_state_topic').value)
        names = self.get_parameter('joint_names').value
        self.joint_names: List[str] = [str(n) for n in names] if names else list(DEFAULT_JOINT_NAMES)
        if len(self.joint_names) != JOINT_COUNT:
            self.get_logger().warn(
                f'joint_names len={len(self.joint_names)}, expected {JOINT_COUNT}; padding/truncating'
            )
            padded = list(DEFAULT_JOINT_NAMES)
            for i, name in enumerate(self.joint_names[:JOINT_COUNT]):
                padded[i] = name
            self.joint_names = padded

        self._angles: Optional[Tuple[float, ...]] = None
        self._vels: Optional[Tuple[float, ...]] = None
        self._sock = None
        self._bind_status()

        self._pub_js = self.create_publisher(JointState, joint_topic, BEST_EFFORT)
        self._pub_basic = self.create_publisher(Int32, basic_topic, 10)
        poll_hz = float(self.get_parameter('poll_hz').value or 10.0)
        period = 1.0 / poll_hz if poll_hz > 0.0 else 0.1
        self.create_timer(period, self._on_poll)
        self.get_logger().info(
            f'lite3_status listening UDP {self.status_port} -> {joint_topic} and {basic_topic}'
        )

    def _bind_status(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(('0.0.0.0', self.status_port))
        except OSError as exc:
            sock.close()
            self.get_logger().error(f'lite3_status: bind {self.status_port} failed: {exc}')
            raise
        sock.settimeout(0.0)
        self._sock = sock

    def _on_poll(self) -> None:
        if self._sock is None:
            return
        got_joints = False
        while True:
            try:
                data, _ = self._sock.recvfrom(2048)
            except BlockingIOError:
                break
            except OSError:
                break
            self._handle_packet(data)
            got_joints = got_joints or parse_joint_angles(data) is not None or parse_joint_velocities(data) is not None
        if got_joints or self._angles is not None:
            self._publish_joint_state()

    def _handle_packet(self, data: bytes) -> None:
        state = parse_basic_state(data)
        if state is not None:
            msg = Int32()
            msg.data = int(state)
            self._pub_basic.publish(msg)
            return
        angles = parse_joint_angles(data)
        if angles is not None:
            self._angles = angles
            return
        vels = parse_joint_velocities(data)
        if vels is not None:
            self._vels = vels

    def _publish_joint_state(self) -> None:
        if self._angles is None:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        msg.position = list(self._angles)
        msg.velocity = list(self._vels) if self._vels is not None else []
        msg.effort = []
        self._pub_js.publish(msg)

    def destroy_node(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        super().destroy_node()


def build_joint_state(
    names: Sequence[str],
    positions: Sequence[float],
    velocities: Optional[Sequence[float]] = None,
) -> JointState:
    msg = JointState()
    msg.name = list(names)
    msg.position = [float(v) for v in positions]
    msg.velocity = [float(v) for v in velocities] if velocities is not None else []
    msg.effort = []
    return msg


def main(args: Optional[list] = None) -> int:
    rclpy.init(args=args)
    try:
        node = Lite3StatusNode()
    except OSError:
        if rclpy.ok():
            rclpy.shutdown()
        return 1
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
