from __future__ import annotations

import time
from typing import Any, Dict, Optional

import rclpy
from geometry_msgs.msg import Twist
from qi_nav_msgs.srv import RobotCommand
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from robot_controller.adapters import create_adapter
from robot_controller.adapters.base import AdapterResult
from robot_controller.command_ids import (
    CMD_ACTION,
    CMD_EMERGENCY,
    CMD_WAYPOINT_ACTION,
    CMD_LIE,
    CMD_PREPARE,
    CMD_STAND,
    CMD_STOP,
    COMMANDS,
    SPEED_RATIOS,
    normalize_robot_type,
)


class RobotControllerNode(Node):
    def __init__(self, adapter=None):
        super().__init__('robot_controller')
        self._declare_params()

        robot_type = normalize_robot_type(self._param_str('robot_type', 'chitu'))
        self.robot_type = robot_type
        self.cmd_timeout = float(self._param('cmd_vel_timeout', 0.5))
        self.auto_stand = bool(self._param('auto_stand_on_cmd_vel', False))
        self.hold_rate = float(self._param('hold_rate', 20.0))
        self.speed_ratio = 1.0
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0
        self._drop_cmd_vel = False
        self._last_msg_time = 0.0
        self._last_logged = None

        params = self._collect_adapter_params(robot_type)
        self.adapter = adapter if adapter is not None else create_adapter(self, robot_type, params)
        self.owns_velocity = bool(getattr(self.adapter, 'owns_velocity', True))

        cmd_topic = self._param_str('cmd_vel_topic', '/cmd_vel')
        remote_topic = self._param_str('remote_cmd_vel_topic', '/ql/nav/robot_cmd_vel')
        self.create_subscription(Twist, cmd_topic, self._on_twist, 10)
        if remote_topic and remote_topic != cmd_topic:
            self.create_subscription(Twist, remote_topic, self._on_twist, 10)

        self.create_service(RobotCommand, '/robot_controller/command', self._on_command)
        if self.hold_rate > 0.0 and self.owns_velocity:
            self.create_timer(1.0 / self.hold_rate, self._on_hold_timer)
        if hasattr(self.adapter, 'heartbeat'):
            self.create_timer(0.5, self._on_heartbeat)

        self.get_logger().info(
            f'robot_controller ready type={robot_type} cmd={cmd_topic} '
            f'remote={remote_topic} timeout={self.cmd_timeout:.2f}s '
            f'auto_stand={self.auto_stand}'
        )

    def _declare_params(self) -> None:
        defaults: Dict[str, Any] = {
            'robot_type': 'chitu',
            'cmd_vel_topic': '/cmd_vel',
            'remote_cmd_vel_topic': '/ql/nav/robot_cmd_vel',
            'cmd_vel_timeout': 0.5,
            'auto_stand_on_cmd_vel': False,
            'hold_rate': 20.0,
            'output_cmd_vel_topic': '/rl_cmd_vel',
            'local_ip': '192.168.168.99',
            'local_port': 43988,
            'dog_ip': '192.168.168.168',
            'max_restart': 5,
            'max_linear_x': 1.0,
            'max_linear_y': 0.5,
            'max_angular_z': 1.5,
            'dog_version': 'new',
            'sdk_send_port': 8081,
            'sdk_recv_port': 8080,
            'pitch_rate': 0.25,
            'height_rate': 0.40,
            'posture_duration': 1.5,
            'posture_repeats': 2,
            'turn_rate': 0.8,
            'turn_duration': 2.0,
            'stand_settle': 3.0,
            'mode': 'auto',
            'invert_y': True,
            'invert_yaw': True,
            'ctrl_ip': '192.168.2.1',
            'ctrl_port': 43893,
            'local_udp_port': 20001,
            'status_port': 43897,
            'state_wait': 2.0,
            'stand_timeout': 15.0,
            'basic_state_topic': '/lite3/basic_state',
            'posture_cmd_topic': '/unitree/posture_cmd',
            'unitree_model': 'go2w',
            'robot_ip': '192.168.123.161',
            'network_interface': 'eth0',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _param(self, name: str, fallback):
        try:
            return self.get_parameter(name).value
        except Exception:
            return fallback

    def _param_str(self, name: str, fallback: str) -> str:
        return str(self._param(name, fallback))

    def _collect_adapter_params(self, robot_type: str) -> Dict[str, Any]:
        params = {
            'output_cmd_vel_topic': self._param_str('output_cmd_vel_topic', '/rl_cmd_vel'),
            'local_ip': self._param_str('local_ip', '192.168.168.99'),
            'local_port': int(self._param('local_port', 43988)),
            'dog_ip': self._param_str('dog_ip', '192.168.168.168'),
            'dog_version': self._param_str('dog_version', 'new'),
            'sdk_send_port': int(self._param('sdk_send_port', 8081)),
            'sdk_recv_port': int(self._param('sdk_recv_port', 8080)),
            'max_restart': int(self._param('max_restart', 5)),
            'max_linear_x': float(self._param('max_linear_x', 1.0)),
            'max_linear_y': float(self._param('max_linear_y', 0.5)),
            'max_angular_z': float(self._param('max_angular_z', 1.5)),
            'pitch_rate': float(self._param('pitch_rate', 0.25)),
            'height_rate': float(self._param('height_rate', 0.40)),
            'posture_duration': float(self._param('posture_duration', 1.5)),
            'posture_repeats': int(self._param('posture_repeats', 2)),
            'turn_rate': float(self._param('turn_rate', 0.8)),
            'turn_duration': float(self._param('turn_duration', 2.0)),
            'stand_settle': float(self._param('stand_settle', 3.0)),
            'mode': self._param_str('mode', 'auto'),
            'invert_y': bool(self._param('invert_y', True)),
            'invert_yaw': bool(self._param('invert_yaw', True)),
            'ctrl_ip': self._param_str('ctrl_ip', '192.168.2.1'),
            'ctrl_port': int(self._param('ctrl_port', 43893)),
            'local_port_udp': int(self._param('local_udp_port', 20001)),
            'local_port': int(self._param('local_port', 43988)),
            'status_port': int(self._param('status_port', 43897)),
            'state_wait': float(self._param('state_wait', 2.0)),
            'stand_timeout': float(self._param('stand_timeout', 15.0)),
            'basic_state_topic': self._param_str('basic_state_topic', '/lite3/basic_state'),
            'posture_cmd_topic': self._param_str('posture_cmd_topic', '/unitree/posture_cmd'),
            'unitree_model': self._param_str('unitree_model', 'go2w'),
        }
        # ysc_lite3 uses local_port for UDP bind; zsl1 uses it for SDK. Split by type.
        if robot_type == 'ysc_lite3':
            params['local_port'] = int(self._param('local_udp_port', 20001))
        if robot_type == 'unitree_b2':
            params['unitree_model'] = 'b2'
        elif robot_type == 'unitree_go2':
            params['unitree_model'] = 'go2'
        elif robot_type == 'unitree_go2w':
            params['unitree_model'] = 'go2w'
        return params

    def _stale(self) -> bool:
        if self._last_msg_time <= 0.0:
            return True
        return (time.monotonic() - self._last_msg_time) > self.cmd_timeout

    def _on_twist(self, msg: Twist) -> None:
        if self._drop_cmd_vel:
            return
        self._vx = float(msg.linear.x)
        self._vy = float(msg.linear.y)
        self._wz = float(msg.angular.z)
        self._last_msg_time = time.monotonic()

    def _on_hold_timer(self) -> None:
        if self._drop_cmd_vel:
            if self.owns_velocity:
                self.adapter.apply_velocity(0.0, 0.0, 0.0)
            return
        if not self.owns_velocity:
            return
        if self._last_msg_time <= 0.0:
            return
        if self._stale():
            self.adapter.apply_velocity(0.0, 0.0, 0.0)
            return
        vx = self._vx * self.speed_ratio
        vy = self._vy * self.speed_ratio
        wz = self._wz * self.speed_ratio
        self.adapter.apply_velocity(vx, vy, wz)
        logged = (round(vx, 3), round(vy, 3), round(wz, 3))
        if logged != self._last_logged:
            self._last_logged = logged
            self.get_logger().info(f'cmd vx={vx:.3f} vy={vy:.3f} wz={wz:.3f}')

    def _on_heartbeat(self) -> None:
        try:
            self.adapter.heartbeat()
        except Exception as exc:
            self.get_logger().error(f'heartbeat error: {exc}')

    def _on_command(self, request, response):
        command = str(request.command or '').strip().lower()
        control_id = int(request.control_id or 0)
        result = self._dispatch(command, control_id)
        response.success = bool(result.success)
        response.message = str(result.message)
        return response

    def _dispatch(self, command: str, control_id: int):
        if command not in COMMANDS:
            return AdapterResult(False, f'unknown command: {command}')
        try:
            if command == CMD_STAND:
                self._drop_cmd_vel = False
                return self.adapter.stand()
            if command == CMD_LIE:
                return self.adapter.lie()
            if command == CMD_EMERGENCY:
                return self.adapter.emergency()
            if command == CMD_STOP:
                self._vx = self._vy = self._wz = 0.0
                if int(control_id) == 1:
                    self._drop_cmd_vel = True
                return self.adapter.stop()
            if command == CMD_PREPARE:
                return self.adapter.prepare()
            if command == CMD_ACTION:
                if control_id in SPEED_RATIOS:
                    self.speed_ratio = SPEED_RATIOS[control_id]
                    extra = self.adapter.execute_action(control_id)
                    msg = f'speed_ratio={self.speed_ratio}'
                    if extra.message:
                        msg = f'{msg}; {extra.message}'
                    return AdapterResult(True, msg)
                return self.adapter.execute_action(control_id)
            if command == CMD_WAYPOINT_ACTION:
                return self.adapter.execute_waypoint_action(control_id)
        except Exception as exc:
            self.get_logger().error(f'command {command} failed: {exc}')
            return AdapterResult(False, str(exc))
        return AdapterResult(False, f'unhandled command: {command}')

    def destroy_node(self):
        try:
            self.adapter.stop()
            self.adapter.shutdown()
        except Exception:
            pass
        super().destroy_node()


def main(args: Optional[list] = None) -> int:
    rclpy.init(args=args)
    node = RobotControllerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
