from __future__ import annotations

import socket
import struct
import time
from typing import Any, Dict, Optional

from std_msgs.msg import Int32

from robot_controller.adapters.base import AdapterResult, RobotAdapter
from robot_controller.command_ids import ACTION_DAMP, ACTION_LIE, ACTION_STAND
from robot_controller.lite3_protocol import parse_basic_state as parse_udp_basic_state

CMD_HEARTBEAT = 0x21040001
CMD_STAND = 0x21010202
CMD_MANUAL_MODE = 0x21010C02
CMD_MOVE_MODE = 0x21010D06
CMD_AUTO_MODE = 0x21010C03
CMD_VX = 0x21010130
CMD_VY = 0x21010131
CMD_WZ = 0x21010135
CMD_AUTO_VX = 0x0140
CMD_AUTO_WZ = 0x0141
CMD_AUTO_VY = 0x0145
CMD_CRAWL = 0x21010406
SCALE_MAX = 32767

STANDING_STATES = {6, 9}
STANDING_UP_STATES = {4, 5}
LYING_STATES = {1, 7}


def clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def vel_to_int(value: float, max_abs: float) -> int:
    if max_abs <= 0.0:
        return 0
    return int(round(clamp(value / max_abs, -1.0, 1.0) * SCALE_MAX))


def parse_basic_state(data: bytes) -> Optional[int]:
    return parse_udp_basic_state(data)


class Lite3Udp:
    def __init__(self, ctrl_ip: str, ctrl_port: int, local_port: int):
        self.ctrl_addr = (ctrl_ip, ctrl_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', local_port))
        self.sock.settimeout(0.0)

    def send_simple(self, code: int, param1: int = 0, param2: int = 0) -> None:
        p1 = param1 & 0xFFFFFFFF if param1 < 0 else param1
        p2 = param2 & 0xFFFFFFFF if param2 < 0 else param2
        self.sock.sendto(struct.pack('<3I', code, p1, p2), self.ctrl_addr)

    def send_complex(self, code: int, value: float) -> None:
        data = struct.pack('<d', float(value))
        header = struct.pack('<3I', code, len(data), 1)
        self.sock.sendto(header + data, self.ctrl_addr)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class YscLite3Adapter(RobotAdapter):
    """UDP bridge for YSC Lite3 (DeepRobotics protocol). 0x21010202 toggles stand/lie — never send blindly."""

    def __init__(self, node: Any, params: Dict[str, Any], udp: Optional[Lite3Udp] = None):
        super().__init__(node, params)
        self.mode = str(params.get('mode', 'auto'))
        self.invert_y = bool(params.get('invert_y', True))
        self.invert_yaw = bool(params.get('invert_yaw', True))
        self.max_vx = float(params.get('max_linear_x', 1.0))
        self.max_vy = float(params.get('max_linear_y', 0.5))
        self.max_wz = float(params.get('max_angular_z', 1.5))
        self.state_wait = float(params.get('state_wait', 2.0))
        self.stand_timeout = float(params.get('stand_timeout', 15.0))
        basic_topic = str(params.get('basic_state_topic', '/lite3/basic_state'))
        self._udp = udp or Lite3Udp(
            str(params.get('ctrl_ip', '192.168.2.1')),
            int(params.get('ctrl_port', 43893)),
            int(params.get('local_port', 20001)),
        )
        self._standing = False
        self._last_state = None
        self._subscribe_basic_state(basic_topic)
        self._apply_mode()
        self.log().info(f'ysc_lite3 adapter ready mode={self.mode} basic_state={basic_topic}')

    def _subscribe_basic_state(self, topic: str) -> None:
        kwargs = {}
        try:
            from rclpy.callback_groups import ReentrantCallbackGroup
            kwargs['callback_group'] = ReentrantCallbackGroup()
        except Exception:
            pass
        self.node.create_subscription(Int32, topic, self._on_basic_state, 10, **kwargs)

    def _on_basic_state(self, msg) -> None:
        try:
            state = int(msg.data)
        except (TypeError, ValueError, AttributeError):
            return
        self._apply_state(state)

    def _apply_state(self, state: Optional[int]) -> Optional[int]:
        if state is None:
            return self._last_state
        self._last_state = state
        self._standing = state in STANDING_STATES
        return state

    def _pump_incoming_state(self, wait: bool = False) -> None:
        pump = getattr(self.node, 'pump_lite3_state', None)
        if not callable(pump):
            return
        try:
            state = pump(wait)
        except TypeError:
            state = pump()
        if state is not None:
            self._apply_state(int(state))

    def poll_state(self) -> Optional[int]:
        self._pump_incoming_state(wait=False)
        return self._last_state

    def _query_state(self, wait_s: Optional[float] = None) -> Optional[int]:
        self.poll_state()
        if self._last_state is not None:
            return self._last_state
        wait = self.state_wait if wait_s is None else wait_s
        if wait <= 0.0:
            return None
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            self._pump_incoming_state(wait=True)
            if self._last_state is not None:
                return self._last_state
            remain = deadline - time.monotonic()
            if remain <= 0.0:
                break
            time.sleep(min(0.05, remain))
        return self._last_state

    def _apply_mode(self) -> None:
        if self.mode == 'auto':
            self._udp.send_simple(CMD_AUTO_MODE)
        else:
            self._udp.send_simple(CMD_MANUAL_MODE)
            time.sleep(0.05)
            self._udp.send_simple(CMD_MOVE_MODE)

    def _send_velocity(self, vx: float, vy: float, wz: float) -> None:
        if self.invert_y:
            vy = -vy
        if self.invert_yaw:
            wz = -wz
        if self.mode == 'auto':
            self._udp.send_complex(CMD_AUTO_VX, vx)
            self._udp.send_complex(CMD_AUTO_VY, vy)
            self._udp.send_complex(CMD_AUTO_WZ, wz)
        else:
            self._udp.send_simple(CMD_VX, vel_to_int(vx, self.max_vx))
            self._udp.send_simple(CMD_VY, vel_to_int(vy, self.max_vy))
            self._udp.send_simple(CMD_WZ, vel_to_int(wz, self.max_wz))

    def _wait_state(self, wanted: set, timeout: float) -> Optional[int]:
        deadline = time.monotonic() + timeout
        last = self._last_state
        while time.monotonic() < deadline:
            self._pump_incoming_state(wait=True)
            last = self._last_state
            if last in wanted:
                return last
            remain = deadline - time.monotonic()
            if remain <= 0.0:
                break
            time.sleep(min(0.05, remain))
        return last

    def is_standing(self) -> bool:
        self.poll_state()
        return self._standing

    def stand(self) -> AdapterResult:
        self._udp.send_simple(CMD_HEARTBEAT)
        state = self._query_state(self.state_wait)
        if state is None:
            self.log().warn('ysc_lite3: no 0x0901, skip stand (0x21010202 is a toggle)')
            return AdapterResult(False, 'ysc_lite3: no 0x0901, skip stand')
        if state in STANDING_STATES:
            self._standing = True
            self._apply_mode()
            return AdapterResult(True, 'ysc_lite3: already standing')
        if state in STANDING_UP_STATES:
            last = self._wait_state(STANDING_STATES, self.stand_timeout)
            self._standing = last in STANDING_STATES
            if self._standing:
                self._apply_mode()
                return AdapterResult(True, 'ysc_lite3: stood up')
            return AdapterResult(False, 'ysc_lite3: wait standing timed out')
        self._udp.send_simple(CMD_STAND)
        last = self._wait_state(STANDING_STATES, self.stand_timeout)
        self._standing = last in STANDING_STATES
        if not self._standing:
            return AdapterResult(False, f'ysc_lite3: stand failed, last_state={last}')
        self._apply_mode()
        return AdapterResult(True, 'ysc_lite3: stand sent')

    def lie(self) -> AdapterResult:
        self.stop()
        self._udp.send_simple(CMD_HEARTBEAT)
        state = self._query_state(self.state_wait)
        if state is None:
            self.log().warn('ysc_lite3: no 0x0901, skip lie (0x21010202 is a toggle)')
            return AdapterResult(False, 'ysc_lite3: no 0x0901, skip lie')
        if state in LYING_STATES:
            self._standing = False
            return AdapterResult(True, 'ysc_lite3: already lying')
        if state not in STANDING_STATES and state not in STANDING_UP_STATES:
            self._standing = False
            return AdapterResult(True, f'ysc_lite3: not standing, skip lie (state={state})')
        self._udp.send_simple(CMD_STAND)
        last = self._wait_state(LYING_STATES, self.stand_timeout)
        self._standing = last in STANDING_STATES
        if self._standing:
            return AdapterResult(False, f'ysc_lite3: lie failed, last_state={last}')
        return AdapterResult(True, 'ysc_lite3: lie sent')

    def crawl(self) -> AdapterResult:
        try:
            self._udp.send_simple(CMD_MANUAL_MODE)
            time.sleep(0.05)
            self._udp.send_simple(CMD_MOVE_MODE)
            self._udp.send_simple(CMD_CRAWL)
        except OSError as exc:
            self.log().error(f'ysc_lite3 crawl failed: {exc}')
            return AdapterResult(False, f'ysc_lite3: crawl failed: {exc}')
        return AdapterResult(True, 'ysc_lite3: crawl gait sent')

    def apply_velocity(self, linear_x: float, linear_y: float, angular_z: float) -> None:
        try:
            self._send_velocity(linear_x, linear_y, angular_z)
        except OSError as exc:
            self.log().error(f'ysc_lite3 UDP send failed: {exc}')

    def stop(self) -> AdapterResult:
        self.apply_velocity(0.0, 0.0, 0.0)
        return AdapterResult(True, 'ysc_lite3: stopped')

    def heartbeat(self) -> None:
        try:
            self.poll_state()
            self._udp.send_simple(CMD_HEARTBEAT)
        except OSError as exc:
            self.log().error(f'ysc_lite3 heartbeat failed: {exc}')

    def execute_action(self, control_id: int) -> AdapterResult:
        if control_id == ACTION_DAMP:
            return self.lie()
        if control_id == ACTION_STAND:
            return self.stand()
        if control_id == ACTION_LIE:
            return self.crawl()
        return AdapterResult(False, f'ysc_lite3: unsupported control_id {control_id}')

    def shutdown(self) -> None:
        self.stop()
        self._udp.close()
