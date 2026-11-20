from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from std_msgs.msg import Int32

from robot_controller.adapters.base import AdapterResult, RobotAdapter
from robot_controller.command_ids import (
    ACTION_DAMP,
    ACTION_LIE,
    ACTION_SPEED_1,
    ACTION_SPEED_2,
    ACTION_SPEED_3,
    ACTION_STAND,
    ACTION_WAVE,
    POSTURE_DAMP,
    POSTURE_HELLO,
    POSTURE_LIE,
    POSTURE_SPEED_1,
    POSTURE_SPEED_2,
    POSTURE_SPEED_3,
    POSTURE_STAND,
    POSTURE_STOP,
)

_SPEED_POSTURE = {
    ACTION_SPEED_1: POSTURE_SPEED_1,
    ACTION_SPEED_2: POSTURE_SPEED_2,
    ACTION_SPEED_3: POSTURE_SPEED_3,
}


class UnitreeAdapter(RobotAdapter):
    """Python facade: C++ sport node owns cmd_vel; this publishes posture_cmd."""

    def __init__(
        self,
        node: Any,
        params: Dict[str, Any],
        publish: Optional[Callable[[int], None]] = None,
    ):
        super().__init__(node, params)
        self.owns_velocity = False
        self.model = str(params.get('unitree_model', 'go2w'))
        topic = str(params.get('posture_cmd_topic', '/unitree/posture_cmd'))
        self._standing = False
        if publish is not None:
            self._publish = publish
        else:
            self._pub = node.create_publisher(Int32, topic, 10)

            def _pub(code: int) -> None:
                msg = Int32()
                msg.data = int(code)
                self._pub.publish(msg)

            self._publish = _pub
        self.log().info(f'Unitree adapter model={self.model} posture_topic={topic}')

    def is_standing(self) -> bool:
        return self._standing

    def _send(self, code: int, message: str) -> AdapterResult:
        self._publish(code)
        return AdapterResult(True, message)

    def stand(self) -> AdapterResult:
        self._standing = True
        return self._send(POSTURE_STAND, f'unitree_{self.model}: StandUp')

    def lie(self) -> AdapterResult:
        self._standing = False
        return self._send(POSTURE_LIE, f'unitree_{self.model}: StandDown')

    def apply_velocity(self, linear_x: float, linear_y: float, angular_z: float) -> None:
        return None

    def stop(self) -> AdapterResult:
        return self._send(POSTURE_STOP, f'unitree_{self.model}: StopMove')

    def emergency(self) -> AdapterResult:
        self.stop()
        return self.lie()

    def execute_action(self, control_id: int) -> AdapterResult:
        if control_id in _SPEED_POSTURE:
            return self._send(_SPEED_POSTURE[control_id], f'unitree: SpeedLevel {control_id}')
        if control_id == ACTION_DAMP:
            if self.model in ('go2', 'go2w'):
                return self.lie()
            self._standing = False
            return self._send(POSTURE_DAMP, f'unitree_{self.model}: Damp')
        if control_id == ACTION_STAND:
            return self.stand()
        if control_id == ACTION_LIE:
            return self.lie()
        if control_id == ACTION_WAVE:
            if self.model == 'b2':
                return AdapterResult(False, 'unitree_b2: Hello/wave not supported')
            return self._send(POSTURE_HELLO, f'unitree_{self.model}: Hello')
        return AdapterResult(False, f'unitree_{self.model}: unsupported control_id {control_id}')
