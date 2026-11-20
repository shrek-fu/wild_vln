from __future__ import annotations

from typing import Any, Dict

from geometry_msgs.msg import Twist

from robot_controller.adapters.base import AdapterResult, RobotAdapter


class ChituAdapter(RobotAdapter):
    """Chitu has no stand/lie SDK; republish Twist to the onboard RL topic."""

    def __init__(self, node: Any, params: Dict[str, Any]):
        super().__init__(node, params)
        self.output_topic = str(params.get('output_cmd_vel_topic', '/rl_cmd_vel'))
        self._pub = node.create_publisher(Twist, self.output_topic, 10)
        self.log().info(f'Chitu adapter: republish Twist -> {self.output_topic}')

    def supports_posture(self) -> bool:
        return False

    def is_standing(self) -> bool:
        return True

    def stand(self) -> AdapterResult:
        self.log().info('Chitu has no stand API; treating as ready')
        return AdapterResult(True, 'chitu: stand not supported, already mobile')

    def lie(self) -> AdapterResult:
        self.stop()
        self.log().info('Chitu has no lie API; velocity zeroed')
        return AdapterResult(True, 'chitu: lie not supported, velocity zeroed')

    def apply_velocity(self, linear_x: float, linear_y: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.y = float(linear_y)
        msg.angular.z = float(angular_z)
        self._pub.publish(msg)

    def stop(self) -> AdapterResult:
        self.apply_velocity(0.0, 0.0, 0.0)
        return AdapterResult(True, 'chitu: stopped')

    def execute_action(self, control_id: int) -> AdapterResult:
        if control_id in (102, 103):
            return self.stand() if control_id == 102 else self.lie()
        return AdapterResult(False, f'chitu: unsupported control_id {control_id}')
