from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class AdapterResult:
    success: bool
    message: str = ''

    def as_tuple(self) -> Tuple[bool, str]:
        return self.success, self.message


class RobotAdapter(ABC):
    """Per-robot backend. The ROS node owns Twist subscription and command service."""

    def __init__(self, node: Any, params: Dict[str, Any]):
        self.node = node
        self.params = params
        self.owns_velocity = True

    def log(self):
        return self.node.get_logger()

    def supports_posture(self) -> bool:
        return True

    def is_standing(self) -> bool:
        return True

    def heartbeat(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    @abstractmethod
    def stand(self) -> AdapterResult:
        raise NotImplementedError

    @abstractmethod
    def lie(self) -> AdapterResult:
        raise NotImplementedError

    @abstractmethod
    def apply_velocity(self, linear_x: float, linear_y: float, angular_z: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> AdapterResult:
        raise NotImplementedError

    def emergency(self) -> AdapterResult:
        self.stop()
        if not self.supports_posture():
            return AdapterResult(True, 'emergency: velocity zeroed')
        result = self.lie()
        if result.success:
            return AdapterResult(True, 'emergency: stopped and lying')
        return result

    def prepare(self) -> AdapterResult:
        if not self.supports_posture():
            return AdapterResult(True, 'prepare: posture not required')
        if self.is_standing():
            return AdapterResult(True, 'prepare: already standing')
        return self.stand()

    def execute_action(self, control_id: int) -> AdapterResult:
        return AdapterResult(False, f'unsupported control_id: {control_id}')

    def execute_waypoint_action(self, control_id: int) -> AdapterResult:
        """Waypoint pose: stop, run the action, then stand. 1043 uses execute_action."""
        stopped = self.stop()
        if not stopped.success:
            return stopped
        result = self.execute_action(control_id)
        if not result.success:
            return result
        if self.supports_posture():
            stood = self.stand()
            if not stood.success:
                return stood
        return result
