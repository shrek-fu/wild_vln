"""Adapter registry."""

from robot_controller.adapters.base import AdapterResult, RobotAdapter
from robot_controller.adapters.chitu import ChituAdapter
from robot_controller.adapters.unitree import UnitreeAdapter
from robot_controller.adapters.ysc_lite3 import YscLite3Adapter
from robot_controller.adapters.zsl1 import Zsl1Adapter
from robot_controller.command_ids import normalize_robot_type

ADAPTERS = {
    'chitu': ChituAdapter,
    'zsl1': Zsl1Adapter,
    'ysc_lite3': YscLite3Adapter,
    'unitree_b2': UnitreeAdapter,
    'unitree_go2': UnitreeAdapter,
    'unitree_go2w': UnitreeAdapter,
}


def create_adapter(node, robot_type: str, params=None) -> RobotAdapter:
    kind = normalize_robot_type(robot_type)
    cls = ADAPTERS[kind]
    return cls(node, params or {})


__all__ = [
    'AdapterResult',
    'RobotAdapter',
    'ChituAdapter',
    'YscLite3Adapter',
    'UnitreeAdapter',
    'Zsl1Adapter',
    'create_adapter',
]
