"""1002/1012 stop latches /cmd_vel until stand. A one-shot stop does not."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if 'qi_nav_msgs' not in sys.modules:
    qi_nav_msgs = ModuleType('qi_nav_msgs')
    qi_srv = ModuleType('qi_nav_msgs.srv')
    qi_srv.RobotCommand = object
    qi_nav_msgs.srv = qi_srv
    sys.modules['qi_nav_msgs'] = qi_nav_msgs
    sys.modules['qi_nav_msgs.srv'] = qi_srv

from robot_controller.adapters.base import AdapterResult
from robot_controller.node import RobotControllerNode

sys.modules.pop('qi_nav_msgs', None)
sys.modules.pop('qi_nav_msgs.srv', None)


def _node():
    node = RobotControllerNode.__new__(RobotControllerNode)
    node._vx = 0.5
    node._vy = 0.1
    node._wz = 0.2
    node._drop_cmd_vel = False
    node._last_msg_time = 1.0
    node._last_logged = None
    node.owns_velocity = True
    node.speed_ratio = 1.0
    node.cmd_timeout = 10.0
    node.applied = []

    class Adapter:
        def stop(self):
            return AdapterResult(True, 'stopped')

        def stand(self):
            return AdapterResult(True, 'stood')

        def apply_velocity(self, vx, vy, wz):
            node.applied.append((vx, vy, wz))

    node.adapter = Adapter()
    node.get_logger = lambda: SimpleNamespace(info=lambda *_a, **_k: None, error=lambda *_a, **_k: None)
    return node


def _twist(x):
    return SimpleNamespace(linear=SimpleNamespace(x=x, y=0.0), angular=SimpleNamespace(z=0.0))


def test_latching_stop_drops_twists_until_stand():
    node = _node()
    node._dispatch('stop', 1)
    node._on_twist(_twist(0.8))
    assert node._vx == 0.0
    node._on_hold_timer()
    assert node.applied[-1] == (0.0, 0.0, 0.0)

    node._dispatch('stand', 0)
    node._on_twist(_twist(0.4))
    assert node._vx == 0.4


def test_one_shot_stop_still_accepts_cmd_vel():
    node = _node()
    node._dispatch('stop', 0)
    node._on_twist(_twist(0.3))
    assert node._vx == 0.3
    assert node._drop_cmd_vel is False
