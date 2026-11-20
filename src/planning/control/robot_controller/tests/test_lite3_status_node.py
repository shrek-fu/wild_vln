import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_controller.lite3_protocol import DEFAULT_JOINT_NAMES, JOINT_COUNT
from robot_controller.status_node import BEST_EFFORT, build_joint_state


def test_joint_state_fields_and_empty_effort():
    positions = [0.1 * i for i in range(JOINT_COUNT)]
    velocities = [0.01 * i for i in range(JOINT_COUNT)]
    msg = build_joint_state(DEFAULT_JOINT_NAMES, positions, velocities)
    assert list(msg.name) == list(DEFAULT_JOINT_NAMES)
    assert list(msg.position) == positions
    assert list(msg.velocity) == velocities
    assert list(msg.effort) == []


def test_joint_state_velocity_empty_when_unknown():
    positions = [0.0] * JOINT_COUNT
    msg = build_joint_state(DEFAULT_JOINT_NAMES, positions, None)
    assert list(msg.velocity) == []
    assert list(msg.effort) == []


def test_joint_states_qos_is_best_effort_depth_5():
    assert BEST_EFFORT.depth == 5
    from rclpy.qos import ReliabilityPolicy
    assert BEST_EFFORT.reliability == ReliabilityPolicy.BEST_EFFORT
