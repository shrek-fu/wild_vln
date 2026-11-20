import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_controller.lite3_protocol import (
    CMD_JOINT_ANGLE,
    CMD_JOINT_VEL,
    CMD_STATE,
    DEFAULT_JOINT_NAMES,
    JOINT_COUNT,
    pack_basic_state,
    pack_joint_doubles,
    parse_basic_state,
    parse_joint_angles,
    parse_joint_velocities,
    unpack_code,
)


def test_parse_basic_state_from_packed_bytes():
    data = pack_basic_state(6)
    assert unpack_code(data) == CMD_STATE
    assert parse_basic_state(data) == 6
    assert parse_joint_angles(data) is None


def test_parse_joint_angles_and_velocities():
    angles = tuple(float(i) * 0.1 for i in range(JOINT_COUNT))
    vels = tuple(float(i) * -0.01 for i in range(JOINT_COUNT))
    ang_pkt = pack_joint_doubles(CMD_JOINT_ANGLE, angles)
    vel_pkt = pack_joint_doubles(CMD_JOINT_VEL, vels)
    assert parse_joint_angles(ang_pkt) == angles
    assert parse_joint_velocities(vel_pkt) == vels
    assert parse_basic_state(ang_pkt) is None


def test_short_packets_return_none():
    assert parse_basic_state(b'\x00' * 8) is None
    assert parse_joint_angles(pack_basic_state(1)[:20]) is None


def test_default_joint_names_count():
    assert len(DEFAULT_JOINT_NAMES) == JOINT_COUNT
