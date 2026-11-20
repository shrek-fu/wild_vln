"""Lite3 UDP status packet parsers (0x0901 / 0x0902 / 0x0903)."""

from __future__ import annotations

import struct
from typing import Optional, Sequence, Tuple

CMD_STATE = 0x0901
CMD_JOINT_ANGLE = 0x0902
CMD_JOINT_VEL = 0x0903
HEADER_LEN = 12
JOINT_COUNT = 12
JOINT_PAYLOAD_LEN = JOINT_COUNT * 8

DEFAULT_JOINT_NAMES: Tuple[str, ...] = (
    'FL_HipX', 'FL_HipY', 'FL_Knee',
    'FR_HipX', 'FR_HipY', 'FR_Knee',
    'HL_HipX', 'HL_HipY', 'HL_Knee',
    'HR_HipX', 'HR_HipY', 'HR_Knee',
)


def unpack_code(data: bytes) -> Optional[int]:
    if len(data) < HEADER_LEN:
        return None
    code, _, _ = struct.unpack('<3I', data[:HEADER_LEN])
    return int(code)


def parse_basic_state(data: bytes) -> Optional[int]:
    if unpack_code(data) != CMD_STATE or len(data) < HEADER_LEN + 4:
        return None
    return int(struct.unpack_from('<i', data, HEADER_LEN)[0])


def _parse_joint_doubles(data: bytes, expected_code: int) -> Optional[Tuple[float, ...]]:
    if unpack_code(data) != expected_code:
        return None
    if len(data) < HEADER_LEN + JOINT_PAYLOAD_LEN:
        return None
    try:
        values = struct.unpack_from('<' + 'd' * JOINT_COUNT, data, HEADER_LEN)
    except struct.error:
        return None
    return tuple(float(v) for v in values)


def parse_joint_angles(data: bytes) -> Optional[Tuple[float, ...]]:
    return _parse_joint_doubles(data, CMD_JOINT_ANGLE)


def parse_joint_velocities(data: bytes) -> Optional[Tuple[float, ...]]:
    return _parse_joint_doubles(data, CMD_JOINT_VEL)


def pack_header(code: int) -> bytes:
    return struct.pack('<3I', int(code) & 0xFFFFFFFF, 0, 0)


def pack_basic_state(state: int) -> bytes:
    return pack_header(CMD_STATE) + struct.pack('<i', int(state))


def pack_joint_doubles(code: int, values: Sequence[float]) -> bytes:
    if len(values) != JOINT_COUNT:
        raise ValueError(f'expected {JOINT_COUNT} joints, got {len(values)}')
    return pack_header(code) + struct.pack('<' + 'd' * JOINT_COUNT, *values)
