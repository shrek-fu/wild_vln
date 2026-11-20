import struct

import numpy as np
import pytest

from scene_memory.perception.lidar_depth import livox_custom_xyz


def _custom_msg_cdr(frame_id: str, points: list[tuple]) -> bytes:
    """Hand-built little-endian CDR of livox_ros_driver2/CustomMsg."""
    out = bytearray(b"\x00\x01\x00\x00")

    def pad(n):
        out.extend(b"\x00" * ((-(len(out) - 4)) % n))

    out += struct.pack("<iI", 12, 500_000_000)
    name = frame_id.encode() + b"\x00"
    out += struct.pack("<I", len(name)) + name
    pad(8)
    out += struct.pack("<Q", 12_500_000_000)
    pad(4)
    out += struct.pack("<I", len(points)) + bytes([0, 0, 0, 0])
    pad(4)
    out += struct.pack("<I", len(points))
    for i, (x, y, z, tag) in enumerate(points):
        pad(4)
        out += struct.pack("<Ifff", i * 1000, x, y, z) + bytes([10, tag, 0])
    return bytes(out)


@pytest.mark.parametrize("frame_id", ["livox_frame", "a", "abcdefgh"])
def test_livox_custom_xyz_reads_points_and_filters(frame_id):
    pts = [(1.0, 2.0, 3.0, 0x00), (0.1, 0.1, 0.1, 0x00), (4.0, 0.0, 0.0, 0x10), (5.0, 5.0, 5.0, 0x20),
           (-2.0, 1.5, 0.25, 0x00)]
    stamp, xyz = livox_custom_xyz(_custom_msg_cdr(frame_id, pts), blind=0.5)
    assert stamp == pytest.approx(12.5)
    np.testing.assert_allclose(xyz, [[1, 2, 3], [4, 0, 0], [-2, 1.5, 0.25]])


def test_livox_custom_xyz_empty():
    stamp, xyz = livox_custom_xyz(_custom_msg_cdr("livox_frame", []))
    assert stamp == pytest.approx(12.5) and xyz.shape == (0, 3)
