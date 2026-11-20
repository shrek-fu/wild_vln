"""PointCloud2 helpers shared by the apply, manual and capture tools."""
from __future__ import annotations

import numpy as np
from sensor_msgs.msg import PointCloud2

MAX_SKEW_NS = 500_000_000


def _xyz_view(msg: PointCloud2) -> tuple[np.ndarray, dict, int]:
    fields = {f.name: f for f in msg.fields}
    for name in ("x", "y", "z"):
        if name not in fields or fields[name].datatype != 7:  # FLOAT32
            raise RuntimeError(f"cloud field {name} is not float32")
    n = int(msg.width) * int(msg.height)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if raw.size != n * int(msg.point_step):
        raise RuntimeError("point cloud data size does not match width*height*point_step")
    return raw.reshape(n, int(msg.point_step)), fields, n


def xyz_from_cloud(msg: PointCloud2) -> np.ndarray:
    buf, fields, n = _xyz_view(msg)
    xyz = np.empty((n, 3), dtype=np.float32)
    for i, name in enumerate(("x", "y", "z")):
        off = int(fields[name].offset)
        xyz[:, i] = np.frombuffer(np.ascontiguousarray(buf[:, off:off + 4]), dtype=np.float32)
    return xyz.astype(np.float64)


def _copy_header(msg: PointCloud2, frame_id: str) -> PointCloud2:
    out = PointCloud2()
    out.header.stamp = msg.header.stamp
    out.header.frame_id = frame_id
    out.height = msg.height
    out.width = msg.width
    out.fields = msg.fields
    out.is_bigendian = msg.is_bigendian
    out.point_step = msg.point_step
    out.row_step = msg.row_step
    out.is_dense = msg.is_dense
    return out


def transform_xyz(msg: PointCloud2, T: np.ndarray, frame_id: str) -> PointCloud2:
    buf, fields, n = _xyz_view(msg)
    buf = buf.copy()
    xyz = xyz_from_cloud(msg)
    moved = (xyz @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
    for i, name in enumerate(("x", "y", "z")):
        off = int(fields[name].offset)
        buf[:, off:off + 4] = np.frombuffer(moved[:, i].tobytes(), dtype=np.uint8).reshape(n, 4)
    out = _copy_header(msg, frame_id)
    out.data = buf.reshape(-1).tobytes()
    return out


def _same_layout(a: PointCloud2, b: PointCloud2) -> bool:
    if a.point_step != b.point_step or a.is_bigendian != b.is_bigendian:
        return False
    if len(a.fields) != len(b.fields):
        return False
    return all(
        (l.name, l.offset, l.datatype, l.count) == (r.name, r.offset, r.datatype, r.count)
        for l, r in zip(a.fields, b.fields)
    )


def concat_clouds(first: PointCloud2, second: PointCloud2, frame_id: str) -> PointCloud2:
    """Stack two clouds that are already in the same frame."""
    if not _same_layout(first, second):
        raise RuntimeError("the two clouds do not share a point layout")
    n1 = int(first.width) * int(first.height)
    n2 = int(second.width) * int(second.height)
    out = PointCloud2()
    out.header.stamp = first.header.stamp
    out.header.frame_id = frame_id
    out.height = 1
    out.width = n1 + n2
    out.fields = first.fields
    out.is_bigendian = first.is_bigendian
    out.point_step = first.point_step
    out.row_step = int(first.point_step) * (n1 + n2)
    out.is_dense = bool(first.is_dense and second.is_dense)
    out.data = bytes(first.data) + bytes(second.data)
    return out


def stamp_ns(msg: PointCloud2) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
