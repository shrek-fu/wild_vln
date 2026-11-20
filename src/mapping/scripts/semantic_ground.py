"""Planning ground from ground_segmentation instead of linefit, for save_map.py.

linefit splits each scan with one line fit per sector against a fixed sensor height, so ramps, bumps and
uneven floors end up in the non-ground map (lethal for the planner) or leave holes in the ground map.
ground_segmentation (pipeline start_ground:=true) keeps a 0.1 m odom grid of cells the camera classes call
ground, each at its own mean lidar height (terrain classes climbable by max_step / max_slope_deg), and
publishes it whole on /ground/traversable_cloud (x y z label).

* ground = those cells, minus the @blocked classes;
* non-ground = DLIO's full map minus the points within ``tol`` of the ground height of a nearby cell, so a
  ramp is not an obstacle to its own neighbouring cells;
* export_traversable_pcd(ground, non-ground) then marks the lethal cells as for QiNav's maps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_NP_TYPES = {("F", 4): np.float32, ("F", 8): np.float64, ("U", 1): np.uint8, ("U", 2): np.uint16,
             ("U", 4): np.uint32, ("U", 8): np.uint64, ("I", 1): np.int8, ("I", 2): np.int16, ("I", 4): np.int32,
             ("I", 8): np.int64}


def read_pcd_xyz(path: str | Path) -> np.ndarray:
    """``(N, 3)`` float64 x y z of an ascii or binary (not binary_compressed) PCD."""
    return read_pcd(path, ("x", "y", "z"))


def read_pcd(path: str | Path, names=("x", "y", "z")) -> np.ndarray:
    """``(N, len(names))`` float64 columns of an ascii or binary (not binary_compressed) PCD."""
    raw = Path(path).read_bytes()
    header, offset = {}, 0
    while "DATA" not in header:
        end = raw.find(b"\n", offset)
        if end < 0:
            raise ValueError(f"{path}: no DATA line in the PCD header")
        key, _, value = raw[offset:end].decode("ascii", "replace").strip().partition(" ")
        offset = end + 1
        if key and not key.startswith("#"):
            header[key] = value.split()
    fields, sizes, types = header["FIELDS"], [int(s) for s in header["SIZE"]], header["TYPE"]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
    n = int(header["POINTS"][0])
    data = header["DATA"][0]
    if data == "ascii":
        cols = np.loadtxt(raw[offset:].decode().splitlines(), ndmin=2)
        starts = np.cumsum([0] + counts[:-1])
        return np.stack([cols[:, starts[fields.index(a)]] for a in names], axis=1).astype(np.float64)
    if data != "binary":
        raise ValueError(f"{path}: DATA {data} not supported (save it binary)")
    # PCL pads unnamed fields as "_"; numpy needs unique names
    fields = [f if f != "_" else f"_pad{i}" for i, f in enumerate(fields)]
    dtype = np.dtype([(f, _NP_TYPES[(t, s)], (c,)) if c > 1 else (f, _NP_TYPES[(t, s)])
                      for f, s, t, c in zip(fields, sizes, types, counts)])
    rec = np.frombuffer(raw, dtype, count=n, offset=offset)
    return np.stack([rec[a] for a in names], axis=1).astype(np.float64)


def write_pcd_xyzi(path: str | Path, xyz: np.ndarray, intensity: np.ndarray | None = None) -> None:
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    inten = np.zeros(len(xyz), np.float32) if intensity is None else np.asarray(intensity, np.float32)
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
              f"TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {len(xyz)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
              f"POINTS {len(xyz)}\nDATA binary\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.column_stack([xyz, inten]).astype(np.float32).tobytes())


def _cell_keys(ij: np.ndarray) -> np.ndarray:
    return (ij[:, 0].astype(np.int64) << 32) | (ij[:, 1].astype(np.int64) & 0xFFFFFFFF)


def near_ground(xyz: np.ndarray, ground: np.ndarray, res: float = 0.1, tol: float = 0.12,
                radius: int = 1) -> np.ndarray:
    """Bool per ``xyz`` point: within ``tol`` (z) of the ground height of a ground cell within ``radius``
    cells (ground = ``(M, 3)`` cell centres at their ground height, on a ``res`` grid)."""
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    ground = np.asarray(ground, np.float64).reshape(-1, 3)
    out = np.zeros(len(xyz), bool)
    if not len(ground) or not len(xyz):
        return out
    gkeys = _cell_keys(np.floor(ground[:, :2] / res).astype(np.int64))
    order = np.argsort(gkeys)
    gkeys, gz = gkeys[order], ground[order, 2]
    ij = np.floor(xyz[:, :2] / res).astype(np.int64)
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            keys = _cell_keys(ij + (di, dj))
            pos = np.minimum(np.searchsorted(gkeys, keys), len(gkeys) - 1)
            hit = gkeys[pos] == keys
            out |= hit & (np.abs(xyz[:, 2] - gz[pos]) < tol)
    return out


def blocked_labels(classes_spec: str, extra: str = "") -> list[int]:
    """Labels (class index + 1) of the spec's @blocked classes plus the comma list ``extra``."""
    from scene_memory.perception.gdino_sam2_detector import read_ground_spec

    entries = read_ground_spec(classes_spec)
    wanted = {c.strip() for c in extra.split(",") if c.strip()}
    return [i + 1 for i, (name, _, flags) in enumerate(entries) if "blocked" in flags or name in wanted]


def grab_ground_cloud(topic: str = "/ground/traversable_cloud", timeout: float = 10.0):
    """``(xyz (M, 3), labels (M,), frame_id)`` of the next ground_segmentation cloud, or None."""
    import rclpy
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import PointCloud2

    got = {}
    rclpy.init(args=[])
    node = rclpy.create_node("stvln_ground_grab")
    try:
        node.create_subscription(PointCloud2, topic, lambda m: got.setdefault("msg", m), QoSProfile(depth=1))
        end = node.get_clock().now().nanoseconds + int(timeout * 1e9)
        while "msg" not in got and node.get_clock().now().nanoseconds < end:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    msg = got.get("msg")
    if msg is None:
        return None
    names = [f.name for f in msg.fields]
    if names[:4] != ["x", "y", "z", "label"] or msg.point_step != 16:
        raise ValueError(f"{topic}: unexpected fields {names} / point_step {msg.point_step}")
    pts = np.frombuffer(bytes(msg.data), np.float32).reshape(-1, 4)
    return pts[:, :3].astype(np.float64), pts[:, 3].astype(np.int64), msg.header.frame_id
