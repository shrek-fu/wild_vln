#!/usr/bin/env python3
"""Grow the navigation map beyond the prior map while the robot drives there, so qi_move_base can plan and
localization_cpp can match off the map.

Every LIO scan (``/lio/cloud_world``, odom) is put into the map frame (TF map -> odom from localization_cpp)
and split into ground (lowest returns of 0.2 m columns near the robot's foot height) and non-ground (above
them). Only voxels away from the prior map's ground are kept, so the prior map and its baked costs never
change, only while ``/stvln/map_extend`` (std_msgs/Bool, goal_server turns it on for an off-map target) is
true, and only while ``/localize/status`` says ``localization_success`` (a drifting pose must not be baked
in). Every ``--period_s`` with new voxels:

* prior ground + new ground, prior non-ground + new non-ground -> ``export_traversable_pcd`` (the same export
  ``stvln.sh map_save`` runs) -> ``mapground`` (latched; perception_3d's static layer rebuilds its graph when
  the point count changes)
* the new voxels -> ``/localize/map_append`` (localization_cpp adds them to its ICP map)
* ``<prefix>_ext/<stamp>/`` holds the new ground / non-ground and the extended traversable cloud (the prior
  files are never written); ``/stvln/map_extent`` reports the counts as JSON.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener

PCD_TYPES = {"F": "f", "U": "u", "I": "i"}


def read_pcd(path: Path) -> np.ndarray:
    """Binary PCD -> structured array (fields as in the file)."""
    with open(path, "rb") as f:
        hdr = {}
        while True:
            key, *val = f.readline().decode("ascii").split()
            hdr[key] = val
            if key == "DATA":
                break
        if hdr["DATA"][0] != "binary":
            raise ValueError(f"{path}: only binary PCD is supported, got {hdr['DATA'][0]}")
        n = int(hdr["POINTS"][0])
        dt = np.dtype([(name, PCD_TYPES[t] + s) for name, s, t in zip(hdr["FIELDS"], hdr["SIZE"], hdr["TYPE"])])
        return np.frombuffer(f.read(n * dt.itemsize), dt, n)


def write_pcd_xyz(path: Path, xyz: np.ndarray) -> None:
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    n = len(xyz)
    header = (f"# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
              f"COUNT 1 1 1\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(xyz.tobytes())


def cloud_xyz(msg: PointCloud2) -> np.ndarray:
    off = {f.name: f.offset for f in msg.fields}
    raw = np.frombuffer(bytes(msg.data), np.uint8).reshape(-1, msg.point_step)
    xyz = np.stack([raw[:, off[k]:off[k] + 4].copy().view(np.float32)[:, 0] for k in "xyz"], axis=1)
    return xyz[np.isfinite(xyz).all(axis=1)]


def cloud_msg(xyz: np.ndarray, frame: str, stamp, intensity: np.ndarray | None = None) -> PointCloud2:
    cols = [np.asarray(xyz, np.float32)] + ([np.asarray(intensity, np.float32)[:, None]] if intensity is not None else [])
    data = np.ascontiguousarray(np.concatenate(cols, axis=1), dtype=np.float32)
    names = "xyz" + ("i" if intensity is not None else "")
    msg = PointCloud2()
    msg.header.frame_id, msg.header.stamp = frame, stamp
    msg.height, msg.width = 1, len(data)
    msg.fields = [PointField(name="intensity" if k == "i" else k, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, k in enumerate(names)]
    msg.point_step, msg.row_step, msg.is_dense, msg.is_bigendian = 4 * len(names), 4 * len(names) * len(data), True, False
    msg.data = data.tobytes()
    return msg


def quat_matrix(x, y, z, w) -> np.ndarray:
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def split_ground(xyz: np.ndarray, foot: np.ndarray, cell: float = 0.2, radius: float = 12.0, slope: float = 0.15,
                 ground_band: float = 0.1, top: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """``(ground, nonground)`` of a map-frame scan: per ``cell`` column, the lowest return is floor when it lies
    within ``max(0.25, slope * distance)`` of the robot's foot height; returns up to ``ground_band`` above it are
    ground, the rest up to ``top`` above the foot non-ground (columns without floor: all non-ground)."""
    d = np.hypot(xyz[:, 0] - foot[0], xyz[:, 1] - foot[1])
    keep = (d < radius) & (xyz[:, 2] < foot[2] + top) & (xyz[:, 2] > foot[2] - 1.0 - slope * radius)
    xyz, d = xyz[keep], d[keep]
    if len(xyz) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    ij = np.floor(xyz[:, :2] / cell).astype(np.int64)
    key = ij[:, 0] * 1_000_003 + ij[:, 1]
    uniq, inv = np.unique(key, return_inverse=True)
    zmin = np.full(len(uniq), np.inf)
    np.minimum.at(zmin, inv, xyz[:, 2])
    dmin = np.full(len(uniq), np.inf)
    np.minimum.at(dmin, inv, d)
    floor = np.abs(zmin - foot[2]) <= np.maximum(0.25, slope * dmin)
    is_ground = floor[inv] & (xyz[:, 2] <= zmin[inv] + ground_band)
    return xyz[is_ground], xyz[~is_ground]


class VoxelSet:
    """Points kept one per ``leaf`` voxel; ``take_new()`` returns those added since the last call."""

    def __init__(self, leaf: float):
        self.leaf, self.keys, self.pts, self._sent = leaf, set(), [], 0

    def add(self, xyz: np.ndarray) -> int:
        if len(xyz) == 0:
            return 0
        k = np.floor(xyz / self.leaf).astype(np.int64)
        flat = (k[:, 0] * 73856093) ^ (k[:, 1] * 19349663) ^ (k[:, 2] * 83492791)
        _, first = np.unique(flat, return_index=True)
        added = [i for i in first if int(flat[i]) not in self.keys]
        self.keys.update(int(flat[i]) for i in added)
        if added:
            self.pts.append(((k[added] + 0.5) * self.leaf).astype(np.float32))
        return len(added)

    def all(self) -> np.ndarray:
        if len(self.pts) > 1:
            self.pts = [np.concatenate(self.pts)]
        return self.pts[0] if self.pts else np.zeros((0, 3), np.float32)

    def take_new(self) -> np.ndarray:
        pts = self.all()
        new, self._sent = pts[self._sent:], len(pts)
        return new


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map_prefix", required=True, help="<dir>/<name> of the prior map (<prefix>_traversable.pcd ...)")
    p.add_argument("--cloud_topic", default="/lio/cloud_world")
    p.add_argument("--status_topic", default="/localize/status")
    p.add_argument("--ground_topic", default="mapground")
    p.add_argument("--append_topic", default="/localize/map_append")
    p.add_argument("--extent_topic", default="/stvln/map_extent")
    p.add_argument("--enable_topic", default="/stvln/map_extend",
                   help="std_msgs/Bool: scans are only added while true (goal_server sets it for off-map targets)")
    p.add_argument("--always_on", action="store_true", help="add scans without waiting for --enable_topic")
    p.add_argument("--map_frame", default="map")
    p.add_argument("--base_frame", default="base_link")
    p.add_argument("--period_s", type=float, default=2.0)
    p.add_argument("--scan_hz", type=float, default=2.0, help="scans used per second")
    p.add_argument("--prior_margin_m", type=float, default=0.5,
                   help="new voxels must be this far (xy) from the prior ground")
    p.add_argument("--min_new", type=int, default=20, help="new ground voxels before an export")
    p.add_argument("--radius_m", type=float, default=12.0, help="scan points used around the robot")
    p.add_argument("--leaf", type=float, default=0.2, help="voxel size of the kept points (export leaf)")
    p.add_argument("--obstacle_half", type=float, default=0.3, help="export_traversable_pcd box (stvln map_save)")
    p.add_argument("--loc_leaf", type=float, default=0.4,
                   help="voxel of the whole-scan points appended to localization (its map_voxel_size)")
    args = p.parse_args(argv)

    prefix = Path(args.map_prefix)
    sem = Path(f"{prefix}_ground_sem.pcd"), Path(f"{prefix}_nonground_sem.pcd")
    plain = Path(f"{prefix}_ground.pcd"), Path(f"{prefix}_nonground.pcd")
    ground_f, nonground_f = sem if all(f.is_file() for f in sem) else plain
    prior_g = read_pcd(ground_f)
    prior_g = np.stack([prior_g[k] for k in "xyz"], axis=1).astype(np.float32)
    prior_n = read_pcd(nonground_f)
    prior_n = np.stack([prior_n[k] for k in "xyz"], axis=1).astype(np.float32)
    prior_tree = cKDTree(prior_g[:, :2])
    prior_map = read_pcd(Path(f"{prefix}_map.pcd"))
    prior_map = np.stack([prior_map[k] for k in "xyz"], axis=1).astype(np.float32)
    prior_map = prior_map[np.isfinite(prior_map).all(axis=1)]
    _, first = np.unique(np.floor(prior_map / args.loc_leaf).astype(np.int64), axis=0, return_index=True)
    map_tree = cKDTree(prior_map[first])
    found = subprocess.run(["ros2", "pkg", "prefix", "perception_3d"], capture_output=True, text=True)
    install = Path(found.stdout.strip()) if found.returncode == 0 else \
        Path(__file__).resolve().parents[2] / "install" / "perception_3d"
    tool = shutil.which("export_traversable_pcd") or str(install / "lib" / "perception_3d" / "export_traversable_pcd")
    if not Path(tool).is_file():
        raise SystemExit(f"map_extender: no export_traversable_pcd ({tool}); build perception_3d")
    out_dir = Path(f"{prefix}_ext") / time.strftime("%Y%m%d_%H%M%S")

    rclpy.init()
    node = rclpy.create_node("map_extender")
    log = node.get_logger()
    tf = Buffer()
    TransformListener(tf, node)
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    pub_ground = node.create_publisher(PointCloud2, args.ground_topic, latched)
    pub_append = node.create_publisher(PointCloud2, args.append_topic, QoSProfile(depth=10,
                                                                               reliability=ReliabilityPolicy.RELIABLE))
    pub_extent = node.create_publisher(String, args.extent_topic, latched)
    ground, nonground, loc = VoxelSet(args.leaf), VoxelSet(args.leaf), VoxelSet(args.loc_leaf)
    st = {"loc_ok_t": -1e9, "scan_t": 0.0, "new": 0, "exports": 0, "on": args.always_on}
    work = Path(tempfile.mkdtemp(prefix="map_extender_"))

    def on_status(msg: String) -> None:
        if msg.data == "localization_success":
            st["loc_ok_t"] = time.monotonic()

    def on_enable(msg: Bool) -> None:
        if bool(msg.data) != st["on"] and not args.always_on:
            st["on"] = bool(msg.data)
            log.info(f"map extension {'ON' if st['on'] else 'off (the extended map stays)'}")

    def on_cloud(msg: PointCloud2) -> None:
        now = time.monotonic()
        if not st["on"] or now - st["scan_t"] < 1.0 / args.scan_hz or now - st["loc_ok_t"] > 5.0:
            return
        try:
            to_map = tf.lookup_transform(args.map_frame, msg.header.frame_id, rclpy.time.Time()).transform
            base = tf.lookup_transform(args.map_frame, args.base_frame, rclpy.time.Time()).transform.translation
        except Exception as exc:  # noqa: BLE001 — no localization yet
            log.warn(f"no TF {args.map_frame} <- {msg.header.frame_id}/{args.base_frame}: {exc}", throttle_duration_sec=10)
            return
        st["scan_t"] = now
        q, t = to_map.rotation, to_map.translation
        xyz = cloud_xyz(msg) @ quat_matrix(q.x, q.y, q.z, q.w).T + np.array([t.x, t.y, t.z])
        foot = np.array([base.x, base.y, base.z])
        g, ng = split_ground(xyz, foot, args.leaf, args.radius_m)
        new = lambda pts: pts[prior_tree.query(pts[:, :2])[0] > args.prior_margin_m] if len(pts) else pts  # noqa: E731
        st["new"] += ground.add(new(g))
        nonground.add(new(ng))
        loc.add(xyz[map_tree.query(xyz, distance_upper_bound=args.loc_leaf)[0] > args.loc_leaf])

    def export() -> None:
        add = loc.take_new()
        if len(add):
            pub_append.publish(cloud_msg(add, args.map_frame, node.get_clock().now().to_msg()))
        if st["new"] < args.min_new:
            return
        t0 = time.monotonic()
        st["new"] = 0
        g, ng = ground.all(), nonground.all()
        write_pcd_xyz(work / "ground.pcd", np.concatenate([prior_g, g]))
        write_pcd_xyz(work / "nonground.pcd", np.concatenate([prior_n, ng]))
        res = subprocess.run([tool, str(work / "ground.pcd"), str(work / "nonground.pcd"), str(work / "trav.pcd"),
                              str(args.leaf), str(args.leaf), "1.5", str(args.obstacle_half)],
                             capture_output=True, text=True)
        if res.returncode != 0:
            log.error(f"export_traversable_pcd failed: {res.stderr.strip()[-300:]}")
            return
        trav = read_pcd(work / "trav.pcd")
        stamp = node.get_clock().now().to_msg()
        pub_ground.publish(cloud_msg(np.stack([trav[k] for k in "xyz"], axis=1), args.map_frame, stamp,
                                     trav["intensity"]))
        out_dir.mkdir(parents=True, exist_ok=True)
        write_pcd_xyz(out_dir / "ground_new.pcd", g)
        write_pcd_xyz(out_dir / "nonground_new.pcd", ng)
        shutil.copyfile(work / "trav.pcd", out_dir / "traversable.pcd")
        st["exports"] += 1
        info = {"new_ground": len(g), "new_nonground": len(ng), "traversable": len(trav),
                "lethal": int((trav["intensity"] < 1.0).sum()), "loc_points": len(loc.all()),
                "exports": st["exports"], "dir": str(out_dir),
                "export_s": round(time.monotonic() - t0, 2)}
        pub_extent.publish(String(data=json.dumps(info)))
        log.info(f"map extended: {info}")

    node.create_subscription(String, args.status_topic, on_status, 10)
    node.create_subscription(Bool, args.enable_topic, on_enable, latched)
    node.create_subscription(PointCloud2, args.cloud_topic, on_cloud, qos_profile_sensor_data)
    node.create_timer(args.period_s, export)
    log.info(f"map_extender: prior {ground_f.name} ({len(prior_g)}) + {nonground_f.name} ({len(prior_n)}); "
             f"new areas -> {args.ground_topic}, {args.append_topic}, {out_dir}")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        shutil.rmtree(work, ignore_errors=True)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
