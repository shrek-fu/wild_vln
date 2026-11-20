"""Synthetic topo-eval bag: a furnished room, a looping robot with drifting LIO odometry and one LIO reset.

Writes ``<out>/bag`` (``/lio/odom``, ``/lio/cloud_world``, ``/detector/detections_json``) and
``<out>/gt_objects.yaml`` (room frame), so ``topo_eval`` can compare mappers on known object centres::

    python3 src/hydra_fake/test/sim_bag.py /tmp/hf_sim --calib_dir data/sessions/depthtest_20261110/calib
    python3 -m scene_memory.mapping.topo_eval /tmp/hf_sim/bag --gt /tmp/hf_sim/gt_objects.yaml \
        --calib_dir data/sessions/depthtest_20261110/calib [--mapper hydra_fake]

Lidar: 360° x (-7°, 52°), 1° bins, nearest return per bin (occlusion), 12 m. Detections: front view
at 2 Hz for objects within 8 m whose centre bin is not occluded, mask = convex hull of the projected box.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC / "scene_memory"))

from scene_memory import paths  # noqa: E402
from scene_memory.common.calib_io import load_rig_calibration  # noqa: E402
from scene_memory.common.labelspace import load_labelspace  # noqa: E402
from scene_memory.mapping.scene_graph import T_odom_optical, pinhole_intrinsics, pose_frame_to_da360  # noqa: E402
from scene_memory.perception.boxer_lift import load_pinhole_views  # noqa: E402

OBJECTS = [  # label, centre xy, size (x, y, z)
    ("chair", (3.0, 2.0), (0.5, 0.5, 0.9)),
    ("chair", (9.5, 8.5), (0.5, 0.5, 0.9)),
    ("table", (6.0, 4.5), (1.2, 0.7, 0.75)),
    ("sofa", (14.0, 1.2), (2.0, 0.9, 0.8)),
    ("tv", (17.6, 6.0), (0.2, 1.2, 0.7)),
    ("trash can", (1.0, 9.0), (0.4, 0.4, 0.6)),
    ("refrigerator", (12.0, 10.5), (0.8, 0.7, 1.8)),
    ("potted plant", (16.5, 9.5), (0.5, 0.5, 1.0)),
    ("bench", (8.0, 1.0), (1.5, 0.4, 0.45)),
]


def box_points(c, s, step=0.04, z0=0.0) -> np.ndarray:
    xs, ys, zs = (np.arange(-v / 2, v / 2 + 1e-6, step) for v in s)
    faces = []
    for x in (-s[0] / 2, s[0] / 2):
        Y, Z = np.meshgrid(ys, zs)
        faces.append(np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], 1))
    for y in (-s[1] / 2, s[1] / 2):
        X, Z = np.meshgrid(xs, zs)
        faces.append(np.stack([X.ravel(), np.full(X.size, y), Z.ravel()], 1))
    X, Y = np.meshgrid(xs, ys)
    faces.append(np.stack([X.ravel(), Y.ravel(), np.full(X.size, s[2] / 2)], 1))
    p = np.concatenate(faces)
    return p + np.array([c[0], c[1], z0 + s[2] / 2])


def world_points() -> np.ndarray:
    parts = []
    W, H, h = 18.0, 11.0, 2.6
    for (x0, y0), (x1, y1) in [((0, 0), (W, 0)), ((W, 0), (W, H)), ((W, H), (0, H)), ((0, H), (0, 0)),
                               ((6.0, 7.0), (6.0, H)), ((11.0, 0), (11.0, 3.5)), ((11.0, 6.0), (13.0, 6.0))]:
        n = int(math.hypot(x1 - x0, y1 - y0) / 0.05)
        t = np.linspace(0, 1, n)
        for z in np.arange(0, h, 0.05):
            parts.append(np.stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0), np.full(n, z)], 1))
    X, Y = np.meshgrid(np.arange(0, W, 0.1), np.arange(0, H, 0.1))
    parts.append(np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1))
    for px, py in [(4.0, 6.5), (13.5, 4.0), (9.0, 6.0)]:
        parts.append(box_points((px, py), (0.4, 0.4, h)))
    for _, c, s in OBJECTS:
        parts.append(box_points(c, s))
    return np.concatenate(parts)


def trajectory(speed=0.5, hz=10.0) -> np.ndarray:
    """Two loops plus a cross-room pass: (t, x, y, yaw) of the LIO frame."""
    way = [(2.0, 4.5), (2.0, 2.0)] + [(5.0, 2.8), (10.0, 2.5), (12.5, 4.8), (16.0, 3.0), (16.0, 8.0), (13.0, 8.5),
                                      (8.0, 9.5), (3.0, 8.0), (2.0, 4.5)] * 2 + [(9.0, 4.8), (15.0, 5.0)]
    pts = []
    for a, b in zip(way[:-1], way[1:]):
        n = max(int(math.hypot(b[0] - a[0], b[1] - a[1]) / (speed / hz)), 1)
        for s in np.arange(n) / n:
            pts.append((a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])))
    pts = np.array(pts)
    d = np.gradient(pts, axis=0)
    yaw = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    k = np.ones(15) / 15
    yaw = np.convolve(np.pad(yaw, 7, mode="edge"), k, mode="valid")
    t = np.arange(len(pts)) / hz
    return np.column_stack([t, pts, yaw])


def T_xyyaw(x, y, z, yaw) -> np.ndarray:
    T = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = [x, y, z]
    return T


def scan(world: np.ndarray, T_w_s: np.ndarray, max_r=12.0, rng=None) -> np.ndarray:
    inv = np.linalg.inv(T_w_s)
    p = world @ inv[:3, :3].T + inv[:3, 3]
    r = np.linalg.norm(p, axis=1)
    el = np.degrees(np.arcsin(np.clip(p[:, 2] / np.maximum(r, 1e-6), -1, 1)))
    keep = (r > 0.3) & (r < max_r) & (el > -7) & (el < 52)
    p, r, el = p[keep], r[keep], el[keep]
    az = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    bins = (np.floor(az + 180).astype(int) % 360) * 100 + np.floor(el + 10).astype(int)
    order = np.lexsort((r, bins))
    first = np.r_[True, bins[order][1:] != bins[order][:-1]]
    sel = order[first]
    out = p[sel]
    if rng is not None:
        out = out + rng.normal(0, 0.02, out.shape)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--calib_dir", default=str(paths.CALIB))
    ap.add_argument("--sensor_height", type=float, default=0.5)
    ap.add_argument("--yaw_drift", type=float, default=0.004, help="rad of odometry yaw error per metre")
    ap.add_argument("--scale_err", type=float, default=0.01)
    ap.add_argument("--reset_at", type=float, default=150.0, help="s; LIO restarts (0 = never)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import rosbag2_py
    from nav_msgs.msg import Odometry
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import String

    rng = np.random.default_rng(args.seed)
    calib = load_rig_calibration(args.calib_dir)
    view = next(v for v in load_pinhole_views(calib.source_dir) if v["name"] == "front")
    T_imu_da360 = pose_frame_to_da360("lio", paths.VIO_CONFIG, paths.LIO_CONFIG, calib)
    names = load_labelspace(str(paths.CONFIG / "prompts.txt")).names[1:]
    W = Hh = 640
    K = pinhole_intrinsics(W, Hh, view["hfov_deg"])
    world = world_points()
    traj = trajectory()
    if args.out.exists():
        shutil.rmtree(args.out / "bag", ignore_errors=True)
    args.out.mkdir(parents=True, exist_ok=True)

    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(args.out / "bag"), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    for name, typ in [("/lio/odom", "nav_msgs/msg/Odometry"), ("/lio/cloud_world", "sensor_msgs/msg/PointCloud2"),
                      ("/detector/detections_json", "std_msgs/msg/String"), ("/lio/reset", "std_msgs/msg/Header")]:
        writer.create_topic(rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format="cdr"))

    t0 = 1_800_000_000.0
    odom = None
    prev = None
    track_of: dict[int, tuple[int, float]] = {}
    next_track = 1
    gap_until = -1.0
    out_rows = []
    for i, (t, x, y, yaw) in enumerate(traj):
        T_true = T_xyyaw(x, y, args.sensor_height, yaw)
        if args.reset_at and prev is not None and prev[0] < args.reset_at <= t:
            odom = np.eye(4)
            gap_until = t + 2.0
            from std_msgs.msg import Header

            hdr = Header(frame_id="odom")
            hdr.stamp.sec, hdr.stamp.nanosec = int(t0 + t), 0
            writer.write("/lio/reset", serialize_message(hdr), int((t0 + t + 1.0) * 1e9))
        elif prev is None:
            odom = np.eye(4)
        else:
            d = np.linalg.inv(prev[1]) @ T_true
            dist = float(np.linalg.norm(d[:2, 3]))
            noisy = T_xyyaw(d[0, 3] * (1 + args.scale_err), d[1, 3] * (1 + args.scale_err), d[2, 3],
                            math.atan2(d[1, 0], d[0, 0]) + args.yaw_drift * dist)
            odom = odom @ noisy
        prev = (t, T_true)
        if t < gap_until:
            continue
        stamp = t0 + t
        ns = int(stamp * 1e9)
        sec, nsec = int(stamp), int(round((stamp - int(stamp)) * 1e9))
        m = Odometry()
        m.header.stamp.sec, m.header.stamp.nanosec = sec, nsec
        m.header.frame_id = "odom"
        from scipy.spatial.transform import Rotation

        q = Rotation.from_matrix(odom[:3, :3]).as_quat()
        m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = map(float, odom[:3, 3])
        m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z, m.pose.pose.orientation.w = map(float, q)
        writer.write("/lio/odom", serialize_message(m), ns)

        pts_s = scan(world, T_true, rng=rng)
        pts_o = (pts_s @ odom[:3, :3].T + odom[:3, 3]).astype(np.float32)
        c = PointCloud2()
        c.header = m.header
        c.height, c.width = 1, len(pts_o)
        c.fields = [PointField(name=n, offset=4 * k, datatype=PointField.FLOAT32, count=1) for k, n in enumerate("xyz")]
        c.point_step, c.row_step, c.is_dense = 12, 12 * len(pts_o), True
        c.data = pts_o.tobytes()
        writer.write("/lio/cloud_world", serialize_message(c), ns)

        if i % 5:
            continue
        T_opt = T_odom_optical(T_true, T_imu_da360, view["R_body_optical"])
        inv = np.linalg.inv(T_opt)
        dets = []
        for oi, (label, cxy, size) in enumerate(OBJECTS):
            center = np.array([cxy[0], cxy[1], size[2] / 2])
            pc = inv[:3, :3] @ center + inv[:3, 3]
            if pc[2] < 0.5 or np.linalg.norm(pc) > 8.0:
                continue
            u = K[0] * pc[0] / pc[2] + K[2]
            v = K[1] * pc[1] / pc[2] + K[3]
            if not (20 < u < W - 20 and 20 < v < Hh - 20):
                continue
            ps = np.linalg.inv(T_true)[:3, :3] @ center + np.linalg.inv(T_true)[:3, 3]
            rc = float(np.linalg.norm(ps))
            dirs = pts_s / np.linalg.norm(pts_s, axis=1, keepdims=True)
            near = dirs @ (ps / rc) > math.cos(math.radians(2.0))
            if near.any() and np.linalg.norm(pts_s[near], axis=1).min() < rc - max(size) * 0.6:
                continue
            corners = np.array([[cxy[0] + sx * size[0] / 2, cxy[1] + sy * size[1] / 2, z]
                                for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, size[2])])
            cc = corners @ inv[:3, :3].T + inv[:3, 3]
            if (cc[:, 2] < 0.2).any():
                continue
            uv = np.stack([K[0] * cc[:, 0] / cc[:, 2] + K[2], K[1] * cc[:, 1] / cc[:, 2] + K[3]], 1)
            uv = np.clip(uv, 0, W - 1)
            import cv2

            hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
            last = track_of.get(oi)
            if last is None or t - last[1] > 3.0:
                track_of[oi] = (next_track, t)
                next_track += 1
            tid = track_of[oi][0]
            track_of[oi] = (tid, t)
            cls = names.index(label)
            dets.append({"name": label, "score": 0.8, "cls": cls, "cls_scores": {str(cls): 0.8},
                         "xyxy": [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())],
                         "view": "front", "track_id": tid, "mask": hull.round(1).tolist()})
        body = {"frame_id": "insta360", "stamp": {"sec": sec, "nanosec": nsec}, "image": {"width": W, "height": Hh},
                "view_image": {"width": W, "height": Hh}, "views": ["front"], "detections": dets}
        writer.write("/detector/detections_json", serialize_message(String(data=json.dumps(body))), ns + 500_000_000)
        out_rows.append(len(dets))
    del writer
    gt = {"frame": "room", "objects": [{"id": f"{label.replace(' ', '_')}_{k}", "label": label, "xy": list(c),
                                        "z": s[2] / 2} for k, (label, c, s) in enumerate(OBJECTS)]}
    import yaml

    (args.out / "gt_objects.yaml").write_text(yaml.safe_dump(gt, sort_keys=False))
    print(f"{args.out}: {len(traj)} poses, {len(out_rows)} detection frames, {sum(out_rows)} detections, "
          f"path {np.sum(np.linalg.norm(np.diff(traj[:, 1:3], axis=0), axis=1)):.0f} m")


if __name__ == "__main__":
    main()
