"""Front-hemisphere depth (DA360 + lidar alignment) on a recorded bag, with a held-out lidar check.

``--lidar_source raw`` (default) projects the raw MID-360 scans from the lidar frame with the camera-lidar
extrinsic alone (``T_front_da360``), no odometry: scans within ``--window_s`` before the image are stacked
as they are, so keep the window short while the robot moves. ``lio`` uses the deskewed ``/lio/cloud_world``
with the LIO pose at the image stamp. Record on the robot while the pipeline runs::

    ros2 bag record -o bag /insta360/equirect/image/compressed /livox/front_lidar/custom

then::

    python3 -m scene_memory.mapping.pano_depth_bag <session>/bag --calib_dir <session>/calib --out_dir /tmp/pano

A random ``--holdout`` share of the lidar pixels is left out of the alignment and used as ground truth.
Errors are reported per range bin for the global scale alone and for the local alignment; ``--save_every``
frames are written as ``rgb | lidar | depth | source`` montages.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from scene_memory import paths
from scene_memory.common.calib_io import load_rig_calibration
from scene_memory.common.pose_buffer import PoseBuffer
from scene_memory.mapping.remap_bag import read_bag
from scene_memory.mapping.scene_graph import pose_frame_to_da360, pose_to_matrix
from scene_memory.perception.lidar_depth import CloudWindow, cloud_xyz_fast, livox_custom_xyz
from scene_memory.perception.pano_depth import (SOURCE_ALIGNED, SOURCE_COARSE, SOURCE_LIDAR, align_local, crop_front,
                                                range_bin_errors, sparse_hemi_depth)

SOURCE_COLORS = {SOURCE_LIDAR: (255, 255, 255), SOURCE_ALIGNED: (80, 200, 80), SOURCE_COARSE: (60, 60, 200)}


def load_da360(model_path: str, device: str | None = None):
    root = str(paths.DA360_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from insta_live.model_runner import DA360Runner

    return DA360Runner(model_path, device=device)


def depth_color(depth: np.ndarray, max_range: float) -> np.ndarray:
    level = np.clip(np.sqrt(np.clip(depth, 0, None) / max_range) * 255.0, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(level, cv2.COLORMAP_TURBO)
    out[depth <= 0] = 0
    return out


def montage(rgb: np.ndarray, sparse: np.ndarray, depth: np.ndarray, source: np.ndarray, max_range: float) -> np.ndarray:
    lidar = rgb // 3
    v, u = np.nonzero(sparse > 0)
    lidar[v, u] = depth_color(sparse, max_range)[v, u]
    src = np.zeros_like(rgb)
    for value, color in SOURCE_COLORS.items():
        src[source == value] = color
    return np.hstack([rgb, lidar, depth_color(depth, max_range), src])


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", type=Path)
    p.add_argument("--calib_dir", default=str(paths.CALIB))
    p.add_argument("--lio_config", default=str(paths.LIO_CONFIG))
    p.add_argument("--model_path", default=str(paths.DA360.with_name("DA360_small.pth")))
    p.add_argument("--image_topic", default="/insta360/equirect/image/compressed")
    p.add_argument("--lidar_source", choices=["raw", "lio"], default="raw")
    p.add_argument("--lidar_topic", default="/livox/front_lidar/custom", help="raw: livox CustomMsg")
    p.add_argument("--blind", type=float, default=0.5, help="m; raw returns closer than this are the rig itself")
    p.add_argument("--odom_topic", default="/lio/odom")
    p.add_argument("--cloud_topic", default="/lio/cloud_world")
    p.add_argument("--out_dir", type=Path, default=Path("/tmp/pano_depth"))
    p.add_argument("--width", type=int, default=0, help="hemisphere side in px (0 = DA360 native)")
    p.add_argument("--window_s", type=float, default=None, help="scans stacked per image (raw 0.3 s, lio 1.0 s)")
    p.add_argument("--max_lag_s", type=float, default=10.0, help="how far images may trail the lidar in the bag")
    p.add_argument("--voxel", type=float, default=0.03)
    p.add_argument("--near", type=float, default=30.0, help="m; aligned region limit")
    p.add_argument("--max_range", type=float, default=60.0)
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--every", type=int, default=2, help="use every n-th image")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    calib = load_rig_calibration(args.calib_dir)
    raw = args.lidar_source == "raw"
    window_s = args.window_s if args.window_s is not None else (0.3 if raw else 1.0)
    T_child_da360 = pose_frame_to_da360("lio", paths.VIO_CONFIG, args.lio_config, calib)
    runner = load_da360(args.model_path)
    side = args.width or runner.height
    rng = np.random.default_rng(args.seed)
    poses = PoseBuffer()
    cloud = CloudWindow(window_s, keep_s=window_s + args.max_lag_s, voxel_m=args.voxel)

    def camera_pose(t: float) -> np.ndarray | None:
        """``T_ref_da360`` of the cloud's frame: the front lidar (raw) or ``odom`` at ``t`` (lio)."""
        if raw:
            return calib.T_front_da360
        T_odom_child = poses.at(t)
        return None if T_odom_child is None else T_odom_child @ T_child_da360
    pending: list[tuple[float, object]] = []
    collected = {"global": ([], []), "local": ([], [])}
    per_frame = []
    n_image = 0

    def process(t: float, msg) -> None:
        T_ref_da360 = camera_pose(t)
        xyz = cloud.points(t)
        if T_ref_da360 is None or len(xyz) == 0:
            return
        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return
        t0 = time.perf_counter()
        rel_full, _ = runner.predict_depth(bgr, is_bgr=True)
        t_net = time.perf_counter() - t0
        rel = cv2.resize(crop_front(rel_full), (side, side), interpolation=cv2.INTER_LINEAR)
        sparse = sparse_hemi_depth(xyz, T_ref_da360, side, side, r_max=args.max_range)
        vv, uu = np.nonzero(sparse > 0)
        held = rng.random(len(vv)) < args.holdout
        truth = np.zeros_like(sparse)
        truth[vv[held], uu[held]] = sparse[vv[held], uu[held]]
        train = sparse.copy()
        train[vv[held], uu[held]] = 0.0
        t1 = time.perf_counter()
        depth, source, stats = align_local(rel, train, near_m=args.near, max_range=args.max_range)
        t_align = time.perf_counter() - t1
        if depth is None:
            return
        glob = rel * np.float32(stats["scale"])
        for key, pred in (("global", glob), ("local", depth)):
            m = truth > 0
            collected[key][0].append(pred[m])
            collected[key][1].append(truth[m])
        stats.update({"t": t, "lidar_px": int(len(vv)), "lidar_share": float(len(vv) / sparse.size),
                      "net_ms": 1e3 * t_net, "align_ms": 1e3 * t_align})
        per_frame.append(stats)
        if len(per_frame) % args.save_every == 1:
            rgb = cv2.resize(crop_front(bgr), (side, side), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(args.out_dir / f"frame_{len(per_frame):04d}.jpg"),
                        montage(rgb, sparse, depth, source, args.max_range))
            np.savez_compressed(args.out_dir / f"frame_{len(per_frame):04d}.npz", depth=depth, source=source,
                                sparse=sparse, relative=rel)

    topics = [args.image_topic] + ([args.lidar_topic] if raw else [args.odom_topic, args.cloud_topic])
    for topic, msg in read_bag(args.bag, topics, raw=frozenset([args.lidar_topic])):
        if topic == args.lidar_topic:
            cloud.add(*livox_custom_xyz(msg, args.blind))
        elif topic == args.odom_topic:
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            poses.add(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec,
                      pose_to_matrix((p.x, p.y, p.z), (q.x, q.y, q.z, q.w)))
        elif topic == args.cloud_topic:
            cloud.add(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec, cloud_xyz_fast(msg))
        else:
            t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
            n_image += 1
            if n_image % args.every == 0:
                pending.append((t, msg))
        while pending and (raw or poses.last_t() >= pending[0][0]) and (cloud.last_t() or 0.0) >= pending[0][0]:
            process(*pending.pop(0))

    report = {"frames": len(per_frame), "side_px": side}
    for key, (preds, truths) in collected.items():
        if preds:
            report[key] = range_bin_errors(np.concatenate(preds), np.concatenate(truths))
    if per_frame:
        for k in ("scale", "lidar_share", "aligned_share", "net_ms", "align_ms"):
            vals = np.array([f[k] for f in per_frame])
            report[k] = {"median": float(np.median(vals)), "min": float(vals.min()), "max": float(vals.max())}
    (args.out_dir / "report.json").write_text(json.dumps({"summary": report, "frames": per_frame}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
