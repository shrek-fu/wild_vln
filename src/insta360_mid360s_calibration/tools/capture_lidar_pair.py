#!/usr/bin/env python3
"""Save live lidar frames and the built-in IMU gravity into calib_capture/lidar/clouds/<scene>/.

For each lidar in config/rig.yaml (front_lidar, and rear_lidar on the dual rig) the scene folder gets:
    <name>.pcd       point cloud in that lidar's frame
    <name>.imu.json  mean accelerometer reading while the rig was still

The MID-360S IMU axes are parallel to the point cloud axes, so the mean
accelerometer vector is the up direction in the lidar frame.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2

import extrinsic_io
from cloud_msg import xyz_from_cloud


class Grabber(Node):
    def __init__(self, rig: dict, frames: int, imu_samples: int) -> None:
        super().__init__("lidar_pair_capture")
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        imu_qos = QoSProfile(depth=400, reliability=ReliabilityPolicy.RELIABLE)
        self.frames = frames
        self.imu_samples = imu_samples
        names = [rig[role]["id"] for role in extrinsic_io.roles(rig)]
        self.bags: dict[str, list[np.ndarray]] = {name: [] for name in names}
        self.acc: dict[str, list[list[float]]] = {name: [] for name in names}
        self.gyro: dict[str, list[list[float]]] = {name: [] for name in names}
        for role in extrinsic_io.roles(rig):
            lidar = rig[role]
            name = lidar["id"]
            self.create_subscription(
                PointCloud2, lidar["points"], lambda msg, key=name: self._on_cloud(key, msg), qos
            )
            self.create_subscription(
                Imu, lidar["imu"], lambda msg, key=name: self._on_imu(key, msg), imu_qos
            )
            self.get_logger().info(
                f"waiting for {frames} frames on {lidar['points']} and {imu_samples} samples on {lidar['imu']}"
            )

    def _on_cloud(self, name: str, msg: PointCloud2) -> None:
        if len(self.bags[name]) < self.frames:
            self.bags[name].append(xyz_from_cloud(msg))

    def _on_imu(self, name: str, msg: Imu) -> None:
        if len(self.acc[name]) >= self.imu_samples:
            return
        a = msg.linear_acceleration
        w = msg.angular_velocity
        self.acc[name].append([a.x, a.y, a.z])
        self.gyro[name].append([w.x, w.y, w.z])

    def ready(self) -> bool:
        return all(len(v) >= self.frames for v in self.bags.values()) and all(
            len(v) >= self.imu_samples for v in self.acc.values()
        )


def save_cloud(path: Path, xyz: np.ndarray, voxel: float) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), pcd):
        raise SystemExit(f"failed to write {path}")
    print(f"wrote {path}  points={len(pcd.points)}")


def save_imu(path: Path, topic: str, acc: list[list[float]], gyro: list[list[float]]) -> None:
    a = np.asarray(acc, dtype=float)
    w = np.asarray(gyro, dtype=float)
    mean = a.mean(axis=0)
    up = mean / np.linalg.norm(mean)
    tilt = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
    doc = {
        "topic": topic,
        "samples": int(len(a)),
        "acc_mean": mean.tolist(),
        "acc_std": a.std(axis=0).tolist(),
        "acc_norm_mean": float(np.linalg.norm(a, axis=1).mean()),
        "gyro_mean_rad_s": w.mean(axis=0).tolist(),
        "up_in_lidar": up.tolist(),
        "z_tilt_from_up_deg": tilt,
        "note": "Livox reports acceleration in g. Only the direction is used.",
    }
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {path}  samples={len(a)}  up={np.round(up, 4).tolist()}  "
        f"z_tilt_deg={tilt:.2f}  acc_std={np.round(a.std(axis=0), 4).tolist()}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--scene", default="live")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--imu-samples", type=int, default=400, help="200 Hz, so 400 is about 2 s")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--format", choices=("ply", "pcd"), default="ply")
    args = ap.parse_args()
    rig = extrinsic_io.load_rig(root)
    rclpy.init()
    node = Grabber(rig, args.frames, args.imu_samples)
    deadline = time.time() + args.timeout
    try:
        while time.time() < deadline and not node.ready():
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        counts = {k: len(v) for k, v in node.bags.items()}
        imu_counts = {k: len(v) for k, v in node.acc.items()}
        node.destroy_node()
        rclpy.shutdown()
    missing = [k for k, n in counts.items() if n < args.frames]
    missing += [k for k, n in imu_counts.items() if n < args.imu_samples]
    if missing:
        raise SystemExit(
            f"timed out, frames={counts} imu={imu_counts}. Start ./calibrate.sh bringup in another terminal; "
            "it remaps the driver topics to /livox/<name>/points and /livox/<name>/imu."
        )
    folder = extrinsic_io.clouds_dir(root) / args.scene
    for role in extrinsic_io.roles(rig):
        name = rig[role]["id"]
        save_cloud(folder / f"{name}.{args.format}", np.vstack(node.bags[name]), args.voxel)
        save_imu(folder / f"{name}.imu.json", rig[role]["imu"], node.acc[name], node.gyro[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
