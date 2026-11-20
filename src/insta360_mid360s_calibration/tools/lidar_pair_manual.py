#!/usr/bin/env python3
"""Keyboard fine adjustment of rear_lidar -> front_lidar, same increments as SensorsCalibration.

The reference manual tool draws both clouds in Pangolin and applies a small
motion in the source frame on each key:

    q/a  +/- roll     w/s  +/- pitch    e/d  +/- yaw
    r/f  +/- x        t/g  +/- y        y/h  +/- z

This host is viewed in Foxglove, so the same keys drive the cloud published
on /calib/rear_lidar. Fixed frame: lidar_level (gravity-aligned, front origin).

Extra keys: n/m degree step, k/l translation step, c reset, o auto-refine
the latest frame pair, z save.
"""
from __future__ import annotations

import select
import sys
import termios
import threading
import tty
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

from lidar_pair_refine import (
    as_T,
    inv,
    load_pair,
    refine_pair,
    rodrigues,
    rotation_angle_deg,
    write_results,
)
import extrinsic_io
from cloud_msg import concat_clouds, transform_xyz, xyz_from_cloud


ROOT = Path(__file__).resolve().parents[1]
AXES = (
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
)


def delta(index: int, sign: float, deg: float, trans_m: float) -> np.ndarray:
    T = np.eye(4)
    if index < 3:
        T[:3, :3] = rodrigues(AXES[index], np.deg2rad(sign * deg))
    else:
        T[index - 3, 3] = sign * trans_m
    return T


class Manual(Node):
    def __init__(self) -> None:
        super().__init__("lidar_pair_manual")
        rig = extrinsic_io.load_rig(ROOT)
        self.pair, self.T, self.T_cad = load_pair(ROOT / "results" / extrinsic_io.EXTRINSIC_FILE)
        self._level_front = as_T(self.pair["R_level_front"])
        self._level_rear = as_T(self.pair["R_level_rear"])
        self.T0 = self.T.copy()
        self.deg = 0.3
        self.trans_m = 0.06
        self._lock = threading.Lock()
        self._latest: dict[str, np.ndarray] = {}
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self._cloud_front: PointCloud2 | None = None
        self._cloud_rear: PointCloud2 | None = None
        self._pub_front = self.create_publisher(PointCloud2, "/calib/front_lidar", qos)
        self._pub_rear = self.create_publisher(PointCloud2, "/calib/rear_lidar", qos)
        self._pub_fused = self.create_publisher(PointCloud2, "/calib/lidar_fused", qos)
        self.create_subscription(PointCloud2, rig["front"]["points"], self._on_front, qos)
        self.create_subscription(PointCloud2, rig["rear"]["points"], self._on_rear, qos)
        self._print_help()
        self._print_state("seed")

    def _print_help(self) -> None:
        print(
            "\n".join(
                [
                    f"Foxglove fixed frame {extrinsic_io.LEVEL_FRAME}",
                    "fused topic /calib/lidar_fused",
                    "separate topics /calib/front_lidar and /calib/rear_lidar",
                    "q/a roll   w/s pitch   e/d yaw    (degrees, source frame)",
                    "r/f x      t/g y       y/h z      (meters, source frame)",
                    "n/m degree step    k/l translation step",
                    "c reset    o auto-refine latest frames    z save    Ctrl-C quit",
                    f"steps: {self.deg:.3f} deg, {self.trans_m * 100:.2f} cm",
                ]
            ),
            flush=True,
        )

    def _print_state(self, reason: str) -> None:
        with self._lock:
            T = self.T.copy()
        t_mm = np.round(T[:3, 3] * 1000.0, 2)
        print(f"{reason}  t_mm={t_mm.tolist()}  deg_step={self.deg:.3f}  cm_step={self.trans_m * 100:.2f}", flush=True)

    def _on_front(self, msg: PointCloud2) -> None:
        cloud = transform_xyz(msg, self._level_front, extrinsic_io.LEVEL_FRAME)
        self._pub_front.publish(cloud)
        self._cloud_front = cloud
        xyz = xyz_from_cloud(msg)
        with self._lock:
            self._latest["front"] = xyz
        self._publish_fused()

    def _on_rear(self, msg: PointCloud2) -> None:
        with self._lock:
            placed = self.T @ self._level_rear
            self._latest["rear"] = xyz_from_cloud(msg)
        cloud = transform_xyz(msg, placed, extrinsic_io.LEVEL_FRAME)
        self._pub_rear.publish(cloud)
        self._cloud_rear = cloud
        self._publish_fused()

    def _publish_fused(self) -> None:
        front = self._cloud_front
        rear = self._cloud_rear
        if front is None or rear is None:
            return
        try:
            self._pub_fused.publish(concat_clouds(front, rear, extrinsic_io.LEVEL_FRAME))
        except RuntimeError as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=5.0)

    def apply_key(self, key: str) -> None:
        table = {
            "q": (0, +1.0),
            "a": (0, -1.0),
            "w": (1, +1.0),
            "s": (1, -1.0),
            "e": (2, +1.0),
            "d": (2, -1.0),
            "r": (3, +1.0),
            "f": (3, -1.0),
            "t": (4, +1.0),
            "g": (4, -1.0),
            "y": (5, +1.0),
            "h": (5, -1.0),
        }
        if key in table:
            index, sign = table[key]
            with self._lock:
                self.T = self.T @ delta(index, sign, self.deg, self.trans_m)
            self._print_state(key)
            return
        if key == "n":
            self.deg = max(0.01, self.deg / 2.0)
        elif key == "m":
            self.deg = min(5.0, self.deg * 2.0)
        elif key == "k":
            self.trans_m = max(0.001, self.trans_m / 2.0)
        elif key == "l":
            self.trans_m = min(0.2, self.trans_m * 2.0)
        elif key == "c":
            with self._lock:
                self.T = self.T0.copy()
            self._print_state("reset")
            return
        elif key == "o":
            self._auto()
            return
        elif key == "z":
            self._save()
            return
        else:
            return
        self._print_state("step")

    def _clouds(self) -> tuple[np.ndarray, np.ndarray] | None:
        with self._lock:
            if "front" not in self._latest or "rear" not in self._latest:
                return None
            return self._latest["front"].copy(), self._latest["rear"].copy()

    def _auto(self) -> None:
        clouds = self._clouds()
        if clouds is None:
            print("auto: waiting for both lidars", flush=True)
            return
        print("auto: multi-stage ICP on the leveled clouds", flush=True)
        with self._lock:
            seed = self.T.copy()
        front = (self._level_front[:3, :3] @ clouds[0].T).T
        rear = (self._level_rear[:3, :3] @ clouds[1].T).T
        try:
            metrics = refine_pair(front, rear, seed)
        except Exception as exc:
            print(f"auto failed: {exc}", flush=True)
            return
        with self._lock:
            self.T = metrics["T"]
        print(
            f"auto fitness={metrics['fitness']:.4f} rmse_m={metrics['inlier_rmse_m']:.4f} "
            f"delta_deg={metrics['rotation_delta_deg']:.3f} "
            f"delta_mm={np.round(metrics['translation_delta_mm'], 2).tolist()}",
            flush=True,
        )
        self._print_state("auto")

    def _save(self) -> None:
        with self._lock:
            T = self.T.copy()
        dT = T @ inv(self.T0)
        metrics = {
            "fitness": float(self.pair.get("quality", {}).get("fitness_within_0_10m", 0.0)),
            "inlier_rmse_m": float(self.pair.get("quality", {}).get("inlier_rmse_mm", 0.0)) / 1000.0,
            "rotation_delta_deg": rotation_angle_deg(dT[:3, :3]),
        }
        path = write_results(
            ROOT,
            self.pair,
            T,
            self.T_cad,
            "manual",
            "SensorsCalibration/lidar2lidar manual_calib keys, source-frame increments, Foxglove view",
            metrics,
        )
        self.pair, _, _ = load_pair(path)
        print(f"saved {path}", flush=True)


def main() -> int:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    rclpy.init()
    node = Manual()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            ready, _, _ = select.select([fd], [], [], 0.2)
            if not ready:
                continue
            key = sys.stdin.read(1)
            if key == "\x03":
                break
            node.apply_key(key)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
