#!/usr/bin/env python3
"""Publish both lidars in the gravity-leveled frame from results/lidar_extrinsic.json.

    p_level = T_level_front * p_front
    p_level = T_level_rear  * p_rear

/extrinsic/front_lidar   front cloud in the level frame
/extrinsic/rear_lidar    rear cloud in the level frame
/extrinsic/lidar_fused   both clouds in one message
/extrinsic/lidar_fused_front  both clouds in the front lidar frame (camera calibration);
                              only built while something subscribes

Single-lidar rig (one lidar in config/rig.yaml): only T_level_front is needed; lidar_fused is the
leveled front cloud and lidar_fused_front the raw front cloud, so downstream topics stay the same.

The level frame has its origin at the front lidar, Z against gravity and the
front lidar's yaw. The raw driver topics are not changed, so capture still
records each lidar in its own frame. The file is read once at start.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

import extrinsic_io
from cloud_msg import MAX_SKEW_NS, concat_clouds, stamp_ns, transform_xyz


def load_level_transforms(path: Path, single: bool) -> tuple[np.ndarray, np.ndarray | None, str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    for key in ("T_level_front",) if single else ("T_level_front", "T_level_rear"):
        if key not in doc:
            how = "./calibrate.sh level <scene>" if single else "./calibrate.sh initial"
            raise SystemExit(f"{path} has no {key}; run {how}")
    T_front = np.array(doc["T_level_front"], dtype=np.float64)
    T_rear = None if single else np.array(doc["T_level_rear"], dtype=np.float64)
    return T_front, T_rear, str(doc.get("status", "unknown"))


class Apply(Node):
    def __init__(self, rig: dict, T_front: np.ndarray, T_rear: np.ndarray | None, status: str) -> None:
        super().__init__("lidar_extrinsic_apply")
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self._single = T_rear is None
        self._T_front = T_front
        self._T_rear = T_rear
        self._T_front_rear = None if self._single else np.linalg.inv(T_front) @ T_rear
        self._front_frame = rig["front"]["frame"]
        self._front: PointCloud2 | None = None
        self._rear: PointCloud2 | None = None
        self._front_raw: PointCloud2 | None = None
        self._rear_in_front: PointCloud2 | None = None
        self._layout_warned = False
        self._pub_front = self.create_publisher(PointCloud2, "/extrinsic/front_lidar", qos)
        self._pub_rear = self.create_publisher(PointCloud2, "/extrinsic/rear_lidar", qos)
        self._pub_fused = self.create_publisher(PointCloud2, "/extrinsic/lidar_fused", qos)
        self._pub_fused_front = self.create_publisher(PointCloud2, "/extrinsic/lidar_fused_front", qos)
        self.create_subscription(PointCloud2, rig["front"]["points"], self._on_front, qos)
        if self._single:
            self.get_logger().info(
                f"single lidar: applying results/{extrinsic_io.EXTRINSIC_FILE} status={status} "
                f"front {rig['front']['points']} rpy_deg={extrinsic_io.rpy_deg(T_front)}  "
                f"frame={extrinsic_io.LEVEL_FRAME}"
            )
            return
        self.create_subscription(PointCloud2, rig["rear"]["points"], self._on_rear, qos)
        self.get_logger().info(
            f"applying results/{extrinsic_io.EXTRINSIC_FILE} status={status} "
            f"front {rig['front']['points']} rpy_deg={extrinsic_io.rpy_deg(T_front)}  "
            f"rear {rig['rear']['points']} rpy_deg={extrinsic_io.rpy_deg(T_rear)} "
            f"t_mm={extrinsic_io.pose(T_rear)['t_mm']}  frame={extrinsic_io.LEVEL_FRAME}"
        )

    def _on_front(self, msg: PointCloud2) -> None:
        cloud = transform_xyz(msg, self._T_front, extrinsic_io.LEVEL_FRAME)
        self._pub_front.publish(cloud)
        self._front = cloud
        self._front_raw = msg
        if self._single:
            self._pub_fused.publish(cloud)
            if self._pub_fused_front.get_subscription_count() > 0:
                self._pub_fused_front.publish(transform_xyz(msg, np.eye(4), self._front_frame))
            return
        self._publish_fused()
        self._publish_fused_front()

    def _on_rear(self, msg: PointCloud2) -> None:
        cloud = transform_xyz(msg, self._T_rear, extrinsic_io.LEVEL_FRAME)
        self._pub_rear.publish(cloud)
        self._rear = cloud
        if self._pub_fused_front.get_subscription_count() > 0:
            self._rear_in_front = transform_xyz(msg, self._T_front_rear, self._front_frame)
        self._publish_fused()

    def _publish_fused(self) -> None:
        if self._front is None or self._rear is None:
            return
        if abs(stamp_ns(self._front) - stamp_ns(self._rear)) > MAX_SKEW_NS:
            return
        try:
            fused = concat_clouds(self._front, self._rear, extrinsic_io.LEVEL_FRAME)
        except RuntimeError as exc:
            if not self._layout_warned:
                self.get_logger().warning(str(exc))
                self._layout_warned = True
            return
        self._pub_fused.publish(fused)

    def _publish_fused_front(self) -> None:
        if self._pub_fused_front.get_subscription_count() == 0:
            return
        if self._front_raw is None or self._rear_in_front is None:
            return
        if abs(stamp_ns(self._front_raw) - stamp_ns(self._rear_in_front)) > MAX_SKEW_NS:
            return
        try:
            fused = concat_clouds(self._front_raw, self._rear_in_front, self._front_frame)
        except RuntimeError:
            return
        self._pub_fused_front.publish(fused)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rig = extrinsic_io.load_rig(root)
    T_front, T_rear, status = load_level_transforms(root / "results" / extrinsic_io.EXTRINSIC_FILE,
                                                    extrinsic_io.single(rig))
    rclpy.init()
    node = Apply(rig, T_front, T_rear, status)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
