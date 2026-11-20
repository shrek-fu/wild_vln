#!/usr/bin/env python3
"""Live DA360 depth from Insta360 equirect, expressed in lidar_level.

Requires:
  - ROS 2 Humble
  - ``./calibrate.sh bringup`` (or camera + apply) from insta360_multi_mid360s_calibration
  - DA360 checkpoint under ``./checkpoints/``
  - Calib files under ``./calib/`` (``scripts/import_calib.sh``)

Publishes:
  /da360/depth              relative camera depth (32FC1)
  /da360/depth_metric       metric camera depth after optional lidar scale (32FC1)
  /da360/depth_global       ||p|| in lidar_level for each pixel (32FC1)
  /da360/depth_viz          KITTI colormap of camera depth (BGR8)
  /da360/points_level       PointCloud2 in frame lidar_level
  /da360/camera_info        equirect camera_info (width/height as intrinsics)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2

from insta_live.calib_io import load_rig_calibration, write_merged_summary
from insta_live.geometry import (
    da360_depth_to_points,
    estimate_scale_from_lidar,
    pack_global_depth_image,
    transform_points,
)
from insta_live.model_runner import DA360Runner
from saver import kitti_colormap


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", default=str(ROOT / "checkpoints" / "DA360_large.pth"))
    p.add_argument("--calib_dir", default=str(ROOT / "calib"))
    p.add_argument("--image_topic", default="/insta360/equirect/image/compressed")
    p.add_argument("--lidar_topic", default="/extrinsic/lidar_fused",
                   help="lidar_level cloud for optional metric scale; empty to disable")
    p.add_argument("--frame_id", default="lidar_level")
    p.add_argument("--max_hz", type=float, default=2.0, help="inference rate cap")
    p.add_argument("--stride", type=int, default=4, help="point cloud pixel stride")
    p.add_argument("--max_range", type=float, default=80.0)
    p.add_argument("--save_dir", default="", help="if set, write latest depth/ply under this dir")
    p.add_argument("--device", default="")
    return p.parse_args()


def _image_msg(header: Header, arr: np.ndarray, encoding: str) -> Image:
    msg = Image()
    msg.header = header
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    if encoding == "32FC1":
        flat = np.ascontiguousarray(arr, dtype=np.float32)
        msg.step = msg.width * 4
    elif encoding == "bgr8":
        flat = np.ascontiguousarray(arr, dtype=np.uint8)
        msg.step = msg.width * 3
    else:
        raise ValueError(encoding)
    msg.data = flat.tobytes()
    return msg


def _cloud_xyz(msg: PointCloud2) -> np.ndarray:
    if hasattr(point_cloud2, "read_points_numpy"):
        pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        if getattr(pts, "dtype", None) is not None and pts.dtype.names:
            return np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float64)
        return np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    rows = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    return np.array([[p[0], p[1], p[2]] for p in rows], dtype=np.float64)


def _make_cloud(header: Header, xyz: np.ndarray, rgb: np.ndarray) -> PointCloud2:
    """Pack XYZ + RGB into PointCloud2 (rgb as packed float32, PCL/Foxglove style)."""
    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint32)
    packed = ((rgb8[:, 0] << 16) | (rgb8[:, 1] << 8) | rgb8[:, 2]).astype(np.uint32)
    packed_f = packed.view(np.float32)
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    n = int(xyz.shape[0])
    buf = np.zeros((n, 4), dtype=np.float32)
    buf[:, 0:3] = xyz.astype(np.float32)
    buf[:, 3] = packed_f
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.is_dense = True
    msg.data = buf.tobytes()
    return msg


class LiveDA360Node(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("da360_insta360_live")
        self.args = args
        self.calib = load_rig_calibration(args.calib_dir)
        write_merged_summary(self.calib, Path(args.calib_dir) / "rig_calib.json")
        self.get_logger().info(self.calib.summary())

        if not Path(args.model_path).is_file():
            raise SystemExit(
                f"missing checkpoint {args.model_path}; run bash scripts/download_models.sh"
            )
        self.get_logger().info(f"loading {args.model_path}")
        self.runner = DA360Runner(args.model_path, device=args.device or None)
        self.get_logger().info(
            f"model ready on {self.runner.device} input {self.runner.width}x{self.runner.height}"
        )

        self._lock = threading.Lock()
        self._latest_bgr: np.ndarray | None = None
        self._latest_stamp = None
        self._lidar_xyz: np.ndarray | None = None
        self._scale: float | None = None
        self._last_infer = 0.0
        self._busy = False

        qos = qos_profile_sensor_data
        self.create_subscription(CompressedImage, args.image_topic, self._on_image, qos)
        if args.lidar_topic:
            self.create_subscription(PointCloud2, args.lidar_topic, self._on_lidar, qos)

        self.pub_depth = self.create_publisher(Image, "/da360/depth", 1)
        self.pub_metric = self.create_publisher(Image, "/da360/depth_metric", 1)
        self.pub_global = self.create_publisher(Image, "/da360/depth_global", 1)
        self.pub_viz = self.create_publisher(Image, "/da360/depth_viz", 1)
        self.pub_cloud = self.create_publisher(PointCloud2, "/da360/points_level", 1)
        self.pub_info = self.create_publisher(CameraInfo, "/da360/camera_info", 1)

        self.create_timer(0.05, self._maybe_infer)
        self.get_logger().info(
            f"waiting for {args.image_topic}"
            + (f" and optional {args.lidar_topic}" if args.lidar_topic else "")
        )

    def _on_image(self, msg: CompressedImage) -> None:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        with self._lock:
            self._latest_bgr = bgr
            self._latest_stamp = msg.header.stamp

    def _on_lidar(self, msg: PointCloud2) -> None:
        try:
            xyz = _cloud_xyz(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"lidar decode failed: {exc}")
            return
        if xyz.size == 0:
            return
        with self._lock:
            self._lidar_xyz = xyz

    def _camera_info(self, stamp, width: int, height: int) -> CameraInfo:
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = "insta360_equirect"
        info.width = width
        info.height = height
        # Equirectangular: store size in K like koide (fx=width, fy=height).
        info.k = [float(width), 0.0, width / 2.0 - 0.5, 0.0, float(height), height / 2.0 - 0.5, 0.0, 0.0, 1.0]
        info.p = [float(width), 0.0, width / 2.0 - 0.5, 0.0, 0.0, float(height), height / 2.0 - 0.5, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.distortion_model = "equirectangular"
        info.d = []
        return info

    def _maybe_infer(self) -> None:
        if self._busy:
            return
        now = time.time()
        if now - self._last_infer < 1.0 / max(self.args.max_hz, 0.1):
            return
        with self._lock:
            bgr = None if self._latest_bgr is None else self._latest_bgr.copy()
            stamp = self._latest_stamp
            lidar = None if self._lidar_xyz is None else self._lidar_xyz.copy()
        if bgr is None or stamp is None:
            return
        self._busy = True
        self._last_infer = now
        try:
            self._infer_once(bgr, stamp, lidar)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"inference failed: {exc}")
        finally:
            self._busy = False

    def _infer_once(self, bgr: np.ndarray, stamp, lidar_xyz: np.ndarray | None) -> None:
        h0, w0 = bgr.shape[:2]
        t0 = time.time()
        depth_rel, rgb = self.runner.predict_depth(bgr, is_bgr=True, out_size=(w0, h0))
        scale = self._scale
        if lidar_xyz is not None and lidar_xyz.shape[0] > 0:
            new_scale = estimate_scale_from_lidar(
                depth_rel, lidar_xyz, self.calib.T_level_da360
            )
            if new_scale is not None:
                scale = new_scale
                self._scale = scale
        depth_metric = depth_rel if scale is None else (depth_rel * scale)

        mask = np.isfinite(depth_metric) & (depth_metric > 0)
        if self.args.max_range > 0 and scale is not None:
            mask &= depth_metric < self.args.max_range

        stride = max(1, int(self.args.stride))
        sel = np.zeros_like(mask)
        sel[::stride, ::stride] = True
        sel &= mask

        pts_cam = da360_depth_to_points(depth_metric, sel)
        pts_level = transform_points(pts_cam, self.calib.T_level_da360)
        cols = rgb[sel]

        depth_global = pack_global_depth_image(depth_metric, self.calib.T_level_da360, mask)

        # Visualization uses relative disparity-like map for stable colours.
        viz = kitti_colormap(1.0 / np.maximum(depth_rel, 1e-3))
        viz_bgr = cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)

        header = Header(stamp=stamp, frame_id=self.args.frame_id)
        for pub, arr in (
            (self.pub_depth, depth_rel.astype(np.float32)),
            (self.pub_metric, depth_metric.astype(np.float32)),
            (self.pub_global, depth_global.astype(np.float32)),
        ):
            pub.publish(_image_msg(header, arr, "32FC1"))
        self.pub_viz.publish(_image_msg(header, viz_bgr, "bgr8"))
        self.pub_cloud.publish(
            _make_cloud(header, pts_level.astype(np.float32), cols.astype(np.float32))
        )
        self.pub_info.publish(self._camera_info(stamp, w0, h0))

        dt = time.time() - t0
        scale_txt = f"{scale:.3f}" if scale is not None else "relative"
        self.get_logger().info(
            f"frame {w0}x{h0} pts={pts_level.shape[0]} scale={scale_txt} {dt*1000:.0f} ms"
        )

        if self.args.save_dir:
            out = Path(self.args.save_dir)
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "latest_rgb.jpg"), bgr)
            cv2.imwrite(str(out / "latest_depth_viz.jpg"), viz_bgr)
            np.save(out / "latest_depth_metric.npy", depth_metric.astype(np.float32))
            np.save(out / "latest_depth_global.npy", depth_global)
            # lightweight ply without open3d dependency at runtime
            self._write_ply(out / "latest_points_level.ply", pts_level, cols)

    @staticmethod
    def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
        rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        n = xyz.shape[0]
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for i in range(n):
                x, y, z = xyz[i]
                r, g, b = rgb8[i]
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")


def main() -> None:
    args = _parse_args()
    rclpy.init()
    node = LiveDA360Node(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
