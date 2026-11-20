#!/usr/bin/env python3
"""Live PromptDA depth: Insta360 pinhole views + Livox cloud in lidar_level.

Requires:
  - ROS 2 Humble
  - ``insta360_views`` (or nav with camera:=true) publishing ``/insta360/<view>/image/compressed``
  - a lidar_level cloud (default ``/extrinsic/lidar_fused``, same as ``da360_depth/live_ros.py``)
  - PromptDA checkpoint under ``checkpoints/`` (``scripts/download_weights.sh``)
  - calib under ``../da360_depth/calib``

Publishes per view:
  /promptda/<view>/depth          metric depth 32FC1 (metres)
  /promptda/<view>/prompt         lidar prompt 32FC1
  /promptda/<view>/depth_viz      TURBO colormap BGR8
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from prompt_depth import build_prompt_depth, load_views
from runner import PromptDARunner, default_ckpt


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib_dir", default=str(ROOT.parent / "da360_depth" / "calib"))
    p.add_argument("--views", default="front", help="comma list: front,left,rear,right")
    p.add_argument("--lidar_topic", default="/extrinsic/lidar_fused")
    p.add_argument("--ckpt", default="")
    p.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    p.add_argument("--max_hz", type=float, default=1.0)
    p.add_argument("--project_scale", type=float, default=1.0 / 7.0)
    p.add_argument("--knn_size", type=int, default=18)
    p.add_argument("--knn_k", type=int, default=4)
    p.add_argument("--device", default="")
    return p.parse_args()


def _image_msg(header: Header, arr: np.ndarray, encoding: str) -> Image:
    msg = Image()
    msg.header = header
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    if encoding == "32FC1":
        flat = np.ascontiguousarray(arr, np.float32)
        msg.step = msg.width * 4
        msg.data = flat.tobytes()
    else:
        flat = np.ascontiguousarray(arr, np.uint8)
        msg.step = msg.width * arr.shape[2]
        msg.data = flat.tobytes()
    return msg


def _colorize(depth: np.ndarray) -> np.ndarray:
    m = depth > 0
    x = np.zeros_like(depth, np.float32)
    if m.any():
        lo, hi = np.percentile(depth[m], [2, 98])
        x[m] = np.clip((depth[m] - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


class PromptDANode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("promptda_livox")
        names = tuple(v.strip() for v in args.views.split(",") if v.strip())
        self.views = {v.name: v for v in load_views(args.calib_dir, names=names)}
        ckpt = Path(args.ckpt) if args.ckpt else default_ckpt(args.encoder)
        self.get_logger().info(f"loading PromptDA {args.encoder} from {ckpt}")
        self.runner = PromptDARunner(ckpt, encoder=args.encoder, device=args.device)
        self.args = args
        self.lock = threading.Lock()
        self.latest_img: dict[str, tuple[float, np.ndarray, Header]] = {}
        self.latest_cloud: tuple[float, np.ndarray] | None = None
        self.last_infer = {n: 0.0 for n in self.views}

        self.pub_depth = {}
        self.pub_prompt = {}
        self.pub_viz = {}
        for name, view in self.views.items():
            self.create_subscription(
                CompressedImage, view.image_topic,
                lambda m, n=name: self._on_image(n, m), qos_profile_sensor_data)
            self.pub_depth[name] = self.create_publisher(Image, f"/promptda/{name}/depth", 1)
            self.pub_prompt[name] = self.create_publisher(Image, f"/promptda/{name}/prompt", 1)
            self.pub_viz[name] = self.create_publisher(Image, f"/promptda/{name}/depth_viz", 1)
            self.get_logger().info(f"view {name}: {view.image_topic}")

        self.create_subscription(PointCloud2, args.lidar_topic, self._on_cloud, qos_profile_sensor_data)
        self.create_timer(0.05, self._tick)
        self.get_logger().info(f"lidar {args.lidar_topic}; max {args.max_hz} Hz/view")

    def _on_image(self, name: str, msg: CompressedImage) -> None:
        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        with self.lock:
            self.latest_img[name] = (t, bgr, msg.header)

    def _on_cloud(self, msg: PointCloud2) -> None:
        pts = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        xyz = np.asarray(pts, np.float64).reshape(-1, 3) if pts else np.zeros((0, 3))
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        with self.lock:
            self.latest_cloud = (t, xyz)

    def _tick(self) -> None:
        with self.lock:
            cloud = self.latest_cloud
            images = dict(self.latest_img)
        if cloud is None or not images:
            return
        _, xyz = cloud
        now = time.monotonic()
        period = 1.0 / max(self.args.max_hz, 0.1)
        for name, (t, bgr, header) in images.items():
            if now - self.last_infer[name] < period:
                continue
            view = self.views[name]
            if bgr.shape[0] != view.height or bgr.shape[1] != view.width:
                bgr = cv2.resize(bgr, (view.width, view.height), interpolation=cv2.INTER_AREA)
            try:
                prompt, stats = build_prompt_depth(
                    xyz, view, project_scale=self.args.project_scale,
                    knn_size=self.args.knn_size, knn_k=self.args.knn_k,
                    out_size=(view.width, view.height))
                if stats["hits"] < 20:
                    self.get_logger().warn(f"{name}: only {stats['hits']} lidar hits in view",
                                           throttle_duration_sec=5.0)
                    continue
                depth = self.runner.predict(bgr, prompt)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"{name}: {exc}", throttle_duration_sec=5.0)
                continue
            self.last_infer[name] = now
            h = Header()
            h.stamp = header.stamp
            h.frame_id = f"insta360_{name}_optical"
            self.pub_depth[name].publish(_image_msg(h, depth, "32FC1"))
            self.pub_prompt[name].publish(_image_msg(h, prompt, "32FC1"))
            self.pub_viz[name].publish(_image_msg(h, _colorize(depth), "bgr8"))
            self.get_logger().info(
                f"{name}: depth {float(depth[depth > 0].min()):.2f}..{float(depth.max()):.2f} m "
                f"(prompt hits={stats['hits']})", throttle_duration_sec=2.0)


def main() -> None:
    args = _parse()
    rclpy.init()
    node = PromptDANode(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
