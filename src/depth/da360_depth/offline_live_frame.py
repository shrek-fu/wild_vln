#!/usr/bin/env python3
"""Run DA360 once on an equirect image (or one ROS compressed frame) with calib.

Examples:
  python offline_live_frame.py --image path/to/pano.jpg
  python offline_live_frame.py --ros-topic /insta360/equirect/image/compressed
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from insta_live.calib_io import load_rig_calibration, write_merged_summary
from insta_live.geometry import (
    da360_depth_to_points,
    pack_global_depth_image,
    transform_points,
)
from insta_live.model_runner import DA360Runner
from saver import kitti_colormap


def grab_ros_compressed(topic: str, timeout: float = 8.0) -> np.ndarray:
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage

    rclpy.init()
    node = rclpy.create_node("da360_grab_frame")
    box: dict = {}

    def cb(msg: CompressedImage) -> None:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        box["bgr"] = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    node.create_subscription(CompressedImage, topic, cb, qos_profile_sensor_data)
    end = __import__("time").time() + timeout
    while __import__("time").time() < end and "bgr" not in box:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    if "bgr" not in box or box["bgr"] is None:
        raise SystemExit(f"no image on {topic}")
    return box["bgr"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", default=str(ROOT / "checkpoints" / "DA360_large.pth"))
    ap.add_argument("--calib_dir", default=str(ROOT / "calib"))
    ap.add_argument("--image", default="")
    ap.add_argument("--ros-topic", default="")
    ap.add_argument("--out_dir", default=str(ROOT / "results" / "live"))
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    calib = load_rig_calibration(args.calib_dir)
    write_merged_summary(calib, Path(args.calib_dir) / "rig_calib.json")
    print(calib.summary())

    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f"cannot read {args.image}")
    elif args.ros_topic:
        bgr = grab_ros_compressed(args.ros_topic)
    else:
        raise SystemExit("pass --image or --ros-topic")

    if not Path(args.model_path).is_file():
        raise SystemExit(f"missing {args.model_path}; run bash scripts/download_models.sh")

    runner = DA360Runner(args.model_path)
    h, w = bgr.shape[:2]
    depth, rgb = runner.predict_depth(bgr, is_bgr=True, out_size=(w, h))
    mask = np.isfinite(depth) & (depth > 0)
    sel = np.zeros_like(mask)
    sel[:: args.stride, :: args.stride] = True
    sel &= mask
    pts = transform_points(da360_depth_to_points(depth, sel), calib.T_level_da360)
    depth_global = pack_global_depth_image(depth, calib.T_level_da360, mask)
    viz = kitti_colormap(1.0 / np.maximum(depth, 1e-3))
    viz_bgr = cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "rgb.jpg"), bgr)
    cv2.imwrite(str(out / "depth_viz.jpg"), viz_bgr)
    np.save(out / "depth_relative.npy", depth)
    np.save(out / "depth_global.npy", depth_global)
    # ASCII PLY
    rgb8 = np.clip(rgb[sel] * 255.0, 0, 255).astype(np.uint8)
    with open(out / "points_level.ply", "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for i in range(pts.shape[0]):
            x, y, z = pts[i]
            r, g, b = rgb8[i]
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
    print(f"wrote {out} ({pts.shape[0]} points in lidar_level, relative scale)")


if __name__ == "__main__":
    main()
