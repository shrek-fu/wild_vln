#!/usr/bin/env python3
"""Offline PromptDA depth: Insta360 pinhole RGB + Livox (lidar_level) prompt.

Examples::

  # one RGB + one PCD (xyz in lidar_level)
  python offline.py --image front.jpg --cloud cloud.npy --view front --save /tmp/depth

  # ROS one-shot: grab one compressed view + one PointCloud2
  python offline.py --ros --view front --save /tmp/depth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompt_depth import build_prompt_depth, load_views  # noqa: E402
from runner import PromptDARunner, default_ckpt  # noqa: E402


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib_dir", default=str(ROOT.parent / "da360_depth" / "calib"))
    p.add_argument("--view", default="front", help="front|left|rear|right")
    p.add_argument("--image", default="", help="BGR image path")
    p.add_argument("--cloud", default="", help=".npy Nx3 in lidar_level, or .pcd xyz")
    p.add_argument("--ros", action="store_true", help="grab one frame from ROS topics")
    p.add_argument("--image_topic", default="", help="override view image topic")
    p.add_argument("--lidar_topic", default="/extrinsic/lidar_fused",
                   help="PointCloud2 in lidar_level (same as da360 live_ros)")
    p.add_argument("--ckpt", default="", help="default: checkpoints/prompt-depth-anything-vits.ckpt")
    p.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    p.add_argument("--project_scale", type=float, default=1.0 / 7.0)
    p.add_argument("--knn_size", type=int, default=18)
    p.add_argument("--knn_k", type=int, default=4)
    p.add_argument("--save", default="/tmp/promptda_livox")
    p.add_argument("--device", default="")
    return p.parse_args()


def _load_cloud(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).reshape(-1, 3)
    if path.suffix == ".pcd":
        # minimal ASCII/binary via open3d if present, else fail clearly
        try:
            import open3d as o3d
            return np.asarray(o3d.io.read_point_cloud(str(path)).points)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"need open3d to read {path}: {exc}") from exc
    raise SystemExit(f"unsupported cloud {path}")


def _ros_once(image_topic: str, lidar_topic: str) -> tuple[np.ndarray, np.ndarray]:
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, PointCloud2

    rclpy.init()
    node = rclpy.create_node("promptda_offline")
    state: dict = {}

    def on_img(msg):
        state["img"] = msg

    def on_cloud(msg):
        state["cloud"] = msg

    node.create_subscription(CompressedImage, image_topic, on_img, qos_profile_sensor_data)
    node.create_subscription(PointCloud2, lidar_topic, on_cloud, qos_profile_sensor_data)
    print(f"[offline] waiting for {image_topic} + {lidar_topic}")
    while rclpy.ok() and ("img" not in state or "cloud" not in state):
        rclpy.spin_once(node, timeout_sec=0.2)
    img_msg, cloud_msg = state["img"], state["cloud"]
    bgr = cv2.imdecode(np.frombuffer(img_msg.data, np.uint8), cv2.IMREAD_COLOR)
    # decode xyz
    from sensor_msgs_py import point_cloud2
    pts = list(point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True))
    xyz = np.asarray(pts, np.float64).reshape(-1, 3) if pts else np.zeros((0, 3))
    node.destroy_node()
    rclpy.shutdown()
    if bgr is None:
        raise SystemExit("failed to decode image")
    return bgr, xyz


def main() -> None:
    args = _parse()
    views = load_views(args.calib_dir, names=(args.view,))
    view = views[0]
    if args.ros:
        topic = args.image_topic or view.image_topic
        bgr, xyz = _ros_once(topic, args.lidar_topic)
    else:
        if not args.image or not args.cloud:
            raise SystemExit("need --image and --cloud, or --ros")
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f"cannot read {args.image}")
        xyz = _load_cloud(Path(args.cloud))

    if bgr.shape[0] != view.height or bgr.shape[1] != view.width:
        bgr = cv2.resize(bgr, (view.width, view.height), interpolation=cv2.INTER_AREA)

    prompt, stats = build_prompt_depth(
        xyz, view, project_scale=args.project_scale, knn_size=args.knn_size, knn_k=args.knn_k,
        out_size=(view.width, view.height))
    print(f"[offline] lidar prompt: {stats}")

    ckpt = Path(args.ckpt) if args.ckpt else default_ckpt(args.encoder)
    runner = PromptDARunner(ckpt, encoder=args.encoder, device=args.device)
    depth = runner.predict(bgr, prompt)

    out = Path(args.save)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{args.view}_depth.npy", depth)
    np.save(out / f"{args.view}_prompt.npy", prompt)
    # viz
    def colorize(d):
        m = d > 0
        if not m.any():
            return np.zeros((*d.shape, 3), np.uint8)
        x = np.zeros_like(d, np.float32)
        lo, hi = np.percentile(d[m], [2, 98])
        x[m] = np.clip((d[m] - lo) / max(hi - lo, 1e-6), 0, 1)
        return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    cv2.imwrite(str(out / f"{args.view}_rgb.jpg"), bgr)
    cv2.imwrite(str(out / f"{args.view}_prompt.jpg"), colorize(prompt))
    cv2.imwrite(str(out / f"{args.view}_depth.jpg"), colorize(depth))
    print(f"[offline] wrote {out}/{{{args.view}_depth.npy,_prompt.npy,*.jpg}} "
          f"depth {depth[depth > 0].min():.2f}..{depth.max():.2f} m")


if __name__ == "__main__":
    main()
