"""Lidar depth for every Insta360 pinhole view.

For each view image, the LIO scans of the last ``--window_s`` (already deskewed and in ``odom``) are
projected with the odometry pose interpolated at the image stamp and the rig calibration:

* ``/insta360/<view>/depth_lidar``: ``32FC1`` metres along the optical axis, 0 = no return
  (nearest return per pixel; ``--splat`` grows returns to small squares);
* ``/insta360/<view>/depth_lidar/overlay/compressed`` (``--overlay``): the view with the returns coloured by
  depth, to check extrinsics and time sync live.
"""

from __future__ import annotations

import argparse
import time
from array import array

import cv2
import numpy as np

from scene_memory import paths
from scene_memory.common.calib_io import load_rig_calibration
from scene_memory.common.pose_buffer import PoseBuffer
from scene_memory.mapping.scene_graph import T_odom_optical, pinhole_intrinsics, pose_frame_to_da360, pose_to_matrix
from scene_memory.perception.boxer_lift import load_pinhole_views
from scene_memory.perception.lidar_depth import CloudWindow, cloud_xyz_fast, view_depth


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib_dir", default=str(paths.CALIB))
    p.add_argument("--pose_source", choices=["vio", "lio"], default="lio")
    p.add_argument("--vio_config", default=str(paths.VIO_CONFIG))
    p.add_argument("--lio_config", default=str(paths.LIO_CONFIG))
    p.add_argument("--odom_topic", default="/lio/odom")
    p.add_argument("--cloud_topic", default="/lio/cloud_world")
    p.add_argument("--views", default="front,left,rear,right", help="comma list; empty = all calibrated views")
    p.add_argument("--max_hz", type=float, default=5.0, help="per view")
    p.add_argument("--window_s", type=float, default=1.0)
    p.add_argument("--max_wait_s", type=float, default=1.0, help="hold an image this long for its odometry")
    p.add_argument("--keep_s", type=float, default=10.0, help="s of scans kept; must exceed the camera latency")
    p.add_argument("--voxel", type=float, default=0.03, help="m; per-scan voxel filter (0 = off)")
    p.add_argument("--max_range", type=float, default=30.0)
    p.add_argument("--splat", type=int, default=0, help="px radius each return covers (0 = single pixel)")
    p.add_argument("--overlay", action="store_true", help="also publish a coloured overlay JPEG per view")
    return p.parse_args(argv)


def colorize(bgr: np.ndarray, depth: np.ndarray, max_range: float) -> np.ndarray:
    out = bgr.copy()
    v, u = np.nonzero(depth > 0)
    if len(u):
        level = np.clip(depth[v, u] / max_range * 255.0, 0, 255).astype(np.uint8)
        colors = cv2.applyColorMap(level.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
        out[v, u] = colors
    return out


def main(argv=None) -> None:
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, Image, PointCloud2

    args = _parse_args(argv)
    calib = load_rig_calibration(args.calib_dir)
    T_child_da360 = pose_frame_to_da360(args.pose_source, args.vio_config, args.lio_config, calib)
    wanted = {v.strip() for v in args.views.split(",") if v.strip()}
    views = [v for v in load_pinhole_views(calib.source_dir) if not wanted or v["name"] in wanted]
    poses = PoseBuffer()
    cloud = CloudWindow(args.window_s, keep_s=args.keep_s, voxel_m=args.voxel)

    rclpy.init()
    node = rclpy.create_node("view_depth")
    log = node.get_logger()
    pubs = {v["name"]: node.create_publisher(Image, f"/insta360/{v['name']}/depth_lidar", 2) for v in views}
    overlay_pubs = {
        v["name"]: node.create_publisher(CompressedImage, f"/insta360/{v['name']}/depth_lidar/overlay/compressed", 2)
        for v in views
    } if args.overlay else {}
    last = {v["name"]: 0.0 for v in views}
    # LIO publishes a scan's pose after processing it, so images usually arrive first: keep the
    # newest image per view until the odometry reaches its stamp.
    pending: dict[str, tuple[dict, object, float]] = {}
    stats = {"published": 0, "no_pose": 0, "no_cloud": 0, "dropped_pending": 0}

    def on_odom(msg) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        poses.add(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec,
                  pose_to_matrix((p.x, p.y, p.z), (q.x, q.y, q.z, q.w)))
        for name, (view, img, t) in list(pending.items()):
            if poses.last_t() >= t:
                del pending[name]
                render(view, img, t)

    def on_cloud(msg) -> None:
        try:
            cloud.add(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec, cloud_xyz_fast(msg))
        except ValueError as exc:
            log.warn(f"cloud decode failed: {exc}", throttle_duration_sec=10.0)

    def on_view(view: dict, msg) -> None:
        name = view["name"]
        if args.max_hz > 0 and time.monotonic() - last[name] < 1.0 / args.max_hz:
            return
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if poses.last_t() < t < poses.last_t() + args.max_wait_s:
            if name in pending:
                stats["dropped_pending"] += 1
            pending[name] = (view, msg, t)
            return
        render(view, msg, t)

    def render(view: dict, msg, t: float) -> None:
        name = view["name"]
        T_odom_child = poses.at(t)
        if T_odom_child is None:
            stats["no_pose"] += 1
            return
        xyz = cloud.points(t)
        if len(xyz) == 0:
            stats["no_cloud"] += 1
            return
        last[name] = time.monotonic()
        w, h = view["width"], view["height"]
        bgr = None
        if args.overlay:
            bgr = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                h, w = bgr.shape[:2]
        K = pinhole_intrinsics(w, h, view["hfov_deg"])
        T = T_odom_optical(T_odom_child, T_child_da360, view["R_body_optical"])
        depth = view_depth(xyz, T, K, w, h, z_max=args.max_range, splat=args.splat)
        out = Image()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = f"insta360_{name}_optical"
        out.height, out.width = depth.shape
        out.encoding = "32FC1"
        out.is_bigendian = 0
        out.step = 4 * depth.shape[1]
        # rclpy checks every element of a bytes value (~1 s per 640x640 float image); array('B') is taken as is.
        out.data = array("B", depth.tobytes())
        pubs[name].publish(out)
        if bgr is not None:
            ok, jpg = cv2.imencode(".jpg", colorize(bgr, depth if args.splat else
                                                    view_depth(xyz, T, K, w, h, z_max=args.max_range, splat=1),
                                                    args.max_range), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                ov = CompressedImage()
                ov.header = out.header
                ov.format = "jpeg"
                ov.data = array("B", jpg.tobytes())
                overlay_pubs[name].publish(ov)
        stats["published"] += 1

    node.create_subscription(Odometry, args.odom_topic, on_odom, 200)
    node.create_subscription(PointCloud2, args.cloud_topic, on_cloud, qos_profile_sensor_data)
    for view in views:
        node.create_subscription(CompressedImage, view["image_topic"], lambda m, v=view: on_view(v, m),
                                 qos_profile_sensor_data)
    node.create_timer(10.0, lambda: log.info(f"depth images {stats} last_odom={poses.last_t():.2f} "
                                             f"last_cloud={cloud.last_t() or 0.0:.2f}"))
    log.info(f"lidar depth for {[v['name'] for v in views]}: {args.odom_topic} ({args.pose_source}) + "
             f"{args.cloud_topic}, window {args.window_s} s, calib {calib.summary()}")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
