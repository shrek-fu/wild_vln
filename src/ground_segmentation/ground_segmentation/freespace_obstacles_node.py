"""ROS 2 node: local obstacles from the semantic freespace, for STVLN's localPlanner instead of linefit.

    python3 src/ground_segmentation/scripts/freespace_obstacles_node.py

Inputs (odom): ``--freespace_topic`` (``ground_node``'s ``/ground/traversable_cloud``, x y z label, the
GroundingDINO + SAM2 ground lifted with the lidar, kept in a ``--window_m`` square around the robot) and every
LIO scan ``--cloud_topic`` (the last ``--scan_window_s`` of them). Each lidar point is measured against the
ground of the nearest freespace cell (``ground_segmentation.freespace``), else the ground under the robot
(``--base_frame`` sits on the ground, base_link_mount.yaml); points ``--h_min``..``--h_max`` above it are obstacles.
No camera frame is waited for: the scans run at lidar rate, the freespace only sets the ground.

Outputs (``--map_frame``):

* ``--out_topic`` (x y z intensity), the localPlanner's ``/terrain_map``: obstacle points with ``z = base_z + h``;
  the planner's height (z minus its base_link ``/localization`` z) is then the height above the freespace, also
  on slopes and ramps;
* ``--blocked_topic``, the localPlanner's ``/added_obstacles``: freespace cells of the ``@blocked`` /
  ``--blocked_classes`` classes within ``--blocked_radius``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from scene_memory import paths
from scene_memory.perception.lidar_depth import CloudWindow, cloud_xyz_fast

from ground_segmentation.freespace import LocalFreespace, obstacle_mask
from ground_segmentation.obstacles_node import _matrix, cloud_msg
from ground_segmentation.segmenters import blocked_classes, class_names


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freespace_topic", default="/ground/traversable_cloud")
    p.add_argument("--cloud_topic", default="/lio/cloud_world")
    p.add_argument("--out_topic", default="/stvln/local_obstacles")
    p.add_argument("--blocked_topic", default="/no_entry_obstacle_pc")
    p.add_argument("--map_frame", default="map")
    p.add_argument("--odom_frame", default="odom")
    p.add_argument("--base_frame", default="base_link")
    p.add_argument("--classes", default=str(paths.CONFIG / "ground_classes.txt"), help="same as ground_node")
    p.add_argument("--blocked_classes", default="", help="comma list, on top of the file's @blocked")
    p.add_argument("--window_m", type=float, default=30.0, help="m; square around the robot")
    p.add_argument("--scan_window_s", type=float, default=0.3, help="s of LIO scans per output (MID-360 density)")
    p.add_argument("--voxel", type=float, default=0.05)
    p.add_argument("--resolution", type=float, default=0.1, help="m; ground_node's --resolution")
    p.add_argument("--search_m", type=float, default=1.0,
                   help="m; a point's ground is the nearest freespace cell within this, else the robot's ground")
    p.add_argument("--h_min", type=float, default=0.15, help="m above the freespace ground")
    p.add_argument("--h_max", type=float, default=1.2,
                   help="m above the freespace ground (overhead clearance; localPlanner maxRelZ 1.2)")
    p.add_argument("--clear_radius", type=float, default=0.45, help="m; never mark points this close to the robot")
    p.add_argument("--blocked_radius", type=float, default=4.0, help="m (localPlanner adjacentRange 4.25)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import rclpy
    from rclpy.duration import Duration
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Header
    from tf2_ros import Buffer, TransformListener

    args = _parse_args(argv)
    classes = class_names(args.classes)
    blocked = [classes.index(c) + 1 for c in blocked_classes(args.classes, args.blocked_classes) if c in classes]
    free = LocalFreespace(args.search_m, blocked, resolution=args.resolution)
    scans = CloudWindow(args.scan_window_s, keep_s=max(1.0, 2 * args.scan_window_s), voxel_m=args.voxel)
    half = 0.5 * args.window_m

    rclpy.init()
    node = rclpy.create_node("freespace_obstacles")
    log = node.get_logger()
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    pub = node.create_publisher(PointCloud2, args.out_topic, 5)
    blocked_pub = node.create_publisher(PointCloud2, args.blocked_topic, 1)
    state = {"free_t": 0.0, "n": 0, "known": 0.0, "ms": 0.0}

    def lookup(target: str, source: str):
        return _matrix(tf_buffer.lookup_transform(target, source, Time(), Duration(seconds=0.1)))

    def on_freespace(msg) -> None:
        names = [f.name for f in msg.fields]
        n = int(msg.width) * int(msg.height)
        cols = np.frombuffer(bytes(msg.data), np.float32, count=n * len(names)).reshape(n, len(names)) if n else \
            np.zeros((0, len(names)), np.float32)
        free.set_cells(cols[:, :3], cols[:, names.index("label")] if "label" in names else None)
        state["free_t"] = time.monotonic()

    def on_cloud(msg) -> None:
        t0 = time.perf_counter()
        try:
            scans.add(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec, cloud_xyz_fast(msg))
        except ValueError as exc:
            log.warn(f"cloud decode failed: {exc}", throttle_duration_sec=10.0)
            return
        try:
            T_odom_base = lookup(args.odom_frame, args.base_frame)
            T_map_odom = lookup(args.map_frame, args.odom_frame)
        except Exception as exc:  # noqa: BLE001 — no localization yet
            log.warn(f"no TF {args.map_frame} <- {args.odom_frame} <- {args.base_frame}: {exc}",
                     throttle_duration_sec=10.0)
            return
        base = T_odom_base[:3, 3]
        robot_ground = free.ground_under(base, base[2])
        pts = scans.points()
        h, known = free.heights(pts, robot_ground)
        keep = obstacle_mask(pts, h, base, h_min=args.h_min, h_max=args.h_max, clear_radius=args.clear_radius,
                             half_m=half)
        base_z_map = (T_map_odom @ np.append(base, 1.0))[2]
        xyz = pts[keep] @ T_map_odom[:3, :3].T + T_map_odom[:3, 3]
        xyz[:, 2] = base_z_map + h[keep]
        header = Header()
        header.stamp, header.frame_id = msg.header.stamp, args.map_frame
        pub.publish(cloud_msg(xyz, header))
        state["n"], state["known"] = int(keep.sum()), float(known.mean()) if len(known) else 0.0
        state["ms"] = 1e3 * (time.perf_counter() - t0)

        cells = free.blocked_xyz
        if len(cells):
            near = np.hypot(cells[:, 0] - base[0], cells[:, 1] - base[1])
            cells = cells[(near <= args.blocked_radius) & (near >= args.clear_radius)]
        cells = cells @ T_map_odom[:3, :3].T + T_map_odom[:3, 3] if len(cells) else np.zeros((0, 3))
        blocked_pub.publish(cloud_msg(cells, header))

    def report() -> None:
        age = time.monotonic() - state["free_t"] if state["free_t"] else float("inf")
        log.info(f"{state['n']} obstacle points; freespace {len(free)} cells (age {age:.1f} s), "
                 f"{100 * state['known']:.0f}% of the points on a freespace ground, {state['ms']:.0f} ms/scan")

    node.create_subscription(PointCloud2, args.freespace_topic, on_freespace, 1)
    node.create_subscription(PointCloud2, args.cloud_topic, on_cloud, qos_profile_sensor_data)
    node.create_timer(10.0, report)
    log.info(f"{args.cloud_topic} against {args.freespace_topic} ({args.window_m:g} m window, obstacles "
             f"{args.h_min}..{args.h_max} m above the freespace) -> {args.out_topic}; blocked "
             f"{[classes[i - 1] for i in blocked]} -> {args.blocked_topic} ({args.map_frame})")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
