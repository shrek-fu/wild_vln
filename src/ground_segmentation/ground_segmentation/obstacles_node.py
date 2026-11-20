"""ROS 2 node: obstacle cells of the live ``/ground/grid`` around the robot -> the local planner's extra obstacles.

    python3 src/ground_segmentation/scripts/ground_obstacles_node.py

``/ground/grid`` (odom, 100 = lidar obstacle, terrain the robot cannot climb or a ``@blocked`` class) is cut to
``--radius`` around ``--base_frame``, moved into ``--map_frame`` with TF and published as ``--out_topic``
(x y z intensity), which STVLN's localPlanner reads as ``/added_obstacles`` (every point an obstacle). Cells
within ``--clear_radius`` of the robot are left out so its own footprint never blocks it; a grid older than
``--max_age_s`` publishes an empty cloud, so stale cells clear.
"""

from __future__ import annotations

import argparse
import time
from array import array

import numpy as np


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid_topic", default="/ground/grid")
    p.add_argument("--out_topic", default="/no_entry_obstacle_pc")
    p.add_argument("--map_frame", default="map")
    p.add_argument("--base_frame", default="base_link")
    p.add_argument("--radius", type=float, default=4.0, help="m around the robot (localPlanner adjacentRange 4.25)")
    p.add_argument("--clear_radius", type=float, default=0.45, help="m; never mark cells this close to the robot")
    p.add_argument("--publish_hz", type=float, default=2.0)
    p.add_argument("--max_age_s", type=float, default=5.0)
    return p.parse_args(argv)


def obstacle_cells(grid) -> np.ndarray:
    """(N, 3) centres of the cells valued 100, in the grid's frame, at its origin height."""
    info = grid.info
    data = np.frombuffer(bytes(grid.data), np.int8).reshape(info.height, info.width)
    rows, cols = np.nonzero(data == 100)
    res = info.resolution
    o = info.origin.position
    return np.column_stack([o.x + (cols + 0.5) * res, o.y + (rows + 0.5) * res, np.full(len(rows), o.z)])


def _matrix(tf) -> np.ndarray:
    t, q = tf.transform.translation, tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), t.x],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), t.y],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), t.z],
        [0.0, 0.0, 0.0, 1.0]])


def cloud_msg(xyz: np.ndarray, header):
    from sensor_msgs.msg import PointCloud2, PointField

    msg = PointCloud2()
    msg.header = header
    msg.height, msg.width = 1, len(xyz)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(("x", "y", "z", "intensity"))]
    msg.is_bigendian, msg.point_step, msg.is_dense = False, 16, True
    msg.row_step = 16 * len(xyz)
    data = np.column_stack([np.asarray(xyz, np.float32).reshape(-1, 3), np.full(len(xyz), 200.0, np.float32)])
    msg.data = array("B", data.tobytes())
    return msg


def main(argv=None) -> None:
    import rclpy
    from nav_msgs.msg import OccupancyGrid
    from rclpy.duration import Duration
    from rclpy.executors import ExternalShutdownException
    from rclpy.time import Time
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Header
    from tf2_ros import Buffer, TransformListener

    args = _parse_args(argv)
    rclpy.init()
    node = rclpy.create_node("ground_obstacles")
    log = node.get_logger()
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    pub = node.create_publisher(PointCloud2, args.out_topic, 1)
    state = {"grid": None, "t": 0.0, "n": 0}

    def on_grid(msg) -> None:
        state["grid"], state["t"] = msg, time.monotonic()

    def publish() -> None:
        header = Header()
        header.stamp, header.frame_id = node.get_clock().now().to_msg(), args.map_frame
        grid = state["grid"]
        xyz = np.zeros((0, 3))
        if grid is not None and time.monotonic() - state["t"] < args.max_age_s:
            try:
                T_map_grid = _matrix(tf_buffer.lookup_transform(args.map_frame, grid.header.frame_id, Time(),
                                                                Duration(seconds=0.2)))
                T_map_base = _matrix(tf_buffer.lookup_transform(args.map_frame, args.base_frame, Time(),
                                                                Duration(seconds=0.2)))
            except Exception as exc:  # noqa: BLE001 — no localization yet
                log.warn(f"no TF {args.map_frame} <- {grid.header.frame_id} / {args.base_frame}: {exc}",
                         throttle_duration_sec=10.0)
            else:
                cells = obstacle_cells(grid)
                if len(cells):
                    xyz = cells @ T_map_grid[:3, :3].T + T_map_grid[:3, 3]
                    d = np.hypot(xyz[:, 0] - T_map_base[0, 3], xyz[:, 1] - T_map_base[1, 3])
                    xyz = xyz[(d <= args.radius) & (d >= args.clear_radius)]
        pub.publish(cloud_msg(xyz, header))
        if len(xyz) != state["n"]:
            state["n"] = len(xyz)
            log.info(f"{len(xyz)} semantic-ground obstacle cells within {args.radius} m", throttle_duration_sec=5.0)

    node.create_subscription(OccupancyGrid, args.grid_topic, on_grid, 1)
    node.create_timer(1.0 / max(args.publish_hz, 0.1), publish)
    log.info(f"{args.grid_topic} obstacles within {args.radius} m of {args.base_frame} -> {args.out_topic} "
             f"({args.map_frame})")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
