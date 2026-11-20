"""Static TF from the camera body ``insta360`` to the frames its images are stamped with.

    insta360 -> insta360_<view>_optical   (front/left/rear/right pinhole views, R_body_optical)
    insta360 -> insta360_equirect         (DA360 frame of the panorama: x right, y down, z forward)

All share the panorama centre, so only rotations. Read from the same calibration the mapper uses,
so Foxglove / RViz see one tree (odom -> livox_frame -> insta360 -> ...).
"""

from __future__ import annotations

import argparse

import numpy as np


def transforms(calib_dir: str, parent: str = "insta360", prefix: str = "insta360") -> list[tuple[str, str, np.ndarray]]:
    """``(parent, child, R_parent_child)`` for every view and the equirect."""
    from scene_memory.common.calib_io import R_BODY_DA360, load_rig_calibration
    from scene_memory.perception.boxer_lift import load_pinhole_views

    calib = load_rig_calibration(calib_dir)
    out = [(parent, f"{prefix}_{v['name']}_optical", v["R_body_optical"]) for v in load_pinhole_views(calib.source_dir)]
    out.append((parent, f"{prefix}_equirect", R_BODY_DA360))
    return out


def main(argv=None) -> None:
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from rclpy.executors import ExternalShutdownException
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

    from scene_memory import paths
    from scene_memory.common.calib_io import rotation_to_quaternion_xyzw

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib_dir", default=str(paths.CALIB))
    p.add_argument("--parent", default="insta360")
    args = p.parse_args(argv)

    rclpy.init()
    node = rclpy.create_node("camera_tf")
    msgs = []
    for parent, child, R in transforms(args.calib_dir, args.parent):
        t = TransformStamped()
        t.header.stamp = node.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = parent, child
        q = rotation_to_quaternion_xyzw(np.asarray(R, dtype=np.float64))
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = map(float, q)
        msgs.append(t)
    broadcaster = StaticTransformBroadcaster(node)
    broadcaster.sendTransform(msgs)
    node.get_logger().info("static TF " + ", ".join(f"{m.header.frame_id}->{m.child_frame_id}" for m in msgs))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
