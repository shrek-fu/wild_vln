"""End-to-end timing of the live pipeline: camera → detector/tracker → lidar depth → LIO → topo map.

For every topic: rate, latency = arrival wall time − header stamp (the capture time for images), and the
longest gap between messages. Sensor topics are taken serialized and only the header stamp is read, so the
probe stays light next to a loaded pipeline.

For every detector frame it also checks what ``topo_map_node`` needs at that stamp: an LIO pose already
published (``pose_ready``) and a lidar scan still inside the cloud window (``cloud_in_window``: the newest
scan is at most ``--cloud_keep_s`` past the image, the topo map's ``CloudWindow`` retention)::

    ros2 run scene_memory latency_probe_node --duration 60 --out /tmp/latency.json
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SENSOR_TOPICS = {
    "camera_front": ("/insta360/front/image/compressed", "sensor_msgs/msg/CompressedImage"),
    "camera_equirect": ("/insta360/equirect/image/compressed", "sensor_msgs/msg/CompressedImage"),
    "lidar_raw": ("/livox/front_lidar/custom", "livox_ros_driver2/msg/CustomMsg"),
    "lidar_imu": ("/livox/front_lidar/imu", "sensor_msgs/msg/Imu"),
    "lio_odom": ("/lio/odom", "nav_msgs/msg/Odometry"),
    "lio_cloud": ("/lio/cloud_world", "sensor_msgs/msg/PointCloud2"),
    "depth_front": ("/insta360/front/depth_lidar", "sensor_msgs/msg/Image"),
}
JSON_TOPICS = {
    "detections": "/detector/detections_json",
    "topo_centers": "/topo/track_centers",
}


def header_stamp(serialized: bytes) -> float:
    """Stamp of a CDR message that starts with ``std_msgs/Header``."""
    sec, nsec = struct.unpack_from("<iI", serialized, 4)
    return sec + 1e-9 * nsec


def summarize(arrivals: list[float], latencies: list[float], duration: float) -> dict:
    if not arrivals:
        return {"n": 0}
    lat = np.asarray(latencies) * 1e3
    gaps = np.diff(arrivals) * 1e3 if len(arrivals) > 1 else np.zeros(1)
    return {"n": len(arrivals), "hz": round(len(arrivals) / duration, 2),
            "lat_med_ms": round(float(np.median(lat)), 1), "lat_p90_ms": round(float(np.percentile(lat, 90)), 1),
            "lat_max_ms": round(float(lat.max()), 1), "lat_first_ms": round(float(lat[0]), 1),
            "lat_last_ms": round(float(lat[-1]), 1), "max_gap_ms": round(float(gaps.max()), 1)}


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--cloud_keep_s", type=float, default=3.0, help="topo_map CloudWindow retention")
    p.add_argument("--out", default="", help="write the report JSON here")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rosidl_runtime_py.utilities import get_message
    from std_msgs.msg import String

    args = _parse_args(argv)
    rclpy.init()
    node = rclpy.create_node("latency_probe")
    arrivals: dict[str, list[float]] = defaultdict(list)
    latencies: dict[str, list[float]] = defaultdict(list)
    newest: dict[str, float] = {}
    sync = defaultdict(int)
    det_lag_behind_lio: list[float] = []

    def record(name: str, stamp: float) -> float:
        now = time.time()
        arrivals[name].append(now)
        latencies[name].append(now - stamp)
        newest[name] = max(newest.get(name, 0.0), stamp)
        return now

    for name, (topic, type_name) in SENSOR_TOPICS.items():
        try:
            msg_type = get_message(type_name)
        except (AttributeError, ModuleNotFoundError, ValueError):
            node.get_logger().warn(f"{type_name} not importable, skipping {topic}")
            continue
        node.create_subscription(msg_type, topic, lambda data, n=name: record(n, header_stamp(data)),
                                 qos_profile_sensor_data, raw=True)

    def on_json(name: str, msg) -> None:
        try:
            body = json.loads(msg.data)
            stamp = body["stamp"]["sec"] + 1e-9 * body["stamp"]["nanosec"]
        except (ValueError, KeyError, TypeError):
            return
        record(name, stamp)
        if name != "detections":
            return
        dets = body.get("detections", [])
        sync["frames"] += 1
        sync["detections"] += len(dets)
        sync["tracked"] += sum(d.get("track_id") is not None for d in dets)
        odom_t, cloud_t = newest.get("lio_odom"), newest.get("lio_cloud")
        if odom_t is not None:
            sync["pose_ready"] += odom_t >= stamp
            det_lag_behind_lio.append(odom_t - stamp)
        if cloud_t is not None:
            sync["cloud_in_window"] += stamp <= cloud_t <= stamp + args.cloud_keep_s

    for name, topic in JSON_TOPICS.items():
        node.create_subscription(String, topic, lambda m, n=name: on_json(n, m), 10)

    t0 = time.time()
    node.get_logger().info(f"probing for {args.duration:.0f} s")
    while rclpy.ok() and time.time() - t0 < args.duration:
        rclpy.spin_once(node, timeout_sec=0.05)
    duration = time.time() - t0
    node.destroy_node()
    rclpy.try_shutdown()

    report = {name: summarize(arrivals[name], latencies[name], duration)
              for name in list(SENSOR_TOPICS) + list(JSON_TOPICS)}
    frames = max(sync["frames"], 1)
    report["sync"] = {
        "detector_frames": sync["frames"],
        "detections_per_frame": round(sync["detections"] / frames, 2),
        "tracked_share": round(sync["tracked"] / max(sync["detections"], 1), 3),
        "pose_ready_share": round(sync["pose_ready"] / frames, 3),
        "cloud_in_window_share": round(sync["cloud_in_window"] / frames, 3),
        "lio_ahead_of_detection_med_s": round(float(np.median(det_lag_behind_lio)), 2) if det_lag_behind_lio else None,
        "cloud_keep_s": args.cloud_keep_s,
    }
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
