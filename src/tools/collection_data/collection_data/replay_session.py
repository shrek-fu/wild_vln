"""Replay a collection_data session as if the camera and VIO were live.

    ros2 run collection_data replay_session <session_dir> [--rate 0.5] [--start 10] [--duration 60]

Publishes, in recorded order and with the recorded header stamps:
    /insta360/equirect/image/compressed            (from the bag)
    /insta360/<view>/image/compressed, camera_info (rendered from the equirect like insta360_views)
    /equirect_vio/odometry, /equirect_vio/path, /tf, /tf_static, /imu/data_raw (from the bag)

--vio rerun drops the recorded odometry / TF and publishes /insta360/equirect/image
(rgb8) for a fresh equirect_vio instead. The bag holds the stitched JPEG at ~15 Hz,
so a rerun sees half the live frame rate.

--lidar_bag <raw session>/bag merges the MID-360S CustomMsg + IMU of the raw recording (a derived
``_proc`` session has none) by receive time, so a Super-LIO started next to the replay runs on them;
--no_odom then leaves out the recorded /lio/*, /equirect_vio/* and /tf (``stvln.sh map_offline``).
"""

import argparse
import heapq
import math
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

EQUIRECT_TOPIC = "/insta360/equirect/image/compressed"
VIO_TOPICS = ("/equirect_vio/odometry", "/equirect_vio/path", "/tf", "/lio/odom", "/lio/cloud_world")
STATIC_TOPICS = ("/tf_static",)
LIDAR_TOPICS = ("/livox/front_lidar/custom", "/livox/front_lidar/imu")
VIEW_YAW = {"front": 0.0, "left": 90.0, "rear": 180.0, "right": -90.0}


class PinholeView:
    """Same sampling as insta360_views View::build (X forward, Y left, Z up, yaw positive left)."""

    def __init__(self, name: str, yaw_deg: float, size: int, hfov_deg: float, eq_w: int, eq_h: int,
                 prefix: str = "insta360"):
        self.name = name
        self.size = size
        self.frame_id = f"{prefix}_{name}_optical"
        self.fx = 0.5 * size / math.tan(0.5 * math.radians(hfov_deg))
        c = 0.5 * size
        yaw = math.radians(yaw_deg)
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
        grid = (np.arange(size) + 0.5 - c) / self.fx
        a, b = np.meshgrid(grid, grid)
        dx = fwd[0] + a * right[0]
        dy = fwd[1] + a * right[1]
        dz = -b
        lon = np.arctan2(dy, dx)
        lat = np.arctan2(dz, np.hypot(dx, dy))
        self.map_u = (eq_w * (0.5 - lon / (2 * math.pi)) - 0.5).astype(np.float32)
        self.map_v = np.clip(eq_h * (0.5 - lat / math.pi) - 0.5, 0, eq_h - 1).astype(np.float32)
        self.eq_shape = (eq_h, eq_w)

    def render(self, equirect_bgr: np.ndarray) -> np.ndarray:
        return cv2.remap(equirect_bgr, self.map_u, self.map_v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def camera_info(self, header) -> CameraInfo:
        info = CameraInfo()
        info.header.stamp = header.stamp
        info.header.frame_id = self.frame_id
        info.width = info.height = self.size
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        c = 0.5 * self.size - 0.5
        info.k = [self.fx, 0.0, c, 0.0, self.fx, c, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [self.fx, 0.0, c, 0.0, 0.0, self.fx, c, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info


def bag_messages(bag_dir: Path, topics, min_period_ns: dict[str, int] | None = None):
    """(bag receive time ns, topic, msg) in recorded order.

    ``min_period_ns`` throttles topics before deserialising them (``/equirect_vio/path`` carries the whole
    trajectory, so at full rate its cost grows until the replay falls behind real time)."""
    import rosbag2_py
    import yaml
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    meta = yaml.safe_load((bag_dir / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=meta["storage_identifier"]),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = [t for t in topics if t in types]
    reader.set_filter(rosbag2_py.StorageFilter(topics=wanted))
    classes = {t: get_message(types[t]) for t in wanted}
    period = min_period_ns or {}
    last: dict[str, int] = {}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic in period:
            if t_ns - last.get(topic, -period[topic]) < period[topic]:
                continue
            last[topic] = t_ns
        yield t_ns, topic, deserialize_message(data, classes[topic]), types[topic]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--rate", type=float, default=1.0, help="playback speed (<1 gives the GPU nodes more frames)")
    ap.add_argument("--start", type=float, default=0.0, help="skip the first N seconds")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = to the end)")
    ap.add_argument("--views", default="front,left,rear,right")
    ap.add_argument("--view_size", type=int, default=640)
    ap.add_argument("--view_hfov_deg", type=float, default=90.0)
    ap.add_argument("--front_yaw_deg", type=float, default=0.0)
    ap.add_argument("--jpeg_quality", type=int, default=80)
    ap.add_argument("--vio", choices=["recorded", "rerun"], default="recorded")
    ap.add_argument("--hold_s", type=float, default=3.0, help="keep the node alive after the last message")
    ap.add_argument("--path_hz", type=float, default=1.0, help="max rate of /equirect_vio/path (0 = drop it)")
    ap.add_argument("--lidar_bag", type=Path, default=None, help="also replay the MID-360S topics of this bag")
    ap.add_argument("--no_odom", action="store_true", help="leave out the recorded odometry and /tf")
    args = ap.parse_args()

    session = args.session.expanduser().resolve()
    bag = session / "bag"
    if not (bag / "metadata.yaml").is_file():
        raise SystemExit(f"no rosbag2 under {bag}")

    rclpy.init()
    node = rclpy.create_node("session_replay")
    log = node.get_logger()
    sensor = qos_profile_sensor_data
    latched = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    pub_eq = node.create_publisher(CompressedImage, EQUIRECT_TOPIC, sensor)
    pub_eq_raw = node.create_publisher(Image, "/insta360/equirect/image", sensor) if args.vio == "rerun" else None
    names = [v.strip() for v in args.views.split(",") if v.strip()]
    view_pubs = {
        n: (node.create_publisher(CompressedImage, f"/insta360/{n}/image/compressed", sensor),
            node.create_publisher(CameraInfo, f"/insta360/{n}/camera_info", sensor))
        for n in names
    }
    views: dict[str, PinholeView] = {}
    other_pubs = {}

    topics = [EQUIRECT_TOPIC, "/imu/data_raw", *STATIC_TOPICS]
    if args.vio == "recorded" and not args.no_odom:
        topics += [t for t in VIO_TOPICS if t != "/equirect_vio/path" or args.path_hz > 0]
    throttle = {"/equirect_vio/path": int(1e9 / args.path_hz)} if args.path_hz > 0 else {}
    messages = bag_messages(bag, topics, throttle)
    if args.lidar_bag is not None:
        lidar_bag = args.lidar_bag.expanduser().resolve()
        if not (lidar_bag / "metadata.yaml").is_file():
            raise SystemExit(f"no rosbag2 under {lidar_bag}")
        messages = heapq.merge(messages, bag_messages(lidar_bag, LIDAR_TOPICS), key=lambda m: m[0])

    log.info(f"replaying {session.name} at {args.rate}x, views={names}, vio={args.vio}"
             + (f", lidar from {args.lidar_bag}" if args.lidar_bag else "") + (", no odometry" if args.no_odom else ""))
    t_first = None
    wall0 = None
    counts: dict[str, int] = {}
    last_report = time.monotonic()
    for t_ns, topic, msg, type_name in messages:
        if not rclpy.ok():
            break
        if t_first is None:
            t_first = t_ns
        rel = (t_ns - t_first) * 1e-9 - args.start
        if rel < 0 and topic not in STATIC_TOPICS:
            continue
        if args.duration > 0 and rel > args.duration:
            break
        rel = max(rel, 0.0)
        if wall0 is None:
            wall0 = time.monotonic()
        delay = wall0 + rel / args.rate - time.monotonic()
        if delay > 0:
            time.sleep(delay)

        if topic == EQUIRECT_TOPIC:
            pub_eq.publish(msg)
            bgr = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            if pub_eq_raw is not None:
                raw = Image()
                raw.header = msg.header
                raw.height, raw.width, raw.encoding, raw.step = h, w, "rgb8", 3 * w
                raw.data = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()
                pub_eq_raw.publish(raw)
            for n, (pub_img, pub_info) in view_pubs.items():
                view = views.get(n)
                if view is None or view.eq_shape != (h, w):
                    view = views[n] = PinholeView(n, args.front_yaw_deg + VIEW_YAW[n], args.view_size,
                                                  args.view_hfov_deg, w, h)
                ok, jpg = cv2.imencode(".jpg", view.render(bgr), [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                if not ok:
                    continue
                out = CompressedImage()
                out.header.stamp = msg.header.stamp
                out.header.frame_id = view.frame_id
                out.format = "jpeg"
                out.data = jpg.tobytes()
                pub_img.publish(out)
                pub_info.publish(view.camera_info(msg.header))
        else:
            pub = other_pubs.get(topic)
            if pub is None:
                qos = latched if topic in STATIC_TOPICS else (
                    2000 if topic in ("/imu/data_raw", "/livox/front_lidar/imu") else 50)
                pub = other_pubs[topic] = node.create_publisher(type(msg), topic, qos)
            pub.publish(msg)
        counts[topic] = counts.get(topic, 0) + 1
        if time.monotonic() - last_report > 10.0:
            last_report = time.monotonic()
            lag = time.monotonic() - (wall0 + rel / args.rate)
            log.info(f"t={rel:7.1f}s  frames={counts.get(EQUIRECT_TOPIC, 0)}  lag={lag:+.2f}s")

    log.info(f"replay done: {counts}")
    end = time.monotonic() + args.hold_s
    while rclpy.ok() and time.monotonic() < end:
        time.sleep(0.1)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
