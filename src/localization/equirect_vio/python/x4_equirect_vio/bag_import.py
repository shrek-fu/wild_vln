"""Import equirectangular images + IMU from a ROS2 bag into an OpenVIO session folder."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .prepare import write_cam_timestamps, write_imu_csv


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def import_ros2_bag(
    *,
    bag_path: Path,
    session_dir: Path,
    image_topic: str = "/camera/image_raw",
    imu_topic: str = "/imu/data_raw",
    width: int = 1440,
    height: int = 720,
) -> dict:
    try:
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as e:
        raise ImportError(
            "import-bag requires a sourced ROS 2 environment (rosbag2_py, rclpy). "
            "source /opt/ros/$ROS_DISTRO/setup.bash first."
        ) from e

    bag_path = bag_path.resolve()
    session_dir = session_dir.resolve()
    panocam = session_dir / "panocam"
    panocam.mkdir(parents=True, exist_ok=True)

    storage = StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    converter = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader = SequentialReader()
    try:
        reader.open(storage, converter)
    except Exception:
        storage = StorageOptions(uri=str(bag_path), storage_id="mcap")
        reader.open(storage, converter)

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if image_topic not in topic_types:
        raise KeyError(f"Image topic {image_topic} not in bag. Have: {sorted(topic_types)}")
    if imu_topic not in topic_types:
        raise KeyError(f"IMU topic {imu_topic} not in bag. Have: {sorted(topic_types)}")

    Image = get_message(topic_types[image_topic])
    Imu = get_message(topic_types[imu_topic])

    stamps: list[float] = []
    imu_rows: list[tuple[float, float, float, float, float, float, float]] = []
    img_i = 0
    t0: float | None = None

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == imu_topic:
            msg = deserialize_message(data, Imu)
            t = _stamp_to_sec(msg.header.stamp)
            if t0 is None:
                t0 = t
            imu_rows.append(
                (
                    t - t0,
                    float(msg.linear_acceleration.x),
                    float(msg.linear_acceleration.y),
                    float(msg.linear_acceleration.z),
                    float(msg.angular_velocity.x),
                    float(msg.angular_velocity.y),
                    float(msg.angular_velocity.z),
                )
            )
        elif topic == image_topic:
            msg = deserialize_message(data, Image)
            t = _stamp_to_sec(msg.header.stamp)
            if t0 is None:
                t0 = t
            img = _image_msg_to_bgr(msg)
            if img is None:
                continue
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            out = panocam / f"{img_i:06d}.png"
            cv2.imwrite(str(out), img)
            stamps.append(t - t0)
            img_i += 1

    if not stamps:
        raise RuntimeError(f"No images on {image_topic}")
    if not imu_rows:
        raise RuntimeError(f"No IMU on {imu_topic}")

    write_cam_timestamps(session_dir / "cam_timestamps.txt", stamps)
    write_imu_csv(session_dir / "imu_data.csv", imu_rows)
    summary = {
        "session_dir": str(session_dir),
        "bag": str(bag_path),
        "image_topic": image_topic,
        "imu_topic": imu_topic,
        "num_images": len(stamps),
        "num_imu": len(imu_rows),
        "width": width,
        "height": height,
        "duration_s": stamps[-1],
    }
    (session_dir / "session_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _image_msg_to_bgr(msg) -> np.ndarray | None:
    h, w = int(msg.height), int(msg.width)
    encoding = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if encoding in ("bgr8", "rgb8"):
        img = data.reshape(h, w, 3)
        if encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    if encoding in ("mono8", "8uc1"):
        gray = data.reshape(h, w)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if encoding in ("jpeg", "jpg"):
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    # compressed image transport sometimes uses sensor_msgs/CompressedImage
    return None