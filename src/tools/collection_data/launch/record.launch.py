"""Record one debug session: Insta360 raw stream + IMU + stitched frames + VIO / Super-LIO trajectory.

    ros2 launch collection_data record.launch.py
    ros2 launch collection_data record.launch.py session_dir:=/data/run1 stream_bitrate_mbps:=20

Session layout (default data/sessions/<time>/, data dir overridable with STVLN_DATA):
    raw/seg_NNN/   pre-stitch fisheye bitstream, video_index.csv, gyro.csv, exposure.csv,
                   frames.csv (device_ts -> ROS time), camera.json (calibration)
    bag/           rosbag2 (mcap). raw_only:=true (default): /imu/data_raw, /livox/front_lidar/{custom,imu}
                   (/livox/{lidar,imu} when QiNav's driver is running), /tf_static. raw_only:=false: stitched JPEG, /imu/data_raw, VIO odometry, /tf, and with
                   start_lio /lio/odom, /lio/cloud_world, /livox/front_lidar/{custom,imu}
    config/        copies of the calibration / camera / VIO configs used
    session.json

Stop with Ctrl+C, then: ros2 run collection_data export_session <session_dir>

The camera accepts only one USB client: stop pipeline.launch.py / other
insta360_views processes first.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Lidar topics of QiNav's own driver: lio.launch.py reuses it instead of starting ours.
QINAV_LIDAR_TOPICS = ["/livox/lidar", "/livox/imu"]
TOPICS = [
    "/insta360/equirect/image/compressed",
    "/imu/data_raw",
    "/equirect_vio/odometry",
    "/equirect_vio/path",
    "/tf",
    "/tf_static",
]
LIO_TOPICS = [
    "/lio/odom",
    "/lio/cloud_world",
    "/livox/front_lidar/custom",
    "/livox/front_lidar/imu",
    *QINAV_LIDAR_TOPICS,
]
# raw_only: sensor output only; LIO / VIO / stitching are re-run offline from these and raw/.
RAW_TOPICS = [
    "/imu/data_raw",
    "/livox/front_lidar/custom",
    "/livox/front_lidar/imu",
    *QINAV_LIDAR_TOPICS,
    "/tf_static",
]
VIEW_TOPICS = [
    f"/insta360/{name}/{suffix}"
    for name in ("front", "left", "rear", "right")
    for suffix in ("image/compressed", "camera_info")
]


def _src_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        for candidate in (parent, parent / "src"):
            if (candidate / "sensors" / "insta360_views" / "config" / "insta360_views.yaml").is_file():
                return candidate
    raise FileNotFoundError("src/sensors/insta360_views not found above the launch file")


def _data_root(src: Path) -> Path:
    return Path(os.environ.get("STVLN_DATA") or src.parent / "data").expanduser()


def _qos_overrides() -> Path:
    pkg = Path(__file__).resolve().parents[1]
    return pkg / "config" / "qos_overrides.yaml"


def _git_rev(path: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=2)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _session(context, *args, **kwargs):
    src = _src_root()
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    session = Path(arg("session_dir") or _data_root(src) / "sessions" / time.strftime("%Y%m%d_%H%M%S"))
    session = session.expanduser().resolve()
    if (session / "bag").exists():
        raise RuntimeError(f"{session}/bag already exists; pick a new session_dir")
    (session / "raw").mkdir(parents=True, exist_ok=True)

    views_config = src / "sensors" / "insta360_views" / "config" / "insta360_views.yaml"
    vio_ws = src / "localization" / "equirect_vio" / "ros2_ws"
    vio_config = vio_ws / "src" / "equirect_vio" / "config" / arg("vio_config")
    models = src.parent / "insta360_driver" / "insta360_media_sdk" / "bin" / "models"

    config_dir = session / "config"
    config_dir.mkdir(exist_ok=True)
    lio_launch = src / "scene_memory" / "launch" / "lio.launch.py"
    lio_config = src / "scene_memory" / "config" / "super_lio_mid360.yaml"
    copies = [views_config, vio_config, lio_config] + sorted((_data_root(src) / "calib").glob("*.json"))
    for f in copies:
        if f.is_file():
            shutil.copy2(f, config_dir / f.name)

    raw_only = arg("raw_only").lower() == "true"
    start_vio = arg("start_vio").lower() == "true" and not raw_only
    start_lio = arg("start_lio").lower() == "true" and not raw_only
    start_lidar = arg("start_lidar").lower() == "true"
    if raw_only:
        topics = list(RAW_TOPICS)
    else:
        topics = (list(TOPICS) + (LIO_TOPICS if start_lio else [])
                  + (VIEW_TOPICS if arg("record_views").lower() == "true" else []))
    camera_params = {
        "model_dir": str(models),
        "raw_dump_dir": str(session / "raw"),
        "stream_resolution": arg("stream_resolution"),
        "stream_bitrate_mbps": float(arg("stream_bitrate_mbps")),
        "publish_equirect": True,
        "publish_vio_topics": True,  # also gates the camera IMU publisher (/imu/data_raw)
    }
    info = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "created_unix": time.time(),
        "camera": camera_params,
        "vio": {"enabled": start_vio, "config": str(vio_config)},
        "lio": {"enabled": start_lio, "config": str(lio_config)},
        "raw_only": raw_only,
        "topics": topics,
        "bag_storage": arg("bag_storage"),
        "git": {"src": _git_rev(src), "vio": _git_rev(vio_ws / "src")},
        "notes": arg("notes"),
    }
    (session / "session.json").write_text(json.dumps(info, indent=2))
    print(f"[collection_data] session -> {session}")

    actions = [
        Node(
            package="insta360_views",
            executable="insta360_views",
            name="insta360_views",
            output="screen",
            parameters=[str(views_config), camera_params],
            # the SDK can abort on a camera that does not answer; a new segment starts under raw/ on restart
            respawn=True,
            respawn_delay=3.0,
        )
    ]
    if start_vio:
        vio_cmd = (
            f"source {vio_ws / 'install' / 'setup.bash'} && exec ros2 run equirect_vio equirect_vio_node --ros-args "
            "-p image_topic:=/insta360/equirect/image -p imu_topic:=/imu/data_raw "
            "-p odom_frame:=odom -p base_frame:=imu -p publish_tf:=true "
            f"-p config:={vio_config}"
        )
        actions.append(ExecuteProcess(cmd=["bash", "-lc", vio_cmd], output="screen"))
    if start_lio or (raw_only and start_lidar):
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(lio_launch)),
            launch_arguments={"start_driver": arg("start_lidar"), "start_lio": str(start_lio).lower()}.items(),
        ))
    actions.append(
        ExecuteProcess(
            cmd=["ros2", "bag", "record", "-s", arg("bag_storage"), "-o", str(session / "bag"),
                 "--qos-profile-overrides-path", str(_qos_overrides()), "--include-hidden-topics", *topics],
            output="screen",
        )
    )
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("session_dir", default_value="",
                                  description="empty = data/sessions/<time>"),
            DeclareLaunchArgument("raw_only", default_value="true",
                                  description="bag only sensor output (camera IMU, MID-360S CustomMsg + IMU, "
                                              "/tf_static) next to raw/; no LIO / VIO, stitched frames not bagged"),
            DeclareLaunchArgument("start_vio", default_value="true"),
            DeclareLaunchArgument("vio_config", default_value="x4_air_equirect_1920.yaml"),
            DeclareLaunchArgument("start_lio", default_value="true",
                                  description="MID-360S + Super-LIO; also bags /lio/* and the raw lidar/IMU"),
            DeclareLaunchArgument("start_lidar", default_value="true",
                                  description="false = MID-360S driver (CustomMsg) already running"),
            # VIO and data/calib calibration assume the 1920_960 template stitch.
            DeclareLaunchArgument("stream_resolution", default_value="1920_960"),
            DeclareLaunchArgument("stream_bitrate_mbps", default_value="10.0",
                                  description="higher bitrate = cleaner raw fisheye for offline SfM"),
            DeclareLaunchArgument("record_views", default_value="false",
                                  description="also bag the 4 pinhole JPEG views"),
            DeclareLaunchArgument("bag_storage", default_value="mcap"),
            DeclareLaunchArgument("notes", default_value=""),
            OpaqueFunction(function=_session),
        ]
    )
