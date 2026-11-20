"""MID-360S driver (CustomMsg) + Super-LIO + the static camera TF.

    ros2 launch scene_memory lio.launch.py            # also included by pipeline.launch.py odom:=lio
                                                      # and collection_data record.launch.py start_lio:=true

* livox_ros_driver2 from the calibration repo (src/insta360_mid360s_calibration/install), lidar IP from its
  config/rig.yaml, host IP picked on the lidar subnet by tools/write_livox_config.py, xfer_format 1:
  /livox/front_lidar/custom (livox CustomMsg) and /livox/front_lidar/imu.
  The calibration bringup (./calibrate.sh bringup, PointCloud2) must not run at the same time.
* super_lio_node (src/localization/super_lio, built into the workspace install) with
  config/super_lio_mid360.yaml: /lio/odom (odom -> livox_frame), /lio/cloud_world, TF odom -> livox_frame.
  Respawned after lio_watchdog_node stops it on divergence (/lio/reset; keep the robot still while it
  re-initialises, ~2 s).
* QiNav starts its own driver (/livox/lidar, /livox/imu) and super_lio_node (same /lio/* topics, frames
  and extrinsic) on a mapping / localization command; whichever of the two is already running is reused
  (scene_memory.common.livox_probe), start_*:=force overrides.
* static TF livox_frame -> insta360 (camera body: X forward = panorama centre, Y left, Z up) from data/calib.
"""

import sys
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch_ros.actions import Node

_PKG_SRC = Path(__file__).resolve().parents[1]
if (_PKG_SRC / "scene_memory" / "paths.py").is_file():
    sys.path.insert(0, str(_PKG_SRC))
from scene_memory import paths  # noqa: E402
from scene_memory.common import livox_probe  # noqa: E402

LIDAR_TOPIC = "/livox/front_lidar/custom"
IMU_TOPIC = "/livox/front_lidar/imu"
QINAV_LIDAR_TOPIC = "/livox/lidar"
QINAV_IMU_TOPIC = "/livox/imu"


def _truthy(value: str) -> bool:
    return str(value).lower() in ("1", "true", "yes")


def _mode(value: str) -> str:
    """true / auto: start unless one is already running; force: always; false: never."""
    value = str(value).lower()
    if value == "force":
        return "force"
    if _truthy(value) or value == "auto":
        return "auto"
    return "off"


def _camera_tf(calib_dir: Path, lio_config: Path) -> list[str] | None:
    from scene_memory.common.calib_io import R_BODY_DA360, load_rig_calibration, rotation_to_quaternion_xyzw
    from scene_memory.mapping.scene_graph import pose_frame_to_da360

    try:
        calib = load_rig_calibration(calib_dir)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[lio] no camera TF: {exc}")
        return None
    T = pose_frame_to_da360("lio", "", lio_config, calib)
    T[:3, :3] = T[:3, :3] @ R_BODY_DA360.T
    q = rotation_to_quaternion_xyzw(T[:3, :3])
    return ["--x", f"{T[0, 3]:.6f}", "--y", f"{T[1, 3]:.6f}", "--z", f"{T[2, 3]:.6f}",
            "--qx", f"{q[0]:.8f}", "--qy", f"{q[1]:.8f}", "--qz", f"{q[2]:.8f}", "--qw", f"{q[3]:.8f}",
            "--frame-id", "livox_frame", "--child-frame-id", "insta360"]


def _actions(context, *args, **kwargs):
    cfg = context.launch_configurations.get
    calib_repo = Path(cfg("calib_repo", str(paths.RIG_CALIB_REPO))).expanduser()
    lio_config = Path(cfg("lio_config", str(paths.LIO_CONFIG))).expanduser()
    driver_setup = calib_repo / "install" / "setup.bash"
    ws_setup = paths.WS_ROOT / "install" / "setup.bash"
    actions = []
    driver_mode, lio_mode = _mode(cfg("start_driver", "true")), _mode(cfg("start_lio", "true"))
    lidar_topic, imu_topic = LIDAR_TOPIC, IMU_TOPIC
    if driver_mode == "auto" and (running := livox_probe.processes(livox_probe.DRIVER_PATTERN)):
        pid, cmd = running[0]
        print(f"[lio] livox driver already running (pid {pid}, {livox_probe.owner(cmd)}); not starting ours")
        driver_mode = "external"
    if lio_mode == "auto" and (running := livox_probe.processes(livox_probe.LIO_PATTERN)):
        pid, cmd = running[0]
        print(f"[lio] super_lio_node already running (pid {pid}, {livox_probe.owner(cmd)}); "
              "reusing its /lio/odom, /lio/cloud_world and TF odom -> livox_frame")
        lio_mode = "off"
    if driver_mode == "external" and lio_mode != "off":
        found = livox_probe.livox_topics()
        lidar_topic, imu_topic = found[0] or QINAV_LIDAR_TOPIC, found[1] or QINAV_IMU_TOPIC
        print(f"[lio] Super-LIO on the running driver's topics: {lidar_topic}, {imu_topic}"
              + ("" if all(found) else " (not discovered yet, QiNav defaults)"))
    if driver_mode in ("auto", "force"):
        rig = yaml.safe_load((calib_repo / "config" / "rig.yaml").read_text())
        ip = cfg("lidar_ip", "") or str(rig["lidars"][0]["ip"])
        tag = ip.replace(".", "_")
        freq = rig.get("livox", {}).get("publish_freq", 10.0)
        driver = (
            f"set -e; source '{driver_setup}'; "
            f"python3 '{calib_repo / 'tools' / 'write_livox_config.py'}'; "
            # exec the binary, not `ros2 run`, which exits on SIGINT without stopping its child.
            'exec "$(ros2 pkg prefix livox_ros_driver2)/lib/livox_ros_driver2/livox_ros_driver2_node" --ros-args '
            f"-p xfer_format:=1 -p multi_topic:=1 -p data_src:=0 -p publish_freq:={float(freq)} "
            f"-p frame_id:=livox_frame -p user_config_path:='{calib_repo / 'config' / 'MID360s_dual.json'}' "
            f"-p cmdline_input_bd_code:=livox0000000001 "
            f"-r /livox/lidar_{tag}:={LIDAR_TOPIC} -r /livox/imu_{tag}:={IMU_TOPIC}"
        )
        actions.append(ExecuteProcess(cmd=["bash", "-lc", driver], output="screen", sigterm_timeout="3",
                                      sigkill_timeout="3"))
    if lio_mode in ("auto", "force"):
        lio = (
            f"set -e; source '{driver_setup}'; source '{ws_setup}'; "
            'exec "$(ros2 pkg prefix super_lio)/lib/super_lio/super_lio_node" '
            f"--ros-args --params-file '{lio_config}' "
            f"-p lio.ros.lidar_topic:={lidar_topic} -p lio.ros.imu_topic:={imu_topic} --log-level info"
        )
        actions.append(ExecuteProcess(cmd=["bash", "-lc", lio], output="screen", respawn=True, respawn_delay=2.0))
        if _truthy(cfg("start_watchdog", "true")):
            actions.append(ExecuteProcess(
                cmd=[paths.PYTHON, str(_PKG_SRC / "scripts" / "lio_watchdog_node.py"),
                     "--max_speed", cfg("lio_max_speed", "5.0")],
                output="screen",
            ))
    tf = _camera_tf(Path(cfg("calib_dir", str(paths.CALIB))), lio_config)
    if tf is not None:
        actions.append(Node(package="tf2_ros", executable="static_transform_publisher", name="livox_to_insta360",
                            arguments=tf, output="log"))
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("start_driver", default_value="true",
                              description="true/auto: unless a livox_ros_driver2_node already runs (QiNav), "
                                          "then Super-LIO uses its topics; force; false: driver on our topics"),
        DeclareLaunchArgument("start_lio", default_value="true",
                              description="true/auto: unless a super_lio_node already runs (reuse its /lio/*); "
                                          "force; false"),
        DeclareLaunchArgument("start_watchdog", default_value="true",
                              description="restart Super-LIO when it diverges (lio_watchdog_node, /lio/reset)"),
        DeclareLaunchArgument("lio_max_speed", default_value="5.0", description="m/s that counts as divergence"),
        DeclareLaunchArgument("lidar_ip", default_value="", description="empty = lidars[0].ip of rig.yaml"),
        DeclareLaunchArgument("calib_repo", default_value=str(paths.RIG_CALIB_REPO)),
        DeclareLaunchArgument("calib_dir", default_value=str(paths.CALIB)),
        DeclareLaunchArgument("lio_config", default_value=str(paths.LIO_CONFIG)),
        OpaqueFunction(function=_actions),
    ])
