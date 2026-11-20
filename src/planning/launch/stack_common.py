"""Pieces shared by STVLN's own mapping (src/mapping/launch/map.launch.py) and navigation (nav.launch.py) stacks:
the QiNav guard, sensors + ground segmentation, the optional camera / battery / model, and map lookup.
"""

import os
import subprocess
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

WS_ROOT = Path(__file__).resolve().parents[3]
LIO_LAUNCH = WS_ROOT / "src" / "scene_memory" / "launch" / "lio.launch.py"
PIPELINE_LAUNCH = WS_ROOT / "src" / "scene_memory" / "launch" / "pipeline.launch.py"
DATA = Path(os.environ.get("STVLN_DATA", WS_ROOT / "data"))
# own maps first (stvln.sh map / map_save), then QiNav's 3D maps
MAPS_DIR = Path(os.environ.get("STVLN_MAPS", DATA / "maps"))
QINAV_MAPS_DIR = Path("~/.ros/nav3d/maps").expanduser()
LIDAR_TOPIC = "/livox/front_lidar/custom"
# executables of the stacks QiNav launches (mapping / localization / navigation / control)
WATCHED = {"livox_ros_driver2_node", "super_lio_node", "dlio_multi_map_node", "octomap_server",
           "global_localization", "transform_fusion", "ground_segmentation_node", "cloud_transformer",
           "ground_trans", "global_planner_node", "pcl_publisher", "qi_move_base_node", "localPlanner",
           "pathFollower", "robot_controller_node", "go2_cmd_vel_node"}


def foreign_processes() -> list[str]:
    """Watched executables running from outside this workspace (QiNav_ws, /opt/sdk_lidar, unitree_ws)."""
    try:
        out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    found = []
    for line in out.splitlines():
        pid, _, args = line.strip().partition(" ")
        for token in args.split()[:2]:
            if os.path.basename(token) in WATCHED:
                if not token.startswith(str(WS_ROOT)):
                    found.append(f"{pid} {token}")
                break
    return found


def _running(*pgrep_args: str) -> list[str]:
    try:
        return subprocess.run(["pgrep", *pgrep_args], capture_output=True, text=True, timeout=3).stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        return []


def include(pkg: str, launch: str, **args) -> IncludeLaunchDescription:
    path = os.path.join(get_package_share_directory(pkg), "launch", launch)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                    launch_arguments={k: str(v) for k, v in args.items()}.items())


def include_file(path: Path, **args) -> IncludeLaunchDescription:
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(str(path)),
                                    launch_arguments={k: str(v) for k, v in args.items()}.items())


def sensors(cfg, linefit: bool = True) -> list:
    """MID-360S driver + Super-LIO (+ camera TF), livox_frame -> base_link, body cloud, linefit ground split
    (/segmentation/ground, /segmentation/obstacle; mapping's DLIO ground map, nav's local_obstacles:=linefit)."""
    robot_type = cfg("robot_type")
    actions = [
        include_file(LIO_LAUNCH, start_driver=cfg("start_driver"), start_lio=cfg("start_lio")),
        include("cloud_preprocess", "livox_base_tf.launch.py", robot_type=robot_type),
        include("cloud_preprocess", "trans_livox.launch.py", robot_type=robot_type, input_topic=LIDAR_TOPIC,
                output_topic="/ql/nav/cloud_in_body"),
    ]
    if linefit:
        actions.append(include("linefit_ground_segmentation_ros", "segmentation.launch.py",
                               ground_height=cfg("ground_height")))
    return actions


def extras(cfg, tag: str) -> list:
    """model:= / battery:= / camera:= actions, each skipped when that piece already runs."""
    actions = []
    if cfg("model") == "true":
        if _running("-f", "__node:=go2w_state_publisher"):
            print(f"[{tag}] Go2-W model already published: not starting another")
        else:
            actions.append(include("go2w_description", "description.launch.py"))
    if cfg("battery") == "true":
        if _running("-x", "go2w_battery_no"):
            print(f"[{tag}] go2w_battery_node already running: not starting another")
        else:
            actions.append(Node(package="go2w_keyboard_control", executable="go2w_battery_node", name="go2w_battery",
                                output="screen", respawn=True, respawn_delay=5.0))
    if cfg("camera") == "true":
        if running := _running("-x", "insta360_views"):
            print(f"[{tag}] insta360_views already running ({', '.join(running)}): not starting the camera")
        else:
            config = WS_ROOT / "src" / "sensors" / "insta360_views" / "config" / "insta360_views.yaml"
            actions.append(Node(package="insta360_views", executable="insta360_views", name="insta360_views",
                                output="screen", respawn=True, respawn_delay=3.0,
                                parameters=[str(config), {"publish_equirect": True, "publish_vio_topics": False}]))
    return actions


def saved_map_prefix(name: str) -> Path:
    """Where map <name> lives: its own folder, MAPS_DIR/<name>/<name>_map.pcd, _scene/, ..."""
    return MAPS_DIR / name / name


def map_prefix(value: str) -> Path:
    """<prefix> of <prefix>_map.pcd: a path prefix, or a map name (MAPS_DIR/<name>/, then the older flat
    MAPS_DIR/<name>_*, then QINAV_MAPS_DIR)."""
    prefix = Path(value).expanduser()
    if prefix.is_absolute() or Path(f"{prefix}_map.pcd").exists():
        return prefix
    for p in (saved_map_prefix(value), MAPS_DIR / value, QINAV_MAPS_DIR / value):
        if Path(f"{p}_map.pcd").is_file():
            return p
    raise RuntimeError(f"map '{value}' not found: no {MAPS_DIR}/{value}/{value}_map.pcd "
                       f"(or {value}_map.pcd in {MAPS_DIR} / {QINAV_MAPS_DIR})")
