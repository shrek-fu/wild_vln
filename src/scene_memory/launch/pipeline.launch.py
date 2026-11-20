"""Camera, odometry (Super-LIO on the MID-360S or equirect VIO), 2D detector + BoT-SORT, layered semantic topo map.

odom:=lio (default): livox_ros_driver2 (CustomMsg) → Super-LIO (/lio/odom, /lio/cloud_world in a
  gravity-aligned odom); the camera pose is livox_frame → front lidar → camera from data/calib.
  view_depth_node publishes /insta360/<view>/depth_lidar and topo_map_node places objects at the
  lidar returns inside their masks. odom:=vio: equirect VIO as before (no lidar needed).

mapper:=topo (default):
  camera + odometry + detector (sam3 | gdino_sam2 | yoloe) / BoT-SORT (front)
  → topo_map_node: object centres from lidar returns (odom:=lio), else by ray
  triangulation over each track's poses, places along the trajectory, regions;
  SuperMap-style on/near edges and Bayesian label fusion. Writes objects_4d.json +
  topo_graph.json to ``scene_dir``.
mapper:=hydra (needs depth):
  + DA360 equirect depth → Hydra ``pipeline.step`` (mesh/places/rooms) + BoxerNet OBBs.
Either mapper publishes object centres that the tracker projects with the VIO
pose as its 3D-to-2D motion prior (SuperMap).
  ``start_dam:=true`` adds a DAM-3B worker that captions saved object views.
  Metric Depth-Anything front mono and the old DAAAM node stay off by default.

Stop any other ``insta360_views`` process first. The camera accepts only one USB client.

Offline: ``session:=data/sessions/<time>`` replays a recorded session
(stitched frames, rendered views, recorded VIO odometry/TF) instead of the camera and
live VIO. Results go to ``<session>/scene4d/<time>`` unless scene_dir is set, and the
launch shuts down when the replay ends (shutdown_on_end).
"""

import os
import sys
import time
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_PKG_SRC = Path(__file__).resolve().parents[1]
if (_PKG_SRC / "scene_memory" / "paths.py").is_file():
    sys.path.insert(0, str(_PKG_SRC))
from scene_memory import paths  # noqa: E402

GROUND_CLASSES_FILE = str(paths.CONFIG / "ground_classes.txt")

RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")


# numpy nodes multiply small matrices (N x 3 clouds); BLAS worker threads only spin
# (Orin NX: 3 x 60 % CPU per node for nothing).
SINGLE_THREAD_BLAS = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
DETECTIONS_TOPIC = "/detector/detections_json"
CENTERS_TOPIC = {"topo": "/topo/track_centers", "hydra": "/hydra/track_centers"}
# qinav: the QiNav 3D stack (its own driver + Super-LIO + ICP on the saved PCD map) is running; its
# /ql/nav/localization is the same livox_frame (MID-360 IMU) pose as /lio/odom, but in its persistent map frame.
ODOM = {
    "vio": {"odom_topic": "/equirect_vio/odometry", "frame_id": "imu", "cloud_topic": "", "pose_source": "vio",
            "world_frame": "odom"},
    "lio": {"odom_topic": "/lio/odom", "frame_id": "livox_frame", "cloud_topic": "/lio/cloud_world",
            "pose_source": "lio", "world_frame": "odom"},
    "qinav": {"odom_topic": "/ql/nav/localization", "frame_id": "livox_frame", "cloud_topic": "/ql/nav/cloud_in_map",
              "pose_source": "lio", "world_frame": "map"},
    # QiNav mapping (api 1031): its Super-LIO remaps /lio/cloud_world -> /cloud_registered, from which octomap
    # (2D <map>.pgm) / dlio_multi_map_node (3D <map>_map.pcd) build the map, so this odom frame is the saved map's.
    "qinav_mapping": {"odom_topic": "/lio/odom", "frame_id": "livox_frame", "cloud_topic": "/cloud_registered",
                      "pose_source": "lio", "world_frame": "odom"},
}
QINAV_ODOM = ("qinav", "qinav_mapping")
LIDAR_TOPIC = "/livox/front_lidar/custom"
LIDAR_IMU_TOPIC = "/livox/front_lidar/imu"


def _staggered(function, slot: int):
    """Start ``function``'s actions ``slot * stagger_s`` seconds after launch. On the Orin NX carrier the 5 V
    input browns out (hard reset) when the GPU models all load at once."""

    def action(context, *args, **kwargs):
        actions = function(context, *args, **kwargs)
        delay = slot * float(context.launch_configurations.get("stagger_s", "0") or 0)
        return [TimerAction(period=delay, actions=actions)] if actions and delay > 0 else actions

    return action


def _truthy(value: str) -> bool:
    return str(value).lower() in ("1", "true", "yes")


def _odom(context) -> str:
    """lio | vio | qinav. A recorded session without /lio/odom replays its VIO odometry."""
    value = context.launch_configurations.get("odom", "lio")
    if value not in ODOM:
        raise RuntimeError(f"odom:={value}; use {', '.join(ODOM)}")
    session = _session(context)
    if value == "lio" and session is not None:
        meta = yaml.safe_load((session / "bag" / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
        topics = {t["topic_metadata"]["name"] for t in meta["topics_with_message_count"]}
        if ODOM["lio"]["odom_topic"] not in topics:
            print(f"[pipeline] {session.name} has no {ODOM['lio']['odom_topic']}; using the recorded VIO")
            return "vio"
    return value


def _pose_args(context, with_cloud: bool) -> list[str]:
    mode = _odom(context)
    args = ["--pose_source", ODOM[mode]["pose_source"], "--odom_topic", ODOM[mode]["odom_topic"]]
    if with_cloud and ODOM[mode]["cloud_topic"]:
        args += ["--cloud_topic", ODOM[mode]["cloud_topic"]]
    return args


def _session(context) -> Path | None:
    raw = context.launch_configurations.get("session", "").strip()
    if not raw:
        return None
    session = Path(raw).expanduser().resolve()
    if not (session / "bag" / "metadata.yaml").is_file():
        raise RuntimeError(f"session {session} has no bag/metadata.yaml")
    return session


def _scene_dir(context) -> str:
    explicit = context.launch_configurations.get("scene_dir", "").strip()
    if explicit:
        return explicit
    session = _session(context)
    if session is not None:
        return str(session / "scene4d" / RUN_STAMP)
    return str(paths.SCENES / RUN_STAMP)


def _models() -> str:
    """Media SDK model dir; empty on Jetson, where insta360_views stitches with its cuda backend."""
    if paths.MEDIA_SDK_MODELS.is_dir():
        return str(paths.MEDIA_SDK_MODELS)
    if paths.JETSON:
        return ""
    raise FileNotFoundError(f"{paths.MEDIA_SDK_MODELS} not found")


def _daaam_action(context, *args, **kwargs):
    daaam = paths.PKG / "scripts" / "daaam_live_node.py"
    calib = paths.CALIB
    daaam_cfg = paths.CONFIG / "daaam_live.yaml"
    view = context.launch_configurations.get("daaam_view", "front")
    # daaam_live only accepts mono|none (its own Metric DA path).
    enable_grounding = context.launch_configurations.get("enable_grounding", "false")
    cmd = [
        paths.PYTHON,
        str(daaam),
        "--calib_dir",
        str(calib),
        "--daaam_config",
        str(daaam_cfg),
        "--view",
        view,
        "--depth_source",
        "mono",
    ]
    if enable_grounding.lower() in ("true", "1", "yes"):
        cmd.append("--enable_grounding")
    return [
        ExecuteProcess(
            cmd=cmd,
            cwd=str(daaam.parent),
            output="screen",
            condition=IfCondition(LaunchConfiguration("start_daaam")),
        )
    ]


def _da360_action(context, *args, **kwargs):
    da360 = paths.DA360_ROOT
    live = da360 / "live_ros.py"
    calib = paths.CALIB
    venv_py = da360 / ".venv" / "bin" / "python"
    py = str(venv_py) if venv_py.is_file() else paths.PYTHON
    model = context.launch_configurations.get(
        "da360_model", str(paths.DA360)
    )
    lidar = context.launch_configurations.get("da360_lidar_topic", "")
    max_hz = context.launch_configurations.get("da360_max_hz", "2.0")
    # Prefer metric topic; lidar optional for median scale.
    cmd = (
        f"set -e; cd '{da360}'; "
        f"exec '{py}' '{live}' "
        f"--model_path '{model}' "
        f"--calib_dir '{calib}' "
        f"--image_topic /insta360/equirect/image/compressed "
        f"--lidar_topic '{lidar}' "
        f"--max_hz {max_hz}"
    )
    return [
        ExecuteProcess(
            cmd=["bash", "-lc", cmd],
            cwd=str(da360),
            output="screen",
            condition=IfCondition(str(_da360_on(context))),
        )
    ]


def _detector_kind(context) -> str:
    cfg = context.launch_configurations.get
    detector = cfg("detector", "auto")
    if detector != "auto":
        return detector
    gdino_model = Path(cfg("gdino_model", str(paths.GDINO)))
    if Path(cfg("sam3_model", str(paths.SAM3))).is_file():
        return "sam3"
    if (gdino_model / "model.safetensors").is_file() or (gdino_model / "pytorch_model.bin").is_file():
        return "gdino_sam2"
    return "yoloe"


def _ground_source(context) -> str:
    """auto: reuse the detector's GroundingDINO + SAM2 pass when it runs gdino_sam2, else an own instance."""
    cfg = context.launch_configurations.get
    source = cfg("ground_segmenter", "auto")
    if source != "auto":
        return source
    return "detector" if _truthy(cfg("start_detector", "true")) and _detector_kind(context) == "gdino_sam2" \
        else "gdino_sam2"


def _detector_action(context, *args, **kwargs):
    live = paths.PKG / "scripts" / "detector_node.py"
    calib = paths.CALIB
    cfg = context.launch_configurations.get
    views = cfg("detector_views", "front")
    max_hz = cfg("detector_max_hz", "5.0")
    prompts = cfg("prompts_file", str(paths.CONFIG / "prompts.txt"))
    distractors = cfg("distractors_file", str(paths.CONFIG / "distractors.txt"))
    yoloe_model = cfg("yoloe_model", str(paths.YOLOE))
    sam3_model = cfg("sam3_model", str(paths.SAM3))
    gdino_model = cfg("gdino_model", str(paths.GDINO))
    sam2_model = cfg("sam2_model", str(paths.SAM2))
    detector = _detector_kind(context)
    if cfg("detector", "auto") == "auto":
        print(f"[pipeline] detector auto -> {detector}")
    conf = cfg("detector_conf", "") or {"sam3": "0.35", "gdino_sam2": "0.35"}.get(detector, "0.25")
    # Front-only detection + tracking → DETECTIONS_TOPIC for Hydra labels.
    # Markers go in the odometry child frame (imu for equirect_vio, livox_frame for Super-LIO).
    cmd = [
        paths.PYTHON,
        str(live),
        "--input",
        "views",
        "--calib_dir",
        str(calib),
        "--views",
        views,
        "--max_hz",
        max_hz,
        "--frame_id",
        ODOM[_odom(context)]["frame_id"],
        *_pose_args(context, with_cloud=False),
        "--no_boxer",
        "--detector",
        detector,
        "--names_file",
        prompts,
        "--distractors_file",
        distractors,
        "--yoloe_model",
        yoloe_model,
        "--sam3_model",
        sam3_model,
        "--gdino_model",
        gdino_model,
        "--sam2_model",
        sam2_model,
        "--gdino_size",
        cfg("gdino_size", "800"),
        "--sam_imgsz",
        cfg("sam_imgsz", "640"),
        "--gdino_precision",
        cfg("gdino_precision", "fp16"),
        "--conf",
        conf,
        "--tracker",
        cfg("tracker", "botsort"),
        "--centers_topic",
        CENTERS_TOPIC.get(_mapper(context), CENTERS_TOPIC["topo"]),
    ]
    if not _truthy(cfg("track_prior", "true")):
        cmd.append("--no_track_prior")
    if not _truthy(cfg("gdino_trt", "true")):
        cmd.append("--no_trt")
    if _truthy(cfg("start_ground", "false")) and _ground_source(context) == "detector":
        cmd += ["--ground_classes", cfg("ground_classes", GROUND_CLASSES_FILE)]
    return [
        ExecuteProcess(
            cmd=cmd,
            cwd=str(live.parent),
            output="screen",
            additional_env={"YOLO_AUTOINSTALL": "false"},
            condition=IfCondition(LaunchConfiguration("start_detector")),
        )
    ]


def _mapper(context) -> str:
    return context.launch_configurations.get("mapper", "topo")


def _da360_on(context) -> bool:
    cfg = context.launch_configurations.get
    value = cfg("start_da360", "auto")
    if value != "auto":
        return _truthy(value)
    return _mapper(context) == "hydra" and cfg("depth_source", "da360") in ("da360", "auto")


def _topo_action(context, *args, **kwargs):
    mapper = _mapper(context)
    if mapper not in ("topo", "hydra_fake"):
        return []
    cfg = context.launch_configurations.get
    script = (paths.SRC / "hydra_fake" / "scripts" / "hydra_fake_node.py" if mapper == "hydra_fake"
              else paths.PKG / "scripts" / "topo_map_node.py")
    cmd = [
        paths.PYTHON,
        str(script),
        "--calib_dir", str(paths.CALIB),
        "--detections_topic", DETECTIONS_TOPIC,
        "--centers_topic", CENTERS_TOPIC["topo"],
        *_pose_args(context, with_cloud=True),
        "--prompts", cfg("prompts_file", ""),
        "--place_spacing", cfg("place_spacing", "1.0"),
        "--region_radius", cfg("region_radius", "4.0"),
        "--dynamic_classes", cfg("dynamic_classes", "person"),
        "--miss_range", cfg("miss_range", "5.0"),
        "--environment", cfg("environment", "auto"),
        "--floor_height", cfg("floor_height", "3.0"),
        "--scene_dir", str(_scene_dir(context)),
        "--world_frame", ODOM[_odom(context)]["world_frame"],
    ]
    if _truthy(cfg("pano_overlay", "true")):
        cmd.append("--pano_overlay")
    return [ExecuteProcess(cmd=cmd, output="screen", sigterm_timeout="30", additional_env=SINGLE_THREAD_BLAS)]


def _view_depth_action(context, *args, **kwargs):
    cfg = context.launch_configurations.get
    mode = _odom(context)
    if not ODOM[mode]["cloud_topic"] or not _truthy(cfg("start_view_depth", "true")):
        return []
    cmd = [
        paths.PYTHON,
        str(paths.PKG / "scripts" / "view_depth_node.py"),
        "--calib_dir", str(paths.CALIB),
        *_pose_args(context, with_cloud=True),
        "--views", cfg("view_depth_views", "front,left,rear,right"),
        "--max_hz", cfg("view_depth_hz", "5.0"),
    ]
    if _truthy(cfg("view_depth_overlay", "false")):
        cmd.append("--overlay")
    return [ExecuteProcess(cmd=cmd, output="screen", additional_env=SINGLE_THREAD_BLAS)]


def _ground_action(context, *args, **kwargs):
    cfg = context.launch_configurations.get
    mode = _odom(context)
    if not ODOM[mode]["cloud_topic"] or not _truthy(cfg("start_ground", "false")):
        return []
    source = _ground_source(context)
    print(f"[pipeline] ground segmenter -> {source}")
    cmd = [
        paths.PYTHON,
        str(paths.SRC / "ground_segmentation" / "scripts" / "ground_node.py"),
        "--calib_dir", str(paths.CALIB),
        *_pose_args(context, with_cloud=True),
        "--segmenter", source,
        # detector: one mask per detector frame, on the detector's views.
        "--views", cfg("detector_views", "front") if source == "detector" else cfg("ground_views", "front,left,rear,right"),
        "--max_hz", "0" if source == "detector" else cfg("ground_hz", "0.5"),
        "--classes", cfg("ground_classes", GROUND_CLASSES_FILE),
        "--blocked_classes", cfg("ground_blocked_classes", ""),
        "--max_step", cfg("ground_max_step", "0.18"),
        "--max_slope_deg", cfg("ground_max_slope_deg", "35"),
    ]
    if _truthy(cfg("ground_overlay", "true")):
        cmd.append("--overlay")
    return [ExecuteProcess(cmd=cmd, output="screen", additional_env=SINGLE_THREAD_BLAS)]


def _goal_server_action(context, *args, **kwargs):
    cfg = context.launch_configurations.get
    value = cfg("start_goal_server", "auto")
    auto = bool(os.environ.get("ZHIPUAI_API_KEY")) or _odom(context) == "qinav"
    if not (auto if value == "auto" else _truthy(value)):
        return []
    cmd = [
        paths.PYTHON,
        str(paths.PKG / "scripts" / "goal_server_node.py"),
        "--scene_dir", str(_scene_dir(context)),
        "--odom_topic", ODOM[_odom(context)]["odom_topic"],
        "--nav2", cfg("nav2", "none"),
        "--model", cfg("glm_model", "glm-4-plus"),
    ]
    qinav = cfg("qinav_goal", "auto")
    if (_odom(context) == "qinav") if qinav == "auto" else _truthy(qinav):
        cmd.append("--qinav")
    if not _truthy(cfg("allow_stairs", "true")):
        cmd.append("--no_stairs")
    if not _truthy(cfg("allow_elevator", "true")):
        cmd.append("--no_elevator")
    return [ExecuteProcess(cmd=cmd, output="screen")]


def _replay_views(context) -> str:
    cfg = context.launch_configurations.get
    value = cfg("replay_views", "auto")
    if value != "auto":
        return value
    views = [v.strip() for v in cfg("detector_views", "front").split(",")]
    if _mapper(context) == "hydra":
        views += [v.strip() for v in cfg("hydra_views", "front").split(",")]
    if _truthy(cfg("start_daaam", "false")):
        views.append(cfg("daaam_view", "front").strip())
    return ",".join(dict.fromkeys(v for v in views if v))


def _hydra_action(context, *args, **kwargs):
    if _mapper(context) != "hydra":
        return []
    hydra = paths.PKG / "scripts" / "hydra_scene_node.py"
    calib = paths.CALIB
    hydra_env = paths.REFER / "hydra_ws" / "setup_hydra_env.bash"
    depth_source = context.launch_configurations.get("depth_source", "da360")
    lidar_topic = context.launch_configurations.get("lidar_topic", "")
    da360_depth = context.launch_configurations.get("da360_depth_topic", "/da360/depth_metric")
    views = context.launch_configurations.get("hydra_views", "front")
    hydra_view = views.split(",")[0].strip() or "front"
    depth_ub = context.launch_configurations.get("depth_ub", "40.0")
    boxer = context.launch_configurations.get("hydra_boxer", "true").lower() in ("1", "true", "yes")
    boxer_hz = context.launch_configurations.get("boxer_max_hz", "2.0")
    scene_dir = _scene_dir(context)
    prompts = context.launch_configurations.get("prompts_file", "")
    # Source Hydra native libs + ROS overlay, then run front RGBD+labels → pipeline.step.
    # BoxerNet runs in the same node on the same RGB/depth/pose for /hydra/objects.
    shell = (
        f"set -e; "
        f"source '{hydra_env}'; "
        f"exec python3 '{hydra}' "
        f"--calib_dir '{calib}' "
        f"--depth_source '{depth_source}' "
        f"--da360_depth_topic '{da360_depth}' "
        f"--depthart_domain {context.launch_configurations.get('depthart_domain', 'outdoor')} "
        f"--depth_ub {depth_ub} "
        f"--views '{views}' "
        f"--hydra_view '{hydra_view}' "
        f"--lidar_topic '{lidar_topic}' "
        f"--detections_topic {DETECTIONS_TOPIC} "
        f"--prompts '{prompts}' "
        f"--hydra_dataset habitat "
        f"--hydra_labelspace ade20k_mp3d "
        f"--boxer_max_hz {boxer_hz} "
        f"--scene_dir '{scene_dir}'"
        + ("" if boxer else " --no_boxer")
    )
    return [
        ExecuteProcess(
            cmd=["bash", "-lc", shell],
            cwd=str(hydra.parent),
            output="screen",
            # SIGINT triggers the final dsg/objects_4d export; give it time before SIGTERM.
            sigterm_timeout="60",
            condition=IfCondition(LaunchConfiguration("start_hydra")),
        )
    ]


def _dam_action(context, *args, **kwargs):
    pkg = paths.PKG
    scene_dir = _scene_dir(context)
    precision = context.launch_configurations.get("dam_precision", "4bit")
    model = context.launch_configurations.get("dam_model", str(paths.DAM))
    # Background DAM-3B worker: describes key views hydra_scene_node saves under scene_dir/views.
    cmd = [
        paths.PYTHON,
        "-m",
        "scene_memory.mapping.describe_objects",
        scene_dir,
        "--watch",
        "5",
        "--precision",
        precision,
        "--model_path",
        model,
    ]
    return [
        ExecuteProcess(
            cmd=cmd,
            cwd=str(pkg),
            output="screen",
            condition=IfCondition(LaunchConfiguration("start_dam")),
        )
    ]


def _source_action(context, *args, **kwargs):
    """Live camera + VIO, or a recorded collection_data session."""
    views_config = paths.VIEWS_CONFIG
    vio_setup = paths.VIO_WS / "install" / "setup.bash"
    vio_config = paths.VIO_CONFIG
    vio_cmd = (
        f"source {vio_setup} && exec ros2 run equirect_vio equirect_vio_node --ros-args "
        "-p image_topic:=/insta360/equirect/image "
        "-p imu_topic:=/imu/data_raw "
        "-p odom_frame:=odom -p base_frame:=imu -p publish_tf:=true "
        f"-p config:={vio_config}"
    )
    cfg = context.launch_configurations.get
    session = _session(context)
    # insta360 -> view / equirect optical frames, so every image topic hangs off the odom tree.
    camera_tf = ExecuteProcess(cmd=[paths.PYTHON, str(paths.PKG / "scripts" / "camera_tf_node.py"),
                                    "--calib_dir", str(paths.CALIB)], output="log")
    if session is None:
        actions = [camera_tf]
        if _truthy(cfg("start_camera", "true")):
            actions.append(
                Node(
                    package="insta360_views",
                    executable="insta360_views",
                    name="insta360_views",
                    output="screen",
                    # The camera SDK occasionally aborts on connect.
                    respawn=True,
                    respawn_delay=3.0,
                    # The raw 22 MB equirect and the camera IMU only feed equirect_vio.
                    parameters=[str(views_config), {"model_dir": _models(),
                                                    "publish_vio_topics": _odom(context) == "vio"}],
                )
            )
        if _odom(context) == "lio":
            if _truthy(cfg("start_lio", "true")):
                actions.append(IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(str(Path(__file__).resolve().parent / "lio.launch.py")),
                    launch_arguments={"start_driver": cfg("start_lidar", "true")}.items(),
                ))
        elif _odom(context) in QINAV_ODOM:
            # QiNav owns the lidar and Super-LIO; only the camera TF (livox_frame -> insta360) is ours.
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(Path(__file__).resolve().parent / "lio.launch.py")),
                launch_arguments={"start_driver": "false", "start_lio": "false"}.items(),
            ))
        elif _truthy(cfg("start_vio", "true")):
            actions.append(ExecuteProcess(cmd=["bash", "-lc", vio_cmd], output="screen"))
        return actions

    # Render the views with the camera settings the session was recorded with.
    recorded = session / "config" / "insta360_views.yaml"
    params = yaml.safe_load((recorded if recorded.is_file() else views_config).read_text())
    params = params.get("insta360_views", {}).get("ros__parameters", {})
    vio_mode = cfg("replay_vio", "recorded")
    replay = ExecuteProcess(
        cmd=[
            paths.PYTHON, str(paths.REPLAY), str(session),
            "--rate", cfg("replay_rate", "1.0"),
            "--start", cfg("replay_start", "0"),
            "--duration", cfg("replay_duration", "0"),
            "--vio", vio_mode,
            "--views", _replay_views(context),
            "--view_size", str(params.get("view_size", 640)),
            "--view_hfov_deg", str(params.get("view_hfov_deg", 90.0)),
            "--front_yaw_deg", str(params.get("front_yaw_deg", 0.0)),
            "--jpeg_quality", str(params.get("jpeg_quality", 80)),
            # Let the slow GPU nodes drain their last frames before shutdown.
            "--hold_s", "10",
            *cfg("replay_extra", "").split(),
        ],
        output="screen",
    )
    actions = [replay, camera_tf]
    if vio_mode == "rerun":
        actions.append(ExecuteProcess(cmd=["bash", "-lc", vio_cmd], output="screen"))
    if _truthy(cfg("shutdown_on_end", "true")):
        actions.append(
            RegisterEventHandler(
                OnProcessExit(target_action=replay, on_exit=[EmitEvent(event=Shutdown(reason="session replay finished"))])
            )
        )
    print(f"[pipeline] offline session {session} -> scene_dir {_scene_dir(context)}")
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "session",
                default_value="",
                description="collection_data session dir; replaces camera + live VIO with the recording",
            ),
            DeclareLaunchArgument("replay_rate", default_value="1.0", description="<1 gives DA360/detector/Boxer more frames"),
            DeclareLaunchArgument("replay_start", default_value="0", description="skip the first N seconds"),
            DeclareLaunchArgument("replay_duration", default_value="0", description="0 = whole session"),
            DeclareLaunchArgument("replay_extra", default_value="",
                                  description="more replay_session options, e.g. '--lidar_bag <raw>/bag --no_odom'"),
            DeclareLaunchArgument(
                "replay_vio",
                default_value="recorded",
                description="recorded = bag odometry/TF | rerun = fresh equirect_vio on the 15 Hz bag frames",
            ),
            DeclareLaunchArgument(
                "replay_views", default_value="auto",
                description="views rendered by the replay; auto = only those the detector/Hydra/DAAAM subscribe to",
            ),
            DeclareLaunchArgument("shutdown_on_end", default_value="true"),
            DeclareLaunchArgument("start_camera", default_value="true"),
            DeclareLaunchArgument(
                "odom", default_value="lio",
                description="lio = MID-360S + Super-LIO (/lio/odom, lidar depth) | qinav = the running QiNav stack's "
                            "/ql/nav/localization + /ql/nav/cloud_in_map (map frame) | qinav_mapping = QiNav 3D "
                            "mapping's /lio/odom + /cloud_registered (frame of the map it saves) | vio = equirect VIO",
            ),
            DeclareLaunchArgument("start_vio", default_value="true", description="odom:=vio only"),
            DeclareLaunchArgument("start_lio", default_value="true", description="odom:=lio: start Super-LIO"),
            DeclareLaunchArgument("start_lidar", default_value="true",
                                  description="odom:=lio: start the MID-360S driver (false = already running)"),
            DeclareLaunchArgument("start_view_depth", default_value="true",
                                  description="odom:=lio: /insta360/<view>/depth_lidar"),
            DeclareLaunchArgument("view_depth_views", default_value="front,left,rear,right"),
            DeclareLaunchArgument("view_depth_hz", default_value="5.0"),
            DeclareLaunchArgument("view_depth_overlay", default_value="false",
                                  description="also publish /insta360/<view>/depth_lidar/overlay/compressed"),
            DeclareLaunchArgument("start_ground", default_value="false",
                                  description="odom:=lio: traversable ground (/ground/grid, /ground/traversable_cloud)"),
            DeclareLaunchArgument(
                "ground_segmenter", default_value="auto",
                description="auto = detector when the detector runs gdino_sam2, else gdino_sam2 | detector (ground "
                            "prompts in the detector's pass, ~0 extra GPU) | gdino_sam2 (own instance)"),
            DeclareLaunchArgument("ground_classes", default_value=GROUND_CLASSES_FILE,
                                  description="ground classes file (class: prompt | prompt  @terrain @blocked per line) "
                                              "or inline '<class>:<prompt>|<prompt>;...'"),
            DeclareLaunchArgument("ground_blocked_classes", default_value="",
                                  description="comma list marked as obstacles in /ground/grid, on top of the file's "
                                              "@blocked classes"),
            DeclareLaunchArgument("ground_max_step", default_value="0.18",
                                  description="m, highest step the robot climbs on @terrain classes (wheeled ~0.03, "
                                              "wheel-legged ~0.12-0.15, legged ~0.18-0.20)"),
            DeclareLaunchArgument("ground_max_slope_deg", default_value="35",
                                  description="steepest ramp / stair pitch over ~0.9 m (wheeled ~10, wheel-legged "
                                              "~25-30, legged ~35); 0 disables"),
            DeclareLaunchArgument("ground_views", default_value="front,left,rear,right",
                                  description="gdino_sam2 only; detector uses detector_views"),
            DeclareLaunchArgument("ground_hz", default_value="0.5", description="per view, gdino_sam2 only"),
            DeclareLaunchArgument("ground_overlay", default_value="true",
                                  description="publish /ground/<view>/overlay/compressed"),
            DeclareLaunchArgument(
                "mapper",
                default_value="topo",
                description="topo = depth-free layered topo graph | hydra_fake = topo + lidar pose graph "
                            "(loops, LIO reset stitching) + free-space places | hydra = DA360 depth + Hydra + BoxerNet",
            ),
            DeclareLaunchArgument("place_spacing", default_value="1.0", description="topo: m between place nodes"),
            DeclareLaunchArgument("region_radius", default_value="4.0", description="topo: place clustering radius"),
            DeclareLaunchArgument("dynamic_classes", default_value="person",
                                  description="topo: tracked but kept out of the static map / navigation"),
            DeclareLaunchArgument("miss_range", default_value="5.0",
                                  description="topo: m; in-view but undetected objects lose existence (0 = off)"),
            DeclareLaunchArgument("environment", default_value="auto",
                                  description="topo: auto (SigLIP indoor/outdoor from the equirect) | indoor | outdoor"),
            DeclareLaunchArgument("floor_height", default_value="3.0",
                                  description="topo: m per storey for stairs / elevator floor counting"),
            DeclareLaunchArgument("allow_stairs", default_value="true", description="goal server: route over stairs"),
            DeclareLaunchArgument("allow_elevator", default_value="true", description="goal server: route via elevators"),
            DeclareLaunchArgument("start_goal_server", default_value="auto",
                                  description="prompt -> goal node; auto = when ZHIPUAI_API_KEY is set or odom:=qinav "
                                              "(without the key only fast-path instructions work)"),
            DeclareLaunchArgument("nav2", default_value="none", description="goal server: none | through_poses | pose"),
            DeclareLaunchArgument("qinav_goal", default_value="auto",
                                  description="goal server sends goals to QiNav (/slam/navigation/execute 1001); "
                                              "auto = with odom:=qinav"),
            DeclareLaunchArgument("glm_model", default_value="glm-4-plus"),
            DeclareLaunchArgument(
                "start_da360", default_value="auto", description="auto = only with mapper:=hydra"
            ),
            DeclareLaunchArgument(
                "da360_model",
                default_value=str(paths.DA360),
            ),
            DeclareLaunchArgument(
                "da360_lidar_topic",
                default_value="",
                description="Optional /extrinsic/lidar_fused for metric scale; empty keeps relative",
            ),
            DeclareLaunchArgument("da360_max_hz", default_value="2.0"),
            DeclareLaunchArgument("da360_depth_topic", default_value="/da360/depth_metric"),
            # 2D detection + tracking → semantic labels for Hydra (DAM/DAAAM off).
            DeclareLaunchArgument("start_detector", default_value="true", description="start the 2D detection node"),
            DeclareLaunchArgument(
                "detector",
                default_value="auto",
                description="sam3 | gdino_sam2 | yoloe | auto (first whose weights exist, in that order)",
            ),
            DeclareLaunchArgument(
                "prompts_file",
                default_value=str(paths.CONFIG / "prompts.txt"),
                description="class list shared by the detector and Hydra",
            ),
            DeclareLaunchArgument(
                "distractors_file",
                default_value=str(paths.CONFIG / "distractors.txt"),
                description="gdino_sam2 look-alike prompts whose boxes are dropped; empty disables",
            ),
            DeclareLaunchArgument("detector_views", default_value="front", description="comma separated pinhole views"),
            DeclareLaunchArgument("detector_max_hz", default_value="5.0"),
            DeclareLaunchArgument(
                "yoloe_model", default_value=str(paths.YOLOE)
            ),
            DeclareLaunchArgument("sam3_model", default_value=str(paths.SAM3)),
            DeclareLaunchArgument("gdino_model", default_value=str(paths.GDINO)),
            DeclareLaunchArgument("sam2_model", default_value=str(paths.SAM2)),
            DeclareLaunchArgument("gdino_size", default_value="800",
                                  description="gdino_sam2: GroundingDINO input side (640 faster, ~1/3 fewer boxes)"),
            DeclareLaunchArgument("sam_imgsz", default_value="640",
                                  description="gdino_sam2: SAM2 input side (1024 model default, ~2.4x slower)"),
            DeclareLaunchArgument("gdino_trt", default_value="true",
                                  description="gdino_sam2: use the TensorRT engines if built "
                                              "(scripts/gdino_trt.py, scripts/sam2_trt.py), else PyTorch"),
            DeclareLaunchArgument("gdino_precision", default_value="fp16",
                                  description="gdino_sam2: GroundingDINO engine fp16 | mixed | int8 "
                                              "(scripts/gdino_trt.py --precision)"),
            DeclareLaunchArgument(
                "detector_conf", default_value="", description="empty: 0.35 sam3 / 0.35 gdino_sam2 / 0.25 yoloe"
            ),
            DeclareLaunchArgument(
                "tracker",
                default_value="botsort",
                description="botsort (+3D pose prior from the mapper's object centres) | bytetrack",
            ),
            DeclareLaunchArgument("track_prior", default_value="true", description="SuperMap 3D-to-2D motion prior"),
            DeclareLaunchArgument("start_daaam", default_value="false"),
            # mapper:=hydra only: front RGB-D + labels → Hydra pipeline.step.
            DeclareLaunchArgument("start_hydra", default_value="true"),
            DeclareLaunchArgument("hydra_views", default_value="front"),
            DeclareLaunchArgument(
                "hydra_boxer",
                default_value="true",
                description="BoxerNet OBBs on /hydra/objects (Hydra DSG boxes stay on /hydra/objects_dsg)",
            ),
            DeclareLaunchArgument("boxer_max_hz", default_value="2.0"),
            DeclareLaunchArgument(
                "scene_dir",
                default_value="",
                description="dsg.json, objects_4d.json, views/, descriptions.json; empty = data/scene4d/<time> "
                "(or <session>/scene4d/<time> offline)",
            ),
            # DAM-3B region captions for the 4D object memory (GPU heavy, off by default).
            DeclareLaunchArgument("start_dam", default_value="false"),
            DeclareLaunchArgument("dam_precision", default_value="4bit", description="4bit (~3.4 GB) | fp16 (~7 GB) | auto"),
            DeclareLaunchArgument("dam_model", default_value=str(paths.DAM)),
            DeclareLaunchArgument("daaam_view", default_value="front"),
            DeclareLaunchArgument("enable_grounding", default_value="false"),
            DeclareLaunchArgument("lidar_topic", default_value=""),
            DeclareLaunchArgument("depth_ub", default_value="40.0"),
            DeclareLaunchArgument(
                "depth_source",
                default_value="da360",
                description="da360 | depthart | mono | lidar | auto — da360 warps /da360/depth_metric to front; "
                            "depthart = DepthART metric TensorRT on the pinhole view (scripts/depthart_trt.py)",
            ),
            DeclareLaunchArgument("depthart_domain", default_value="outdoor", description="indoor | outdoor"),
            DeclareLaunchArgument(
                "stagger_s", default_value="8" if paths.JETSON else "0",
                description="seconds between starting the GPU nodes (detector, DA360, ground segmenter, DAM) "
                            "so their model loads do not overlap; 0 = all at once",
            ),
            OpaqueFunction(function=_source_action),
            OpaqueFunction(function=_staggered(_da360_action, 2)),
            OpaqueFunction(function=_staggered(_detector_action, 1)),
            OpaqueFunction(function=_staggered(_daaam_action, 4)),
            OpaqueFunction(function=_hydra_action),
            OpaqueFunction(function=_topo_action),
            OpaqueFunction(function=_view_depth_action),
            OpaqueFunction(function=_staggered(_ground_action, 3)),
            OpaqueFunction(function=_goal_server_action),
            OpaqueFunction(function=_staggered(_dam_action, 4)),
        ]
    )
