"""STVLN's own 3D navigation stack (QiNav's planners / localization / Go2-W control, run without QiNav).

    src/scripts/stvln.sh nav <map> [--bg] [control:=true] [k:=v ...]
    ros2 launch src/planning/launch/nav.launch.py map:=5f_test_3d

<map> is a map name: data/maps/<map>/ (stvln.sh map / map_save; older flat data/maps/<map>_*), then
~/.ros/nav3d/maps (QiNav), or a path
prefix (<prefix>_map.pcd, <prefix>_traversable.pcd).

* sensors: scene_memory lio.launch.py (MID-360S driver, Super-LIO /lio/odom /lio/cloud_world /lio/base/odom,
  camera TF)
* cloud_preprocess: static TF livox_frame -> base_link (base_link_mount.yaml[robot_type]),
  /livox/front_lidar/custom -> /ql/nav/cloud_in_body
* local obstacles (local_obstacles:=freespace, default): camera freespace (src/ground_segmentation ground_node,
  GroundingDINO + SAM2 ground classes lifted with the lidar, freespace_views at freespace_hz) kept only in a
  freespace_window (30 m) square around the robot -> /ground/traversable_cloud; freespace_obstacles measures
  every LIO scan against that ground (points 0.15-1.2 m above it) -> /stvln/local_obstacles, the
  localPlanner's /terrain_map, and @blocked / ground_blocked_classes cells -> /no_entry_obstacle_pc
  (/added_obstacles). No linefit: its per-scan ground line splits ramps, kerbs and low objects wrongly.
  local_obstacles:=linefit: QiNav's linefit + ground_trans -> /non_ground_in_map instead (semantic_local:=true
  adds the camera ground's obstacle cells on top, ground_obstacles)
* localization_cpp: ICP on <map>_map.pcd -> TF map -> odom, /localization, /ql/nav/localization; initial pose
  from /initialpose (RViz / Foxglove "2D Pose Estimate") when the robot did not start at the map origin
* auto_initial_pose (auto_init:=false to skip; src/localization/scripts/auto_initial_pose.py): publishes that
  /initialpose itself - last saved pose (data/loc/<map>_last_pose.json) first, else a global search of the
  accumulated LIO cloud over the map's traversable cells (gn10-pointcloud-localization style); again when
  global_localization reports localization_failed
* semantic_map_viz (semantic_viz:=false to skip): the map's saved semantic layer (places, rooms, object
  boxes + labels from <map>_topo_graph.json, frame map) -> /topo/markers; add it as a 3D panel topic
* goal_server (vln:=false to skip): VLN on the same semantic map (<map>_scene, the frame of <map>_map.pcd):
  `ros2 topic pub --once /scene_memory/query std_msgs/String "{data: '去电视那里'}"` -> answer on
  /scene_memory/answer, route /scene_memory/path, goal -> /p2p_move_base, events /scene_memory/nav_status.
  On arrival the target is checked in the Insta360 views (arrival_check:=false to skip): verified / far /
  seen / not_seen. A target the map lacks (or a person / moved object) is looked for around the robot and
  driven to (search:=false to skip): searching / far / seen / not_found. In no view at all, frontiers of the
  planning ground are explored, best VLM-rated direction first (explore:=false to skip): exploring.
* map_extender (extend_map:=false to skip): off-map targets are followed while the planning ground and the
  localization map grow with the live scans (only beyond the prior map, only while localized); the extension
  is saved to <map>_ext/<stamp>/ and is not loaded next time.
  Free-form instructions need the GLM key: stvln.sh nav <map> --glm (else only the rule / SigLIP fast path)
* qi_move_base: DDDMR global planner on <map>_traversable.pcd + CMU local planner -> /stvln/cmd_vel;
  goals on /clicked_point (PointStamped) or the /p2p_move_base action (/goal_pose_3d is taken: clicked2goal
  reads PoseStamped there, p2p_move_base writes PointStamped)
* go2w_battery_node (battery:=false to skip): rt/lowstate on eth0 -> /go2w/battery (BatteryState),
  /go2w/diagnostics (WARN <= 30 %, ERROR <= 15 %)
* insta360_views (camera:=false to skip; skipped when one already runs): panorama
  /insta360/equirect/image/compressed, views /insta360/<view>/image/compressed
* go2w_description: Go2-W model on /robot_description under base_link (model:=false to skip); in Foxglove
  (robot bridge, port 8766) add a URDF layer from topic /robot_description
* control:=true: robot_controller (/stvln/robot_controller/command) + go2_cmd_vel_node
  (/stvln/cmd_vel -> unitree_sdk2 SportClient on eth0). Default off: the planners run, the dog does not move.
  The dog stands up (StandUp + SpeedLevel 1) by itself when a point goal (/clicked_point) or a semantic goal
  (/scene_memory/goal) arrives, or on the first motion command, unless it already stands (auto_prepare:=false:
  only by hand, `ros2 topic pub --once /stvln/unitree/posture_cmd std_msgs/msg/Int32 "{data: 1}"`;
  0 StopMove, 1 StandUp + BalanceStand, 2 StandDown, 3 Damp, 5 BalanceStand, 21-23 SpeedLevel)

QiNav's qi-nav-server keeps running; nothing here uses the names it drives (/ql/nav/robot_cmd_vel,
/robot_controller/command, /unitree/posture_cmd, /cmd_vel), and the launch refuses to start while any of
QiNav's driver / LIO / localization / planner / controller processes run.
"""

import json
import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, Shutdown
from launch_ros.actions import Node

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stack_common import WS_ROOT, extras, foreign_processes, include, map_prefix, sensors  # noqa: E402

CMD_VEL = "/stvln/cmd_vel"
POSTURE = "/stvln/unitree/posture_cmd"


def resolve_map(value: str) -> tuple[Path, Path]:
    prefix = map_prefix(value)
    map_pcd = Path(f"{prefix}_map.pcd")
    ground = Path(f"{prefix}_traversable.pcd")
    if not ground.is_file():
        ground = Path(f"{prefix}_ground.pcd")
    if not ground.is_file():
        raise RuntimeError(f"[nav] no {prefix}_traversable.pcd / _ground.pcd for planning")
    return map_pcd, ground


def semantic_scene(prefix: str) -> Path:
    """<prefix>_scene (stvln.sh map_save), else the scene_dir <prefix>_semantic.json names (qinav_bridge)."""
    scene = Path(f"{prefix}_scene")
    meta = Path(f"{prefix}_semantic.json")
    if not (scene / "topo_graph.json").is_file() and meta.is_file():
        scene = Path(json.loads(meta.read_text()).get("scene_dir") or scene)
    return scene


def _actions(context, *args, **kwargs):
    cfg = context.launch_configurations.get
    if foreign := foreign_processes():
        print("[nav] QiNav (or another) navigation stack is running; stop it first:\n  " + "\n  ".join(foreign),
              file=sys.stderr)
        return [Shutdown(reason="QiNav stack running")]
    map_pcd, ground = resolve_map(cfg("map", ""))
    robot_type, ground_height = cfg("robot_type"), cfg("ground_height")
    local = cfg("local_obstacles")
    if local not in ("freespace", "linefit"):
        return [Shutdown(reason=f"local_obstacles:={local}: freespace or linefit")]
    freespace = local == "freespace"
    print(f"[nav] map {map_pcd}, planning ground {ground}, robot {robot_type}, "
          f"control {'ON' if cfg('control') == 'true' else 'off (planners only)'}, local obstacles: "
          + (f"camera freespace ({cfg('freespace_views')} @ {cfg('freespace_hz')} Hz, "
             f"{cfg('freespace_window')} m window)" if freespace else "linefit"))
    if freespace and cfg("camera") != "true":
        print("[nav] local_obstacles:=freespace without camera:=true: needs insta360_views running, else the "
              "obstacles fall back to the ground under the robot", file=sys.stderr)

    actions = sensors(cfg, linefit=not freespace) + [
        include("localization_cpp", "localization.launch.py", map=map_pcd, odom_topic="/lio/base/odom",
                cloud_topic="/lio/cloud_world"),
        include("qi_move_base", "qi_move_base.launch.py", mode="all", map_file="", ground_file=ground,
                ground_height=ground_height, max_speed=cfg("max_speed"), cmd_vel_topic=CMD_VEL,
                goal_tolerance_xy=cfg("goal_tolerance_xy"), goal_tolerance_yaw=cfg("goal_tolerance_yaw"),
                obstacle_topic="/stvln/local_obstacles" if freespace else "/non_ground_in_map"),
    ] + extras(cfg, "nav")
    python = WS_ROOT / ".venv" / "bin" / "python"
    python = str(python if python.is_file() else sys.executable)
    ground_src = WS_ROOT / "src" / "ground_segmentation" / "scripts"
    single_thread = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    if not freespace:
        actions.append(include("cloud_preprocess", "trans_ground_to_map.launch.py"))
    if cfg("semantic_viz") == "true":
        actions.append(ExecuteProcess(
            cmd=[python, "-m", "scene_memory.nodes.semantic_map_viz_node",
                 "--map_prefix", str(map_pcd)[:-len("_map.pcd")]],
            name="semantic_map_viz", output="screen"))
    if freespace or cfg("semantic_local") == "true":
        window = cfg("freespace_window") if freespace else "0"
        actions.append(ExecuteProcess(
            cmd=[python, str(ground_src / "ground_node.py"), "--segmenter", "gdino_sam2",
                 "--views", cfg("freespace_views"), "--max_hz", cfg("freespace_hz"), "--window_m", window,
                 "--pose_source", "lio", "--odom_topic", "/lio/odom", "--cloud_topic", "/lio/cloud_world",
                 "--blocked_classes", cfg("ground_blocked_classes"), "--overlay"],
            name="semantic_ground", output="screen", additional_env=single_thread))
    if freespace:
        actions.append(ExecuteProcess(
            cmd=[python, str(ground_src / "freespace_obstacles_node.py"), "--window_m", cfg("freespace_window"),
                 "--blocked_classes", cfg("ground_blocked_classes")],
            name="freespace_obstacles", output="screen", additional_env=single_thread))
    elif cfg("semantic_local") == "true":
        actions.append(ExecuteProcess(cmd=[python, str(ground_src / "ground_obstacles_node.py")],
                                      name="ground_obstacles", output="screen"))
    if cfg("extend_map") == "true":
        actions.append(ExecuteProcess(
            cmd=[python, str(WS_ROOT / "src" / "mapping" / "scripts" / "map_extender.py"),
                 "--map_prefix", str(map_pcd)[:-len("_map.pcd")]],
            name="map_extender", output="screen"))
    scene = semantic_scene(str(map_pcd)[:-len("_map.pcd")])
    if cfg("vln") == "true":
        if (scene / "topo_graph.json").is_file() and (scene / "objects_4d.json").is_file():
            agent = (f"{os.environ.get('GLM_MODEL', '?')} on {os.environ['GLM_BASE_URL']}" if os.environ.get("GLM_BASE_URL")
                     else "GLM on" if os.environ.get("ZHIPUAI_API_KEY") else "GLM off: --glm or --vlm")
            print(f"[nav] vln: goal_server on {scene} ({agent})")
            actions.append(ExecuteProcess(
                cmd=[python, "-m", "scene_memory.nodes.goal_server_node", "--scene_dir", str(scene),
                     "--odom_topic", "/ql/nav/localization", "--frame_id", "map", "--p2p",
                     "--arrival_check", "on" if cfg("arrival_check") == "true" else "off",
                     "--search", "on" if cfg("search") == "true" else "off",
                     "--extend_map", "on" if cfg("extend_map") == "true" else "off",
                     "--explore", "on" if cfg("explore") == "true" else "off"],
                name="goal_server", output="screen"))
        else:
            print(f"[nav] vln: no semantic map {scene}/topo_graph.json + objects_4d.json; goal_server off")
    if cfg("auto_init") == "true":
        actions.append(ExecuteProcess(
            cmd=[python,
                 str(WS_ROOT / "src" / "localization" / "scripts" / "auto_initial_pose.py"),
                 "--map_prefix", str(map_pcd)[:-len("_map.pcd")],
                 "--odom_topic", "/lio/base/odom", "--cloud_topic", "/lio/cloud_world"],
            name="auto_initial_pose", output="screen"))
    if cfg("control") == "true":
        params = os.path.join(get_package_share_directory("robot_controller"), "config", f"{robot_type}.yaml")
        topics = {"cmd_vel_topic": CMD_VEL, "posture_cmd_topic": POSTURE}
        auto = cfg("auto_prepare") == "true"
        stand_up_on = {"auto_prepare_motion": auto, "goal_point_topic": "/clicked_point" if auto else "",
                       "goal_pose_topic": "/scene_memory/goal" if auto else ""}
        actions += [
            Node(package="robot_controller", executable="robot_controller_node", name="robot_controller",
                 output="screen",
                 parameters=[params, {"robot_type": robot_type, "remote_cmd_vel_topic": "", **topics}],
                 remappings=[("/robot_controller/command", "/stvln/robot_controller/command")]),
            Node(package="go2w_keyboard_control", executable="go2_cmd_vel_node", name="go2_cmd_vel_node",
                 output="screen",
                 parameters=[params, {**topics, **stand_up_on}]),
        ]
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("map", description="map name in data/maps (stvln.sh map) or ~/.ros/nav3d/maps "
                                                     "(QiNav), or a path prefix"),
        DeclareLaunchArgument("control", default_value="false",
                              description="true: drive the Go2-W (robot_controller + go2_cmd_vel_node)"),
        DeclareLaunchArgument("robot_type", default_value="unitree_go2w"),
        DeclareLaunchArgument("battery", default_value="true",
                              description="Go2-W battery (rt/lowstate) -> /go2w/battery, /go2w/diagnostics"),
        DeclareLaunchArgument("camera", default_value="true",
                              description="Insta360 X4 Air: panorama /insta360/equirect/image/compressed + views"),
        DeclareLaunchArgument("model", default_value="true",
                              description="publish the Go2-W model (/robot_description) for Foxglove / RViz"),
        DeclareLaunchArgument("auto_prepare", default_value="true",
                              description="control:=true: StandUp + SpeedLevel(1) when a goal arrives "
                                          "(/clicked_point, /scene_memory/goal) or on the first motion command, "
                                          "if not standing yet; false = send posture 1 on "
                                          "/stvln/unitree/posture_cmd first"),
        DeclareLaunchArgument("semantic_viz", default_value="true",
                              description="saved semantic map (<map>_topo_graph.json) -> /topo/markers"),
        DeclareLaunchArgument("local_obstacles", default_value="freespace",
                              description="freespace: local planner obstacles = LIO points above the camera "
                                          "freespace (GroundingDINO + SAM2 on the GPU, MAXN); linefit: QiNav's "
                                          "linefit ground split"),
        DeclareLaunchArgument("freespace_views", default_value="front",
                              description="segmented views (comma list of front,left,rear,right)"),
        DeclareLaunchArgument("freespace_hz", default_value="0.5", description="segmented frames/s per view"),
        DeclareLaunchArgument("freespace_window", default_value="30.0",
                              description="m; side of the freespace square kept around the robot"),
        DeclareLaunchArgument("semantic_local", default_value="false",
                              description="local_obstacles:=linefit only: add the camera ground's obstacle cells"),
        DeclareLaunchArgument("ground_blocked_classes", default_value="",
                              description="extra classes that are always obstacles (e.g. grass), besides @blocked"),
        DeclareLaunchArgument("vln", default_value="true",
                              description="goal_server on <map>_scene: /scene_memory/query -> /p2p_move_base"),
        DeclareLaunchArgument("arrival_check", default_value="true",
                              description="after arriving, find + lidar-range the target in the Insta360 views "
                                          "(qwen3-vl at STVLN_VLM_URL); verified/far/seen/not_seen on nav_status"),
        DeclareLaunchArgument("extend_map", default_value="true",
                              description="map_extender: while goal_server follows an off-map target, grow the "
                                          "planning ground (mapground) and localization's map with the live LIO "
                                          "scans; saved to <map>_ext/<stamp>/, the prior map files stay as they are"),
        DeclareLaunchArgument("search", default_value="true",
                              description="with arrival_check: look around for (and go to) a target the semantic "
                                          "map lacks, or a dynamic/absent one, before its old spot"),
        DeclareLaunchArgument("explore", default_value="true",
                              description="with search: target in no view -> WildOS-style frontier exploration "
                                          "(VLM-rated directions), looking again at each frontier"),
        DeclareLaunchArgument("auto_init", default_value="true",
                              description="auto_initial_pose: last saved pose, else global search -> /initialpose"),
        DeclareLaunchArgument("ground_height", default_value="-0.5", description="QiNav backend_3d default"),
        DeclareLaunchArgument("max_speed", default_value="0.5"),
        DeclareLaunchArgument("goal_tolerance_xy", default_value=""),
        DeclareLaunchArgument("goal_tolerance_yaw", default_value=""),
        DeclareLaunchArgument("start_driver", default_value="true"),
        DeclareLaunchArgument("start_lio", default_value="true"),
        OpaqueFunction(function=_actions),
    ])
