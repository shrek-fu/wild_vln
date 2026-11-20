"""STVLN's own point-cloud + semantic mapping (QiNav's 3D mapping chain, run without QiNav).

    src/scripts/stvln.sh map <name> [--bg] [semantic:=false] [k:=v ...]
    src/scripts/stvln.sh map_save <name>      # while mapping: PCD maps + semantic files -> data/maps/<name>/

* sensors (planning/launch/stack_common.py): MID-360S driver + Super-LIO (/lio/cloud_world, /lio/base/odom in
  odom), livox_frame -> base_link, body cloud, linefit ground segmentation
* cloud_preprocess ground_trans: /segmentation/{ground,obstacle} -> /ground_in_odom, /non_ground_in_odom
* direct_lidar_inertial_odometry dlio_multi_map_node: accumulates the registered / ground / non-ground clouds
  in odom, /stvln/laser_map for display, /save_pcd for map_save
* semantic:=true: scene_memory pipeline (odom:=lio on the same Super-LIO) -> data/maps/<name>/<name>_scene, so the
  topo graph / objects share the PCD map's frame (odom here = map for nav.launch.py's localization)
* ground:=semantic (default, needs semantic:=true): the pipeline's ground_segmentation node
  (/ground/traversable_cloud); map_save builds <name>_traversable.pcd from it instead of linefit's ground,
  which breaks on ramps and uneven floors (linefit still feeds DLIO's own ground / non-ground maps)
* camera / battery / model as in nav.launch.py
* offline (stvln.sh map_offline <session> <name>, e.g. on the PC): session:=<raw>_proc raw_session:=<raw>;
  replay_session plays the derived session (panorama + views) together with the raw bag's MID-360S topics,
  and the same chain as live (Super-LIO, linefit, DLIO, pipeline) maps them, so the PCD map and the semantic
  map share one Super-LIO run (one frame)

Refuses to start while QiNav's driver / LIO / mapping / localization / planner / controller processes run.
"""

import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, Shutdown
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "planning" / "launch"))
from stack_common import (PIPELINE_LAUNCH, WS_ROOT, extras, foreign_processes, include,  # noqa: E402
                          include_file, saved_map_prefix, sensors)

REPLAY = WS_ROOT / "src" / "tools" / "collection_data" / "collection_data" / "replay_session.py"
PYTHON = WS_ROOT / ".venv" / "bin" / "python" if (WS_ROOT / ".venv" / "bin" / "python").is_file() else Path(sys.executable)


def _actions(context, *args, **kwargs):
    cfg = context.launch_configurations.get
    if foreign := foreign_processes():
        print("[map] QiNav (or another) lidar / mapping / navigation stack is running; stop it first:\n  "
              + "\n  ".join(foreign), file=sys.stderr)
        return [Shutdown(reason="QiNav stack running")]
    name = cfg("name", "").strip()
    if not name or "/" in name:
        return [Shutdown(reason=f"name:={name!r}: a plain map name")]
    prefix = saved_map_prefix(name)
    if Path(f"{prefix}_map.pcd").exists() and cfg("overwrite") != "true":
        print(f"[map] {prefix}_map.pcd exists (overwrite:=true to replace it on map_save)", file=sys.stderr)
        return [Shutdown(reason="map exists")]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    scene_dir = Path(f"{prefix}_scene")
    semantic = cfg("semantic") == "true"
    ground = cfg("ground")
    if ground == "semantic" and not semantic:
        print("[map] ground:=semantic needs semantic:=true (the detector's GroundingDINO + SAM2 pass); planning "
              "ground falls back to linefit", file=sys.stderr)
        ground = "linefit"
    print(f"[map] mapping '{name}' -> {prefix.parent}/  (save: stvln.sh map_save {name}); semantic "
          f"{'-> ' + str(scene_dir) if semantic else 'off'}; planning ground: {ground}")

    session, live = cfg("session", "").strip(), cfg
    if session:
        # Offline: the replay is the lidar driver and the camera; Super-LIO runs on the replayed CustomMsg.
        proc, raw = Path(session).expanduser().resolve(), Path(cfg("raw_session")).expanduser().resolve()
        if not (raw / "bag" / "metadata.yaml").is_file():
            return [Shutdown(reason=f"raw_session:={raw}: no bag/metadata.yaml with the MID-360S topics")]
        offline = {"start_driver": "false", "start_lio": "true", "camera": "false", "battery": "false"}
        live = lambda key, default=None: offline.get(key, cfg(key, default))  # noqa: E731
        replay_extra = f"--lidar_bag {raw / 'bag'} --no_odom"
        print(f"[map] offline: replaying {proc} + lidar of {raw} at {cfg('replay_rate')}x")

    share = FindPackageShare("direct_lidar_inertial_odometry")
    actions = sensors(live) + [
        include("cloud_preprocess", "trans_ground_to_odom.launch.py", ground_height=cfg("ground_height")),
        Node(package="direct_lidar_inertial_odometry", executable="dlio_multi_map_node", output="screen",
             parameters=[PathJoinSubstitution([share, "cfg", "dlio.yaml"]),
                         PathJoinSubstitution([share, "cfg", "params_mapping.yaml"]),
                         {"map/prior_map": "", "map/prior_ground": "", "map/prior_nonground": "",
                          "map/wait_for_localization": False}],
             remappings=[("cloud_map", "/lio/cloud_world"), ("cloud_ground", "/ground_in_odom"),
                         ("cloud_non_ground", "/non_ground_in_odom"), ("odom", "/lio/base/odom"),
                         ("map", "/stvln/laser_map")]),
    ] + extras(live, "map")
    offline_args = dict(session=proc, replay_extra=replay_extra, replay_rate=cfg("replay_rate"),
                        shutdown_on_end="false") if session else {}
    if semantic:
        actions.append(include_file(PIPELINE_LAUNCH, odom="lio", mapper=cfg("mapper"), scene_dir=scene_dir,
                                    start_lio="false", start_lidar="false", start_camera="false",
                                    start_ground=str(ground == "semantic").lower(),
                                    ground_max_step=cfg("ground_max_step"),
                                    ground_max_slope_deg=cfg("ground_max_slope_deg"), **offline_args))
    elif session:
        actions.append(ExecuteProcess(
            cmd=[str(PYTHON), str(REPLAY), str(proc), "--views", "", "--rate", cfg("replay_rate"),
                 "--hold_s", "10", *replay_extra.split()],
            name="session_replay", output="screen"))
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("name", description="map name: data/maps/<name>/<name>_map.pcd ... on map_save"),
        DeclareLaunchArgument("semantic", default_value="true", description="also build the semantic map"),
        DeclareLaunchArgument("mapper", default_value="topo", description="semantic mapper (topo | hydra_fake)"),
        DeclareLaunchArgument("overwrite", default_value="false"),
        DeclareLaunchArgument("robot_type", default_value="unitree_go2w"),
        DeclareLaunchArgument("ground", default_value="semantic", choices=["semantic", "linefit"],
                              description="planning ground for map_save: semantic = ground_segmentation (camera "
                                          "classes + per-cell lidar height, follows ramps / uneven floors), "
                                          "linefit = DLIO's ground map"),
        DeclareLaunchArgument("ground_max_step", default_value="0.18", description="m, terrain classes"),
        DeclareLaunchArgument("ground_max_slope_deg", default_value="35"),
        DeclareLaunchArgument("ground_height", default_value="-0.5", description="QiNav backend_3d default"),
        DeclareLaunchArgument("camera", default_value="true"),
        DeclareLaunchArgument("battery", default_value="true"),
        DeclareLaunchArgument("model", default_value="true"),
        DeclareLaunchArgument("session", default_value="",
                              description="offline: derived session (stvln.sh derive -> <session>_proc) to replay "
                                          "instead of the live camera / lidar (stvln.sh map_offline)"),
        DeclareLaunchArgument("raw_session", default_value="",
                              description="offline: the raw recording whose bag holds the MID-360S topics"),
        DeclareLaunchArgument("replay_rate", default_value="1.0"),
        DeclareLaunchArgument("start_driver", default_value="true"),
        DeclareLaunchArgument("start_lio", default_value="true"),
        OpaqueFunction(function=_actions),
    ])
