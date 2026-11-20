"""Map-free JanusVLN walking: the .87 brain picks each step, the dog executes it with a lidar guard.

    src/scripts/stvln.sh janus_nav --instruction="walk forward and go to the door" [--bg] [control:=true]
    ros2 launch src/planning/launch/janus_nav.launch.py instruction:="..." control:=true

No map, no localization: works anywhere.

* sensors: MID-360S driver + Super-LIO (/lio/base/odom closes each step), static TF livox_frame -> base_link,
  /livox/front_lidar/custom -> /ql/nav/cloud_in_body (base_link, ground z=0, body cropped) for the guard
* insta360_views (camera:=false to skip; skipped when one already runs): /insta360/front/image/compressed
* janus client (refer/JanusVLN/deploy/dog_front_client.py): front frame -> brain (.87:8088) -> one action;
  MOVE_FORWARD 0.25 m only when the corridor ahead is clear (else one 15° turn toward the emptier side),
  stops mid-step when something comes within 0.3 m; TURN 15°; STOP ends the episode once qwen3-vl (.209
  Ollama, verify:=false to skip) sees the goal in the frame, saved to data/janus/. A new instruction on
  /janus/instruction (stvln.sh janus_go "..."; "stop" to halt) aborts the step and starts a new episode.
  /janus/path, /janus/step_goal (odom), /janus/guard (base_link) for Foxglove / RViz
* control:=true: robot_controller + go2_cmd_vel_node (/stvln/cmd_vel -> unitree_sdk2 SportClient on eth0);
  the dog stands up on the first motion command (auto_prepare). Default off: the brain runs and the guard is
  logged, the dog does not move.

Refuses to start while QiNav's driver / LIO / planner / controller processes run.
"""

import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, Shutdown
from launch_ros.actions import Node

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stack_common import WS_ROOT, extras, foreign_processes, sensors  # noqa: E402

CMD_VEL = "/stvln/cmd_vel"
POSTURE = "/stvln/unitree/posture_cmd"
CLIENT = WS_ROOT / "refer" / "JanusVLN" / "deploy" / "dog_front_client.py"


def _actions(context, *args, **kwargs):
    cfg = context.launch_configurations.get
    if foreign := foreign_processes():
        print("[janus_nav] QiNav (or another) navigation stack is running; stop it first:\n  "
              + "\n  ".join(foreign), file=sys.stderr)
        return [Shutdown(reason="QiNav stack running")]
    instruction = cfg("instruction", "").strip()
    control = cfg("control") == "true"
    print(f"[janus_nav] brain {cfg('brain')}, control {'ON' if control else 'off (brain + guard only)'}, "
          + (f"instruction \"{instruction}\"" if instruction else "idle until /janus/instruction"))

    actions = sensors(cfg, linefit=False) + extras(cfg, "janus_nav")
    client = ["python3", str(CLIENT), "--brain", cfg("brain"), "--instruction", instruction,
              "--cmd-vel-topic", CMD_VEL, "--on-blocked", cfg("on_blocked"), "--max-steps", cfg("max_steps")]
    if control:
        client.append("--execute")
    if cfg("verify") != "true":
        client.append("--no-verify")
    actions.append(ExecuteProcess(cmd=client, name="janus_client", output="screen"))

    if control:
        robot_type = cfg("robot_type")
        params = os.path.join(get_package_share_directory("robot_controller"), "config", f"{robot_type}.yaml")
        topics = {"cmd_vel_topic": CMD_VEL, "posture_cmd_topic": POSTURE}
        actions += [
            Node(package="robot_controller", executable="robot_controller_node", name="robot_controller",
                 output="screen",
                 parameters=[params, {"robot_type": robot_type, "remote_cmd_vel_topic": "", **topics}],
                 remappings=[("/robot_controller/command", "/stvln/robot_controller/command")]),
            Node(package="go2w_keyboard_control", executable="go2_cmd_vel_node", name="go2_cmd_vel_node",
                 output="screen",
                 parameters=[params, {**topics, "auto_prepare_motion": cfg("auto_prepare") == "true",
                                      "goal_point_topic": "", "goal_pose_topic": ""}]),
        ]
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("instruction", default_value="",
                              description="first English instruction (R2R / RxR style); later ones on "
                                          "/janus/instruction (stvln.sh janus_go \"...\")"),
        DeclareLaunchArgument("brain", default_value=os.environ.get("STVLN_JANUS_URL", "http://192.168.110.87:8088")),
        DeclareLaunchArgument("control", default_value="false",
                              description="true: drive the Go2-W (robot_controller + go2_cmd_vel_node)"),
        DeclareLaunchArgument("on_blocked", default_value="turn",
                              description="MOVE_FORWARD into an obstacle: turn (toward the emptier side) or stop"),
        DeclareLaunchArgument("max_steps", default_value="400"),
        DeclareLaunchArgument("verify", default_value="true",
                              description="confirm each STOP with qwen3-vl on the .209 Ollama (STVLN_VLM_URL); "
                                          "a rejected STOP turns 15° and goes on"),
        DeclareLaunchArgument("robot_type", default_value="unitree_go2w"),
        DeclareLaunchArgument("auto_prepare", default_value="true",
                              description="StandUp + SpeedLevel(1) on the first motion command if not standing"),
        DeclareLaunchArgument("camera", default_value="true"),
        DeclareLaunchArgument("battery", default_value="true"),
        DeclareLaunchArgument("model", default_value="true"),
        DeclareLaunchArgument("start_driver", default_value="true"),
        DeclareLaunchArgument("start_lio", default_value="true"),
        OpaqueFunction(function=_actions),
    ])
