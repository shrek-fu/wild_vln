import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_PKG_SRC = Path(__file__).resolve().parents[1]
if (_PKG_SRC / "scene_memory" / "paths.py").is_file():
    sys.path.insert(0, str(_PKG_SRC))
from scene_memory import paths  # noqa: E402


def generate_launch_description() -> LaunchDescription:
    declared = [
        DeclareLaunchArgument("image_topic", default_value="/insta360/equirect/image/compressed"),
        DeclareLaunchArgument("calib_dir", default_value=str(paths.CALIB)),
        DeclareLaunchArgument("input", default_value="views"),
        DeclareLaunchArgument("checkpoint", default_value=""),
        DeclareLaunchArgument("device", default_value=""),
        DeclareLaunchArgument("mode", default_value="tiles"),
        DeclareLaunchArgument("names", default_value=""),
    ]
    arguments = [
        "--image_topic", LaunchConfiguration("image_topic"),
        "--calib_dir", LaunchConfiguration("calib_dir"),
        "--input", LaunchConfiguration("input"),
        "--mode", LaunchConfiguration("mode"),
        "--names", LaunchConfiguration("names"),
        "--device", LaunchConfiguration("device"),
    ]
    # Empty checkpoint keeps the node's default weights path.
    return LaunchDescription(
        declared
        + [
            Node(
                package="scene_memory",
                executable="detector_node",
                name="detector_live",
                output="screen",
                arguments=arguments,
            )
        ]
    )
