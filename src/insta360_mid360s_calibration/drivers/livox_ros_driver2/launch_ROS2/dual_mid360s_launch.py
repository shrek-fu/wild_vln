"""Two MID-360S, one topic per lidar. Config lives in this calibration project."""
import os

from launch import LaunchDescription
from launch_ros.actions import Node

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONFIG = os.path.join(PROJECT, "config", "MID360s_dual.json")


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="livox_ros_driver2",
            executable="livox_ros_driver2_node",
            name="livox_lidar_publisher",
            output="screen",
            parameters=[{
                "xfer_format": 0,
                "multi_topic": 1,
                "data_src": 0,
                "publish_freq": 10.0,
                "output_data_type": 0,
                "frame_id": "livox_frame",
                "lvx_file_path": "/tmp/livox_unused.lvx",
                "user_config_path": CONFIG,
                "cmdline_input_bd_code": "livox0000000001",
            }],
        )
    ])
