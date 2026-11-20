"""Go2-W model on /robot_description + static TF base_link -> go2w_* links (legs at the stand pose).

Foxglove 3D panel: add a URDF layer with source "Topic" /robot_description. The meshes are package:// URIs
fetched through the bridge's assets capability: the robot's foxglove-bridge service (8766) resolves them via
the drop-in /etc/systemd/system/foxglove-bridge.service.d/stvln-go2w-model.conf (AMENT_PREFIX_PATH +
CycloneDDS on lo and the default interface); stvln.sh foxglove (8767) works without it.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    urdf = Path(get_package_share_directory("go2w_description")) / "urdf" / "go2w.urdf"
    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher", name="go2w_state_publisher",
             output="screen", parameters=[{"robot_description": urdf.read_text(), "publish_frequency": 1.0}]),
    ])
