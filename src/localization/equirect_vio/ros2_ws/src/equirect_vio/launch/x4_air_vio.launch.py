from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("equirect_vio")
    # Match MediaSDK template output (1920×960), same as calib project.
    default_config = os.path.join(pkg_share, "config", "x4_air_equirect_1920.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            # Stitched equirectangular from x4_air_equirect.launch.xml
            DeclareLaunchArgument("image_topic", default_value="/equirectangular/image"),
            DeclareLaunchArgument("imu_topic", default_value="/imu/data_raw"),
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            DeclareLaunchArgument("base_frame", default_value="imu"),
            DeclareLaunchArgument("publish_tf", default_value="true"),
            Node(
                package="equirect_vio",
                executable="equirect_vio_node",
                name="equirect_vio",
                output="screen",
                parameters=[
                    {
                        "config": LaunchConfiguration("config"),
                        "image_topic": LaunchConfiguration("image_topic"),
                        "imu_topic": LaunchConfiguration("imu_topic"),
                        "odom_frame": LaunchConfiguration("odom_frame"),
                        "base_frame": LaunchConfiguration("base_frame"),
                        "publish_tf": LaunchConfiguration("publish_tf"),
                    }
                ],
            ),
        ]
    )