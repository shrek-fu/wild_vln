from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ground_height = LaunchConfiguration("ground_height")

    declare_ground_height_cmd = DeclareLaunchArgument(
        "ground_height", default_value="-0.5", description="Height of the ground plane")

    ground_to_odom_node = Node(
        package="cloud_preprocess",
        executable="ground_trans",
        name="ground_to_odom",
        output="screen",
        parameters=[{
            "input_topic":             '/segmentation/ground',
            "output_topic":            '/ground_in_odom',
            "non_ground_input_topic":  '/segmentation/obstacle',
            "non_ground_output_topic": '/non_ground_in_odom',
            "source_frame":            'base_link',
            "target_frame":            'odom',
            "ground_height":           ground_height,
            "fill_radius":             0.8,
            "fill_resolution":         0.15,
        }],
    )

    return LaunchDescription([
        declare_ground_height_cmd,
        ground_to_odom_node,
    ])
