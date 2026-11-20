from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ground_to_map_node = Node(
        package="cloud_preprocess",
        executable="ground_trans",
        name="ground_to_map",
        output="screen",
        parameters=[{
            "input_topic":             '/segmentation/ground',
            "output_topic":            '/ground_in_map',
            "non_ground_input_topic":  '/segmentation/obstacle',
            "non_ground_output_topic": '/non_ground_in_map',
            "source_frame":            'base_link',
            "target_frame":            'map',
            "fill_radius":             -1.0,
        }],
    )

    return LaunchDescription([
        ground_to_map_node,
    ])
