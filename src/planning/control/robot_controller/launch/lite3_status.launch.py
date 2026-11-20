from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_controller',
            executable='lite3_status_node',
            name='lite3_status',
            output='screen',
        ),
    ])
