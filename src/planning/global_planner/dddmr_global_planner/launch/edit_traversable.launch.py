import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    ql_share = get_package_share_directory('qi_move_base')
    traversable_file = LaunchConfiguration('traversable_file').perform(context).strip()
    rviz = LaunchConfiguration('rviz').perform(context).strip().lower() == 'true'
    rviz_cfg = os.path.join(ql_share, 'rviz', 'nav3d.rviz')
    nodes = [
        Node(
            package='global_planner',
            executable='edit_traversable.py',
            name='traversable_editor',
            output='screen',
            parameters=[{
                'traversable_file': traversable_file,
                'global_frame': 'map',
            }],
        ),
    ]
    if rviz:
        nodes.append(
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', rviz_cfg],
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'traversable_file',
            default_value='',
            description='Absolute path to *_traversable.pcd',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Open RViz with intensity-colored /map/dGraph',
        ),
        OpaqueFunction(function=_setup),
    ])
