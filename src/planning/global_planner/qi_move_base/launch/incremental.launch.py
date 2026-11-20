import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    ql_share = get_package_share_directory('qi_move_base')
    default_rviz_cfg = os.path.join(ql_share, 'rviz', 'mapping.rviz')

    ground_height = LaunchConfiguration('ground_height')
    rviz = LaunchConfiguration('rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_prefix = LaunchConfiguration('map_prefix')
    robot_type = LaunchConfiguration('robot_type')

    declare_ground_height_arg = DeclareLaunchArgument(
        'ground_height',
        default_value='0.0',
        description='Height of the ground plane',
    )
    declare_rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Enable RViz',
    )
    declare_rviz_cfg_arg = DeclareLaunchArgument(
        'rviz_cfg',
        default_value=default_rviz_cfg,
        description='RViz config path for incremental mapping',
    )
    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time (for bag replay)',
    )
    declare_map_prefix_arg = DeclareLaunchArgument(
        'map_prefix',
        default_value='',
        description='Full-map prefix. Example: /path/to/lobby -> lobby_map.pcd',
    )
    declare_robot_type_arg = DeclareLaunchArgument(
        'robot_type',
        default_value='zsl1',
        description='Robot type passed to super_lio relocation',
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ql_share, 'launch', 'slam.launch.py')
        ),
        launch_arguments={
            'mode': 'incremental',
            'use_sim_time': use_sim_time,
            'ground_height': ground_height,
            'rviz': rviz,
            'rviz_cfg': rviz_cfg,
            'map_prefix': map_prefix,
            'robot_type': robot_type,
        }.items(),
    )

    return LaunchDescription([
        declare_ground_height_arg,
        declare_rviz_arg,
        declare_rviz_cfg_arg,
        declare_use_sim_time_arg,
        declare_map_prefix_arg,
        declare_robot_type_arg,
        slam_launch,
    ])
