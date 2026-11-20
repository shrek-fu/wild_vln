import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    ql_share = get_package_share_directory('qi_move_base')

    default_rviz_cfg = os.path.join(ql_share, 'rviz', 'nav3d.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time')
    map_prefix = LaunchConfiguration('map_prefix')
    map_pcd = LaunchConfiguration('map_pcd')
    traversable_pcd = LaunchConfiguration('traversable_pcd')
    ground_height = LaunchConfiguration('ground_height')
    max_speed = LaunchConfiguration('max_speed')
    goal_tolerance_xy = LaunchConfiguration('goal_tolerance_xy')
    goal_tolerance_yaw = LaunchConfiguration('goal_tolerance_yaw')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    use_no_entry_layer = LaunchConfiguration('use_no_entry_layer')
    no_entry_zone_yaml = LaunchConfiguration('no_entry_zone_yaml')
    rviz = LaunchConfiguration('rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')

    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time (for bag replay)',
    )
    declare_map_prefix_arg = DeclareLaunchArgument(
        'map_prefix',
        default_value='',
        description='Map file prefix. Example: /path/to/01 -> /path/to/01_map.pcd / /path/to/01_traversable.pcd',
    )
    declare_map_pcd_arg = DeclareLaunchArgument(
        'map_pcd',
        default_value=[map_prefix, '_map.pcd'],
        description='Path to *_map.pcd',
    )
    declare_ground_pcd_arg = DeclareLaunchArgument(
        'ground_pcd',
        default_value=[map_prefix, '_ground.pcd'],
        description='Path to *_ground.pcd (mapping intermediate, unused by nav)',
    )
    declare_non_ground_pcd_arg = DeclareLaunchArgument(
        'non_ground_pcd',
        default_value=[map_prefix, '_nonground.pcd'],
        description='Path to *_nonground.pcd (mapping intermediate, unused by nav)',
    )
    declare_traversable_pcd_arg = DeclareLaunchArgument(
        'traversable_pcd',
        default_value=[map_prefix, '_traversable.pcd'],
        description='Path to *_traversable.pcd (ground xyz + dGraph intensity)',
    )
    declare_ground_height_arg = DeclareLaunchArgument(
        'ground_height',
        default_value='0.0',
        description='Height of the ground plane',
    )
    declare_max_speed_arg = DeclareLaunchArgument(
        'max_speed',
        default_value='0.5',
        description='Maximum speed of the robot in m/s',
    )
    declare_goal_tolerance_xy_arg = DeclareLaunchArgument(
        'goal_tolerance_xy',
        default_value='',
        description='XY goal tolerance in meters; empty keeps yaml defaults',
    )
    declare_goal_tolerance_yaw_arg = DeclareLaunchArgument(
        'goal_tolerance_yaw',
        default_value='',
        description='Yaw goal tolerance in radians; empty keeps yaml defaults',
    )
    declare_cmd_vel_topic_arg = DeclareLaunchArgument(
        'cmd_vel_topic',
        default_value='/cmd_vel',
        description='Velocity output topic for p2p_move_base/localPlanner/pathFollower',
    )
    declare_use_no_entry_layer_arg = DeclareLaunchArgument(
        'use_no_entry_layer',
        default_value='false',
        description='Use path_planning_on_no_entry_layer.yaml when true; default is static-layer config',
    )
    declare_no_entry_zone_yaml_arg = DeclareLaunchArgument(
        'no_entry_zone_yaml',
        default_value=[map_prefix, '_no_entry.yaml'],
        description='Path to no-entry zone yaml; default {map_prefix}_no_entry.yaml',
    )
    declare_rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Enable RViz',
    )
    declare_rviz_cfg_arg = DeclareLaunchArgument(
        'rviz_cfg',
        default_value=default_rviz_cfg,
        description='RViz config path for nav3d flow',
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ql_share, 'launch', 'slam.launch.py')
        ),
        launch_arguments={
            'mode': 'relocalize',
            'use_sim_time': use_sim_time,
            'map_prefix': map_prefix,
            'map_pcd': map_pcd,
            'ground_height': ground_height,
            'rviz': rviz,
            'rviz_cfg': rviz_cfg,
        }.items(),
    )

    qi_move_base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ql_share, 'launch', 'qi_move_base.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'mode': 'all',
            'map_file': '',
            'ground_file': traversable_pcd,
            'ground_height': ground_height,
            'max_speed': max_speed,
            'goal_tolerance_xy': goal_tolerance_xy,
            'goal_tolerance_yaw': goal_tolerance_yaw,
            'cmd_vel_topic': cmd_vel_topic,
            'use_no_entry_layer': use_no_entry_layer,
            'no_entry_zone_yaml': no_entry_zone_yaml,
        }.items(),
    )

    return LaunchDescription([
        declare_use_sim_time_arg,
        declare_map_prefix_arg,
        declare_map_pcd_arg,
        declare_ground_pcd_arg,
        declare_non_ground_pcd_arg,
        declare_traversable_pcd_arg,
        declare_ground_height_arg,
        declare_max_speed_arg,
        declare_goal_tolerance_xy_arg,
        declare_goal_tolerance_yaw_arg,
        declare_cmd_vel_topic_arg,
        declare_use_no_entry_layer_arg,
        declare_no_entry_zone_yaml_arg,
        declare_rviz_arg,
        declare_rviz_cfg_arg,
        slam_launch,
        qi_move_base_launch,
    ])
