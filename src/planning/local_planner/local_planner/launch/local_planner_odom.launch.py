import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.actions import DeclareLaunchArgument
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('local_planner')
    config_file = os.path.join(pkg_share, 'config', 'local_planner.yaml')
    paths_folder = os.path.join(pkg_share, 'paths')

    ground_height_arg = DeclareLaunchArgument(
        "ground_height", default_value="-0.25", description="Height of the ground plane")
    
    max_speed_arg = DeclareLaunchArgument(
        "max_speed", default_value="0.7", description="Maximum speed of the robot in m/s")

    cmd_vel_topic_arg = DeclareLaunchArgument(
        'cmd_vel_topic',
        default_value='/robot_cmd_vel',
        description='Velocity output topic used by localPlanner/pathFollower',
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time (for bag replay)',
    )

    ground_height = LaunchConfiguration("ground_height")
    max_speed = LaunchConfiguration("max_speed")
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    use_sim_time = LaunchConfiguration('use_sim_time')

    obstacle_height = ParameterValue(PythonExpression([ground_height, ' + 0.2']), value_type=float)

    return LaunchDescription([
        ground_height_arg,
        max_speed_arg,
        cmd_vel_topic_arg,
        use_sim_time_arg,
        # ── localPlanner ─────────────────────────────────────────────────
        Node(
            package='local_planner',
            executable='localPlanner',
            name='localPlanner',
            output='screen',
            parameters=[
                config_file,
                {'use_sim_time': use_sim_time},
                {'pathFolder': paths_folder},  # paths 路径由 launch 动态填入
                {'maxSpeed': max_speed},
                {'groundHeightThre': ground_height},
                {'obstacleHeightThre': obstacle_height},
            ],
            remappings=[
                ('/state_estimation', '/ql/nav/odometry'),
                ('/registered_scan',  '/cloud_registered'),
                ('/terrain_map',      '/cloud_registered'),
                ('/added_obstacles',  '/no_entry_obstacle_pc'),
                ('/global_path',      '/ql/nav/global_path'),
                ('/cmd_vel',          cmd_vel_topic),
                # ('/way_point',      '/clicked_point'),
                # ('/goal_pose',      '/goal_pose_3d'),
            ],
        ),

        # ── pathFollower ─────────────────────────────────────────────────
        Node(
            package='local_planner',
            executable='pathFollower',
            name='pathFollower',
            output='screen',
            parameters=[
                config_file,
                {'use_sim_time': use_sim_time},
                {'maxSpeed': max_speed},
            ],
            remappings=[
                ('/state_estimation', '/ql/nav/odometry'),
                ('/cmd_vel',          cmd_vel_topic),
            ],
        ),
    ])
