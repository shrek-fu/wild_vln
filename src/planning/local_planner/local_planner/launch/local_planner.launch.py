import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.parameter_descriptions import ParameterValue

MIN_GOAL_TOLERANCE_XY = 0.2


def generate_launch_description():
    pkg_share = get_package_share_directory('local_planner')
    config_file = os.path.join(pkg_share, 'config', 'local_planner.yaml')
    paths_folder = os.path.join(pkg_share, 'paths')

    ground_height_arg = DeclareLaunchArgument(
        "ground_height", default_value="0.0", description="Height of the ground plane")

    max_speed_arg = DeclareLaunchArgument(
        "max_speed", default_value="0.5", description="Maximum speed of the robot in m/s")

    goal_tolerance_xy_arg = DeclareLaunchArgument(
        'goal_tolerance_xy',
        default_value='',
        description='XY arrival distance in meters; also keeps pathFollower stop inside that radius',
    )
    goal_tolerance_yaw_arg = DeclareLaunchArgument(
        'goal_tolerance_yaw',
        default_value='',
        description='Yaw goal tolerance in radians; empty keeps yaml defaults',
    )

    cmd_vel_topic_arg = DeclareLaunchArgument(
        'cmd_vel_topic',
        default_value='/cmd_vel',
        description='Velocity output topic used by localPlanner/pathFollower',
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time (for bag replay)',
    )

    obstacle_topic_arg = DeclareLaunchArgument(
        'obstacle_topic',
        default_value='/non_ground_in_map',
        description="Obstacle cloud (map) for /registered_scan + /terrain_map: linefit's /non_ground_in_map or "
                    "the freespace obstacles /stvln/local_obstacles",
    )

    ground_height = LaunchConfiguration("ground_height")
    max_speed = LaunchConfiguration("max_speed")
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    use_sim_time = LaunchConfiguration('use_sim_time')
    obstacle_topic = LaunchConfiguration('obstacle_topic')

    obstacle_height = ParameterValue(PythonExpression([ground_height, ' + 0.2']), value_type=float)

    def _nodes(context, *args, **kwargs):
        xy_text = LaunchConfiguration('goal_tolerance_xy').perform(context).strip()
        yaw_text = LaunchConfiguration('goal_tolerance_yaw').perform(context).strip()
        planner_params = [
            config_file,
            {'use_sim_time': use_sim_time},
            {'pathFolder': paths_folder},
            {'maxSpeed': max_speed},
            {'groundHeightThre': ground_height},
            {'obstacleHeightThre': obstacle_height},
        ]
        follower_params = [
            config_file,
            {'use_sim_time': use_sim_time},
            {'maxSpeed': max_speed},
        ]
        if xy_text:
            xy = max(MIN_GOAL_TOLERANCE_XY, float(xy_text))
            planner_params.append({'goalReachedDis': xy})
            follower_params.append({'stopDisThre': min(0.3, max(0.02, xy * 0.5))})
        if yaw_text:
            planner_params.append({'headingAlignThre': float(yaw_text)})

        planner = Node(
            package='local_planner',
            executable='localPlanner',
            name='localPlanner',
            output='screen',
            parameters=planner_params,
            remappings=[
                ('/state_estimation', '/localization'),
                ('/registered_scan',  obstacle_topic),
                ('/terrain_map',      obstacle_topic),
                ('/added_obstacles',  '/no_entry_obstacle_pc'),
                ('/global_path',      '/ql/nav/global_path'),
                ('/cmd_vel',          cmd_vel_topic),
            ],
        )
        follower = Node(
            package='local_planner',
            executable='pathFollower',
            name='pathFollower',
            output='screen',
            parameters=follower_params,
            remappings=[
                ('/state_estimation', '/localization'),
                ('/cmd_vel',          cmd_vel_topic),
            ],
        )
        return [planner, follower]

    return LaunchDescription([
        ground_height_arg,
        max_speed_arg,
        goal_tolerance_xy_arg,
        goal_tolerance_yaw_arg,
        cmd_vel_topic_arg,
        use_sim_time_arg,
        obstacle_topic_arg,
        OpaqueFunction(function=_nodes),
    ])
