"""
qi_move_base.launch.py

Launches the navigation stack with selectable scope via `mode`:
  - all     : global planning + local planning + qi_move_base FSM
  - global  : global_planner.launch.py + qi_move_base + clicked2goal
  - local   : local_planner.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

MIN_GOAL_TOLERANCE_XY = 0.2


def generate_launch_description():

    # ── Package share directories ────────────────────────────────────────────
    ql_share = get_package_share_directory('qi_move_base')
    gp_share = get_package_share_directory('global_planner')
    lp_share = get_package_share_directory('local_planner')

    # ── Launch arguments ─────────────────────────────────────────────────────
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='all',
        description="Navigation stack mode: 'all', 'global', or 'local'",
    )
    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value='',
        description='Path to the map PCD file')
    ground_file_arg = DeclareLaunchArgument(
        'ground_file',
        default_value='',
        description='Path to the ground / traversable PCD file')
    ground_height_arg = DeclareLaunchArgument(
        'ground_height',
        default_value='0.0',
        description='Height of the ground plane')
    max_speed_arg = DeclareLaunchArgument(
        'max_speed',
        default_value='0.5',
        description='Maximum speed of the robot in m/s')
    goal_tolerance_xy_arg = DeclareLaunchArgument(
        'goal_tolerance_xy',
        default_value='',
        description='XY goal tolerance in meters; empty keeps yaml defaults',
    )
    goal_tolerance_yaw_arg = DeclareLaunchArgument(
        'goal_tolerance_yaw',
        default_value='',
        description='Yaw goal tolerance in radians; empty keeps yaml defaults',
    )
    cmd_vel_topic_arg = DeclareLaunchArgument(
        'cmd_vel_topic',
        default_value='/cmd_vel',
        description='Velocity output topic used by p2p_move_base/localPlanner/pathFollower',
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time (for bag replay)',
    )
    obstacle_topic_arg = DeclareLaunchArgument(
        'obstacle_topic',
        default_value='/non_ground_in_map',
        description='Local planner obstacle cloud (map): /non_ground_in_map (linefit) or /stvln/local_obstacles',
    )
    use_no_entry_layer_arg = DeclareLaunchArgument(
        'use_no_entry_layer',
        default_value='false',
        description='Use path_planning_on_no_entry_layer.yaml when true; default is static-layer config',
    )
    no_entry_zone_yaml_arg = DeclareLaunchArgument(
        'no_entry_zone_yaml',
        default_value='',
        description='Absolute path to no_entry_layer.yaml; forwarded to global_planner/perception',
    )

    mode = LaunchConfiguration('mode')
    map_file = LaunchConfiguration('map_file')
    ground_file = LaunchConfiguration('ground_file')
    ground_height = LaunchConfiguration('ground_height')
    max_speed = LaunchConfiguration('max_speed')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_no_entry_layer = LaunchConfiguration('use_no_entry_layer')
    no_entry_zone_yaml = LaunchConfiguration('no_entry_zone_yaml')

    enable_global = IfCondition(
        PythonExpression(["'", mode, "' == 'all' or '", mode, "' == 'global'"])
    )
    enable_local = IfCondition(
        PythonExpression(["'", mode, "' == 'all' or '", mode, "' == 'local'"])
    )

    ql_params_file = os.path.join(ql_share, 'config', 'qi_move_base.yaml')

    def _tolerance_actions(context, *args, **kwargs):
        xy_text = LaunchConfiguration('goal_tolerance_xy').perform(context).strip()
        yaw_text = LaunchConfiguration('goal_tolerance_yaw').perform(context).strip()
        p2p_parameters = [
            ql_params_file,
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
        if xy_text:
            xy = max(MIN_GOAL_TOLERANCE_XY, float(xy_text))
            xy_text = str(xy)
            p2p_parameters.append({'goal_reached_dis': xy})
        if yaw_text:
            p2p_parameters.append({'heading_align_thre': float(yaw_text)})

        local_arguments = {
            'use_sim_time': use_sim_time,
            'ground_height': ground_height,
            'max_speed': max_speed,
            'cmd_vel_topic': cmd_vel_topic,
            'obstacle_topic': LaunchConfiguration('obstacle_topic'),
            'goal_tolerance_xy': xy_text,
            'goal_tolerance_yaw': yaw_text,
        }
        qi_move_base_node = Node(
            package='qi_move_base',
            executable='qi_move_base_node',
            output='screen',
            parameters=p2p_parameters,
            remappings=[
                ('/state_estimation', '/localization'),
                ('/cmd_vel', cmd_vel_topic),
                ('/global_path', '/ql/nav/global_path'),
            ],
            condition=enable_global,
        )
        local_planner_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(lp_share, 'launch', 'local_planner.launch.py')
            ),
            launch_arguments=local_arguments.items(),
            condition=enable_local,
        )
        return [qi_move_base_node, local_planner_launch]

    clicked2goal_node = Node(
        package='qi_move_base',
        executable='clicked2goal.py',
        name='clicked2goal',
        output='screen',
        condition=enable_global,
    )

    # ── Global stack: delegate to global_planner.launch.py ───────────────────
    global_planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gp_share, 'launch', 'global_planner.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'map_file': map_file,
            'ground_file': ground_file,
            'ground_height': ground_height,
            'use_no_entry_layer': use_no_entry_layer,
            'no_entry_zone_yaml': no_entry_zone_yaml,
        }.items(),
        condition=enable_global,
    )

    return LaunchDescription([
        mode_arg,
        use_sim_time_arg,
        use_no_entry_layer_arg,
        no_entry_zone_yaml_arg,
        map_file_arg,
        ground_file_arg,
        ground_height_arg,
        max_speed_arg,
        goal_tolerance_xy_arg,
        goal_tolerance_yaw_arg,
        cmd_vel_topic_arg,
        obstacle_topic_arg,
        OpaqueFunction(function=_tolerance_actions),
        clicked2goal_node,
        global_planner_launch,
    ])
