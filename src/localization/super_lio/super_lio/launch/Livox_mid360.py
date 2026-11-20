import math
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def livox_to_base_link(lidar_x, lidar_y, lidar_height, pitch_deg):
    """Pose of base_link in livox_frame.

    lidar_x, lidar_y and lidar_height are the livox_frame origin in base_link.
    pitch_deg is positive when livox +X pitches down about base_link +Y.
    """
    pitch = math.radians(float(pitch_deg))
    cosine = math.cos(pitch)
    sine = math.sin(pitch)
    tx = -float(lidar_x) * cosine + float(lidar_height) * sine
    ty = -float(lidar_y)
    tz = -float(lidar_x) * sine - float(lidar_height) * cosine
    half = pitch / 2.0
    return (tx, ty, tz, 0.0, -math.sin(half), 0.0, math.cos(half))


def _format_component(value):
    return f'{float(value):.6f}'


def _launch_base_link_tf(context):
    robot_type = LaunchConfiguration('robot_type').perform(context).strip().lower() or 'zsl1'
    pkg_super_lio = get_package_share_directory('super_lio')
    mount_path = os.path.join(pkg_super_lio, 'config', 'base_link_mount.yaml')
    with open(mount_path, 'r', encoding='utf-8') as mount_file:
        mounts = yaml.safe_load(mount_file) or {}
    mount = mounts.get(robot_type)
    if not isinstance(mount, dict):
        known = ', '.join(sorted(str(key) for key in mounts))
        return [
            LogInfo(msg=f'[Livox_mid360] Unsupported robot_type: {robot_type}. Supported: {known}'),
            Shutdown(reason=f'Unsupported robot_type: {robot_type}'),
        ]

    tx, ty, tz, qx, qy, qz, qw = livox_to_base_link(
        mount['lidar_x'], mount['lidar_y'], mount['lidar_height'], mount['pitch_deg'])
    return [
        LogInfo(
            msg=(
                f'[Livox_mid360] {robot_type} livox_frame -> base_link: '
                f'{_format_component(tx)} {_format_component(ty)} {_format_component(tz)} '
                f'{_format_component(qx)} {_format_component(qy)} {_format_component(qz)} {_format_component(qw)}'
            )
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='imu_to_base_link',
            arguments=[
                _format_component(tx),
                _format_component(ty),
                _format_component(tz),
                _format_component(qx),
                _format_component(qy),
                _format_component(qz),
                _format_component(qw),
                'livox_frame',
                'base_link',
            ],
        ),
    ]


def generate_launch_description():
    pkg_super_lio = get_package_share_directory('super_lio')
    config_yaml = os.path.join(pkg_super_lio, 'config', 'livox_360.yaml')
    rviz_config_file = os.path.join(pkg_super_lio, 'rviz', 'lio.rviz')

    declare_rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to start RVIZ2'
    )
    rviz_flag = LaunchConfiguration('rviz')

    declare_save_map_arg = DeclareLaunchArgument(
        'save_map',
        default_value='false',
        description='Whether to save the map'
    )
    save_map_flag = LaunchConfiguration('save_map')

    declare_pub_map_arg = DeclareLaunchArgument(
        'pub_map',
        default_value='false',
        description='Whether to publish the map'
    )
    pub_map_flag = LaunchConfiguration('pub_map')

    declare_robot_type_arg = DeclareLaunchArgument(
        'robot_type',
        default_value='zsl1',
        description='Robot type used to look up livox_frame to base_link in base_link_mount.yaml',
    )

    super_lio_node = Node(
        package='super_lio',
        executable='super_lio_node',
        name='super_lio_node',
        output='screen',
        parameters=[config_yaml,
                    {'lio.map.save_map': save_map_flag},
                    {'lio.output.ml_map': pub_map_flag}],
        arguments=['--ros-args', '--log-level', 'info'],
        remappings=[
            ('/lio/cloud_world', '/cloud_registered'),
            ("/lio/base/odom", "/ql/nav/odometry"),
            ("/lio/cloud_all", "/ql/nav/laser_map"),
        ]
    )

    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='super_lio',
        arguments=['-d', rviz_config_file, '--ros-args', '--log-level', 'warn'],
        condition=IfCondition(rviz_flag),
        remappings=[
            ('/lio/cloud_world', '/cloud_registered'),
        ]
    )

    ld = LaunchDescription()
    ld.add_action(declare_rviz_arg)
    ld.add_action(declare_save_map_arg)
    ld.add_action(declare_pub_map_arg)
    ld.add_action(declare_robot_type_arg)
    ld.add_action(super_lio_node)
    ld.add_action(rviz2_node)
    ld.add_action(OpaqueFunction(function=_launch_base_link_tf))
    return ld
