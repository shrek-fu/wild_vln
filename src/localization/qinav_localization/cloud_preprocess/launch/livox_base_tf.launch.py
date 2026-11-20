from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from cloud_preprocess.mount import format_component, livox_to_base_link, load_mount


def _launch_setup(context):
    robot_type = LaunchConfiguration('robot_type').perform(context).strip().lower() or 'zsl1'
    mount, mounts = load_mount(robot_type)
    if mount is None:
        known = ', '.join(sorted(str(key) for key in mounts))
        return [
            LogInfo(msg=f'[livox_base_tf] Unsupported robot_type: {robot_type}. Supported: {known}'),
            Shutdown(reason=f'Unsupported robot_type: {robot_type}'),
        ]

    tx, ty, tz, qx, qy, qz, qw = livox_to_base_link(
        mount['lidar_x'], mount['lidar_y'], mount['lidar_height'], mount['pitch_deg'])
    return [
        LogInfo(
            msg=(
                f'[livox_base_tf] {robot_type} livox_frame -> base_link: '
                f'{format_component(tx)} {format_component(ty)} {format_component(tz)} '
                f'{format_component(qx)} {format_component(qy)} {format_component(qz)} '
                f'{format_component(qw)}'
            )
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='imu_to_base_link',
            arguments=[
                format_component(tx),
                format_component(ty),
                format_component(tz),
                format_component(qx),
                format_component(qy),
                format_component(qz),
                format_component(qw),
                'livox_frame',
                'base_link',
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_type',
            default_value='zsl1',
            description='Robot type used to look up livox_frame to base_link',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
