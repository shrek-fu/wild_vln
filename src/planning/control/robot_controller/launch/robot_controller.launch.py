import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from robot_controller.command_ids import CONFIG_FILES, is_unitree, normalize_robot_type

_UNITREE_NODE = {
    'unitree_b2': ('b2_sport_control', 'b2_cmd_vel_node'),
    'unitree_go2': ('go2w_keyboard_control', 'go2_cmd_vel_node'),
    'unitree_go2w': ('go2w_keyboard_control', 'go2_cmd_vel_node'),
}


def _config_path(robot_type: str) -> str:
    share = get_package_share_directory('robot_controller')
    return os.path.join(share, 'config', CONFIG_FILES[robot_type])


def generate_launch_description():
    robot_type_arg = DeclareLaunchArgument(
        'robot_type',
        default_value='chitu',
        description='chitu | zsl1 | ysc_lite3 | unitree_b2 | unitree_go2 | unitree_go2w',
    )

    def build_nodes(context):
        robot_type = normalize_robot_type(LaunchConfiguration('robot_type').perform(context))
        params_file = _config_path(robot_type)
        actions = [
            Node(
                package='robot_controller',
                executable='robot_controller_node',
                name='robot_controller',
                output='screen',
                parameters=[params_file, {'robot_type': robot_type}],
            )
        ]
        if not is_unitree(robot_type):
            return actions

        package_name, executable = _UNITREE_NODE[robot_type]
        try:
            get_package_share_directory(package_name)
        except PackageNotFoundError:
            actions.append(
                LogInfo(
                    msg=(
                        f'[robot_controller] {package_name} not found; '
                        'source robots/unitree_ws before launching unitree types'
                    )
                )
            )
            return actions

        actions.append(
            Node(
                package=package_name,
                executable=executable,
                name=executable,
                output='screen',
                parameters=[
                    params_file,
                    {
                        'cmd_vel_topic': '/cmd_vel',
                        'posture_cmd_topic': '/unitree/posture_cmd',
                        'auto_prepare_motion': False,
                    },
                ],
            )
        )
        return actions

    return LaunchDescription([
        robot_type_arg,
        OpaqueFunction(function=build_nodes),
    ])
