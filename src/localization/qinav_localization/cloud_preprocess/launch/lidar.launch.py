import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    package_share = get_package_share_directory('cloud_preprocess')
    livox_pkg = get_package_share_directory('livox_ros_driver2')
    launch_file = LaunchConfiguration('launch_file')
    enable_livox_transform = LaunchConfiguration('enable_livox_transform')
    robot_type = LaunchConfiguration('robot_type')

    return LaunchDescription([
        DeclareLaunchArgument(
            'launch_file',
            default_value='msg_MID360s_launch.py',
            description='Livox launch file name',
        ),
        DeclareLaunchArgument(
            'enable_livox_transform',
            default_value='false',
            description='Transform /livox/lidar into base_link and publish /ql/nav/cloud_in_body',
        ),
        DeclareLaunchArgument(
            'robot_type',
            default_value='zsl1',
            description='Robot type used to look up the Livox mount',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    livox_pkg,
                    'launch_ROS2',
                    launch_file,
                ])
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(package_share, 'launch', 'livox_base_tf.launch.py')
            ),
            launch_arguments={
                'robot_type': robot_type,
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(package_share, 'launch', 'trans_livox.launch.py')
            ),
            launch_arguments={
                'robot_type': robot_type,
            }.items(),
            condition=IfCondition(enable_livox_transform),
        ),
    ])
