
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue



def generate_launch_description():
    # Getting directories and launch-files
    bringup_dir = get_package_share_directory('linefit_ground_segmentation_ros')
    params_file = os.path.join(bringup_dir, 'launch', 'segmentation_params.yaml')

    ground_height = LaunchConfiguration("ground_height")

    declare_ground_height_cmd = DeclareLaunchArgument(
        "ground_height", default_value="0.0", description="Height of the ground plane")

    sensor_height = ParameterValue(PythonExpression(['-', ground_height]), value_type=float)

    # Nodes launching commands
    node_start_cmd = Node(
            package='linefit_ground_segmentation_ros',
            executable='ground_segmentation_node',
            output='screen',
            arguments=['--ros-args', '--log-level', 'error'],
            parameters=[params_file,
                        {'sensor_height': sensor_height},
            ],)


    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_ground_height_cmd)
    ld.add_action(node_start_cmd)


    return ld
