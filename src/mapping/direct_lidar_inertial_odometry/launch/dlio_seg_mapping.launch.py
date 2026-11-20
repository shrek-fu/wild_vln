#
#   Copyright (c)     
#
#   The Verifiable & Control-Theoretic Robotics (VECTR) Lab
#   University of California, Los Angeles
#
#   Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez
#   Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu
#

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition   
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    current_pkg = FindPackageShare('direct_lidar_inertial_odometry')

    # Set default arguments
    rviz = LaunchConfiguration('rviz', default='false')
    prior_map = LaunchConfiguration('prior_map')
    prior_ground = LaunchConfiguration('prior_ground')
    prior_nonground = LaunchConfiguration('prior_nonground')
    wait_for_localization = LaunchConfiguration('wait_for_localization')

    # Define arguments
    declare_rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value=rviz,
        description='Launch RViz'
    )
    declare_prior_map_arg = DeclareLaunchArgument(
        'prior_map',
        default_value='',
        description='Prior full-scene PCD. Empty starts a new map.',
    )
    declare_prior_ground_arg = DeclareLaunchArgument(
        'prior_ground',
        default_value='',
        description='Prior ground PCD.',
    )
    declare_prior_nonground_arg = DeclareLaunchArgument(
        'prior_nonground',
        default_value='',
        description='Prior non-ground PCD.',
    )
    declare_wait_for_localization_arg = DeclareLaunchArgument(
        'wait_for_localization',
        default_value='false',
        description='Insert new scans only after /localize/status localization_success.',
    )

    # Load parameters
    dlio_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'dlio.yaml'])
    dlio_params_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'params_mapping.yaml'])

    # DLIO Mapping Node
    dlio_map_node = Node(
        package='direct_lidar_inertial_odometry',
        executable='dlio_multi_map_node',
        output='screen',
        parameters=[
            dlio_yaml_path,
            dlio_params_yaml_path,
            {
                'map/prior_map': prior_map,
                'map/prior_ground': prior_ground,
                'map/prior_nonground': prior_nonground,
                'map/wait_for_localization': ParameterValue(
                    wait_for_localization, value_type=bool,
                ),
            },
        ],
        remappings=[
            ('cloud_map', '/cloud_registered'),
            ('cloud_ground', '/ground_in_odom'),
            ('cloud_non_ground', '/non_ground_in_odom'),
            ('odom', '/ql/nav/odometry'),
            ('map', '/ql/nav/laser_map'),
        ]
    )

    # RViz node
    rviz_config_path = PathJoinSubstitution([current_pkg, 'launch', 'dlio.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='dlio_rviz',
        arguments=['-d', rviz_config_path],
        output='log',
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    return LaunchDescription([
        declare_rviz_arg,
        declare_prior_map_arg,
        declare_prior_ground_arg,
        declare_prior_nonground_arg,
        declare_wait_for_localization_arg,
        dlio_map_node,
        rviz_node
    ])
