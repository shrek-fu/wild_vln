import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    ql_share = get_package_share_directory('qi_move_base')
    dlio_share = get_package_share_directory('direct_lidar_inertial_odometry')
    lio_share = get_package_share_directory('super_lio')
    linefit_share = get_package_share_directory('linefit_ground_segmentation_ros')
    localization_share = get_package_share_directory('localization_cpp')
    cloud_share = get_package_share_directory('cloud_preprocess')

    default_rviz_cfg = os.path.join(ql_share, 'rviz', 'nav3d.rviz')

    mode = LaunchConfiguration('mode')
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_prefix = LaunchConfiguration('map_prefix')
    map_pcd = LaunchConfiguration('map_pcd')
    ground_height = LaunchConfiguration('ground_height')
    rviz = LaunchConfiguration('rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')
    robot_type = LaunchConfiguration('robot_type')

    is_mapping = IfCondition(PythonExpression(["'", mode, "' == 'mapping'"]))
    is_relocalize = IfCondition(PythonExpression(["'", mode, "' == 'relocalize'"]))
    is_incremental = IfCondition(PythonExpression(["'", mode, "' == 'incremental'"]))
    is_not_incremental = IfCondition(PythonExpression(["'", mode, "' != 'incremental'"]))
    is_odom_ground = IfCondition(PythonExpression([
        "'", mode, "' == 'mapping' or '", mode, "' == 'incremental'",
    ]))

    declare_mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='relocalize',
        description="SLAM mode: 'mapping', 'relocalize', or 'incremental'",
    )
    declare_robot_type_arg = DeclareLaunchArgument(
        'robot_type',
        default_value='zsl1',
        description='Robot type passed to the Livox static TF and body cloud',
    )
    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time (for bag replay)',
    )
    declare_map_prefix_arg = DeclareLaunchArgument(
        'map_prefix',
        default_value='',
        description='Map file prefix. Example: /path/to/01 -> /path/to/01_map.pcd',
    )
    declare_map_pcd_arg = DeclareLaunchArgument(
        'map_pcd',
        default_value=[map_prefix, '_map.pcd'],
        description='Path to *_map.pcd (relocalize mode)',
    )
    declare_ground_height_arg = DeclareLaunchArgument(
        'ground_height',
        default_value='0.0',
        description='Height of the ground plane',
    )
    declare_rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Enable RViz',
    )
    declare_rviz_cfg_arg = DeclareLaunchArgument(
        'rviz_cfg',
        default_value=default_rviz_cfg,
        description='RViz config path',
    )

    livox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(cloud_share, 'launch', 'lidar.launch.py')
        ),
        launch_arguments={
            'robot_type': robot_type,
            'enable_livox_transform': 'true',
        }.items(),
    )

    lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(lio_share, 'launch', 'Livox_mid360.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'rviz': 'false',
            'pub_map': 'false',
        }.items(),
        condition=is_not_incremental,
    )

    reloc_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(lio_share, 'launch', 'relocation.py')
        ),
        launch_arguments={
            'rviz': 'false',
            'octomap': 'false',
            'map_pcd': [map_prefix, '_map.pcd'],
        }.items(),
        condition=is_incremental,
    )

    mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(dlio_share, 'launch', 'dlio_seg_mapping.launch.py')
        ),
        launch_arguments={
            'rviz': 'false',
        }.items(),
        condition=is_mapping,
    )

    incremental_mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(dlio_share, 'launch', 'dlio_seg_mapping.launch.py')
        ),
        launch_arguments={
            'rviz': 'false',
            'prior_map': [map_prefix, '_map.pcd'],
            'prior_ground': [map_prefix, '_ground.pcd'],
            'prior_nonground': [map_prefix, '_nonground.pcd'],
            'wait_for_localization': 'true',
        }.items(),
        condition=is_incremental,
    )

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(localization_share, 'launch', 'localization.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'map': map_pcd,
        }.items(),
        condition=is_relocalize,
    )

    linefit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(linefit_share, 'launch', 'segmentation.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'ground_height': ground_height,
        }.items(),
    )

    trans_ground_to_odom_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(cloud_share, 'launch', 'trans_ground_to_odom.launch.py')
        ),
        launch_arguments={
            'ground_height': ground_height,
        }.items(),
        condition=is_odom_ground,
    )

    trans_ground_to_map_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(cloud_share, 'launch', 'trans_ground_to_map.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
        }.items(),
        condition=is_relocalize,
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_cfg],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(rviz),
    )

    return LaunchDescription([
        declare_mode_arg,
        declare_robot_type_arg,
        declare_use_sim_time_arg,
        declare_map_prefix_arg,
        declare_map_pcd_arg,
        declare_ground_height_arg,
        declare_rviz_arg,
        declare_rviz_cfg_arg,
        livox_launch,
        rviz_node,
        lio_launch,
        reloc_launch,
        mapping_launch,
        incremental_mapping_launch,
        localization_launch,
        linefit_launch,
        trans_ground_to_odom_launch,
        trans_ground_to_map_launch,
    ])
