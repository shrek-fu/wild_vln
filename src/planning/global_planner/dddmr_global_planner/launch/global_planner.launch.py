import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _write_no_entry_override_params(no_entry_zone_yaml: str) -> str:
    """Write a real params file so multi-node overrides are not filtered by launch Node name."""
    payload = {
        'perception_3d_global': {
            'ros__parameters': {
                'no_entry_layer.no_entry_zone_pcd_file_dir': no_entry_zone_yaml,
            }
        }
    }
    fd, path = tempfile.mkstemp(prefix='no_entry_zone_override_', suffix='.yaml')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def _launch_setup(context, *args, **kwargs):
    gp_share = get_package_share_directory('global_planner')

    map_file = LaunchConfiguration('map_file')
    ground_file = LaunchConfiguration('ground_file')
    ground_height = LaunchConfiguration('ground_height')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_no_entry_layer = LaunchConfiguration('use_no_entry_layer')
    no_entry_zone_yaml = LaunchConfiguration('no_entry_zone_yaml').perform(context).strip()

    gp_params_no_entry_file = os.path.join(gp_share, 'config', 'path_planning_on_no_entry_layer.yaml')
    gp_params_static_file = os.path.join(gp_share, 'config', 'path_planning_on_static_layer.yaml')
    gp_params_file = PythonExpression([
        "'", gp_params_no_entry_file, "' if '", use_no_entry_layer,
        "' == 'true' else '", gp_params_static_file, "'"
    ])

    map_file_str = LaunchConfiguration('map_file').perform(context).strip()
    pcl_params = {
        'map_file': map_file,
        'ground_file': ground_file,
    }
    if not map_file_str:
        # Baked *_traversable.pcd: keep intensity and point order.
        pcl_params['map_down_sample'] = 0.0
        pcl_params['ground_down_sample'] = 0.0

    common_overrides = {
        'use_sim_time': use_sim_time,
        'ground_height': ground_height,
        'map_file': map_file,
        'ground_file': ground_file,
    }

    global_planner_params = [gp_params_file, common_overrides]
    if no_entry_zone_yaml:
        # Must be a params *file path* (not a nested dict). Nested dicts keyed by
        # perception_3d_global are dropped because this Node action has no matching name.
        global_planner_params.append(_write_no_entry_override_params(no_entry_zone_yaml))

    return [
        Node(
            package='global_planner',
            executable='pcl_publisher',
            name='pcl_publisher',
            output='screen',
            parameters=[
                gp_params_file,
                pcl_params,
            ],
        ),
        Node(
            package='global_planner',
            executable='global_planner_node',
            output='screen',
            parameters=global_planner_params,
            remappings=[
                ('/segmented_cloud_pure', '/segmentation/obstacle'),
                ('/global_path', '/ql/nav/global_path'),
            ],
        ),
    ]


def generate_launch_description():
    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value='',
        description='Path to the map PCD file',
    )
    ground_file_arg = DeclareLaunchArgument(
        'ground_file',
        default_value='',
        description='Path to the ground PCD file',
    )
    ground_height_arg = DeclareLaunchArgument(
        'ground_height',
        default_value='-0.5',
        description='Height of the ground plane',
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time (for bag replay)',
    )
    use_no_entry_layer_arg = DeclareLaunchArgument(
        'use_no_entry_layer',
        default_value='false',
        description='Use path_planning_on_no_entry_layer.yaml when true; default is static-layer config',
    )
    no_entry_zone_yaml_arg = DeclareLaunchArgument(
        'no_entry_zone_yaml',
        default_value='',
        description='Absolute path to no_entry_layer.yaml; overrides perception no_entry_zone_pcd_file_dir',
    )

    return LaunchDescription([
        map_file_arg,
        ground_file_arg,
        ground_height_arg,
        use_sim_time_arg,
        use_no_entry_layer_arg,
        no_entry_zone_yaml_arg,
        OpaqueFunction(function=_launch_setup),
    ])
