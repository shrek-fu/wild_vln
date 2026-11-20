from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from cloud_preprocess.mount import body_box, load_mount


def _launch_setup(context):
    robot_type = LaunchConfiguration('robot_type').perform(context).strip().lower() or 'zsl1'
    input_topic = LaunchConfiguration('input_topic').perform(context)
    output_topic = LaunchConfiguration('output_topic').perform(context)

    mount, mounts = load_mount(robot_type)
    if mount is None:
        known = ', '.join(sorted(str(key) for key in mounts))
        return [
            LogInfo(msg=f'[trans_livox] Unsupported robot_type: {robot_type}. Supported: {known}'),
            Shutdown(reason=f'Unsupported robot_type: {robot_type}'),
        ]

    length, width, height = body_box(mount)
    return [
        Node(
            package='cloud_preprocess',
            executable='cloud_transformer',
            name='livox_cloud_transformer',
            output='screen',
            parameters=[{
                'input_msg_type': 'custom_msg',
                'input_topic': input_topic,
                'publish_map': False,
                'publish_body': True,
                'qos_best_effort': True,
                'robot_type': robot_type,
                'crop_body': True,
                'body_length': length,
                'body_width': width,
                'body_height': height,
            }],
            remappings=[
                ('/cloud_in_body', output_topic),
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_type',
            default_value='zsl1',
            description='Robot type used to look up the Livox body box',
        ),
        DeclareLaunchArgument(
            'input_topic',
            default_value='/livox/lidar',
            description='Input Livox CustomMsg topic (frame_id should be livox_frame)',
        ),
        DeclareLaunchArgument(
            'output_topic',
            default_value='/ql/nav/cloud_in_body',
            description='Output PointCloud2 topic transformed to base_link with the body cropped out',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
