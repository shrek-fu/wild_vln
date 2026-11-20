from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            #SetEnvironmentVariable(name="RMW_IMPLEMENTATION", value="rmw_fastrtps_cpp"),
            Node(
                package="go2w_keyboard_control",
                executable="go2_cmd_vel_node",
                name="go2_cmd_vel_node",
                output="screen",
                parameters=[
                    {
                        "robot_ip": "192.168.123.161",
                        "network_interface": "enx9c69d3025bcd",
                        "cmd_vel_topic": "/cmd_vel",
                        "posture_cmd_topic": "/unitree/posture_cmd",
                        "cmd_vel_timeout": 0.5,
                        "publish_rate": 50.0,
                        "max_linear_x": 0.6,
                        "max_linear_y": 0.4,
                        "max_angular_z": 0.8,
                        "auto_stand_up": False,
                        "auto_prepare_motion": True,
                    }
                ],
            ),
        ]
    )
