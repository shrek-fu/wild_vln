from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="go2w_keyboard_control",
                executable="go2w_keyboard_node",
                name="go2w_keyboard_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "robot_ip": "192.168.123.161",
                        "network_interface": "enx9c69d3025bcd",
                        "linear_speed": 0.35,
                        "lateral_speed": 0.20,
                        "yaw_speed": 0.60,
                        # Match Unitree SDK sport examples: repeated Move() ~50 Hz while moving.
                        "move_send_hz": 50.0,
                    }
                ],
            )
        ]
    )
