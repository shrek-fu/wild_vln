# go2w_keyboard_control

ROS2 Humble keyboard teleop node for Go2W (wired target IP `192.168.123.161`).
It can also consume external velocity topic `/ql/nav/robot_cmd_vel` (`geometry_msgs/msg/Twist`).

## Build

```bash
cd /home/foo/00WS/00qiRobotics/16LM_AC1_GOW2_DEV
source /opt/ros/humble/setup.bash
colcon build --packages-select go2w_keyboard_control
```

## Run

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch go2w_keyboard_control go2w_keyboard.launch.py
```

## Keyboard mapping

- `1`: stand up
- `2`: stand down
- `w/s`: forward/backward
- `q/e`: left/right shift
- `a/d`: turn left/right
- `space` or `x`: stop move
- `h`: print help

## Notes

- Default wired interface is `eno1`; adjust `network_interface` parameter if needed.
- Default cmd topic is `/ql/nav/robot_cmd_vel`, timeout is `0.30` s.
- `linear.x / linear.y / angular.z` are mapped to `Move(vx, vy, wz)` and clamped by
  `linear_speed / lateral_speed / yaw_speed`.
- Keep robot in a safe area before sending motion commands.
