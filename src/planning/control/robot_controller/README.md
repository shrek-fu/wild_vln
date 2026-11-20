# robot_controller

独立于 `nav_server` 的机器狗控制器。导航栈只发速度；本节点按 `robot_type` 把 Twist 和起立/趴下/急停落到具体机型。

## 架构

```
Nav2 / qi_move_base  --Twist-->  /cmd_vel
nav_server MQTT 摇杆  --Twist-->  /ql/nav/robot_cmd_vel
nav_server 1002/1003/1041-1043 --srv-->  /robot_controller/command
                                              |
                                    robot_controller_node
                                              |
                    +------------+------------+------------+
                    |            |            |            |
                 chitu        zsl1      ysc_lite3      unitree_*
              转发 /rl_cmd_vel   SDK        UDP       C++ SportClient
```

本包在主工作空间 `guide_ws/src` 中编译。宇树 C++ 节点在独立工作空间（默认 `robots/unitree_ws`，可用 `UNITREE_WS` 改路径），**不必**和本仓放在一起。

## 支持的机型

| `robot_type` | 别名 | 速度 | 起立 / 趴下 | 说明 |
|---|---|---|---|---|
| `chitu` | | 转发到 `/rl_cmd_vel` | 无（102/103 记日志后返回成功） | 机载 RL 控制器自己吃速度 |
| `zsl1` | | `HighLevel.move` | `standUp` / `lieDown`（匍匐） / `passive`（完全趴下） | SDK 子进程 |
| `ysc_lite3` | `lite3` | UDP，默认 `auto` 模式 | 查 `0x0901` 再发 `0x21010202`（切换，禁止盲发） | 绝影 Lite3 协议 |
| `unitree_b2` | `b2` | C++ `Move()` | `StandUp` / `StandDown` | 需 source 宇树 workspace |
| `unitree_go2` | `go2` | 同上（go2 SportClient） | 同上 | 与 go2w 共用 `go2_cmd_vel_node` |
| `unitree_go2w` | `go2w` | 同上 | 同上 | |

机型参数文件在 `config/`：`chitu.yaml`、`zsl1.yaml`、`ysc_lite3.yaml`、`unitree_b2.yaml`、`unitree_go2.yaml`、`unitree_go2w.yaml`。IP、网卡、限速、超时都在对应 YAML 里改。

## 编译与启动

主仓：

```bash
cd /path/to/guide_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select qi_nav_msgs robot_controller
source install/setup.bash
```

宇树（仅 `unitree_*` 需要；工作空间可在任意路径）：

```bash
cd /path/to/unitree_ws    # 默认 guide_ws/robots/unitree_ws
colcon build --packages-select b2_sport_control go2w_keyboard_control
source install/setup.bash
```

启动（`ROBOT_TYPE` 与 `nav_server` 同源，读 `~/.qi_nav_env.sh`）：

```bash
export ROBOT_TYPE=ysc_lite3          # 或 chitu / zsl1 / unitree_go2w ...
./scripts/run_robot_controller.sh
```

或直接 launch：

```bash
ros2 launch robot_controller robot_controller.launch.py robot_type:=ysc_lite3
```

宇树工作空间不在默认路径时：

```bash
export UNITREE_WS=/opt/unitree_ws
./scripts/run_robot_controller.sh
```

不要做成开机自启。速度通道只转发 Twist，不触发起立。

## 是否拉起狗控

nav_server 未检测到 `robot_controller_node` 时，会 `ros2 launch robot_controller` 并 `prepare`。已在跑则只复用。

| api_id | 名称 | 拉起进程 | 说明 |
|---:|---|:---:|---|
| **1041** | 进入遥控 | 是 | `prepare` 后再起雷达 |
| **1011** | `launch_all` 导航+定位 | 是 | `launch_nav_and_localization` |
| **1007** | 导航+定位并自动重定位 | 是 | 内部先走 1011 同一条路径 |
| **1008** | 执行编排 | 仅含 GoToGoal 时 | 无导航点则不拉 |
| **1012** | `kill_all` 关闭导航 | 杀掉进程 | 关栈名单含 `robot_controller_node` |
| **1061** | `activate_nav_mode` | 否 | 只激活/拉导航栈 |
| **1001 / 1004** | 开始/继续导航 | 否 | 只起行为树，假定栈已就绪 |
| **1002 / 1003** | 急停 / 解除急停 | 否 | 1002 发 `stop`（`control_id=1` 丢掉后续速度）再 `lie`。1003 `stand` 打开速度锁 |
| **1042 / 1043** | 结束遥控 / 遥控动作 | 1042 关掉残留进程 | 1043 要求已处于 1041 的 `robot_in_control`；1042 在 1041 失败后仍会杀掉已拉起的狗控 |
| **1031** | 建图 | 否 | |
| **1021 / 1022** | 纯定位 | 否 | |
| MQTT `cmdVel` | 摇杆 | 否 | 只发 `/ql/nav/robot_cmd_vel` |

手工启动：`./scripts/run_robot_controller.sh`。

## ROS 接口

### 订阅（速度）

| 话题 | 类型 | 来源 |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | 导航 |
| `/ql/nav/robot_cmd_vel` | `geometry_msgs/Twist` | 平台遥控摇杆（MQTT → nav_server） |

超时（默认 0.5 s）自动清零。速度热路径不查站立、不 `prepare`（`auto_stand_on_cmd_vel` 默认 false）。起立由 1011/1007/1008（含 GoToGoal）或 1041 的 `prepare` 完成，1061 不起立。chitu 无姿态接口，`is_standing()` 恒为 True。

宇树速度由 C++ 节点直接订 `/cmd_vel`，Python 节点只转发姿态。

### 发布（ysc_lite3 状态）

`lite3_status_node` 独占 UDP `43897`，由 `scripts/run_nav_server.sh` 在 `ROBOT_TYPE` 为 `ysc_lite3` / `lite3` 时拉起（已在跑则跳过；脚本退出时停掉自己起的那份）。不随 1031，也不进 `kill_all`。`nav_server_main` 不再拉该节点。狗控不绑 43897，stand/lie 订 `/lite3/basic_state`。

| 话题 | 类型 | QoS | 说明 |
|---|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | depth=5 BEST_EFFORT | `name[]` / `position[]`(0x0902) / `velocity[]`(0x0903) / `effort[]` 为空 |
| `/lite3/basic_state` | `std_msgs/Int32` | reliable | 内部：0x0901 `basic_state`，给起立判断 |

手工（无 nav_server 时）：

```bash
ros2 launch robot_controller lite3_status.launch.py
# 或
ros2 run robot_controller lite3_status_node
```

不要和调试脚本 `0901_0902_0903_V20153.py` 同时占 43897。

### 服务（姿态 / 急停 / 动作）

- 名称：`/robot_controller/command`
- 类型：`qi_nav_msgs/srv/RobotCommand`

```
string command      # stand | lie | emergency | stop | prepare | action
int32 control_id    # command=action 时使用
---
bool success
string message
```

示例：

```bash
ros2 service call /robot_controller/command qi_nav_msgs/srv/RobotCommand "{command: stand, control_id: 0}"
ros2 service call /robot_controller/command qi_nav_msgs/srv/RobotCommand "{command: emergency, control_id: 0}"
ros2 service call /robot_controller/command qi_nav_msgs/srv/RobotCommand "{command: action, control_id: 102}"
```

`stop` 清速度。`control_id=1` 时丢掉后续 `/cmd_vel`，直到 `stand`（1002/1012）。`control_id=0` 只清一次（1063）。`lie` 是趴下。`emergency` 仍是立刻停再趴，只用于拉栈失败。

### 1043 `control_id`

| ID | 含义 | chitu | zsl1 | ysc_lite3 | unitree |
|---:|---|---|---|---|---|
| 1 / 2 / 3 | 速度倍率 1.0 / 1.5 / 2.0 | 软件倍率 | 软件倍率 | 软件倍率 | 转发 `SpeedLevel` |
| 101 | 阻尼 / 完全趴下 | 不支持 | `passive` | 查状态后趴下 | go2/go2w：`StandDown`（同 103）；b2：`Damp` |
| 102 | 起立 | 记日志成功 | `standUp` | 查状态后起立 | `StandUp` |
| 103 | 匍匐 | 只停车 | `lieDown` | 匍匐 `0x21010406` | `StandDown` |
| 104 | 挥手 | 不支持 | `shakeHand` | 不支持 | go2/go2w `Hello`；b2 不支持 |
| 105 | 俯仰身形（正反一组） | 不支持 | `attitudeControl` 正反循环后 `standUp` | 不支持 | 不支持 |
| 106 | 高低身形（正反一组） | 不支持 | 同上，height 轴 | 不支持 | 不支持 |
| 107 / 108 | 水平转身 左 / 右 | 不支持 | 踏步 `move(0,0,±wz)` | 不支持 | 不支持 |

机型不支持的 ID 返回 `success=false`。zsl1 的 105/106 是一组正反开环速率，重复次数由 `posture_repeats` 决定，结束后 `standUp`；107/108 是踏步转向，结束后停车。速率、时长和组数在 `config/zsl1.yaml`：`pitch_rate`、`height_rate`、`turn_rate`、`posture_duration`、`posture_repeats`、`turn_duration`。若实机俯仰反了，把 `pitch_rate` 设为负数。

## 与 nav_server 的关系

平台 API 仍走 nav_server，本节点只执行狗侧动作：

| api_id | nav_server | 拉起进程 | robot_controller |
|---:|---|:---:|---|
| 1002 紧急停止 | 先锁速度，取消目标不挡住停车，停稳后趴下 | 否 | `stop`（`control_id=1`）再 `lie` |
| 1003 解除急停 | 调本服务 | 否 | `command=stand`，打开速度锁 |
| 1007 / 1011 | 导航+定位 | 是 | `command=prepare` |
| 1008 编排（含 GoToGoal） | 同 1011 | 是 | `command=prepare` |
| 1012 关闭导航 | 先锁速度，停稳后趴下，再关栈 | 杀掉进程 | `stop`（`control_id=1`）再 `lie`，然后杀进程 |
| 1061 激活导航 | 只拉导航栈 | 否 | 不调用 |
| 1041 进入遥控 | 可拉雷达，再 `prepare` | 是 | `command=prepare` |
| 1042 结束遥控 | `lie` 后关栈；1041 失败留下的进程也会关 | 关掉残留 | `command=lie`（zsl1=`passive`）；未连接但节点仍在则直接杀进程 |
| 1043 遥控动作 | 转发 `robot_control_id` | 否 | `command=action` |
| 遥控摇杆 | 发布 `/ql/nav/robot_cmd_vel` | 否 | 订阅后驱动 |

导航输出统一为 `/cmd_vel`。chitu 由本节点转发到 `/rl_cmd_vel`。

## 机型注意

**chitu**  
无起立/趴下。导航和遥控速度都进本节点，再发到 `/rl_cmd_vel`。

**zsl1**  
YAML 里改 `local_ip` / `local_port` / `dog_ip`。SDK 在子进程中，启动时不自动起立，等 `prepare` / `stand` 或首帧速度触发。`lieDown` 是匍匐（1043 id=103）；`passive` 才是阻尼完全趴下。`command: lie` / 1042 / 1043 id=101 都走 `passive`。1002 和 1012 先发零速度并锁住后续速度，停稳后再 `lie`（`passive`）。拉栈失败仍走立刻 `emergency`。1043 的 105/106 走 `attitudeControl` 正反循环（Python 参数顺序为 roll/pitch/yaw/height），动作期间不发 `move`；结束后 `standUp`。107/108 用 `move` 踏步转向。

**ysc_lite3**  
起立指令 `0x21010202` 是站/趴切换。收不到 `0x0901`（经 `/lite3/basic_state`）时**不会盲发**。默认 `mode: auto`（浮点 m/s，适合导航）。手动遥控可把 YAML 改成 `mode: manual`。`invert_y` / `invert_yaw` 默认 true，把 ROS REP-103 对到运控约定。1043：101 趴下、102 起立、103 匍匐（先切手动+移动再发 `0x21010406`）。`command: lie` / 1042 仍直接 `lie()`，不走 103。1002 和 1012 先发零速度并锁住后续速度，停稳后再切换趴下。状态节点由 `./scripts/run_nav_server.sh` 拉起，`robot_controller.launch.py` 不重复占 43897。

**unitree_***  
需已 source 宇树 install。与 nav_server 共用 `~/.qi_nav_env.sh` 中的 CycloneDDS（勿在 launch 里强制 Fast DDS，否则 1041/1011 无法调用 `/robot_controller/command`）。`robot_ip`、`network_interface` 写在对应 YAML。姿态走 `/unitree/posture_cmd`（`std_msgs/Int32`）。

## 常见问题

### `motion requested but robot is not prepared`

- **现象**：`go2_cmd_vel_node` 收到非零 `/cmd_vel` 后打印该 WARN，机器狗不动。
- **原因**：本仓库 launch 将 `auto_prepare_motion` 设为 `False`；需先 prepare/stand，`motion_prepared_` 才为 true。
- **解决**：
  - 通过 nav_server **1041 / 1011 / 1007 / 1008**（会自动调 `prepare`）
  - 或手工：`ros2 service call /robot_controller/command qi_nav_msgs/srv/RobotCommand "{command: prepare, control_id: 0}"`
  - 确认 C++ 日志出现 `posture StandUp -> 0`（或返回值 ≥ 0）后再发速度

### nav_server 1041 报 `robot_controller service unavailable` 或 `command timed out`

- **现象**：1041 拉起进程后仍失败，或 prepare 超时。
- **常见原因**：
  1. `robot_controller.launch.py` 强制 `rmw_fastrtps_cpp`，与 nav_server 的 CycloneDDS 不在同一 ROS 图（已修复：与外部 `unitree_ws` 一致，沿用环境 RMW）
  2. 节点刚启动、服务尚未就绪（已加长等待）
  3. 在 nav_server 服务回调内 `spin_until_future_complete` 会导致 1042/1043/重复 1041 死锁（已改为 subprocess `ros2 service call`）
- **解决**：`colcon build` 后重启 nav_server；确认 `~/.qi_nav_env.sh` 已 source `unitree_ws`。

## 目录

```
src/robot_controller/
  robot_controller/node.py
  robot_controller/status_node.py
  robot_controller/lite3_protocol.py
  robot_controller/adapters/{base,chitu,zsl1,ysc_lite3,unitree}.py
  launch/robot_controller.launch.py
  launch/lite3_status.launch.py
  config/*.yaml
  tests/test_adapters.py
```
