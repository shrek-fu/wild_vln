# EquirectangularVIO — Insta360 X4 Air 全景 VIO（自包含）

本仓库**内嵌** [OpenVIO](third_party/openvio)（MIT）源码与依赖，面向 **Insta360 X4 / X4 Air** 的 equirectangular + IMU 视觉惯性里程计。  
拷贝本目录即可移植，**不再依赖**同级的 `../openvio`。

```text
Insta360 X4 Air
  → equirectangular 图像 + IMU
  → third_party/openvio (EQUIRECTANGULAR)
  → TUM 轨迹 / ROS2 Odometry + TF
```

## 目录

```text
EquirectangularVIO/
├── third_party/openvio/     # 完整 OpenVIO 源码 + ceres（已 vendoring）
├── driver/                  # CameraSDK + MediaSDK(symlink) + insta360_ros_driver
├── config/                  # X4 Air YAML（1280 / 1440）
├── python/x4_equirect_vio/  # 导入 / 离线跑
├── ros2_ws/src/             # equirect_vio + symlink → driver/
├── scripts/                 # build_openvio / build_ros2 / check_env
├── CMakeLists.txt           # 顶层：编译 vendored OpenVIO
└── data/
```

## 一键编译

### 1) 离线 `vio` 工具（不需要 ROS）

```bash
cd EquirectangularVIO
./scripts/build_openvio.sh
# 产物: build/openvio/bin/vio
```

依赖：CMake ≥ 3.16、C++17、OpenCV 4、Eigen3、libpng。  
Ceres / spdlog 已在 `third_party/openvio/thirdparty/`。

### 2) ROS 2 实时节点

```bash
source /opt/ros/humble/setup.bash   # 或 jazzy
./scripts/build_ros2.sh
source ros2_ws/install/setup.bash
```

之前报错 `ros2_ws/install/setup.bash: No such file` 是因为还没 `colcon build`；跑完 `build_ros2.sh` 就会生成。

## 离线跑

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/python:$PYTHONPATH

python -m x4_equirect_vio.cli all \
  --video data/input/session.mp4 \
  --camera-data data/input/session_cameradata.csv \
  --session-dir data/raw/session01 \
  --output data/processed/session01/traj.tum \
  --width 1440 --height 720
```

或直接：

```bash
./build/openvio/bin/vio odometry \
  --dataset_path data/raw/session01 \
  --config_path config/x4_air_equirect_1440.yaml \
  --output_path data/processed/session01/traj.tum
```

## 实时（自带 driver/ 相机栈）

```bash
source ros2_ws/install/setup.bash
# 首次：driver/insta360_ros_driver/setup.sh 装 udev（USB=Android）

# 终端 1 — MediaSDK template 全景 /equirectangular/image (1920×960) + IMU
ros2 launch insta360_ros_driver x4_air_equirect.launch.xml

# 终端 2 — OpenVIO EQUIRECTANGULAR @ 1920×960
ros2 launch equirect_vio x4_air_vio.launch.py
```

Foxglove 订阅 **`/equirectangular/image`**（官方 MediaSDK 拼接，与标定工程同几何）。

详见 [`driver/README.md`](driver/README.md)。

## 标定

`config/*.yaml` 里 `rig.imu_from_camera` 与 IMU 噪声是占位符，上机前请标定。分辨率须与 YAML 一致（推荐 **1440×720**）。

## 许可证

- 本仓库集成代码：MIT（见 `LICENSE`）
- Vendored OpenVIO：MIT（见 `third_party/openvio/LICENSE`）
- Ceres Solver 等第三方：见各自目录许可
- `driver/Linux_CameraSDK-*`：Insta360 专有 SDK，勿公开再分发
- `driver/insta360_ros_driver`：见其 `LICENSE.txt`

## 与外部 `src/openvio` 的关系

| | 说明 |
|---|---|
| `src/openvio` | 上游参考副本，可继续单独更新 |
| `EquirectangularVIO/third_party/openvio` | **本工程实际编译使用的副本** |

更新算法时：把上游改动同步进 `third_party/openvio` 后重新 `./scripts/build_openvio.sh` / `./scripts/build_ros2.sh`。
