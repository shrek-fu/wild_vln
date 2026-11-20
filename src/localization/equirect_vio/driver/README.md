# driver — Insta360 X4 Air CameraSDK + MediaSDK + ROS 驱动

自包含相机层。等距柱状图默认用 **官方 MediaSDK `RealTimeStitcher`**（与
`insta360_multi_mid360s_calibration` 相同），不再用 DIY 双鱼眼投影。

```text
driver/
├── Linux_CameraSDK-2.1.8/          # CameraSDK 2.1.8 x86_64（含 X4_Air json）
├── insta360_media_sdk -> …         # 符号链接到 <ws>/insta360_driver/insta360_media_sdk
└── insta360_ros_driver/            # ROS 驱动 + media_sdk_stitcher
```

## 数据流（默认）

```text
X4 Air USB
  → CameraSDK → /dual_fisheye/image/compressed + /imu/data_raw
              → /insta360/camera_preview_info
  → MediaSDK RealTimeStitcher (stitch_type=template, 1920×960)
              → /equirectangular/image
```

## 编译

```bash
# MediaSDK 需已链接到 driver/insta360_media_sdk
./scripts/build_ros2.sh
```

若未找到 MediaSDK，`media_sdk_stitcher` 不会编进包，launch 会失败。

## 运行

```bash
source ros2_ws/install/setup.bash
# 首次：./driver/insta360_ros_driver/setup.sh  （USB=Android）

ros2 launch insta360_ros_driver x4_air_equirect.launch.xml
ros2 launch equirect_vio x4_air_vio.launch.py   # 默认 config 1920×960
```

## 与标定工程对齐

| 项 | 标定工程 | 本仓库默认 |
|---|---|---|
| 拼接 | MediaSDK `RealTimeStitcher` | 同左（`media_sdk_stitcher`） |
| `stitch_type` | `template` | `template` |
| 全景尺寸 | 1920×960 | 1920×960 |
| 输出 topic | `/insta360/equirect/image/compressed` | `/equirectangular/image`（raw Image） |

DIY 节点 `dual_fisheye2equirectangular_cpp` 仍保留，但不作为默认 launch。

## 许可证

CameraSDK / MediaSDK 为 Insta360 专有许可；`insta360_ros_driver` 见其 `LICENSE.txt`。
