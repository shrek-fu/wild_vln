# Integration notes

## OpenVIO as estimator (vendored)

EquirectangularVIO vendors the full OpenVIO tree at [`third_party/openvio`](../third_party/openvio):

- Camera model: `EQUIRECTANGULAR` (dual-lens stitch + online stitching parameter `k`)
- Online: cam–IMU extrinsics, time offsets, dual-lens time offset
- Offline binary: `build/openvio/bin/vio` (from `./scripts/build_openvio.sh`)
- Live API: `vio::VisualInertialOdometry::{AddImu,ProcessImage}`
- ROS2 CMake links `EQUIRECT_ROOT/third_party/openvio` only — no sibling `../openvio` required


## X4 Air data path

Preferred live stack with `x4_slam`:

```text
insta360_ros_driver  →  /dual_fisheye/image/compressed + /imu/data_raw
                     →  /insta360/camera_preview_info
media_sdk_stitcher   →  /equirectangular/image   (MediaSDK RealTimeStitcher, template, 1920×960)
equirect_vio_node    ←  /equirectangular/image + /imu/data_raw
  (camera.model=EQUIRECTANGULAR, 1920×960)
                     →  /equirect_vio/odometry

Same stitcher path as insta360_multi_mid360s_calibration (official MediaSDK, not DIY projection).

Launch:
  ros2 launch insta360_ros_driver x4_air_equirect.launch.xml
  ros2 launch equirect_vio x4_air_vio.launch.py
```

Image resolution **must** match `camera.width/height` in the YAML (default 1440×720).

## Calibration checklist

1. Fix equirect resolution in the driver and YAML.
2. Calibrate `rig.imu_from_camera` (do not leave identity for production).
3. Set IMU noise densities from a static recording.
4. Tune `mask_rectangles` for mount / body occlusion.
5. Keep a few seconds of near-static motion at start, then excite with translation + rotation for init.
