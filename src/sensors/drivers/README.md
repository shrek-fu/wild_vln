# Sensor drivers

- **Insta360 SDKs** live at the workspace root: `<ws>/insta360_driver/`
  (`insta360_sdk/`, `insta360_media_sdk/`). The `insta360_*` entries here are
  symlinks so relative paths still resolve. Binaries are not in the public repo.
- `livox_lidar_sdk/`、`livox_ros_driver2/`：Livox MID-360S 驱动（仍在本目录）。

Jetson 上 Media SDK 无 aarch64 构建，`insta360_views` 用 Camera SDK 的 `lib_aarch64/` + `insta360_cuda_stitch`。
