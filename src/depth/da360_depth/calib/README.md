# Calibration inputs for live DA360

Copied from `../insta360_multi_mid360s_calibration/results/export/` by:

```bash
bash scripts/import_calib.sh
```

| File | Meaning |
|---|---|
| `insta360_intrinsics.json` | Equirect size + four view pinholes |
| `insta360_extrinsic.json` | Camera body → `front_lidar` / `lidar_level` |
| `lidar_extrinsic.json` | Dual-lidar extrinsic + **IMU gravity leveling** |
| `rig_calib.json` | Merged frame chain (`T_level_da360`, …) written on import |

Refresh this folder after re-calibrating the rig.
