#!/usr/bin/env python3
"""Single-lidar rig: gravity leveling of the front lidar from its built-in IMU.

With one MID-360S there is no lidar-lidar registration. The only lidar extrinsic is

    p_level = T_level_front * p_front

where T_level_front removes roll and pitch (the up vector of the IMU becomes +Z); yaw and origin stay
those of the lidar. The floor of the captured cloud is used as an independent check: after leveling,
its normal should point straight up.

Reads calib_capture/lidar/clouds/<scene>/front_lidar.{pcd,imu.json} (./calibrate.sh level <scene>
captures them), writes results/lidar_extrinsic.json.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

import extrinsic_io
from lidar_pair_refine import gravity_level, rotation_angle_deg


def floor_check(xyz: np.ndarray, T_level: np.ndarray) -> dict:
    """Largest near-horizontal plane below the lidar after leveling: its tilt and height."""
    p = xyz @ T_level[:3, :3].T
    p = p[np.isfinite(p).all(axis=1) & (np.linalg.norm(p, axis=1) > 0.5)]
    low = p[p[:, 2] < np.percentile(p[:, 2], 30)]
    if len(low) < 200:
        return {"found": False}
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(low))
    plane, inliers = pcd.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=500)
    n = np.asarray(plane[:3], dtype=float)
    n /= np.linalg.norm(n)
    if n[2] < 0:
        n, d = -n, -plane[3]
    else:
        d = plane[3]
    tilt = float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0))))
    return {"found": tilt < 15.0, "tilt_deg": round(tilt, 3), "lidar_height_mm": round(float(d) * 1000.0, 1),
            "inliers": int(len(inliers))}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()
    rig = extrinsic_io.load_rig(root)
    if not extrinsic_io.single(rig):
        raise SystemExit("config/rig.yaml has two lidars; use ./calibrate.sh refine-auto")
    name = rig["front"]["id"]
    cloud = extrinsic_io.clouds_dir(root) / args.scene / f"{name}.pcd"
    if not cloud.is_file():
        raise SystemExit(f"missing {cloud}; run ./calibrate.sh level {args.scene} with the lidar running")
    T_level_front, info = gravity_level(cloud)
    imu = json.loads(cloud.with_name(f"{name}.imu.json").read_text(encoding="utf-8"))
    acc_std = float(np.linalg.norm(imu.get("acc_std", [0, 0, 0])))
    xyz = np.asarray(o3d.io.read_point_cloud(str(cloud)).points)
    floor = floor_check(xyz, T_level_front)
    roll, pitch, _ = extrinsic_io.rpy_deg(T_level_front)
    doc = {
        "schema": extrinsic_io.SCHEMA,
        "status": "imu_level",
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "single lidar: roll and pitch from the MID-360S IMU gravity while the rig is still; "
                  "no lidar-lidar registration",
        "lidars": extrinsic_io.lidars_block(rig),
        "convention": {**extrinsic_io.CONVENTION, "equation": "p_level = T_level_front * p_front"},
        "result": {
            "T_level_front": {
                **extrinsic_io.pose(T_level_front),
                "meaning": f"front lidar into {extrinsic_io.LEVEL_FRAME} (gravity-aligned); apply uses it",
            },
        },
        "leveling": {"front": {**info, "roll_deg": roll, "pitch_deg": pitch,
                               "tilt_deg": round(rotation_angle_deg(T_level_front[:3, :3]), 3)}},
        "quality": {"scene": args.scene, "imu_acc_std_g": round(acc_std, 5), "floor_after_leveling": floor},
        "T_level_front": extrinsic_io.matrix(T_level_front),
    }
    path = root / "results" / extrinsic_io.EXTRINSIC_FILE
    extrinsic_io.write(path, doc)
    print(f"wrote {path}")
    print(f"front lidar tilt {doc['leveling']['front']['tilt_deg']} deg (roll {roll}, pitch {pitch}); "
          f"IMU acc std {acc_std:.4f} g")
    # The MID-360S IMU shakes with the lidar's own motor (about 0.08 g std when standing still), so
    # judge the mean: its angular standard error over the samples.
    n = max(int(imu.get("samples", 400)), 1)
    tilt_err = float(np.degrees(acc_std / np.sqrt(n)))
    print(f"gravity direction standard error about {tilt_err:.2f} deg over {n} IMU samples")
    if acc_std > 0.2 or tilt_err > 0.3:
        print("WARNING: the IMU was not still; keep the rig still and capture a new scene")
    design = yaml.safe_load((root / "config" / "rig.yaml").read_text(encoding="utf-8")) \
        .get("geometry", {}).get("lidar_tilt_deg")
    if design is not None:
        diff = doc["leveling"]["front"]["tilt_deg"] - float(design)
        print(f"design tilt {float(design):.1f} deg, measured - design = {diff:+.2f} deg "
              "(includes the slope of the surface the rig stands on)")
        if abs(diff) > 3.0:
            print("WARNING: more than 3 deg from the drawing; is the rig standing level and the lidar seated?")
    if floor.get("found"):
        print(f"floor check: normal {floor['tilt_deg']} deg from vertical, lidar {floor['lidar_height_mm']} mm "
              "above the floor")
        if floor["tilt_deg"] > 1.0:
            print("WARNING: the floor is not level after leveling; is the floor flat and visible in this scene?")
    else:
        print("floor check: no floor plane found (the tilted lidar may not see the floor here)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
