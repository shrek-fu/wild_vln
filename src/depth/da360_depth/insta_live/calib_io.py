"""Load Insta360 + dual MID-360S calibration exported for DA360."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# DA360 saver.py unprojects equirect depth in an optical-like frame:
#   +Z = panorama centre (forward), +X = right, +Y = down.
# Calibration body frame (FlowState off):
#   +X = panorama centre (forward), +Y = left, +Z = up.
# So p_body = R_BODY_DA360 @ p_da360 (same as R_body_optical at yaw 0).
R_BODY_DA360 = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _as_T(mat: list[list[float]]) -> np.ndarray:
    T = np.asarray(mat, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"expected 4x4 matrix, got {T.shape}")
    return T


def _T_from_R_t(R: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


@dataclass
class RigCalibration:
    """Camera / lidar / IMU-level transforms used by the live DA360 node."""

    equirect_width: int
    equirect_height: int
    stitch_type: str
    views: dict[str, Any]
    T_front_insta360: np.ndarray
    T_level_insta360: np.ndarray
    T_level_front: np.ndarray
    T_level_rear: np.ndarray | None  # None on the single-lidar rig
    T_front_rear: np.ndarray | None
    leveling: dict[str, Any]
    T_body_da360: np.ndarray
    T_level_da360: np.ndarray
    T_front_da360: np.ndarray
    source_dir: Path
    camera_extrinsic_status: str
    lidar_extrinsic_status: str

    def summary(self) -> str:
        t = self.T_level_insta360[:3, 3] * 1000.0
        return (
            f"equirect {self.equirect_width}x{self.equirect_height} stitch={self.stitch_type}; "
            f"T_level_insta360 t_mm={np.round(t, 1).tolist()}; "
            f"cam={self.camera_extrinsic_status} lidar={self.lidar_extrinsic_status}"
        )


def load_rig_calibration(calib_dir: str | Path) -> RigCalibration:
    """Read results/export copies under ``calib_dir`` and build frame chain.

    Required files:
      - insta360_intrinsics.json
      - insta360_extrinsic.json
      - lidar_extrinsic.json  (IMU gravity leveling + T_level_*)
    """
    root = Path(calib_dir).resolve()
    intr = json.loads((root / "insta360_intrinsics.json").read_text(encoding="utf-8"))
    extr = json.loads((root / "insta360_extrinsic.json").read_text(encoding="utf-8"))
    lidar = json.loads((root / "lidar_extrinsic.json").read_text(encoding="utf-8"))

    stream = intr["stream"]
    T_front_insta360 = _as_T(extr["T_front_insta360"])
    T_level_insta360 = _as_T(extr["T_level_insta360"])
    T_level_front = _as_T(lidar["T_level_front"])
    T_level_rear = _as_T(lidar["T_level_rear"]) if "T_level_rear" in lidar else None
    T_front_rear = _as_T(lidar["T_front_rear"]) if "T_front_rear" in lidar else None
    T_body_da360 = _T_from_R_t(R_BODY_DA360)
    T_level_da360 = T_level_insta360 @ T_body_da360
    T_front_da360 = T_front_insta360 @ T_body_da360

    return RigCalibration(
        equirect_width=int(stream["equirect_width"]),
        equirect_height=int(stream["equirect_height"]),
        stitch_type=str(stream.get("stitch_type", "")),
        views=intr["views"],
        T_front_insta360=T_front_insta360,
        T_level_insta360=T_level_insta360,
        T_level_front=T_level_front,
        T_level_rear=T_level_rear,
        T_front_rear=T_front_rear,
        leveling=lidar.get("leveling", {}),
        T_body_da360=T_body_da360,
        T_level_da360=T_level_da360,
        T_front_da360=T_front_da360,
        source_dir=root,
        camera_extrinsic_status=str(extr.get("status", "")),
        lidar_extrinsic_status=str(lidar.get("status", "")),
    )


def write_merged_summary(calib: RigCalibration, out_path: str | Path) -> None:
    """Write a single JSON that documents the frame chain for downstream tools."""
    out = {
        "schema": "da360_insta360_rig/v1",
        "source_dir": str(calib.source_dir),
        "equirect": {
            "width": calib.equirect_width,
            "height": calib.equirect_height,
            "stitch_type": calib.stitch_type,
            "model": "equirectangular",
            "note": "intrinsic model is image size only; no radial distortion",
        },
        "frames": {
            "insta360_body": "X forward=panorama centre, Y left, Z up (FlowState off)",
            "da360_optical": "X right, Y down, Z forward (DA360 unprojection)",
            "lidar_level": "origin at front lidar, Z against gravity, yaw of front lidar",
            "front_lidar": "raw front MID-360S",
        },
        "imu": {
            "source": "MID-360S built-in IMU gravity at lidar leveling time",
            "leveling": calib.leveling,
            "note": "camera IMU / FlowState is off; gravity alignment uses lidar IMU via T_level_*",
        },
        "views": calib.views,
        "T_body_da360": calib.T_body_da360.tolist(),
        "T_front_insta360": calib.T_front_insta360.tolist(),
        "T_level_insta360": calib.T_level_insta360.tolist(),
        "T_front_da360": calib.T_front_da360.tolist(),
        "T_level_da360": calib.T_level_da360.tolist(),
        "T_level_front": calib.T_level_front.tolist(),
        "T_level_rear": None if calib.T_level_rear is None else calib.T_level_rear.tolist(),
        "T_front_rear": None if calib.T_front_rear is None else calib.T_front_rear.tolist(),
        "status": {
            "camera": calib.camera_extrinsic_status,
            "lidar": calib.lidar_extrinsic_status,
        },
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
