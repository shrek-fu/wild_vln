"""Insta360 + MID-360S rig transforms, same files as DA360 ``insta_live.calib_io``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Equirect unprojection frame used by DA360 saver.py:
#   +Z panorama centre (forward), +X right, +Y down.
# Stitched camera body (FlowState off):
#   +X panorama centre (forward), +Y left, +Z up.
# p_body = R_BODY_DA360 @ p_da360.
R_BODY_DA360 = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _as_T(mat: list) -> np.ndarray:
    T = np.asarray(mat, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"expected 4x4, got {T.shape}")
    return T


def _T_from_R(R: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


@dataclass
class RigCalibration:
    equirect_width: int
    equirect_height: int
    stitch_type: str
    T_front_insta360: np.ndarray
    T_level_insta360: np.ndarray
    T_body_da360: np.ndarray
    T_level_da360: np.ndarray
    T_front_da360: np.ndarray
    source_dir: Path
    camera_extrinsic_status: str

    def summary(self) -> str:
        t_mm = self.T_level_insta360[:3, 3] * 1000.0
        return (
            f"equirect {self.equirect_width}x{self.equirect_height} stitch={self.stitch_type}; "
            f"T_level_insta360 t_mm={np.round(t_mm, 1).tolist()}; "
            f"cam={self.camera_extrinsic_status}"
        )


def load_rig_calibration(calib_dir: str | Path) -> RigCalibration:
    """Read ``insta360_intrinsics.json``, ``insta360_extrinsic.json``, ``lidar_extrinsic.json``.

    ``lidar_extrinsic.json`` is required so the gravity chain matches DA360, even though
    detection only applies ``T_level_insta360``.
    """
    root = Path(calib_dir).expanduser().resolve()
    intr = json.loads((root / "insta360_intrinsics.json").read_text(encoding="utf-8"))
    extr = json.loads((root / "insta360_extrinsic.json").read_text(encoding="utf-8"))
    lidar_path = root / "lidar_extrinsic.json"
    if not lidar_path.is_file():
        raise FileNotFoundError(f"missing {lidar_path} (IMU leveling, same folder DA360 uses)")
    json.loads(lidar_path.read_text(encoding="utf-8"))

    stream = intr["stream"]
    T_front_insta360 = _as_T(extr["T_front_insta360"])
    T_level_insta360 = _as_T(extr["T_level_insta360"])
    T_body_da360 = _T_from_R(R_BODY_DA360)
    return RigCalibration(
        equirect_width=int(stream["equirect_width"]),
        equirect_height=int(stream["equirect_height"]),
        stitch_type=str(stream.get("stitch_type", "")),
        T_front_insta360=T_front_insta360,
        T_level_insta360=T_level_insta360,
        T_body_da360=T_body_da360,
        T_level_da360=T_level_insta360 @ T_body_da360,
        T_front_da360=T_front_insta360 @ T_body_da360,
        source_dir=root,
        camera_extrinsic_status=str(extr.get("status", "")),
    )


def rotation_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to quaternion ``(x, y, z, w)``."""
    m = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / np.linalg.norm(q)
