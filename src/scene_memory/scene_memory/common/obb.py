"""Gravity-aligned 7-DoF boxes, the same search as Boxer ``ObbTW.fit_points``."""

from __future__ import annotations

import numpy as np

# Boxer searches yaw in [0, pi) because a box matches itself after a half turn.
_YAW_STEPS = 20


def yaw_to_quaternion_xyzw(yaw: float) -> np.ndarray:
    """Rotation about ``lidar_level`` +Z, matching Boxer's ``Rz(yaw)``."""
    half = 0.5 * float(yaw)
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float64)


def fit_gravity_obb(points: np.ndarray, yaw_steps: int = _YAW_STEPS) -> dict | None:
    """Minimum-volume box whose vertical axis stays aligned with +Z.

    Returns center ``(3,)``, size ``(sx, sy, sz)`` and ``yaw`` about +Z.
    Size is expressed in the yaw-rotated object frame, as in Boxer after ``center()``.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return None
    keep = np.isfinite(pts).all(axis=1)
    pts = pts[keep]
    if pts.shape[0] < 2:
        return None

    best_volume = np.inf
    best: dict | None = None
    for yaw in np.linspace(0.0, np.pi, int(yaw_steps), endpoint=False):
        cosine, sine = np.cos(yaw), np.sin(yaw)
        x_obj = cosine * pts[:, 0] + sine * pts[:, 1]
        y_obj = -sine * pts[:, 0] + cosine * pts[:, 1]
        z_obj = pts[:, 2]
        low = np.array([x_obj.min(), y_obj.min(), z_obj.min()])
        high = np.array([x_obj.max(), y_obj.max(), z_obj.max()])
        size = high - low
        volume = float(np.prod(np.maximum(size, 0.0)))
        if volume >= best_volume:
            continue
        center_obj = 0.5 * (low + high)
        center = np.array(
            [
                cosine * center_obj[0] - sine * center_obj[1],
                sine * center_obj[0] + cosine * center_obj[1],
                center_obj[2],
            ]
        )
        best_volume = volume
        best = {
            "center": center,
            "size": size,
            "yaw": float(yaw),
            "num_points": int(pts.shape[0]),
        }
    return best
