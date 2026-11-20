"""Time-indexed odometry poses with interpolation (LIO publishes at the 10 Hz scan rate)."""

from __future__ import annotations

import bisect

import numpy as np

from scene_memory.common.calib_io import rotation_to_quaternion_xyzw


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def interpolate_pose(T0: np.ndarray, T1: np.ndarray, a: float) -> np.ndarray:
    """Pose at fraction ``a`` from ``T0`` to ``T1``: linear translation, slerp rotation."""
    q0 = rotation_to_quaternion_xyzw(T0[:3, :3])
    q1 = rotation_to_quaternion_xyzw(T1[:3, :3])
    dot = float(q0 @ q1)
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        theta = np.arccos(dot)
        q = (np.sin((1 - a) * theta) * q0 + np.sin(a * theta) * q1) / np.sin(theta)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_to_matrix(q / np.linalg.norm(q))
    T[:3, 3] = (1 - a) * T0[:3, 3] + a * T1[:3, 3]
    return T


def pose_at(stamps: list[float], poses: list[np.ndarray], t: float, max_dt: float = 0.1) -> np.ndarray | None:
    """Interpolated between the bracketing poses (gap up to ``2 max_dt``), else the nearest within ``max_dt``."""
    i = bisect.bisect_left(stamps, t)
    if 0 < i < len(stamps) and stamps[i] - stamps[i - 1] <= 2 * max_dt:
        a = (t - stamps[i - 1]) / max(stamps[i] - stamps[i - 1], 1e-9)
        return interpolate_pose(poses[i - 1], poses[i], a)
    best = min((j for j in (i - 1, i) if 0 <= j < len(stamps)), key=lambda j: abs(stamps[j] - t), default=None)
    if best is None or abs(stamps[best] - t) > max_dt:
        return None
    return poses[best]


class PoseBuffer:
    def __init__(self, capacity: int = 4000):
        self.capacity = int(capacity)
        self.stamps: list[float] = []
        self.poses: list[np.ndarray] = []

    def add(self, t: float, T: np.ndarray) -> bool:
        if self.stamps and t <= self.stamps[-1]:
            return False
        self.stamps.append(float(t))
        self.poses.append(T)
        if len(self.stamps) > self.capacity:
            drop = self.capacity // 4
            del self.stamps[:drop], self.poses[:drop]
        return True

    def at(self, t: float, max_dt: float = 0.1) -> np.ndarray | None:
        return pose_at(self.stamps, self.poses, t, max_dt)

    def last_t(self) -> float:
        return self.stamps[-1] if self.stamps else 0.0
