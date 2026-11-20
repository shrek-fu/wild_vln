"""Equirectangular pixels ↔ unit rays, matching DA360 ``insta_live.geometry``."""

from __future__ import annotations

import numpy as np


def equirect_angles(u: np.ndarray, v: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel centres to ``(theta, phi)`` used by DA360 unprojection.

    ``theta`` is polar angle from +Y down, ``phi`` is azimuth from +Z toward +X.
    """
    theta = np.pi - (v + 0.5) * np.pi / float(height)
    phi = (u + 0.5) * (2.0 * np.pi) / float(width) - np.pi
    return theta, phi


def equirect_uv_to_da360_rays(u: np.ndarray, v: np.ndarray, width: int, height: int) -> np.ndarray:
    """Unit rays in the DA360 optical frame (+X right, +Y down, +Z forward)."""
    theta, phi = equirect_angles(u, v, width, height)
    rays = np.stack(
        [
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
            np.sin(theta) * np.cos(phi),
        ],
        axis=-1,
    )
    return rays.astype(np.float64)


def rays_to_level(rays_da360: np.ndarray, T_level_da360: np.ndarray) -> np.ndarray:
    """Rotate DA360 rays into ``lidar_level``. Translation does not affect direction."""
    R = np.asarray(T_level_da360, dtype=np.float64)[:3, :3]
    return np.asarray(rays_da360, dtype=np.float64) @ R.T


def camera_origin_level(T_level_da360: np.ndarray) -> np.ndarray:
    return np.asarray(T_level_da360, dtype=np.float64)[:3, 3].copy()


def project_level_points_to_equirect(
    points_level: np.ndarray,
    T_level_da360: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project ``lidar_level`` points to equirect pixels. Same inverse as DA360.

    Returns ``u``, ``v``, ``depth`` for points with positive range. ``u`` is wrapped
    into ``[0, width)``.
    """
    points = np.asarray(points_level, dtype=np.float64).reshape(-1, 3)
    T_da360_level = np.linalg.inv(np.asarray(T_level_da360, dtype=np.float64))
    pts = points @ T_da360_level[:3, :3].T + T_da360_level[:3, 3]
    depth = np.linalg.norm(pts, axis=1)
    valid = depth > 1e-6
    x, y, z, depth = pts[valid, 0], pts[valid, 1], pts[valid, 2], depth[valid]
    theta = np.arccos(np.clip(y / depth, -1.0, 1.0))
    phi = np.arctan2(x, z)
    v = (np.pi - theta) * height / np.pi - 0.5
    u = (phi + np.pi) * width / (2.0 * np.pi) - 0.5
    return np.mod(u, width), v, depth


def azimuth_elevation(ray_level: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth about +Z and elevation from the XY plane, ``lidar_level`` (X forward, Y left, Z up)."""
    x, y, z = ray_level[..., 0], ray_level[..., 1], ray_level[..., 2]
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.hypot(x, y))
    return azimuth, elevation
