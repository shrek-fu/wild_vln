"""Equirect depth ↔ 3D helpers and lidar scale alignment."""

from __future__ import annotations

import numpy as np


def da360_depth_to_points(depth: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Unproject equirectangular depth with the same convention as ``saver.py``.

    Returns Nx3 points in the DA360 optical frame (+Z forward, +X right, +Y down).
    """
    h, w = depth.shape
    theta = np.pi - np.arange(h, dtype=np.float64).reshape(h, 1) * np.pi / h - np.pi / h / 2.0
    theta = np.repeat(theta, w, axis=1)
    phi = np.arange(w, dtype=np.float64).reshape(1, w) * 2.0 * np.pi / w + np.pi / w - np.pi
    phi = np.repeat(phi, h, axis=0)

    x = depth * np.sin(theta) * np.sin(phi)
    y = depth * np.cos(theta)
    z = depth * np.sin(theta) * np.cos(phi)
    if mask is None:
        return np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return np.stack([x[mask], y[mask], z[mask]], axis=1)


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to Nx3 points."""
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def depth_from_level_points(points_level: np.ndarray) -> np.ndarray:
    """Per-point range from the lidar_level origin (global depth)."""
    return np.linalg.norm(points_level, axis=1)


def pack_global_depth_image(
    depth_cam: np.ndarray,
    T_level_da360: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build an HxW map of ||p_level|| for each valid pixel (0 elsewhere)."""
    h, w = depth_cam.shape
    if mask is None:
        mask = np.isfinite(depth_cam) & (depth_cam > 0)
    pts = da360_depth_to_points(depth_cam, mask)
    pts_level = transform_points(pts, T_level_da360)
    ranges = depth_from_level_points(pts_level)
    out = np.zeros((h, w), dtype=np.float32)
    out[mask] = ranges.astype(np.float32)
    return out


def project_level_points_to_equirect(
    points_level: np.ndarray,
    T_level_da360: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project lidar_level points into the equirect; return u, v, depth_in_camera.

    Points behind the unprojection singularity or outside the image are dropped.
    """
    T_da360_level = np.linalg.inv(T_level_da360)
    pts = transform_points(points_level, T_da360_level)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    depth = np.linalg.norm(pts, axis=1)
    valid = depth > 1e-6
    x, y, z, depth = x[valid], y[valid], z[valid], depth[valid]
    # Inverse of saver unprojection: theta = acos(y/r), phi = atan2(x, z)
    theta = np.arccos(np.clip(y / depth, -1.0, 1.0))
    phi = np.arctan2(x, z)
    # saver: theta = pi - (v+0.5)*pi/h  => v = (pi - theta)*h/pi - 0.5
    # saver: phi = (u+0.5)*2pi/w - pi   => u = (phi+pi)*w/(2pi) - 0.5
    v = (np.pi - theta) * height / np.pi - 0.5
    u = (phi + np.pi) * width / (2.0 * np.pi) - 0.5
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    return ui[inside], vi[inside], depth[inside].astype(np.float32)


def estimate_scale_from_lidar(
    depth_cam: np.ndarray,
    points_level: np.ndarray,
    T_level_da360: np.ndarray,
    min_depth: float = 1.0,
    max_depth: float = 40.0,
) -> float | None:
    """Median scale so that ``scale * depth_cam`` matches lidar ranges in camera.

    Returns None if there are too few overlaps.
    """
    h, w = depth_cam.shape
    u, v, lidar_depth = project_level_points_to_equirect(
        points_level, T_level_da360, h, w
    )
    if u.size < 50:
        return None
    pred = depth_cam[v, u]
    keep = (
        np.isfinite(pred)
        & (pred > 1e-3)
        & (lidar_depth > min_depth)
        & (lidar_depth < max_depth)
    )
    if keep.sum() < 50:
        return None
    ratios = lidar_depth[keep] / pred[keep]
    scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= 0:
        return None
    return scale
