"""Lift one equirect YOLOE box to a gravity-aligned 3D box with lidar points."""

from __future__ import annotations

import numpy as np

from scene_memory.common.geometry import project_level_points_to_equirect
from scene_memory.common.obb import fit_gravity_obb


def _inside_parts(u: np.ndarray, v: np.ndarray, parts: list[tuple[float, float, float, float]]) -> np.ndarray:
    inside = np.zeros(u.shape[0], dtype=bool)
    for x1, y1, x2, y2 in parts:
        inside |= (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return inside


def points_in_detection(
    points_level: np.ndarray,
    T_level_da360: np.ndarray,
    parts: list[tuple[float, float, float, float]],
    width: int,
    height: int,
    depth_extent: float,
    max_range: float,
) -> np.ndarray:
    """Lidar points that land in the 2D box and on its nearest depth slab.

    The slab starts at the near end of the frustum so the wall behind the object
    does not stretch the box. This is the geometric stand-in for Boxer's
    semi-dense points associated with one 2D detection.
    """
    if points_level.size == 0 or not parts:
        return np.zeros((0, 3), dtype=np.float64)
    u, v, depth = project_level_points_to_equirect(points_level, T_level_da360, height, width)
    finite = np.isfinite(depth) & (depth > 0.05) & (depth < max_range) & (v >= 0.0) & (v < height)
    u, v, depth = u[finite], v[finite], depth[finite]
    source = np.asarray(points_level, dtype=np.float64).reshape(-1, 3)[finite]
    chosen = source[_inside_parts(u, v, parts)]
    chosen_depth = depth[_inside_parts(u, v, parts)]
    if chosen.shape[0] == 0:
        return chosen
    near = float(np.percentile(chosen_depth, 10.0))
    return chosen[chosen_depth <= near + depth_extent]


def lift_detection(
    points_level: np.ndarray,
    T_level_da360: np.ndarray,
    parts: list[tuple[float, float, float, float]],
    width: int,
    height: int,
    depth_extent: float = 1.5,
    max_range: float = 20.0,
    min_points: int = 20,
    max_size: float = 4.0,
) -> dict | None:
    """Fit one Boxer-style gravity OBB, or return None when the points are too few or too large."""
    pts = points_in_detection(
        points_level, T_level_da360, parts, width, height, depth_extent, max_range
    )
    if pts.shape[0] < min_points:
        return None
    obb = fit_gravity_obb(pts)
    if obb is None:
        return None
    if float(np.max(obb["size"])) > max_size:
        return None
    obb["num_points"] = int(pts.shape[0])
    return obb


def attach_obbs(
    detections: list[dict],
    points_level: np.ndarray | None,
    T_level_da360: np.ndarray,
    width: int,
    height: int,
    depth_extent: float,
    max_range: float,
    min_points: int,
    max_size: float,
) -> int:
    """Write ``det['obb']`` for detections that have enough lidar support. Returns how many."""
    if points_level is None or len(detections) == 0:
        return 0
    fitted = 0
    for det in detections:
        parts = det.get("parts") or []
        obb = lift_detection(
            points_level,
            T_level_da360,
            parts,
            width,
            height,
            depth_extent=depth_extent,
            max_range=max_range,
            min_points=min_points,
            max_size=max_size,
        )
        if obb is None:
            det.pop("obb", None)
            continue
        det["obb"] = {
            "center": [float(v) for v in obb["center"]],
            "size": [float(v) for v in obb["size"]],
            "yaw": float(obb["yaw"]),
            "num_points": int(obb["num_points"]),
        }
        fitted += 1
    return fitted
