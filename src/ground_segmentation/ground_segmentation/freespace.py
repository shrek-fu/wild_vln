"""ROS-free core of ``freespace_obstacles_node``: lidar points measured against the semantic freespace.

The camera ground segmentation (``GroundMapper``: GroundingDINO + SAM2 labels lifted with the lidar) gives the
traversable cells around the robot with their ground height. Every lidar point gets its height above the
nearest such cell (within ``search_m``); far from any cell (unseen yet, or outside the camera views) the
ground under the robot is the reference. Points ``h_min``..``h_max`` above that ground are obstacles, so ramps,
kerbs and uneven floors keep their own ground instead of linefit's per-scan line fit.

The cells are rasterised once per freespace update and a distance transform stores each pixel's nearest
cell, so a scan costs one array lookup per point.
"""

from __future__ import annotations

import numpy as np


class LocalFreespace:
    def __init__(self, search_m: float = 1.0, blocked_labels=(), robot_radius_m: float = 1.0,
                 resolution: float = 0.1) -> None:
        self.search_m = float(search_m)
        self.blocked = tuple(int(b) for b in blocked_labels)
        self.robot_radius_m = float(robot_radius_m)
        self.res = float(resolution)
        self.n = 0
        self.origin = np.zeros(2)
        self.ground = None  # (H, W) height of the nearest cell, NaN beyond search_m
        self.cell_z = None  # (H, W) height of the cell itself, NaN where none
        self.blocked_xyz = np.zeros((0, 3))

    def __len__(self) -> int:
        return self.n

    def set_cells(self, xyz: np.ndarray, labels: np.ndarray | None = None) -> None:
        """Freespace cell centres at their ground height (odom) and their labels (class index + 1)."""
        from scipy.ndimage import distance_transform_edt

        xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
        labels = np.zeros(len(xyz), np.int64) if labels is None else np.asarray(labels).astype(np.int64)
        self.n = len(xyz)
        self.blocked_xyz = xyz[np.isin(labels, self.blocked)] if self.blocked else np.zeros((0, 3))
        if not len(xyz):
            self.ground = self.cell_z = None
            return
        pad = int(np.ceil(self.search_m / self.res)) + 1
        self.origin = np.floor(xyz[:, :2].min(axis=0) / self.res) * self.res - pad * self.res
        ij = np.floor((xyz[:, :2] - self.origin) / self.res).astype(np.int64)
        w, h = (ij.max(axis=0) + pad + 1).tolist()
        cell_z = np.full((h, w), np.nan)
        cell_z[ij[:, 1], ij[:, 0]] = xyz[:, 2]
        dist, (ri, ci) = distance_transform_edt(np.isnan(cell_z), return_indices=True)
        ground = cell_z[ri, ci]
        ground[dist * self.res > self.search_m] = np.nan
        self.cell_z, self.ground = cell_z, ground

    def _pixels(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ij = np.floor((np.asarray(xy, np.float64).reshape(-1, 2) - self.origin) / self.res).astype(np.int64)
        h, w = self.ground.shape
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
        return ij[:, 1], ij[:, 0], inside

    def ground_under(self, xy, fallback_z: float) -> float:
        """Median freespace height within ``robot_radius_m`` of ``xy``, else ``fallback_z``."""
        if self.cell_z is None:
            return float(fallback_z)
        r = int(np.ceil(self.robot_radius_m / self.res))
        row, col, inside = self._pixels(np.asarray(xy, np.float64)[:2])
        if not inside[0]:
            return float(fallback_z)
        patch = self.cell_z[max(row[0] - r, 0):row[0] + r + 1, max(col[0] - r, 0):col[0] + r + 1]
        patch = patch[np.isfinite(patch)]
        return float(np.median(patch)) if len(patch) else float(fallback_z)

    def heights(self, pts: np.ndarray, robot_ground_z: float) -> tuple[np.ndarray, np.ndarray]:
        """Height of each point above its ground and whether that ground is a freespace cell (else the
        robot's)."""
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        ground = np.full(len(pts), float(robot_ground_z))
        known = np.zeros(len(pts), bool)
        if self.ground is not None and len(pts):
            row, col, inside = self._pixels(pts[:, :2])
            z = np.full(len(pts), np.nan)
            z[inside] = self.ground[row[inside], col[inside]]
            known = np.isfinite(z)
            ground[known] = z[known]
        return pts[:, 2] - ground, known


def obstacle_mask(pts: np.ndarray, h: np.ndarray, robot_xy, *, h_min: float, h_max: float, clear_radius: float,
                  half_m: float) -> np.ndarray:
    """Points ``h_min``..``h_max`` above the ground, inside the window, outside the robot's own footprint."""
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    rel = pts[:, :2] - np.asarray(robot_xy, np.float64)[:2]
    return ((h > h_min) & (h < h_max) & np.all(np.abs(rel) <= half_m, axis=1)
            & (np.hypot(rel[:, 0], rel[:, 1]) >= clear_radius))
