"""Free-space places and object approach poses from the lidar map (Hydra's GVD places, 2D and lidar only).

The world-frame lidar map is rasterised at ``res``: the floor under every cell is the nearest trajectory
pose minus the sensor height (estimated from the returns around the trajectory), returns
``obstacle_band`` above it are obstacles, returns within ``ground_tol`` of it are observed floor, and
the trajectory swept with ``robot_radius`` is free. ``clearance`` is the distance to the nearest
obstacle; a cell is traversable when free with clearance >= ``robot_radius`` and connected to the
trajectory.

* places: in every ``spacing`` block the traversable cell with the largest clearance (a coarse
  Voronoi skeleton), linked to its 8 neighbour blocks when the straight segment stays traversable;
* approach: per object, the nearest traversable cell between ``standoff`` of its footprint with a
  clear line towards it; the heading faces the object. This is the navigation goal for the object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    origin: np.ndarray
    res: float
    obstacle: np.ndarray
    free: np.ndarray
    clearance: np.ndarray
    traversable: np.ndarray
    sensor_height: float

    def cell(self, xy) -> tuple[int, int]:
        ij = np.floor((np.asarray(xy, dtype=np.float64)[:2] - self.origin) / self.res).astype(int)
        return int(ij[1]), int(ij[0])

    def xy(self, r: int, c: int) -> np.ndarray:
        return self.origin + (np.array([c, r]) + 0.5) * self.res

    def inside(self, r: int, c: int) -> bool:
        return 0 <= r < self.free.shape[0] and 0 <= c < self.free.shape[1]


def estimate_sensor_height(points: np.ndarray, traj: np.ndarray, radius: float = 1.2, default: float = 0.5) -> float:
    from scipy.spatial import cKDTree

    if len(points) < 100 or len(traj) == 0:
        return default
    tree = cKDTree(traj[:, :2])
    d, i = tree.query(points[:, :2], distance_upper_bound=radius)
    ok = np.isfinite(d)
    if ok.sum() < 50:
        return default
    dz = points[ok, 2] - traj[i[ok], 2]
    below = dz[dz < -0.1]
    return float(-np.percentile(below, 10)) if len(below) > 50 else default


def build_grid(points: np.ndarray, traj: np.ndarray, res: float = 0.1, robot_radius: float = 0.35,
               obstacle_band: tuple[float, float] = (0.15, 1.6), ground_tol: float = 0.12, max_range: float = 15.0,
               sensor_height: float | None = None, margin: float = 1.0) -> Grid | None:
    import cv2
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree

    if len(traj) == 0 or len(points) == 0:
        return None
    h = estimate_sensor_height(points, traj) if sensor_height is None else sensor_height
    tree = cKDTree(traj[:, :2])
    d, i = tree.query(points[:, :2], distance_upper_bound=max_range)
    ok = np.isfinite(d)
    pts, near = points[ok], traj[i[ok]]
    if not len(pts):
        return None
    lo = np.minimum(pts[:, :2].min(0), traj[:, :2].min(0)) - margin
    hi = np.maximum(pts[:, :2].max(0), traj[:, :2].max(0)) + margin
    shape = (int(math.ceil((hi[1] - lo[1]) / res)), int(math.ceil((hi[0] - lo[0]) / res)))
    rc = np.floor((pts[:, :2] - lo) / res).astype(int)[:, ::-1]
    height = pts[:, 2] - (near[:, 2] - h)
    obstacle = np.zeros(shape, bool)
    ground = np.zeros(shape, bool)
    ob = (height > obstacle_band[0]) & (height < obstacle_band[1])
    obstacle[rc[ob, 0], rc[ob, 1]] = True
    gr = np.abs(height) < ground_tol
    ground[rc[gr, 0], rc[gr, 1]] = True
    swept = np.zeros(shape, np.uint8)
    tr = np.floor((traj[:, :2] - lo) / res).astype(int)
    for a, b in zip(tr[:-1], tr[1:]):
        cv2.line(swept, tuple(map(int, a)), tuple(map(int, b)), 1, thickness=max(1, int(2 * robot_radius / res)))
    free = (ground | swept.astype(bool))
    free = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool) & ~obstacle
    clearance = distance_transform_edt(~obstacle) * res
    trav = free & (clearance >= robot_radius)
    n, labels = cv2.connectedComponents(trav.astype(np.uint8), connectivity=8)
    seeds = {int(labels[r, c]) for c, r in tr if 0 <= r < shape[0] and 0 <= c < shape[1] and labels[r, c] > 0}
    trav = np.isin(labels, list(seeds)) if seeds else trav
    return Grid(lo, res, obstacle, free, clearance, trav, h)


def _line_clear(grid: Grid, a, b, mask: np.ndarray, stop_m: float = 0.0) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    n = max(int((length - stop_m) / (grid.res * 0.5)), 1)
    for s in np.linspace(0.0, max(length - stop_m, 0.0) / max(length, 1e-9), n + 1):
        r, c = grid.cell(a + s * (b - a))
        if not grid.inside(r, c) or not mask[r, c]:
            return False
    return True


def free_places(grid: Grid, spacing: float = 1.0) -> tuple[list[dict], list[tuple[int, int]]]:
    step = max(int(round(spacing / grid.res)), 1)
    H, W = grid.traversable.shape
    blocks: dict[tuple[int, int], int] = {}
    places: list[dict] = []
    for br in range(0, H, step):
        for bc in range(0, W, step):
            t = grid.traversable[br: br + step, bc: bc + step]
            if t.sum() < 0.15 * step * step:
                continue
            cl = np.where(t, grid.clearance[br: br + step, bc: bc + step], -1.0)
            r, c = np.unravel_index(int(np.argmax(cl)), cl.shape)
            blocks[(br // step, bc // step)] = len(places)
            places.append({"id": len(places), "xy": grid.xy(br + r, bc + c).round(3).tolist(),
                           "clearance": round(float(cl[r, c]), 3)})
    edges = []
    for (i, j), a in blocks.items():
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
            b = blocks.get((i + di, j + dj))
            if b is not None and _line_clear(grid, places[a]["xy"], places[b]["xy"], grid.traversable):
                edges.append((a, b))
    return places, edges


def approach(grid: Grid, center, size, places: list[dict], standoff: tuple[float, float] = (0.4, 1.5)) -> dict | None:
    """Navigation goal for one object: nearest reachable cell around its footprint with a clear view."""
    c = np.asarray(center, float)[:2]
    radius = 0.5 * float(np.hypot(size[0], size[1])) if size is not None else 0.3
    radius = min(radius, 1.5)
    r0, c0 = grid.cell(c)
    reach = int(math.ceil((radius + standoff[1]) / grid.res))
    rs, cs = np.mgrid[max(r0 - reach, 0): min(r0 + reach + 1, grid.free.shape[0]),
                      max(c0 - reach, 0): min(c0 + reach + 1, grid.free.shape[1])]
    if not rs.size:
        return None
    xy = grid.origin + (np.stack([cs, rs], -1) + 0.5) * grid.res
    d = np.linalg.norm(xy - c, axis=-1)
    ok = grid.traversable[rs, cs] & (d >= radius + standoff[0]) & (d <= radius + standoff[1])
    order = np.argsort(np.where(ok, d - 0.2 * np.minimum(grid.clearance[rs, cs], 1.0), np.inf), axis=None)
    see = ~grid.obstacle
    for flat in order[:400]:
        if not ok.flat[flat]:
            break
        p = xy.reshape(-1, 2)[flat]
        if not _line_clear(grid, p, c, see, stop_m=radius + 0.1):
            continue
        place = None
        if places:
            pd = [float(np.linalg.norm(np.subtract(q["xy"], p))) for q in places]
            for k in np.argsort(pd)[:5]:
                if _line_clear(grid, places[k]["xy"], p, grid.traversable):
                    place = int(places[k]["id"])
                    break
        return {"xy": p.round(3).tolist(), "yaw": round(math.atan2(c[1] - p[1], c[0] - p[0]), 4),
                "standoff_m": round(float(d.flat[flat]), 3), "clearance_m": round(float(grid.clearance[rs, cs].flat[flat]), 3),
                "free_place": place}
    return None
