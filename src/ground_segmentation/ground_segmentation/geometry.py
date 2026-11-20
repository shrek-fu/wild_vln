"""Ground plane and 3D lifting of a 2D traversable mask.

A plane is ``(n, d)`` with ``n . p + d = 0``, unit ``n`` pointing up (``n_z > 0``) in ``odom``, so
``n . p + d`` is the height of ``p`` above the ground.
"""

from __future__ import annotations

import cv2
import numpy as np


def fit_ground_plane(xyz: np.ndarray, dist: float = 0.05, iters: int = 200, max_tilt_deg: float = 10.0,
                     min_inliers: int = 30, low_share: float = 0.3, prior=None, prior_tol: float = 0.25,
                     prior_angle_deg: float = 5.0, rng: np.random.Generator | None = None):
    """RANSAC plane within ``max_tilt_deg`` of horizontal (``odom`` z is gravity-aligned), refined by least
    squares on its inliers.

    The ground is the lowest large surface: of the hypotheses with at least ``low_share`` of the best
    support, the lowest (at the points' centroid) wins, so a desk top or a leaked bag does not.
    With ``prior`` (a plane), only points within ``prior_tol`` of it are used and the result must stay
    within ``prior_angle_deg`` of its normal.
    Returns ``(n, d, inliers)`` (inliers over the input points) or None."""
    pts = np.asarray(xyz, np.float64).reshape(-1, 3)
    if prior is not None and len(pts):
        near = np.abs(heights(pts, prior)) < prior_tol
        fit = fit_ground_plane(pts[near], dist, iters, max_tilt_deg, min_inliers, low_share, rng=rng)
        if fit is None or float(fit[0] @ np.asarray(prior[0], np.float64)) < np.cos(np.deg2rad(prior_angle_deg)):
            return None
        n, d, _ = fit
        return n, d, np.abs(pts @ n + d) < dist
    if len(pts) < max(3, min_inliers):
        return None
    rng = rng or np.random.default_rng(0)
    tri = pts[rng.integers(0, len(pts), size=(iters, 3))]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(norm, 1e-12)
    n *= np.sign(n[:, 2:3] + 1e-12)
    ok = (norm[:, 0] > 1e-9) & (n[:, 2] >= np.cos(np.deg2rad(max_tilt_deg)))
    if not ok.any():
        return None
    n = n[ok]
    d = -np.einsum("ij,ij->i", n, tri[ok, 0])
    counts = (np.abs(pts @ n.T + d) < dist).sum(axis=0)
    if counts.max() < min_inliers:
        return None
    strong = np.flatnonzero(counts >= max(min_inliers, low_share * counts.max()))
    c = pts.mean(axis=0)
    z_at_c = -(n[strong, :2] @ c[:2] + d[strong]) / n[strong, 2]
    best = int(strong[z_at_c.argmin()])
    inliers = np.abs(pts @ n[best] + d[best]) < dist
    for _ in range(2):
        sel = pts[inliers]
        c = sel.mean(axis=0)
        normal = np.linalg.svd(sel - c, full_matrices=False)[2][-1]
        normal *= np.sign(normal[2] + 1e-12)
        if normal[2] < np.cos(np.deg2rad(max_tilt_deg)):
            normal = n[best]
        offset = -float(normal @ c)
        refit = np.abs(pts @ normal + offset) < dist
        if refit.sum() < min_inliers:
            break
        inliers = refit
    return normal, offset, inliers


def heights(xyz: np.ndarray, plane) -> np.ndarray:
    n, d = plane[0], plane[1]
    return np.asarray(xyz, np.float64).reshape(-1, 3) @ np.asarray(n, np.float64) + float(d)


def local_slope_deg(top: np.ndarray, res: float, radius: int = 2, min_cells: int = 6) -> np.ndarray:
    """Per cell of a height raster (NaN = empty): slope of the plane fitted to the occupied cells in the
    ``(2 radius + 1)²`` window, degrees; NaN where too few cells or they are collinear."""
    occ = ~np.isnan(top)
    ii, jj = np.indices(top.shape, dtype=np.float64)
    x, y, z = ii * res, jj * res, np.where(occ, top, 0.0)
    m = occ.astype(np.float64)
    k = (2 * int(radius) + 1,) * 2

    def box(a):
        return cv2.boxFilter(a, -1, k, normalize=False, borderType=cv2.BORDER_CONSTANT)

    n = box(m)
    sx, sy, sz = box(m * x), box(m * y), box(z)
    nn = np.maximum(n, 1.0)
    cxx = box(m * x * x) - sx * sx / nn
    cyy = box(m * y * y) - sy * sy / nn
    cxy = box(m * x * y) - sx * sy / nn
    cxz = box(x * z) - sx * sz / nn
    cyz = box(y * z) - sy * sz / nn
    det = cxx * cyy - cxy * cxy
    ok = occ & (n >= min_cells) & (det > 1e-3 * (res * res * n) ** 2)
    det = np.where(ok, det, 1.0)
    a = (cyy * cxz - cxy * cyz) / det
    b = (cxx * cyz - cxy * cxz) / det
    return np.where(ok, np.degrees(np.arctan(np.hypot(a, b))), np.nan)


def step_reachable(xy: np.ndarray, h: np.ndarray, res: float = 0.1, max_step: float = 0.22,
                   radius: int = 2, max_slope_deg: float | None = None, slope_window: float = 0.9) -> np.ndarray:
    """Bool per point: surface the robot can climb. Points more than ``max_step`` above their cell's lowest
    point are vertical structure; the remaining cells (height = highest kept point) must be reachable from
    plane level (|h| <= max_step) through neighbours within ``radius`` cells, each hop changing height by at
    most ``max_step``, and (``max_slope_deg``) not be steeper than that over ``slope_window`` metres (wide
    enough to span several stair treads, so stairs read as their pitch). Rejects table
    tops and shelves that a detector calls stairs, and stairs / ramps beyond the robot's limits."""
    if len(h) == 0:
        return np.zeros(0, bool)
    ij = np.floor(np.asarray(xy)[:, :2] / res).astype(np.int64)
    ij -= ij.min(axis=0) - radius
    shape = tuple(ij.max(axis=0) + radius + 1)
    flat = np.ravel_multi_index(ij.T, shape)
    zmin = np.full(shape[0] * shape[1], np.inf)
    np.minimum.at(zmin, flat, h)
    low = h <= zmin[flat] + max_step
    top = np.full(zmin.shape, np.nan)
    np.fmax.at(top, flat[low], h[low])
    top = top.reshape(shape)
    occ = ~np.isnan(top)
    if max_slope_deg is not None:
        slope = local_slope_deg(top, res, max(1, int(round(0.5 * slope_window / res))))
        occ &= ~(slope > max_slope_deg)
        top = np.where(occ, top, np.nan)
    reached = occ & (np.abs(top) <= max_step)
    r = int(radius)
    while True:
        grow = np.zeros_like(reached)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di == 0 and dj == 0:
                    continue
                src_r = np.roll(np.roll(reached, di, 0), dj, 1)
                src_z = np.roll(np.roll(top, di, 0), dj, 1)
                grow |= src_r & (np.abs(top - src_z) <= max_step)
        grow &= occ & ~reached
        if not grow.any():
            break
        reached |= grow
    return low & reached.ravel()[flat]


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.astype(bool)
    k = 2 * int(px) + 1
    return cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def ray_plane_fill(mask: np.ndarray, K, T_odom_optical: np.ndarray, plane, stride: int = 8, max_range: float = 6.0,
                   sparse_depth: np.ndarray | None = None, block_ratio: float = 0.85, return_pixels: bool = False):
    """``(M, 3)`` odom ground points where the camera rays of mask pixels (every ``stride`` px) hit the plane,
    within ``max_range`` m horizontally of the camera. With ``sparse_depth`` (lidar optical z, 0 = none), a ray
    is dropped when a lidar return in that pixel is nearer than ``block_ratio`` of the hit (something stands
    in front of the ground there). ``return_pixels``: also the ``(u, v)`` int arrays of the kept rays."""
    h, w = mask.shape
    vv, uu = np.mgrid[stride // 2:h:stride, stride // 2:w:stride]
    keep = mask[vv, uu]
    u, v = uu[keep].astype(np.float64), vv[keep].astype(np.float64)
    if len(u) == 0:
        empty = np.zeros((0, 3))
        return (empty, (np.zeros(0, int), np.zeros(0, int))) if return_pixels else empty
    fx, fy, cx, cy = K
    T = np.asarray(T_odom_optical, np.float64)
    rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=1) @ T[:3, :3].T
    origin = T[:3, 3]
    n, d = np.asarray(plane[0], np.float64), float(plane[1])
    denom = rays @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -(origin @ n + d) / denom
    ok = np.isfinite(t) & (t > 0) & (denom < 0)
    pts = origin + t[:, None] * rays
    ok &= np.hypot(pts[:, 0] - origin[0], pts[:, 1] - origin[1]) <= max_range
    if sparse_depth is not None:
        z = sparse_depth[v.astype(int), u.astype(int)]
        ok &= ~((z > 0) & (z < block_ratio * t))
    if return_pixels:
        return pts[ok], (u[ok].astype(int), v[ok].astype(int))
    return pts[ok]


def min_depth_image(u, v, z, width: int, height: int, radius: int = 4) -> np.ndarray:
    """Nearest lidar optical z per pixel, grown to ``(2 radius + 1)`` squares; 0 = no return."""
    depth = np.full((height, width), np.inf, np.float32)
    np.minimum.at(depth, (v, u), np.asarray(z, np.float32))
    if radius > 0:
        k = 2 * int(radius) + 1
        depth = -cv2.dilate(-depth, np.ones((k, k), np.uint8))
    depth[~np.isfinite(depth) | (depth > 1e6)] = 0.0
    return depth
