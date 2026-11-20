"""Object centroids from 2D tracks and VIO poses, without depth.

Each track id accumulates one viewing ray per detection (camera centre and the
direction through the mask centroid, both in odom). With enough parallax the
rays are triangulated; before that, the range along the latest ray comes from
the class's typical size and the box height (or width when the box is cut by
the image border).
"""

from __future__ import annotations

import numpy as np

# Typical (width, depth, height) in metres, used for the range prior and as the
# box size of the exported objects.
CLASS_SIZE_M: dict[str, tuple[float, float, float]] = {
    "person": (0.5, 0.3, 1.7),
    "door": (0.9, 0.1, 2.0),
    "chair": (0.5, 0.5, 0.9),
    "table": (1.2, 0.7, 0.75),
    "sofa": (1.8, 0.9, 0.85),
    "bed": (1.6, 2.0, 0.6),
    "stairs": (1.2, 3.0, 2.5),
    "backpack": (0.3, 0.2, 0.45),
    "bottle": (0.08, 0.08, 0.25),
    "cup": (0.08, 0.08, 0.1),
    "potted plant": (0.4, 0.4, 0.8),
    "tv": (1.0, 0.1, 0.6),
    "refrigerator": (0.7, 0.7, 1.8),
    "sink": (0.6, 0.5, 0.3),
    "toilet": (0.4, 0.7, 0.8),
    "laptop": (0.35, 0.25, 0.25),
    "book": (0.2, 0.05, 0.25),
    "clock": (0.3, 0.05, 0.3),
    "trash can": (0.35, 0.35, 0.6),
    "elevator": (1.2, 0.3, 2.2),
    "robot": (0.8, 0.6, 1.2),
    "car": (1.8, 4.5, 1.5),
    "bicycle": (0.6, 1.8, 1.1),
    "bench": (1.5, 0.5, 0.5),
}
DEFAULT_SIZE_M = (0.5, 0.5, 0.5)


def class_size(name: str) -> tuple[float, float, float]:
    return CLASS_SIZE_M.get(str(name), DEFAULT_SIZE_M)


def class_max_size(name: str, scale: float = 1.5, margin_m: float = 0.4, z_scale: float = 1.3,
                   z_margin_m: float = 0.3) -> tuple[float, float, float]:
    """Largest plausible axis-aligned box of a class: the typical size scaled up, with either horizontal side
    along x or y (yaw is unknown)."""
    w, d, h = class_size(name)
    side = max(w, d) * scale + margin_m
    return side, side, h * z_scale + z_margin_m


def pixel_center(det: dict) -> np.ndarray:
    """Mask centroid (vertex mean of the contour), else the box centre."""
    mask = det.get("mask")
    if isinstance(mask, (list, tuple)) and len(mask) >= 3:
        return np.asarray(mask, dtype=np.float64).reshape(-1, 2).mean(axis=0)
    x1, y1, x2, y2 = (float(v) for v in det["xyxy"])
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5])


def viewing_ray(uv, K, T_world_optical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(origin, unit direction)`` in world for pixel ``uv``."""
    fx, fy, cx, cy = K
    ray = np.array([(uv[0] - cx) / fx, (uv[1] - cy) / fy, 1.0])
    T = np.asarray(T_world_optical, dtype=np.float64)
    d = T[:3, :3] @ (ray / np.linalg.norm(ray))
    return T[:3, 3].copy(), d


def _solve(origins: np.ndarray, dirs: np.ndarray) -> np.ndarray | None:
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    if np.linalg.cond(A) > 1e6:
        return None
    return np.linalg.solve(A, b)


def max_parallax_deg(dirs: np.ndarray) -> float:
    dots = np.clip(dirs @ dirs.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(dots.min())))


def _inliers(X: np.ndarray, origins: np.ndarray, dirs: np.ndarray, max_err_deg: float, min_range_m: float) -> np.ndarray:
    v = X - origins
    along = np.einsum("ij,ij->i", v, dirs)
    cos = along / np.maximum(np.linalg.norm(v, axis=1), 1e-9)
    return (along >= min_range_m) & (np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) <= max_err_deg)


def triangulate_rays(
    origins: np.ndarray,
    dirs: np.ndarray,
    min_rays: int = 3,
    min_parallax_deg: float = 3.0,
    max_err_deg: float = 2.0,
    min_range_m: float = 0.3,
    max_range_m: float = 20.0,
    max_pairs: int = 60,
) -> np.ndarray | None:
    """Ray intersection: best-supported pair hypothesis, then least squares on its inliers.

    A ray counts as an inlier when the point lies in front of it and within
    ``max_err_deg`` of its direction, so one jumpy mask centroid cannot drag the result.
    """
    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    dirs = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
    n = len(dirs)
    if n < min_rays or max_parallax_deg(dirs) < min_parallax_deg:
        return None
    cos_min = np.cos(np.radians(min_parallax_deg))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if dirs[i] @ dirs[j] <= cos_min]
    if len(pairs) > max_pairs:
        pick = np.random.default_rng(0).choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[k] for k in pick]
    best = None
    for i, j in pairs:
        X = _solve(origins[[i, j]], dirs[[i, j]])
        if X is None:
            continue
        mask = _inliers(X, origins, dirs, max_err_deg, min_range_m)
        if best is None or mask.sum() > best.sum():
            best = mask
    if best is None or best.sum() < min_rays or max_parallax_deg(dirs[best]) < min_parallax_deg:
        return None
    X = _solve(origins[best], dirs[best])
    if X is None or not _inliers(X, origins[best], dirs[best], 2 * max_err_deg, min_range_m).all():
        return None
    if np.linalg.norm(X - origins[best].mean(axis=0)) > max_range_m:
        return None
    return X


def prior_depth(name: str, xyxy, K, width: int, height: int, margin: float = 3.0) -> float | None:
    """Optical-axis depth from the class height (or width if the box is cut top/bottom)."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    fx, fy = K[0], K[1]
    w_m, d_m, h_m = class_size(name)
    cut_v = y1 <= margin or y2 >= height - 1 - margin
    cut_h = x1 <= margin or x2 >= width - 1 - margin
    if not cut_v and y2 - y1 > 4:
        return fy * h_m / (y2 - y1)
    if not cut_h and x2 - x1 > 4:
        return fx * 0.5 * (w_m + d_m) / (x2 - x1)
    return None


class CentroidLifter:
    """Per-track ray memory; ``observe`` returns the current best centre of the track."""

    def __init__(
        self,
        min_rays: int = 3,
        min_parallax_deg: float = 3.0,
        min_prior_obs: int = 5,
        max_rays: int = 60,
        forget_s: float = 120.0,
    ):
        self.min_rays = int(min_rays)
        self.min_parallax_deg = float(min_parallax_deg)
        # Class-size ranges scatter widely (about a third within ±30 % of the
        # triangulated range), so a track needs this many frames before its
        # median prior position is trusted; 1–2 frame tracks are mostly glimpses.
        self.min_prior_obs = int(min_prior_obs)
        self.max_rays = int(max_rays)
        self.forget_s = float(forget_s)
        self._rays: dict[int, list[tuple[float, np.ndarray, np.ndarray, np.ndarray | None]]] = {}

    def observe(
        self,
        track_id: int,
        name: str,
        xyxy,
        uv,
        K,
        T_world_optical: np.ndarray,
        width: int,
        height: int,
        stamp_s: float,
    ) -> tuple[np.ndarray | None, str | None]:
        """``(centre, source)`` with source ``tri`` (triangulated) or ``prior`` (class size)."""
        origin, direction = viewing_ray(uv, K, T_world_optical)
        z = prior_depth(name, xyxy, K, width, height)
        optical_z = np.asarray(T_world_optical, dtype=np.float64)[:3, 2]
        guess = None if z is None else origin + direction * (z / max(float(direction @ optical_z), 1e-3))
        rays = self._rays.setdefault(int(track_id), [])
        rays.append((float(stamp_s), origin, direction, guess))
        del rays[: max(0, len(rays) - self.max_rays)]
        X = triangulate_rays(
            np.array([r[1] for r in rays]),
            np.array([r[2] for r in rays]),
            min_rays=self.min_rays,
            min_parallax_deg=self.min_parallax_deg,
        )
        if X is not None:
            return X, "tri"
        guesses = [r[3] for r in rays if r[3] is not None]
        if len(rays) < self.min_prior_obs or len(guesses) < self.min_rays:
            return None, None
        return np.median(np.stack(guesses), axis=0), "prior"

    def forget(self, stamp_s: float) -> None:
        stale = [tid for tid, rays in self._rays.items() if stamp_s - rays[-1][0] > self.forget_s]
        for tid in stale:
            del self._rays[tid]
