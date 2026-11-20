"""Front-hemisphere metric depth: DA360 shape pinned to the LIO cloud.

The front hemisphere is the middle half of the Insta360 equirect (azimuth -90..90 deg about the DA360 +Z,
full polar range), so a crop ``[W/4, 3W/4)`` of the equirect image lines up pixel for pixel. Depth is the
range ``||p||`` from the camera centre, as DA360 predicts it.

* ``hemi_project`` / ``sparse_hemi_depth``: z-buffered lidar range, with returns that sit just behind a
  nearer one in their neighbourhood dropped (background leaking past foreground edges through the
  lidar-camera parallax);
* ``align_local``: ``depth = relative * exp(log_scale(u, v))``. The log scale is the robust global median of
  ``log(lidar / relative)`` plus a smooth residual field (normalised convolution of the per-pixel residuals),
  so every region within lidar reach is pinned by its own returns. Pixels with a consistent return keep the
  lidar value. ``source`` marks each pixel: lidar, aligned (near and supported), or coarse (global scale only).
"""

from __future__ import annotations

import cv2
import numpy as np

SOURCE_NONE, SOURCE_LIDAR, SOURCE_ALIGNED, SOURCE_COARSE = 0, 1, 2, 3


def crop_front(equirect: np.ndarray) -> np.ndarray:
    """Front hemisphere columns of a full equirect image (or depth map)."""
    w = equirect.shape[1]
    return equirect[:, w // 4: w // 4 + w // 2]


def hemi_project(xyz_ref: np.ndarray, T_ref_da360: np.ndarray, width: int, height: int,
                 r_min: float = 0.3, r_max: float = 60.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(u, v, r)`` integer pixels in the front-hemisphere equirect for points in front of the camera.

    ``ref`` is the cloud's frame: the front lidar for raw scans (``T_front_da360``), ``odom`` for LIO clouds."""
    T = np.linalg.inv(np.asarray(T_ref_da360, dtype=np.float64)).astype(np.float32)
    p = np.asarray(xyz_ref, dtype=np.float32).reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
    r = np.linalg.norm(p, axis=1)
    keep = (r > r_min) & (r < r_max) & (p[:, 2] >= 0.0)
    p, r = p[keep], r[keep]
    theta = np.arccos(np.clip(p[:, 1] / r, -1.0, 1.0))
    phi = np.arctan2(p[:, 0], p[:, 2])
    u = np.rint((phi + 0.5 * np.pi) * width / np.pi - 0.5).astype(np.int32)
    v = np.rint((np.pi - theta) * height / np.pi - 0.5).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[inside], v[inside], r[inside]


def sparse_hemi_depth(xyz_ref: np.ndarray, T_ref_da360: np.ndarray, width: int, height: int,
                      r_min: float = 0.3, r_max: float = 60.0, occlusion_px: int = 2,
                      occlusion_rel: float = 0.08) -> np.ndarray:
    """Sparse range image (float32, 0 = no return). A return is dropped when a return within
    ``occlusion_px`` is nearer by more than ``occlusion_rel`` of its range (plus 0.1 m)."""
    u, v, r = hemi_project(xyz_ref, T_ref_da360, width, height, r_min, r_max)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(depth, (v, u), r.astype(np.float32))
    if occlusion_px > 0:
        k = 2 * int(occlusion_px) + 1
        near = cv2.erode(depth, np.ones((k, k), np.uint8), borderType=cv2.BORDER_REPLICATE)
        depth[depth > near * (1.0 + occlusion_rel) + 0.1] = np.inf
    depth[~np.isfinite(depth)] = 0.0
    return depth


def _gate(residual: np.ndarray, min_tolerance: float) -> np.ndarray:
    """Inliers of a residual within 3 MAD of its median (at least ``log(min_tolerance)``)."""
    center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - center)))
    return np.abs(residual - center) <= max(3.0 * 1.4826 * mad, float(np.log(min_tolerance)))


def _residual_field(vv, uu, resid, shape, cell_px, sigma_cells) -> tuple[np.ndarray, np.ndarray]:
    """Normalised convolution of per-return residuals on a ``cell_px`` grid: ``(field, support)``."""
    gh, gw = -(-shape[0] // cell_px), -(-shape[1] // cell_px)
    resid_sum = np.zeros((gh, gw), np.float32)
    count = np.zeros((gh, gw), np.float32)
    np.add.at(resid_sum, (vv // cell_px, uu // cell_px), resid.astype(np.float32))
    np.add.at(count, (vv // cell_px, uu // cell_px), 1.0)
    ksize = 2 * int(np.ceil(3 * sigma_cells)) + 1
    resid_b = cv2.GaussianBlur(resid_sum, (ksize, ksize), sigma_cells, borderType=cv2.BORDER_REPLICATE)
    count_b = cv2.GaussianBlur(count, (ksize, ksize), sigma_cells, borderType=cv2.BORDER_REPLICATE)
    return np.where(count_b > 1e-6, resid_b / np.maximum(count_b, 1e-6), 0.0).astype(np.float32), count_b


def align_local(relative: np.ndarray, sparse: np.ndarray, near_m: float = 30.0, cell_px: int = 8,
                sigma_cells: float = 1.5, min_support: float = 0.5, min_points: int = 50,
                min_tolerance: float = 1.3, keep_lidar: bool = True,
                max_range: float = 80.0) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Metric range from a relative map and sparse lidar on the same grid.

    Returns ``(depth, source, stats)``; ``depth``/``source`` are None with fewer than ``min_points`` returns.
    ``support`` is the smoothed number of returns per cell: below ``min_support`` the region keeps the
    global scale and is marked coarse, as is anything beyond ``near_m``.
    """
    rel = np.asarray(relative, dtype=np.float32)
    if rel.shape != sparse.shape:
        rel = cv2.resize(rel, (sparse.shape[1], sparse.shape[0]), interpolation=cv2.INTER_LINEAR)
    h, w = sparse.shape
    ok = (sparse > 0) & np.isfinite(rel) & (rel > 1e-6)
    n = int(ok.sum())
    if n < min_points:
        return None, None, {"points": n}
    vv, uu = np.nonzero(ok)
    log_ratio = np.log(sparse[ok] / rel[ok])
    # Regions may legitimately disagree on scale, so the global gate is wide and the tight gate is
    # applied to what is left after the regional field.
    good = np.abs(log_ratio - np.median(log_ratio)) <= np.log(4.0)
    center = float(np.median(log_ratio[good]))
    field, _ = _residual_field(vv[good], uu[good], log_ratio[good] - center, (h, w), cell_px, sigma_cells)
    local = log_ratio - center - field[vv // cell_px, uu // cell_px]
    good &= _gate(local, min_tolerance)
    center = float(np.median(log_ratio[good]))
    field, count_b = _residual_field(vv[good], uu[good], log_ratio[good] - center, (h, w), cell_px, sigma_cells)
    # Thin support fades to the global scale instead of extrapolating one return's residual.
    blend = np.clip(count_b / max(min_support, 1e-6), 0.0, 1.0)
    field *= blend
    field_px = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)
    support_px = cv2.resize(count_b, (w, h), interpolation=cv2.INTER_LINEAR)

    depth = (rel * np.exp(center + field_px)).astype(np.float32)
    source = np.full((h, w), SOURCE_COARSE, np.uint8)
    source[(support_px >= min_support) & (depth <= near_m)] = SOURCE_ALIGNED
    if keep_lidar:
        lv, lu = vv[good], uu[good]
        depth[lv, lu] = sparse[lv, lu]
        source[lv, lu] = SOURCE_LIDAR
    bad = ~np.isfinite(depth) | (depth <= 0) | (depth > max_range)
    depth[bad] = 0.0
    source[bad] = SOURCE_NONE
    stats = {"points": n, "inliers": int(good.sum()), "scale": float(np.exp(center)),
             "aligned_share": float(np.mean(source == SOURCE_ALIGNED) + np.mean(source == SOURCE_LIDAR))}
    return depth, source, stats


def range_bin_errors(pred: np.ndarray, truth: np.ndarray, bins=(0.0, 5.0, 10.0, 30.0, np.inf)) -> list[dict]:
    """Per range bin of ``truth`` (> 0): count, median and 90th percentile absolute error (m), mean abs rel."""
    ok = (truth > 0) & (pred > 0)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = ok & (truth >= lo) & (truth < hi)
        if not m.any():
            out.append({"bin": f"{lo:g}-{hi:g}", "n": 0})
            continue
        err = np.abs(pred[m] - truth[m])
        out.append({"bin": f"{lo:g}-{hi:g}", "n": int(m.sum()), "abs_med": float(np.median(err)),
                    "abs_p90": float(np.percentile(err, 90)), "rel_mean": float(np.mean(err / truth[m]))})
    return out
