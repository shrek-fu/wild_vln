"""Build a PromptDA-style lidar prompt depth from Livox points on an Insta360 pinhole view.

Pipeline (same idea as RoboSense / PromptDA ``test_on_pc.py``):

1. Transform points into the view's optical frame (``T_level_optical`` or ``T_odom_optical``).
2. Project with the view ``K`` into a (optionally downscaled) image → sparse z-buffer.
3. Fill holes with KNN pooling so the network gets a dense-enough prompt.
4. The metric min/max of the filled map is kept for denormalization after PromptDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class ViewGeom:
    name: str
    width: int
    height: int
    K: np.ndarray          # 3x3
    T_level_optical: np.ndarray  # 4x4: p_level = T @ p_optical
    image_topic: str = ""


def load_views(calib_dir: str | Path, names: tuple[str, ...] | None = None) -> list[ViewGeom]:
    """Read ``insta360_intrinsics.json`` + ``insta360_extrinsic.json`` / ``lidar_extrinsic.json``."""
    import json
    import sys

    root = Path(calib_dir).expanduser().resolve()
    da360 = root
    # Prefer da360_depth/calib helpers when present.
    parent = Path(__file__).resolve().parents[1] / "da360_depth"
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    from insta_live.calib_io import load_rig_calibration

    calib = load_rig_calibration(root if (root / "insta360_intrinsics.json").is_file() else da360 / "calib")
    wanted = set(names) if names else None
    out: list[ViewGeom] = []
    for name, view in calib.views.items():
        if wanted is not None and name not in wanted:
            continue
        R = np.asarray(view["R_body_optical"], np.float64)
        T_body_optical = np.eye(4)
        T_body_optical[:3, :3] = R
        # p_level = T_level_insta360 @ p_body; p_body = R_body_optical @ p_optical
        T_level_optical = calib.T_level_insta360 @ T_body_optical
        K = np.asarray(view["K"], np.float64)
        out.append(ViewGeom(
            name=name,
            width=int(view["width"]),
            height=int(view["height"]),
            K=K,
            T_level_optical=T_level_optical,
            image_topic=str(view.get("image_topic", f"/insta360/{name}/image/compressed")),
        ))
    if not out:
        raise RuntimeError(f"no views in {calib_dir} (wanted {names})")
    return out


def knn_pooling(sparse: np.ndarray, pool_size: int = 18, k: int = 4) -> np.ndarray:
    """Fill zero pixels with the mean depth of the k nearest non-zero neighbours in a window."""
    sparse = np.asarray(sparse, np.float32)
    h, w = sparse.shape
    out = sparse.copy()
    ys, xs = np.nonzero(sparse == 0)
    if len(ys) == 0:
        return out
    radius = pool_size // 2
    for y, x in zip(ys.tolist(), xs.tolist()):
        top, bot = max(y - radius, 0), min(y + radius + 1, h)
        left, right = max(x - radius, 0), min(x + radius + 1, w)
        region = sparse[top:bot, left:right]
        nz = np.flatnonzero(region)
        if nz.size == 0:
            continue
        ry, rx = np.divmod(nz, region.shape[1])
        d2 = (ry + top - y) ** 2 + (rx + left - x) ** 2
        take = np.argsort(d2)[:k]
        out[y, x] = float(region.ravel()[nz[take]].mean())
    return out


def knn_pooling_fast(sparse: np.ndarray, pool_size: int = 18, k: int = 4) -> np.ndarray:
    """Fill empty cells from neighbouring lidar hits.

    * ``k == 1``: nearest-seed fill (scipy EDT), capped to ``pool_size`` pixels.
    * ``k > 1``: PromptDA-style windowed mean of the k nearest hits (slower, reference match).
    """
    sparse = np.asarray(sparse, np.float32)
    mask = sparse > 0
    if not mask.any():
        return sparse.copy()
    if k > 1:
        return knn_pooling(sparse, pool_size, k)
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return knn_pooling(sparse, pool_size, k=1)
    d, (iy, ix) = distance_transform_edt(~mask, return_indices=True)
    filled = sparse.copy()
    near = (~mask) & (d <= pool_size)
    filled[near] = sparse[iy[near], ix[near]]
    return filled.astype(np.float32)


def project_sparse_depth(
    xyz_level: np.ndarray,
    view: ViewGeom,
    *,
    scale: float = 1.0,
    z_min: float = 0.3,
    z_max: float = 40.0,
) -> np.ndarray:
    """Sparse optical-z depth image; ``scale`` < 1 shrinks the raster (PromptDA uses ~1/7)."""
    xyz = np.asarray(xyz_level, np.float64).reshape(-1, 3)
    if not len(xyz):
        h = max(1, int(view.height * scale))
        w = max(1, int(view.width * scale))
        return np.zeros((h, w), np.float32)

    T_opt_level = np.linalg.inv(view.T_level_optical)
    cam = xyz @ T_opt_level[:3, :3].T + T_opt_level[:3, 3]
    z = cam[:, 2]
    keep = (z > z_min) & (z < z_max)
    cam, z = cam[keep], z[keep]

    K = view.K.copy()
    K[:2] *= scale
    h = max(1, int(round(view.height * scale)))
    w = max(1, int(round(view.width * scale)))
    u = np.rint(K[0, 0] * cam[:, 0] / z + K[0, 2]).astype(np.int32)
    v = np.rint(K[1, 1] * cam[:, 1] / z + K[1, 2]).astype(np.int32)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[inside], v[inside], z[inside].astype(np.float32)

    depth = np.full((h, w), np.inf, np.float32)
    if len(z):
        np.minimum.at(depth, (v, u), z)
    depth[~(depth < z_max)] = 0.0
    return depth


def build_prompt_depth(
    xyz_level: np.ndarray,
    view: ViewGeom,
    *,
    project_scale: float = 1.0 / 7.0,
    knn_size: int = 18,
    knn_k: int = 4,
    out_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(prompt_depth HxW metres, stats)`` resized to the RGB / network size.

    ``out_size`` is ``(width, height)``; default is the view's native resolution.
    """
    sparse = project_sparse_depth(xyz_level, view, scale=project_scale)
    n_hits = int((sparse > 0).sum())
    # KNN on the downscaled raster (PromptDA / RoboSense), then upsample to the RGB size.
    dense = knn_pooling(sparse, knn_size, knn_k) if n_hits else sparse
    ow = view.width if out_size is None else int(out_size[0])
    oh = view.height if out_size is None else int(out_size[1])
    if dense.shape != (oh, ow):
        dense = cv2.resize(dense, (ow, oh), interpolation=cv2.INTER_LINEAR)
        # bilinear bleed from holes → tiny depths; drop them
        dense[dense < 0.05] = 0.0
    stats = {
        "hits": n_hits,
        "filled": int((dense > 0).sum()),
        "min": float(dense[dense > 0].min()) if (dense > 0).any() else 0.0,
        "max": float(dense.max()) if (dense > 0).any() else 0.0,
        "shape": dense.shape,
    }
    return dense.astype(np.float32), stats
