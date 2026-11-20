"""Metric depth for the pinhole views from the LIO cloud.

Super-LIO publishes every deskewed scan already in ``odom`` (``/lio/cloud_world``). A short window of
scans is projected into a view with that view's ``T_odom_optical``:

* ``view_depth``: sparse z-buffer depth image (nearest return per pixel, metres along optical z);
* ``mask_depth``: an object's points inside its (eroded) mask, reduced to the nearest dense cluster, so
  background returns that leak through the mask edge or past a thin object do not pull the centre back;
* ``scale_relative_depth``: dense metric depth from a relative map (DA360 / mono) by a robust lidar scale
  (median of log ratios with a MAD gate, as in IADC-GO2-EDU ``prepare_x5_mid360_refinement.py``).
"""

from __future__ import annotations

import bisect
from collections import deque

import cv2
import numpy as np

_FIELD_DTYPES = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32, 6: np.uint32, 7: np.float32,
                 8: np.float64}


def cloud_xyz_fast(msg) -> np.ndarray:
    """``(N, 3) float32`` from a ``sensor_msgs/PointCloud2`` without per-point Python."""
    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in "xyz"):
        raise ValueError("cloud has no x/y/z")
    n = int(msg.width) * int(msg.height)
    if n == 0:
        return np.zeros((0, 3), np.float32)
    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [np.dtype(_FIELD_DTYPES[fields[k].datatype]).newbyteorder(">" if msg.is_bigendian else "<")
                    for k in "xyz"],
        "offsets": [fields[k].offset for k in "xyz"],
        "itemsize": int(msg.point_step),
    })
    rows = np.frombuffer(bytes(msg.data), dtype=dtype, count=n)
    xyz = np.stack([rows["x"], rows["y"], rows["z"]], axis=1).astype(np.float32)
    return xyz[np.isfinite(xyz).all(axis=1)]


# livox_ros_driver2/CustomPoint in CDR: uint32 offset_time, float32 x y z, uint8 reflectivity tag line, padded
# to 20 bytes because the next offset_time is 4-aligned.
_LIVOX_POINT = np.dtype({"names": ["offset_time", "x", "y", "z", "reflectivity", "tag"],
                         "formats": ["<u4", "<f4", "<f4", "<f4", "u1", "u1"],
                         "offsets": [0, 4, 8, 12, 16, 17], "itemsize": 20})


def livox_custom_xyz(serialized: bytes, blind: float = 0.5) -> tuple[float, np.ndarray]:
    """``(stamp, (N, 3) float32)`` from a serialized ``livox_ros_driver2/CustomMsg`` (raw subscription or bag
    bytes), without building 20k Python point objects. Keeps returns tagged as normal or low-confidence
    (``tag & 0x30`` in {0x00, 0x10}, as Super-LIO does) beyond ``blind`` metres."""
    data = memoryview(serialized)
    if bytes(data[:2]) != b"\x00\x01":
        raise ValueError("expected little-endian CDR")
    base = 4

    def align(pos: int, n: int) -> int:
        return pos + (-(pos - base)) % n

    sec, nsec, name_len = np.frombuffer(data, "<i4", 1, base)[0], *np.frombuffer(data, "<u4", 2, base + 4)
    pos = align(base + 12 + int(name_len), 8) + 8  # timebase
    pos = align(pos, 4) + 4 + 1 + 3  # point_num, lidar_id, rsvd
    pos = align(pos, 4)
    count = int(np.frombuffer(data, "<u4", 1, pos)[0])
    pos += 4
    stamp = float(sec) + 1e-9 * float(nsec)
    if count == 0:
        return stamp, np.zeros((0, 3), np.float32)
    need = pos + count * _LIVOX_POINT.itemsize
    buf = bytes(data) + b"\x00" * max(0, need - len(data))
    pts = np.frombuffer(buf, _LIVOX_POINT, count, pos)
    xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=1)
    tag = pts["tag"] & 0x30
    keep = ((tag == 0x00) | (tag == 0x10)) & (np.einsum("ij,ij->i", xyz, xyz) > blind * blind) & np.isfinite(xyz).all(1)
    return stamp, xyz[keep]


class CloudWindow:
    """Recent LIO scans in ``odom``; ``points(t)`` returns the scans within ``window_s`` around ``t``."""

    def __init__(self, window_s: float = 1.0, keep_s: float = 3.0, voxel_m: float = 0.0):
        self.window_s = float(window_s)
        self.keep_s = float(keep_s)
        self.voxel_m = float(voxel_m)
        self._t: deque[float] = deque()
        self._xyz: deque[np.ndarray] = deque()

    def add(self, t: float, xyz_odom: np.ndarray) -> None:
        xyz = np.asarray(xyz_odom, dtype=np.float32).reshape(-1, 3)
        if self.voxel_m > 0 and len(xyz):
            keys = np.floor(xyz / self.voxel_m).astype(np.int64)
            _, first = np.unique(keys, axis=0, return_index=True)
            xyz = xyz[np.sort(first)]
        if self._t and t < self._t[-1]:
            self._t.clear()
            self._xyz.clear()
        self._t.append(float(t))
        self._xyz.append(xyz)
        while self._t and self._t[-1] - self._t[0] > self.keep_s:
            self._t.popleft()
            self._xyz.popleft()

    def last_t(self) -> float | None:
        return self._t[-1] if self._t else None

    def points(self, t: float | None = None) -> np.ndarray:
        """Scans stamped in ``[t - window_s, t + 0.1]`` (``t`` = None: the newest window)."""
        if not self._t:
            return np.zeros((0, 3), np.float32)
        stamps = list(self._t)
        t_ref = stamps[-1] if t is None else float(t)
        lo = bisect.bisect_left(stamps, t_ref - self.window_s)
        hi = bisect.bisect_right(stamps, t_ref + 0.1)
        if hi <= lo:
            return np.zeros((0, 3), np.float32)
        chunks = list(self._xyz)[lo:hi]
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]


def project_points(xyz_odom: np.ndarray, T_odom_optical: np.ndarray, K, width: int, height: int,
                   z_min: float = 0.3, z_max: float = 30.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(u, v, z, xyz_odom)`` of the points that land in the image, ``u, v`` integer pixels."""
    T = np.linalg.inv(np.asarray(T_odom_optical, dtype=np.float64)).astype(np.float32)
    xyz = np.asarray(xyz_odom, dtype=np.float32).reshape(-1, 3)
    cam = xyz @ T[:3, :3].T + T[:3, 3]
    z = cam[:, 2]
    keep = (z > z_min) & (z < z_max)
    cam, z, xyz = cam[keep], z[keep], xyz[keep]
    fx, fy, cx, cy = K
    u = np.rint(fx * cam[:, 0] / z + cx).astype(np.int32)
    v = np.rint(fy * cam[:, 1] / z + cy).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[inside], v[inside], z[inside], xyz[inside]


def view_depth(xyz_odom: np.ndarray, T_odom_optical: np.ndarray, K, width: int, height: int,
               z_min: float = 0.3, z_max: float = 30.0, splat: int = 0) -> np.ndarray:
    """Sparse metric depth (float32, 0 = no return). ``splat`` > 0 grows each return to a
    ``(2 splat + 1)`` square, nearest still wins, for a denser preview or label lookups."""
    u, v, z, _ = project_points(xyz_odom, T_odom_optical, K, width, height, z_min, z_max)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(depth, (v, u), z.astype(np.float32))
    if splat > 0:
        k = 2 * int(splat) + 1
        depth = -cv2.dilate(-depth, np.ones((k, k), np.uint8))
    # dilate pads float borders with -FLT_MAX, which survives the negation as a finite value.
    depth[~(depth <= z_max)] = 0.0
    return depth


def mask_image(det: dict, width: int, height: int, erode_px: int = 3) -> np.ndarray:
    """Boolean mask from the detection polygon (else its box), eroded so edge pixels do not catch background."""
    canvas = np.zeros((height, width), dtype=np.uint8)
    poly = det.get("mask")
    if isinstance(poly, (list, tuple)) and len(poly) >= 3:
        cv2.fillPoly(canvas, [np.rint(np.asarray(poly, dtype=np.float64).reshape(-1, 2)).astype(np.int32)], 1)
    else:
        x1, y1, x2, y2 = (int(round(float(c))) for c in det["xyxy"])
        # A box holds more background than a mask: keep its middle half.
        dx, dy = (x2 - x1) // 4, (y2 - y1) // 4
        canvas[max(y1 + dy, 0):max(y2 - dy, 0), max(x1 + dx, 0):max(x2 - dx, 0)] = 1
    if erode_px > 0 and canvas.sum() > 400:
        k = 2 * int(erode_px) + 1
        eroded = cv2.erode(canvas, np.ones((k, k), np.uint8))
        if eroded.sum() > 100:
            canvas = eroded
    return canvas.astype(bool)


def nearest_cluster(z: np.ndarray, min_points: int, gap_m: float = 0.25, min_share: float = 0.15) -> np.ndarray:
    """Indices of the nearest depth cluster holding at least ``max(min_points, min_share * N)`` points.

    Points are split where consecutive sorted depths jump by more than ``gap_m`` plus 3 % of the range,
    so far clusters may be sparser without merging into the near one.
    """
    if len(z) < min_points:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(z)
    zs = z[order]
    cuts = np.flatnonzero(np.diff(zs) > gap_m + 0.03 * zs[:-1]) + 1
    need = max(int(min_points), int(np.ceil(min_share * len(z))))
    start = 0
    for end in list(cuts) + [len(zs)]:
        if end - start >= need:
            return order[start:end]
        start = end
    return np.zeros(0, dtype=np.int64)


def mask_depth(projected, width: int, height: int, det: dict, min_points: int = 12) -> dict | None:
    """Lidar-measured object from ``project_points`` output (one projection per view and frame):
    ``center`` (odom), ``depth`` (optical z), ``extent`` (odom 5-95 % span), ``n_points``, ``points`` (odom);
    None when the mask holds too few returns."""
    u, v, z, xyz = projected
    if len(u) == 0:
        return None
    inside = mask_image(det, width, height)[v, u]
    if inside.sum() < min_points:
        return None
    z, xyz = z[inside], xyz[inside]
    idx = nearest_cluster(z, min_points)
    if len(idx) == 0:
        return None
    pts = xyz[idx].astype(np.float64)
    lo, hi = np.percentile(pts, [5, 95], axis=0)
    return {"center": np.median(pts, axis=0), "depth": float(np.median(z[idx])),
            "extent": (hi - lo).tolist(), "n_points": int(len(idx)), "points": pts}


def scale_relative_depth(relative: np.ndarray, sparse: np.ndarray, min_points: int = 50,
                         min_tolerance: float = 1.5) -> tuple[np.ndarray | None, dict]:
    """Metric dense depth = ``scale * relative`` with ``scale`` from the lidar returns that agree.

    The log ratio lidar / relative is centred on its median and points outside 3 MAD (at least
    ``log(min_tolerance)``) are rejected before the final median. Returns ``(depth or None, stats)``.
    """
    rel = np.asarray(relative, dtype=np.float32)
    if rel.shape != sparse.shape:
        rel = cv2.resize(rel, (sparse.shape[1], sparse.shape[0]), interpolation=cv2.INTER_LINEAR)
    ok = (sparse > 0) & np.isfinite(rel) & (rel > 1e-6)
    n = int(ok.sum())
    if n < min_points:
        return None, {"points": n}
    log_ratio = np.log(sparse[ok] / rel[ok])
    center = float(np.median(log_ratio))
    mad = float(np.median(np.abs(log_ratio - center)))
    tol = max(3.0 * 1.4826 * mad, float(np.log(min_tolerance)))
    good = np.abs(log_ratio - center) <= tol
    scale = float(np.exp(np.median(log_ratio[good])))
    return (scale * rel).astype(np.float32), {"points": n, "inliers": int(good.sum()), "scale": scale,
                                              "log_mad": mad}
