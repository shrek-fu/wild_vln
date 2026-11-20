"""Sparse 2D log-odds traversability grid in ``odom``, with a ground class per cell.

Each frame a cell gets at most one vote: ``+hit`` when ground points fall in it, ``obstacle`` (negative,
stronger) when obstacle points do; obstacle votes override ground votes in the same frame. Cells keep the
mean ground height (stairs and ramps keep their z) and per-class vote counts; the cell class is the majority
(class index + 1, 0 = none yet).
"""

from __future__ import annotations

import numpy as np


def _keys(xy: np.ndarray, res: float) -> np.ndarray:
    ij = np.floor(np.asarray(xy, np.float64)[:, :2] / res).astype(np.int64)
    return (ij[:, 0] << 32) | (ij[:, 1] & 0xFFFFFFFF)


def _unkey(keys: np.ndarray) -> np.ndarray:
    i = keys >> 32
    j = (keys & 0xFFFFFFFF).astype(np.int64)
    j = np.where(j >= 1 << 31, j - (1 << 32), j)
    return np.stack([i, j], axis=1)


def _nearest_z(zimg: np.ndarray, valid: np.ndarray, x: int, y: int, radius: int = 2) -> float:
    ys, xs = slice(max(y - radius, 0), y + radius + 1), slice(max(x - radius, 0), x + radius + 1)
    patch = valid[ys, xs].astype(bool)
    return float(zimg[ys, xs][patch].mean()) if patch.any() else float(zimg[valid.astype(bool)].mean())


class TraversableGrid:
    def __init__(self, resolution: float = 0.1, hit: float = 0.6, obstacle: float = -1.2, clamp=(-3.0, 3.0),
                 free_thresh: float = 1.0, occ_thresh: float = -1.0, n_classes: int = 1) -> None:
        self.res = float(resolution)
        self.hit, self.obstacle = float(hit), float(obstacle)
        self.lo_min, self.lo_max = (float(c) for c in clamp)
        self.free_thresh, self.occ_thresh = float(free_thresh), float(occ_thresh)
        self.n_classes = int(n_classes)
        self.keys = np.zeros(0, np.int64)
        self.logodds = np.zeros(0, np.float32)
        self.z_sum = np.zeros(0, np.float64)
        self.z_n = np.zeros(0, np.float64)
        self.votes = np.zeros((0, self.n_classes + 1), np.int32)

    def __len__(self) -> int:
        return len(self.keys)

    def _index(self, keys: np.ndarray) -> np.ndarray:
        """Indices of ``keys`` (unique), inserting the missing ones."""
        pos = np.searchsorted(self.keys, keys)
        found = (pos < len(self.keys)) & (self.keys[np.minimum(pos, len(self.keys) - 1)] == keys) if len(
            self.keys) else np.zeros(len(keys), bool)
        new = keys[~found]
        if len(new):
            at = np.searchsorted(self.keys, new)
            self.keys = np.insert(self.keys, at, new)
            self.logodds = np.insert(self.logodds, at, 0.0)
            self.z_sum = np.insert(self.z_sum, at, 0.0)
            self.z_n = np.insert(self.z_n, at, 0.0)
            self.votes = np.insert(self.votes, at, 0, axis=0)
        return np.searchsorted(self.keys, keys)

    def update(self, ground_xyz: np.ndarray, obstacle_xyz: np.ndarray, ground_cls: np.ndarray | None = None) -> None:
        """``ground_cls``: label per ground point (class index + 1); default all class 1."""
        ground_xyz = np.asarray(ground_xyz, np.float64).reshape(-1, 3)
        obstacle_xyz = np.asarray(obstacle_xyz, np.float64).reshape(-1, 3)
        okeys = np.unique(_keys(obstacle_xyz, self.res)) if len(obstacle_xyz) else np.zeros(0, np.int64)
        if len(ground_xyz):
            cls = np.ones(len(ground_xyz), np.int64) if ground_cls is None else \
                np.clip(np.asarray(ground_cls, np.int64), 0, self.n_classes)
            ukeys, inv = np.unique(_keys(ground_xyz, self.res), return_inverse=True)
            zmean = np.bincount(inv, weights=ground_xyz[:, 2]) / np.bincount(inv)
            counts = np.zeros((len(ukeys), self.n_classes + 1), np.int64)
            np.add.at(counts, (inv, cls), 1)
            counts[:, 0] = 0
            frame_cls = counts.argmax(axis=1)
            keep = ~np.isin(ukeys, okeys)
            ukeys, zmean, frame_cls = ukeys[keep], zmean[keep], frame_cls[keep]
            if len(ukeys):
                idx = self._index(ukeys)
                self.logodds[idx] = np.clip(self.logodds[idx] + self.hit, self.lo_min, self.lo_max)
                self.z_sum[idx] += zmean
                self.z_n[idx] += 1.0
                labelled = frame_cls > 0
                self.votes[idx[labelled], frame_cls[labelled]] += 1
        if len(okeys):
            idx = self._index(okeys)
            self.logodds[idx] = np.clip(self.logodds[idx] + self.obstacle, self.lo_min, self.lo_max)

    def crop(self, center_xy, half_m: float) -> int:
        """Drop the cells outside the square of half side ``half_m`` around ``center_xy``; returns how many."""
        if not len(self.keys):
            return 0
        xy = (_unkey(self.keys) + 0.5) * self.res
        keep = np.all(np.abs(xy - np.asarray(center_xy, np.float64)[:2]) <= half_m, axis=1)
        dropped = int((~keep).sum())
        if dropped:
            self.keys, self.logodds = self.keys[keep], self.logodds[keep]
            self.z_sum, self.z_n, self.votes = self.z_sum[keep], self.z_n[keep], self.votes[keep]
        return dropped

    def cell_class(self) -> np.ndarray:
        """Majority label per cell (0 = no class vote yet)."""
        if not len(self.keys):
            return np.zeros(0, np.int64)
        cls = self.votes[:, 1:].argmax(axis=1) + 1
        return np.where(self.votes[:, 1:].max(axis=1) > 0, cls, 0)

    def _traversable_sel(self, label: int | None = None) -> np.ndarray:
        sel = (self.logodds >= self.free_thresh) & (self.z_n > 0)
        if label is not None:
            sel &= self.cell_class() == label
        return sel

    def traversable(self, with_class: bool = False):
        """``(M, 3)`` cell centres with their mean ground height, for cells above ``free_thresh``
        (``with_class``: also their labels)."""
        sel = self._traversable_sel()
        ij = _unkey(self.keys[sel])
        xyz = np.column_stack([(ij + 0.5) * self.res, self.z_sum[sel] / self.z_n[sel]])
        return (xyz, self.cell_class()[sel]) if with_class else xyz

    def obstacles(self) -> np.ndarray:
        """``(M, 2)`` centres of the cells at or below ``occ_thresh``."""
        return (_unkey(self.keys[self.logodds <= self.occ_thresh]) + 0.5) * self.res

    def outlines(self, label: int | None = None, min_cells: int = 6,
                 simplify_cells: float = 1.0) -> list[tuple[np.ndarray, bool]]:
        """Closed boundaries of the traversable regions (of class ``label``, or all) as ``[(xyz (N, 3), is_hole)]``
        in odom, each point at the ground height of its boundary cell. Holes are areas enclosed by the region;
        regions and holes smaller than ``min_cells`` are dropped; ``simplify_cells`` is the Douglas-Peucker
        tolerance."""
        import cv2

        sel = self._traversable_sel(label)
        if not sel.any():
            return []
        ij = _unkey(self.keys[sel])
        z = self.z_sum[sel] / self.z_n[sel]
        lo = ij.min(axis=0) - 1
        w, h = (ij.max(axis=0) - lo + 2).tolist()
        img = np.zeros((h, w), np.uint8)
        zimg = np.zeros((h, w), np.float64)
        img[ij[:, 1] - lo[1], ij[:, 0] - lo[0]] = 1
        zimg[ij[:, 1] - lo[1], ij[:, 0] - lo[0]] = z
        contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c, info in zip(contours, hierarchy[0] if hierarchy is not None else []):
            if cv2.contourArea(c) < min_cells:
                continue
            c = cv2.approxPolyDP(c, simplify_cells, True).reshape(-1, 2)
            if len(c) < 3:
                continue
            zs = np.array([_nearest_z(zimg, img, x, y) for x, y in c])
            xy = (c + lo + 0.5) * self.res
            out.append((np.column_stack([xy, zs]), info[3] >= 0))
        return out

    def _raster(self, values: np.ndarray):
        ij = _unkey(self.keys)
        lo = ij.min(axis=0)
        w, h = (ij.max(axis=0) - lo + 1).tolist()
        data = np.full((h, w), -1, np.int8)
        data[ij[:, 1] - lo[1], ij[:, 0] - lo[0]] = values
        return (float(lo[0] * self.res), float(lo[1] * self.res)), int(w), int(h), data.ravel()

    def occupancy(self, blocked: tuple[int, ...] = ()):
        """``(origin_xy, width, height, data)`` for a ``nav_msgs/OccupancyGrid``: 0 traversable, 100 obstacle or
        a ``blocked`` class, -1 unknown or undecided; row-major from ``origin_xy``, x along columns."""
        if not len(self.keys):
            return (0.0, 0.0), 0, 0, np.zeros(0, np.int8)
        val = np.full(len(self.keys), -1, np.int8)
        val[self.logodds >= self.free_thresh] = 0
        if blocked:
            val[(self.logodds >= self.free_thresh) & np.isin(self.cell_class(), blocked)] = 100
        val[self.logodds <= self.occ_thresh] = 100
        return self._raster(val)

    def class_occupancy(self):
        """Same layout as ``occupancy``: traversable cells hold their label (1..n_classes), obstacles 100,
        everything else -1."""
        if not len(self.keys):
            return (0.0, 0.0), 0, 0, np.zeros(0, np.int8)
        val = np.full(len(self.keys), -1, np.int8)
        sel = self.logodds >= self.free_thresh
        val[sel] = self.cell_class()[sel]
        val[self.logodds <= self.occ_thresh] = 100
        return self._raster(val)
