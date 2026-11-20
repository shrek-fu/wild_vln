"""Per-object lidar points and their 3D box (SuperMap instance layer).

Each detection's mask cluster (``mask_depth`` points, odom frame) is voxelised into its object's set.
A voxel's hit count grows each time a detection sees it, so returns that only one mask caught
(background at the mask edge, a passer-by) stay weak. The box is the axis-aligned extent
(SuperMap's On/Under relations use z-min/z-max and xy overlap) of the supported voxels, with the
outer ``trim`` percentiles cut.
"""

from __future__ import annotations

import numpy as np

_OFFSET = 1 << 20


def _pack(ijk: np.ndarray) -> np.ndarray:
    ijk = ijk.astype(np.int64) + _OFFSET
    return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]


def _unpack(ids: np.ndarray) -> np.ndarray:
    mask = (1 << 21) - 1
    return np.stack([(ids >> 42) & mask, (ids >> 21) & mask, ids & mask], axis=1) - _OFFSET


class MapCloud:
    """Voxel-downsampled accumulated scan map (display background). Scans are merged in batches."""

    def __init__(self, voxel_m: float = 0.1, max_voxels: int = 2_000_000):
        self.voxel_m = float(voxel_m)
        self.max_voxels = int(max_voxels)
        self._ids = np.zeros(0, dtype=np.int64)
        self._pending: list[np.ndarray] = []

    def add(self, xyz: np.ndarray) -> None:
        if len(self._ids) < self.max_voxels and len(xyz):
            self._pending.append(np.unique(_pack(np.floor(np.asarray(xyz, dtype=np.float64) / self.voxel_m))))

    def points(self) -> np.ndarray:
        if self._pending:
            self._ids = np.unique(np.concatenate([self._ids, *self._pending]))
            self._pending = []
        return ((_unpack(self._ids) + 0.5) * self.voxel_m).astype(np.float32)


class InstancePoints:
    def __init__(self, voxel_m: float = 0.05, max_voxels: int = 3000, min_hits: int = 2, min_voxels: int = 10,
                 trim: float = 2.0):
        self.voxel_m = float(voxel_m)
        self.max_voxels = int(max_voxels)
        self.min_hits = int(min_hits)
        self.min_voxels = int(min_voxels)
        self.trim = float(trim)
        # key -> (packed voxel ids sorted, xyz sums, point counts, hits)
        self._sets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._sets

    def add(self, key: str, xyz: np.ndarray, replace: bool = False) -> None:
        """One observation. ``replace`` keeps only this one (moving objects)."""
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        if not len(xyz):
            return
        ids, inv = np.unique(_pack(np.floor(xyz / self.voxel_m)), return_inverse=True)
        sums = np.zeros((len(ids), 3))
        np.add.at(sums, inv, xyz)
        counts = np.bincount(inv, minlength=len(ids)).astype(np.float64)
        hits = np.ones(len(ids), dtype=np.int32)
        old = None if replace else self._sets.get(key)
        if old is not None:
            ids, sums, counts, hits = self._union(old, (ids, sums, counts, hits))
        if len(ids) > self.max_voxels:
            keep = np.sort(np.argsort(-hits, kind="stable")[: self.max_voxels])
            ids, sums, counts, hits = ids[keep], sums[keep], counts[keep], hits[keep]
        self._sets[key] = (ids, sums, counts, hits)

    @staticmethod
    def _union(a, b):
        ids = np.concatenate([a[0], b[0]])
        uniq, inv = np.unique(ids, return_inverse=True)
        sums = np.zeros((len(uniq), 3))
        np.add.at(sums, inv, np.concatenate([a[1], b[1]]))
        counts = np.bincount(inv, weights=np.concatenate([a[2], b[2]]), minlength=len(uniq))
        hits = np.bincount(inv, weights=np.concatenate([a[3], b[3]]), minlength=len(uniq)).astype(np.int32)
        return uniq, sums, counts, hits

    def merge(self, into: str, other: str) -> None:
        b = self._sets.pop(other, None)
        if b is None:
            return
        a = self._sets.get(into)
        self._sets[into] = b if a is None else self._union(a, b)

    def drop(self, key: str) -> None:
        self._sets.pop(key, None)

    def points(self, key: str, supported_only: bool = True) -> np.ndarray:
        """Voxel centroids (odom), the supported ones when there are enough of them."""
        s = self._sets.get(key)
        if s is None:
            return np.zeros((0, 3))
        ids, sums, counts, hits = s
        xyz = sums / counts[:, None]
        if supported_only:
            strong = hits >= self.min_hits
            if strong.sum() >= self.min_voxels:
                xyz = xyz[strong]
        return xyz

    def voxels(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """All voxel centroids (odom) of an object and how many observations hit each."""
        s = self._sets.get(key)
        if s is None:
            return np.zeros((0, 3)), np.zeros(0, dtype=np.int32)
        ids, sums, counts, hits = s
        return sums / counts[:, None], hits

    def box(self, key: str, anchor=None, max_size=None) -> dict | None:
        """``{center, size, n_voxels}`` AABB in odom, None until ``min_voxels`` voxels are known. With ``anchor``
        (the object's centre) and ``max_size`` (its class's largest plausible box) only voxels inside that box
        around the anchor count: background caught at mask edges cannot inflate it."""
        xyz = self.points(key)
        if anchor is not None and max_size is not None and len(xyz):
            inside = np.all(np.abs(xyz - np.asarray(anchor, dtype=np.float64)) <= 0.5 * np.asarray(max_size), axis=1)
            xyz = xyz[inside]
        if len(xyz) < self.min_voxels:
            return None
        lo, hi = np.percentile(xyz, [self.trim, 100.0 - self.trim], axis=0)
        size = np.maximum(hi - lo, self.voxel_m)
        return {"center": ((lo + hi) * 0.5).tolist(), "size": size.tolist(), "n_voxels": int(len(xyz))}
