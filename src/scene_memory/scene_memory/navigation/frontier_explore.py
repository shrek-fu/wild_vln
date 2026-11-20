"""Where to look next for a target nobody can see: WildOS-style frontier exploration on the planning ground.

Frontiers are free planning-ground cells (``mapground``, intensity 9999) next to unknown space (no ground at
all), clustered, and kept only when the unknown continues ``depth_m`` outward (holes inside a mapped area are
not frontiers). Each frontier gets a semantic score from the VLM's per-view "would walking this way find it"
ratings, interpolated by bearing, and remembered WildOS-style: a frontier's score is only replaced when it is
rated from closer than before, frontiers that were reached or failed (and new ones within ``separation_m`` of
them) are dropped, and unrated ones get ``default_score``. The pick minimises WildOS's planner cost

    path_length_over_free_ground * (1 - score_factor * ln(score))
"""

from __future__ import annotations

import heapq
import math

import numpy as np
from scipy import ndimage

N8 = [(-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)), (0, -1, 1.0), (0, 1, 1.0),
      (1, -1, math.sqrt(2)), (1, 0, 1.0), (1, 1, math.sqrt(2))]


class GroundGrid:
    """2D grid of the planning ground: ``known`` (any ground point), ``free`` (non-lethal ground)."""

    def __init__(self, xyz: np.ndarray, free: np.ndarray, cell: float = 0.2, pad_m: float = 3.0):
        xyz = np.asarray(xyz, dtype=np.float64)
        self.cell = cell
        self.origin = xyz[:, :2].min(axis=0) - pad_m
        shape = np.ceil((xyz[:, :2].max(axis=0) + pad_m - self.origin) / cell).astype(int) + 1
        self.shape = (int(shape[0]), int(shape[1]))
        rc = self.cells(xyz)
        self.known = np.zeros(self.shape, bool)
        self.known[rc[:, 0], rc[:, 1]] = True
        self.free = np.zeros(self.shape, bool)
        self.free[rc[free, 0], rc[free, 1]] = True
        self.z = np.full(self.shape, np.nan)
        self.z[rc[:, 0], rc[:, 1]] = xyz[:, 2]
        # one-cell gaps between export voxels are not unknown space
        self.known = ndimage.binary_closing(self.known, iterations=1) | self.known

    def cells(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, np.shape(xy)[-1])[:, :2]
        return np.clip(np.floor((xy - self.origin) / self.cell).astype(int), 0, np.array(self.shape) - 1)

    def xy(self, r, c) -> np.ndarray:
        return self.origin + (np.stack([r, c], axis=-1) + 0.5) * self.cell

    def distance_from(self, xy, snap_m: float = 1.5) -> np.ndarray:
        """Path length (m) over free cells from the free cell nearest ``xy`` (inf where unreachable)."""
        dist = np.full(self.shape, np.inf)
        r0, c0 = self.cells(xy)[0]
        k = int(snap_m / self.cell)
        win = self.free[max(r0 - k, 0):r0 + k + 1, max(c0 - k, 0):c0 + k + 1]
        rs, cs = np.nonzero(win)
        if len(rs) == 0:
            return dist
        i = int(np.argmin((rs + max(r0 - k, 0) - r0) ** 2 + (cs + max(c0 - k, 0) - c0) ** 2))
        s = (int(rs[i] + max(r0 - k, 0)), int(cs[i] + max(c0 - k, 0)))
        dist[s] = 0.0
        heap = [(0.0, s)]
        rows, cols = self.shape
        while heap:
            d, (r, c) = heapq.heappop(heap)
            if d > dist[r, c]:
                continue
            for dr, dc, w in N8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and self.free[nr, nc]:
                    nd = d + w * self.cell
                    if nd < dist[nr, nc]:
                        dist[nr, nc] = nd
                        heapq.heappush(heap, (nd, (nr, nc)))
        return dist


def find_frontiers(grid: GroundGrid, depth_m: float = 1.0, min_cells: int = 4) -> list[dict]:
    """Clusters of free cells bordering unknown space that stays unknown ``depth_m`` outward:
    ``{xy, z, cells: (K, 2), bearing (outward, rad), size_m}``."""
    unknown = ~grid.known
    border = grid.free & ndimage.binary_dilation(unknown, structure=np.ones((3, 3), bool))
    rs, cs = np.nonzero(border)
    if len(rs) == 0:
        return []
    # outward direction of each border cell: mean offset to unknown cells within 2 cells
    k = 2
    pad = np.pad(unknown, k, constant_values=True)
    offs = [(dr, dc) for dr in range(-k, k + 1) for dc in range(-k, k + 1) if dr or dc]
    vec = np.zeros((len(rs), 2))
    for dr, dc in offs:
        vec += pad[rs + k + dr, cs + k + dc][:, None] * np.array([dr, dc], float)
    norm = np.linalg.norm(vec, axis=1)
    ok = norm > 0
    vec[ok] /= norm[ok, None]
    # keep cells whose outward ray stays unknown for depth_m (else: a hole or a narrow gap)
    steps = max(int(round(depth_m / grid.cell)), 1)
    deep = ok.copy()
    for s in range(2, steps + 1):
        rr = np.clip(np.rint(rs + vec[:, 0] * s).astype(int), 0, grid.shape[0] - 1)
        cc = np.clip(np.rint(cs + vec[:, 1] * s).astype(int), 0, grid.shape[1] - 1)
        deep &= unknown[rr, cc]
    mask = np.zeros(grid.shape, bool)
    mask[rs[deep], cs[deep]] = True
    vec_map = np.zeros(grid.shape + (2,))
    vec_map[rs, cs] = vec
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    out = []
    for lab in range(1, n + 1):
        r, c = np.nonzero(labels == lab)
        if len(r) < min_cells:
            continue
        xy = grid.xy(r, c)
        centre = xy.mean(axis=0)
        i = int(np.argmin(np.linalg.norm(xy - centre, axis=1)))  # a member cell: free, on the frontier
        v = vec_map[r, c].mean(axis=0)
        out.append({"xy": [float(xy[i, 0]), float(xy[i, 1])], "z": float(grid.z[r[i], c[i]]),
                    "cell": (int(r[i]), int(c[i])), "bearing": float(math.atan2(v[1], v[0])),
                    "size_m": round(len(r) * grid.cell, 2)})
    return out


def view_score(bearing: float, view_bearings: dict[str, float], ratings: dict[str, float],
               sharpness: float = 2.0) -> float | None:
    """Rating (0..1) in direction ``bearing`` from per-view ratings, cos^sharpness-weighted; None if no view
    faces it."""
    num = den = 0.0
    for name, s in ratings.items():
        if name not in view_bearings:
            continue
        w = max(math.cos(bearing - view_bearings[name]), 0.0) ** sharpness
        num, den = num + w * s, den + w
    return num / den if den > 1e-6 else None


class FrontierMemory:
    """WildOS's frontier bookkeeping: scores rated from closer replace older ones; reached / failed spots
    suppress frontiers within ``separation_m``."""

    def __init__(self, separation_m: float = 0.75, default_score: float = 0.3, match_m: float = 1.0):
        self.separation_m, self.default_score, self.match_m = separation_m, default_score, match_m
        self.rated: list[dict] = []  # {xy, score, from_m}
        self.removed: list[np.ndarray] = []

    def reset(self) -> None:
        self.rated, self.removed = [], []

    def remove(self, xy) -> None:
        self.removed.append(np.asarray(xy[:2], dtype=np.float64))

    def is_removed(self, xy) -> bool:
        p = np.asarray(xy[:2], dtype=np.float64)
        return any(np.linalg.norm(p - q) < self.separation_m for q in self.removed)

    def rate(self, xy, score: float, from_m: float) -> None:
        p = np.asarray(xy[:2], dtype=np.float64)
        for r in self.rated:
            if np.linalg.norm(r["xy"] - p) < self.match_m:
                if from_m < r["from_m"]:
                    r.update(xy=p, score=score, from_m=from_m)
                return
        self.rated.append({"xy": p, "score": score, "from_m": from_m})

    def score(self, xy) -> float:
        p = np.asarray(xy[:2], dtype=np.float64)
        best = None
        for r in self.rated:
            d = np.linalg.norm(r["xy"] - p)
            if d < self.match_m and (best is None or d < best[0]):
                best = (d, r["score"])
        return best[1] if best else self.default_score


def pick_frontier(frontiers: list[dict], dist: np.ndarray, memory: FrontierMemory, score_factor: float = 0.5,
                  min_dist_m: float = 1.0) -> dict | None:
    """Reachable, not removed frontier with the lowest ``path * (1 - score_factor * ln(score))``."""
    best = None
    for f in frontiers:
        d = float(dist[f["cell"]])
        if not math.isfinite(d) or d < min_dist_m or memory.is_removed(f["xy"]):
            continue
        s = min(max(memory.score(f["xy"]), 0.05), 1.0)
        cost = d * (1.0 - score_factor * math.log(s))
        if best is None or cost < best["cost"]:
            best = {**f, "path_m": round(d, 2), "score": round(s, 2), "cost": round(cost, 2)}
    return best
