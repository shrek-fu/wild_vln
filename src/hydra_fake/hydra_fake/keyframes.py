"""Keyframe pose graph over the LIO odometry: lidar loop closures and re-localisation after LIO resets.

Hydra keeps its scene graph attached to a pose graph and deforms it on loop closure. Here the LIO
odometry (gravity aligned, so 4 DoF: x, y, z, yaw) is cut into keyframes every ``kf_dist_m`` /
``kf_yaw_deg``; each keyframe owns the scans received while it was the newest one, as a voxel submap in
its own frame. When a keyframe is complete it is ICP-registered (4 DoF, point to point) against the
submaps of older keyframes near its current estimate:

* ``loop``: same odometry segment, older than ``loop_min_dt_s``;
* ``reloc``: first keyframe after an odometry restart (``new_segment``), searched in a wider radius;
  until it succeeds the new segment hangs on a weak "robot did not move" edge.

An accepted match adds an edge and re-optimises the graph (``scipy.optimize.least_squares``, Huber loss
against wrong matches). ``correction(k)`` maps odometry-frame points of keyframe ``k``'s time to the
corrected world frame; ``version`` counts optimisations so users can re-anchor what they derived.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np


def yaw_of(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def pose4(T: np.ndarray) -> np.ndarray:
    return np.array([T[0, 3], T[1, 3], T[2, 3], yaw_of(T[:3, :3])])


def T4(x) -> np.ndarray:
    c, s = math.cos(x[3]), math.sin(x[3])
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = x[:3]
    return T


def rel4(a, b) -> np.ndarray:
    """``b`` in the frame of ``a`` (x, y, z, yaw)."""
    c, s = math.cos(a[3]), math.sin(a[3])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, b[2] - a[2], float(wrap(b[3] - a[3]))])


def voxel_down(xyz: np.ndarray, voxel: float) -> np.ndarray:
    if not len(xyz):
        return xyz.reshape(0, 3).astype(np.float32)
    key = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return xyz[idx].astype(np.float32)


@dataclass
class Keyframe:
    id: int
    t: float
    seg: int
    T_odom: np.ndarray
    x: np.ndarray
    pending: list = field(default_factory=list)
    pts: np.ndarray | None = None
    reloc: bool = False
    done: bool = False


@dataclass
class Edge:
    i: int
    j: int
    z: np.ndarray
    sigma: np.ndarray
    kind: str


def icp4(src: np.ndarray, tree, tgt: np.ndarray, iters: int = 30, max_d: tuple[float, float] = (1.0, 0.25),
         inlier_m: float = 0.2) -> tuple[np.ndarray, float, float, int]:
    """4 DoF ICP of world points ``src`` onto ``tgt`` (KD-tree ``tree``). Returns
    ``(D, fitness, rmse, inliers)``: ``D`` (4x4, yaw only) moves ``src`` onto ``tgt``."""
    D = np.eye(4)
    cur = src.astype(np.float64)
    for it in range(iters):
        d_max = max_d[0] + (max_d[1] - max_d[0]) * min(it / max(iters * 0.6, 1), 1.0)
        dist, idx = tree.query(cur, distance_upper_bound=d_max)
        ok = np.isfinite(dist)
        if ok.sum() < 50:
            break
        a, b = cur[ok], tgt[idx[ok]]
        ma, mb = a[:, :2].mean(0), b[:, :2].mean(0)
        H = (a[:, :2] - ma).T @ (b[:, :2] - mb)
        th = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        step = np.eye(4)
        step[:2, :2] = R
        step[:2, 3] = mb - R @ ma
        step[2, 3] = float(np.median(b[:, 2] - a[:, 2]))
        cur = cur @ step[:3, :3].T + step[:3, 3]
        D = step @ D
        if abs(th) < 1e-4 and np.linalg.norm(step[:3, 3]) < 1e-3:
            break
    dist, _ = tree.query(cur, distance_upper_bound=inlier_m)
    ok = np.isfinite(dist)
    n = int(ok.sum())
    return D, n / max(len(cur), 1), float(np.sqrt(np.mean(dist[ok] ** 2))) if n else float("inf"), n


class KeyframeGraph:
    def __init__(self, kf_dist_m: float = 1.0, kf_yaw_deg: float = 20.0, voxel_m: float = 0.1,
                 max_points: int = 15000, loop_radius_m: float = 4.0, loop_min_dt_s: float = 30.0,
                 reloc_radius_m: float = 8.0, min_fitness: float = 0.55, min_inliers: int = 400,
                 max_rmse_m: float = 0.12, max_jump_m: float = 3.0, context: int = 3, loop_every: int = 3,
                 odom_sigma: tuple[float, float] = (0.02, 0.005), loop_sigma: tuple[float, float] = (0.08, 0.02),
                 reset_sigma: tuple[float, float] = (2.0, 0.6)) -> None:
        self.kf_dist_m, self.kf_yaw = kf_dist_m, math.radians(kf_yaw_deg)
        self.voxel_m, self.max_points = voxel_m, max_points
        self.loop_radius_m, self.loop_min_dt_s, self.reloc_radius_m = loop_radius_m, loop_min_dt_s, reloc_radius_m
        self.min_fitness, self.min_inliers, self.max_rmse_m, self.max_jump_m = min_fitness, min_inliers, max_rmse_m, max_jump_m
        self.context = context
        self.loop_every = max(int(loop_every), 1)
        self._last_try = -10**9
        self.odom_sigma, self.loop_sigma, self.reset_sigma = odom_sigma, loop_sigma, reset_sigma
        self.kfs: list[Keyframe] = []
        self.edges: list[Edge] = []
        self.seg = 0
        self.version = 0
        self._new_seg = False
        self._unlocalised = False
        self._last_T: np.ndarray | None = None
        self._last_world: np.ndarray | None = None
        self.stats = {"loops": 0, "relocs": 0, "rejected": 0, "segments": 1, "optimisations": 0}
        self.events: list[dict] = []

    # --- input -------------------------------------------------------------------------------------
    def new_segment(self) -> None:
        """Odometry restarted: the next pose starts a new frame."""
        if self.kfs:
            self._last_world = self.world_pose(self._last_T) if self._last_T is not None else None
        self._new_seg = True

    def on_odom(self, t: float, T: np.ndarray) -> int:
        """Returns the id of the keyframe active at ``t``."""
        if self._new_seg or not self.kfs:
            self._add_kf(t, T, reset=bool(self.kfs))
        else:
            last = self.kfs[-1]
            d = rel4(pose4(last.T_odom), pose4(T))
            if np.hypot(d[0], d[1]) >= self.kf_dist_m or abs(d[2]) >= self.kf_dist_m or abs(d[3]) >= self.kf_yaw:
                self._add_kf(t, T)
        self._last_T = T
        return self.kfs[-1].id

    def on_cloud(self, xyz_odom: np.ndarray) -> None:
        if not self.kfs or not len(xyz_odom):
            return
        kf = self.kfs[-1]
        Ti = np.linalg.inv(kf.T_odom)
        kf.pending.append(voxel_down(np.asarray(xyz_odom, np.float64) @ Ti[:3, :3].T + Ti[:3, 3], self.voxel_m))
        # A standing robot never closes its keyframe: fold the scans so the submap stays voxel-bounded.
        if len(kf.pending) >= 10:
            kf.pending = [voxel_down(np.concatenate(kf.pending), self.voxel_m)]

    def _add_kf(self, t: float, T: np.ndarray, reset: bool = False) -> None:
        if self.kfs:
            self._finish(self.kfs[-1])
        o = pose4(T)
        if reset:
            self.seg += 1
            self.stats["segments"] += 1
            guess = pose4(self._last_world) if self._last_world is not None else o
            kf = Keyframe(len(self.kfs), t, self.seg, T.copy(), guess, reloc=True)
            prev = self.kfs[-1]
            self.edges.append(Edge(prev.id, kf.id, rel4(prev.x, guess), np.array(self.reset_sigma), "reset"))
            self.events.append({"t": t, "event": "segment", "kf": kf.id})
            self._unlocalised = True
        else:
            if self.kfs:
                prev = self.kfs[-1]
                z = rel4(pose4(prev.T_odom), o)
                x = pose4(T4(prev.x) @ T4(z))
                sig = np.array([self.odom_sigma[0] + 0.01 * np.hypot(z[0], z[1]), self.odom_sigma[1]])
                self.edges.append(Edge(prev.id, len(self.kfs), z, sig, "odom"))
            else:
                x = o
            kf = Keyframe(len(self.kfs), t, self.seg, T.copy(), x, reloc=self._unlocalised)
        self.kfs.append(kf)
        self._new_seg = False

    def _finish(self, kf: Keyframe) -> None:
        if kf.done:
            return
        kf.done = True
        pts = np.concatenate(kf.pending) if kf.pending else np.zeros((0, 3), np.float32)
        kf.pending = []
        pts = voxel_down(pts, self.voxel_m)
        if len(pts) > self.max_points:
            pts = pts[np.random.default_rng(kf.id).choice(len(pts), self.max_points, replace=False)]
        kf.pts = pts
        # Re-localisation is tried on every keyframe until it succeeds; loops every ``loop_every``.
        if len(pts) >= self.min_inliers and (kf.reloc or kf.id - self._last_try >= self.loop_every):
            self._match(kf)

    # --- frames ------------------------------------------------------------------------------------
    def correction(self, k: int) -> np.ndarray:
        """4x4: odometry frame (of keyframe ``k``'s segment, near its time) → corrected world."""
        kf = self.kfs[k]
        return T4(kf.x) @ np.linalg.inv(T4(pose4(kf.T_odom)))

    def kf_at(self, t: float) -> int | None:
        """Keyframe of the current segment active at ``t`` (the newest one not after ``t``)."""
        if not self.kfs:
            return None
        times = [k.t for k in self.kfs]
        i = max(bisect.bisect_right(times, t) - 1, 0)
        while i < len(self.kfs) - 1 and self.kfs[i].seg != self.seg:
            i += 1
        return self.kfs[i].id

    def world_pose(self, T_odom: np.ndarray, k: int | None = None) -> np.ndarray:
        k = self.kfs[-1].id if k is None else k
        return self.correction(k) @ T_odom

    def kf_world_points(self, k: int) -> np.ndarray:
        """Submap of keyframe ``k`` in the world (the scans so far while it is still the newest)."""
        kf = self.kfs[k]
        pts = kf.pts if kf.pts is not None else (np.concatenate(kf.pending) if kf.pending else None)
        if pts is None or not len(pts):
            return np.zeros((0, 3))
        T = self.correction(k) @ kf.T_odom
        return pts.astype(np.float64) @ T[:3, :3].T + T[:3, 3]

    def world_points(self, voxel: float = 0.1) -> np.ndarray:
        parts = [p for p in (self.kf_world_points(k.id) for k in self.kfs) if len(p)]
        return voxel_down(np.concatenate(parts), voxel) if parts else np.zeros((0, 3), np.float32)

    # --- loop closure ------------------------------------------------------------------------------
    def _candidates(self, kf: Keyframe) -> list[int]:
        radius = self.reloc_radius_m if kf.reloc else self.loop_radius_m
        out = []
        for o in self.kfs[: kf.id]:
            if o.pts is None or not len(o.pts):
                continue
            if kf.reloc:
                if o.seg == kf.seg:
                    continue
            elif o.seg == kf.seg and kf.t - o.t < self.loop_min_dt_s:
                continue
            d = np.hypot(o.x[0] - kf.x[0], o.x[1] - kf.x[1])
            if d < radius and abs(o.x[2] - kf.x[2]) < 1.5:
                out.append((d, o.id))
        return [i for _, i in sorted(out)]

    def _match(self, kf: Keyframe) -> None:
        from scipy.spatial import cKDTree

        cands = self._candidates(kf)
        if not cands:
            return
        self._last_try = kf.id
        j = cands[0]
        ctx = [o.id for o in self.kfs if o.seg == self.kfs[j].seg and abs(o.id - j) <= self.context and o.id < kf.id]
        tgt = voxel_down(np.concatenate([self.kf_world_points(i) for i in ctx]), self.voxel_m).astype(np.float64)
        src = self.kf_world_points(kf.id)
        if len(tgt) < self.min_inliers:
            return
        D, fitness, rmse, n = icp4(src, cKDTree(tgt), tgt, max_d=(2.0 if kf.reloc else 1.0, 0.25))
        shift = float(np.linalg.norm(D[:3, 3]))
        max_shift = self.reloc_radius_m if kf.reloc else self.max_jump_m
        ok = fitness >= self.min_fitness and n >= self.min_inliers and rmse <= self.max_rmse_m and shift <= max_shift
        self.events.append({"t": kf.t, "event": "reloc" if kf.reloc else "loop", "kf": kf.id, "target": j,
                            "fitness": round(fitness, 3), "rmse": round(rmse, 3), "shift": round(shift, 3),
                            "accepted": bool(ok)})
        if not ok:
            self.stats["rejected"] += 1
            return
        measured = pose4(D @ T4(kf.x))
        self.edges.append(Edge(j, kf.id, rel4(self.kfs[j].x, measured), np.array(self.loop_sigma),
                               "reloc" if kf.reloc else "loop"))
        self.stats["relocs" if kf.reloc else "loops"] += 1
        if kf.reloc and kf.seg == self.seg:
            self._unlocalised = False
        kf.reloc = False
        self.optimise()

    def optimise(self) -> None:
        from scipy.optimize import least_squares
        from scipy.sparse import lil_matrix

        n = len(self.kfs)
        if n < 2:
            return
        x0 = np.stack([k.x for k in self.kfs])
        free = x0[1:].ravel()

        def unpack(v):
            return np.vstack([x0[:1], v.reshape(-1, 4)])

        w = [np.array([1 / e.sigma[0]] * 3 + [1 / e.sigma[1]]) for e in self.edges]

        def residuals(v):
            X = unpack(v)
            r = np.empty((len(self.edges), 4))
            for m, e in enumerate(self.edges):
                d = rel4(X[e.i], X[e.j]) - e.z
                d[3] = wrap(d[3])
                r[m] = d * w[m]
            return r.ravel()

        S = lil_matrix((4 * len(self.edges), 4 * (n - 1)), dtype=int)
        for m, e in enumerate(self.edges):
            for k in (e.i, e.j):
                if k > 0:
                    S[4 * m: 4 * m + 4, 4 * (k - 1): 4 * k] = 1
        sol = least_squares(residuals, free, jac_sparsity=S, loss="huber", f_scale=3.0, max_nfev=50, x_scale="jac")
        X = unpack(sol.x)
        for k, x in zip(self.kfs, X):
            k.x = np.array([x[0], x[1], x[2], float(wrap(x[3]))])
        self.version += 1
        self.stats["optimisations"] += 1
