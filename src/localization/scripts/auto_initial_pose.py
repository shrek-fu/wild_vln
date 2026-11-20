#!/usr/bin/env python3
"""Automatic initial pose for nav.launch.py's localization (localization_cpp's global_localization only runs
local ICP from an /initialpose and waits for one until then).

The global search follows gn10-pointcloud-localization's GlobalSearcher (refer/gn10-pointcloud-localization,
Apache-2.0): truncated distance-field cost, coarse grid over x y yaw, top-K minima with NMS, refinement,
top-1 / top-2 uniqueness, trial verification. Run on the CPU (the GPU belongs to the vision models), over
the map's traversable cells instead of a whole field.

1. last pose: <data>/loc/<map>_last_pose.json, saved every few seconds while localized, searched in a small
   window first (most runs start where the last one stopped);
2. global: the Super-LIO cloud accumulated for --accum_s around the robot (odom is gravity-aligned, so
   x y yaw, z = ground height + the robot's base height) against a distance field of <map>_map.pcd, over the
   non-lethal cells of <map>_traversable.pcd (or _ground.pcd) plus the mapping trajectory <map>_poses.pcd;
   the best must beat the best distinct other minimum by --uniqueness_ratio, else nothing is published
   (ambiguous place: move a bit, it retries, or give /initialpose by hand);
3. the pose goes to /initialpose; global_localization's next /localize/icp_score (fitness, rmse) must pass its
   thresholds, else the search runs again after --retry_s with a fresh cloud.

/localize/status localization_failed (lost) runs the global search again, as does any message on
/auto_initial_pose/trigger (std_msgs/Empty). Status text on /auto_initial_pose/status.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

WS_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WS_ROOT / "src" / "mapping" / "scripts"))
from semantic_ground import read_pcd, read_pcd_xyz  # noqa: E402

DATA = Path(os.environ.get("STVLN_DATA", WS_ROOT / "data"))


def rotz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def voxel_down(xyz: np.ndarray, leaf: float) -> np.ndarray:
    if len(xyz) == 0 or leaf <= 0:
        return xyz
    _, first = np.unique(np.floor(xyz / leaf).astype(np.int64), axis=0, return_index=True)
    return xyz[np.sort(first)]


class DistanceField:
    """Truncated Euclidean distance (m) to the nearest map point on a voxel grid; outside the grid = dmax."""

    def __init__(self, xyz: np.ndarray, res: float = 0.2, dmax: float = 1.0, max_voxels: float = 1.2e7) -> None:
        from scipy.ndimage import distance_transform_edt

        xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        pad = dmax + 2 * res
        lo, hi = xyz.min(axis=0) - pad, xyz.max(axis=0) + pad
        while np.prod(np.ceil((hi - lo) / res) + 1) > max_voxels:
            res *= 1.25
        self.res, self.dmax, self.lo = float(res), float(dmax), lo
        self.shape = tuple((np.ceil((hi - lo) / res) + 1).astype(int))
        occ = np.ones(self.shape, bool)
        occ[tuple(np.floor((xyz - lo) / res).astype(int).T)] = False
        self.d = np.minimum(distance_transform_edt(occ, sampling=res), dmax).astype(np.float32)
        self._flat = self.d.ravel()
        self._strides = np.array([self.shape[1] * self.shape[2], self.shape[2], 1], np.int64)

    def nearest(self, pts: np.ndarray) -> np.ndarray:
        ijk = np.floor((pts - self.lo.astype(np.float32)) / np.float32(self.res)).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < np.array(self.shape)), axis=-1)
        out = np.full(ok.shape, self.dmax, np.float32)
        out[ok] = self._flat[ijk[ok] @ self._strides]
        return out

    def linear(self, pts: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates

        flat = pts.reshape(-1, 3)
        coords = ((flat - self.lo) / self.res - 0.5).T
        return map_coordinates(self.d, coords, order=1, mode="constant", cval=self.dmax).reshape(pts.shape[:-1])


class Candidate:
    __slots__ = ("cost", "x", "y", "z", "yaw", "inliers")

    def __init__(self, cost, x, y, z, yaw, inliers=0.0):
        self.cost, self.x, self.y, self.z, self.yaw, self.inliers = float(cost), x, y, z, float(wrap(yaw)), inliers

    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def __repr__(self) -> str:
        return (f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f}, {math.degrees(self.yaw):.0f} deg) "
                f"cost {self.cost:.3f} inliers {self.inliers:.2f}")


def coarse_search(df: DistanceField, pts: np.ndarray, cands: np.ndarray, yaws: np.ndarray, top_k: int = 8,
                  nms_xy: float = 1.0, nms_yaw: float = math.radians(30), chunk: int = 2048) -> list[Candidate]:
    """Mean clipped distance of ``pts`` (local, base at origin) at every candidate position x yaw; the
    ``top_k`` lowest that are ``nms_xy`` m or ``nms_yaw`` apart."""
    pts32, cands32 = pts.astype(np.float32), cands.astype(np.float32)
    costs = np.empty((len(yaws), len(cands)), np.float32)
    for a, yaw in enumerate(yaws):
        P = pts32 @ rotz(yaw).T.astype(np.float32)
        for s in range(0, len(cands), chunk):
            c = cands32[s:s + chunk]
            costs[a, s:s + len(c)] = df.nearest(P[None] + c[:, None]).mean(axis=1)
    picked: list[Candidate] = []
    for flat in np.argsort(costs, axis=None):
        a, ci = divmod(int(flat), len(cands))
        x, y, z = cands[ci]
        if all(math.hypot(x - p.x, y - p.y) > nms_xy or abs(wrap(yaws[a] - p.yaw)) > nms_yaw for p in picked):
            picked.append(Candidate(costs[a, ci], x, y, z, yaws[a]))
            if len(picked) >= top_k:
                break
    return picked


def pose_cost(df: DistanceField, pts: np.ndarray, poses: np.ndarray, inlier_d: float):
    """``poses`` (M, 4) x y z yaw -> (mean clipped distance, inlier fraction) per pose, trilinear."""
    c, s = np.cos(poses[:, 3]), np.sin(poses[:, 3])
    x = c[:, None] * pts[:, 0] - s[:, None] * pts[:, 1] + poses[:, :1]
    y = s[:, None] * pts[:, 0] + c[:, None] * pts[:, 1] + poses[:, 1:2]
    z = np.broadcast_to(pts[:, 2], x.shape) + poses[:, 2:3]
    d = df.linear(np.stack([x, y, z], axis=-1))
    return d.mean(axis=1), (d < inlier_d).mean(axis=1)


def refine(df: DistanceField, pts: np.ndarray, cand: Candidate, inlier_d: float = 0.25,
           steps=((0.2, 0.1, math.radians(4)), (0.05, 0.03, math.radians(1))), max_iter: int = 8) -> Candidate:
    """Hill climb on the trilinear cost over x y z yaw, coarse then fine steps."""
    best = np.array([cand.x, cand.y, cand.z, cand.yaw])
    grid = np.stack(np.meshgrid(*[[-1, 0, 1]] * 4, indexing="ij"), axis=-1).reshape(-1, 4).astype(np.float64)
    cost, inl = pose_cost(df, pts, best[None], inlier_d)
    for sxy, sz, syaw in steps:
        delta = grid * np.array([sxy, sxy, sz, syaw])
        for _ in range(max_iter):
            poses = best + delta
            costs, inliers = pose_cost(df, pts, poses, inlier_d)
            i = int(costs.argmin())
            if costs[i] >= cost[0] - 1e-5:
                break
            best, cost, inl = poses[i], costs[i:i + 1], inliers[i:i + 1]
    return Candidate(cost[0], best[0], best[1], best[2], best[3], float(inl[0]))


def pick_unique(refined: list[Candidate], ratio: float, min_inliers: float, sep_xy: float = 1.0,
                sep_yaw: float = math.radians(20)) -> tuple[Candidate | None, Candidate | None, str]:
    """(best, runner-up distinct from it, reason) ; best is None when it fails the inlier floor or the
    top-1 / top-2 margin."""
    if not refined:
        return None, None, "no candidates"
    ranked = sorted(refined, key=lambda c: c.cost)
    best = ranked[0]
    second = next((c for c in ranked[1:] if math.hypot(c.x - best.x, c.y - best.y) > sep_xy
                   or abs(wrap(c.yaw - best.yaw)) > sep_yaw), None)
    if best.inliers < min_inliers:
        return None, second, f"best inliers {best.inliers:.2f} < {min_inliers}"
    if second is not None and best.cost > ratio * second.cost:
        return None, second, f"ambiguous: best {best.cost:.3f} vs {second.cost:.3f} (ratio {ratio})"
    return best, second, "ok"


def grid_candidates(ground: np.ndarray, base_height: float, step: float, extra: np.ndarray | None = None):
    """One candidate base position per ``step`` cell of the ground points (z = ground + base height), plus
    ``extra`` positions (mapping trajectory, already at base height)."""
    keys = np.floor(ground[:, :2] / step).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    out = ground[first] + np.array([0.0, 0.0, base_height])
    if extra is not None and len(extra):
        keys = np.floor(extra[:, :2] / step).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        out = np.vstack([out, extra[first]])
    return out


def base_height_from_poses(poses: np.ndarray, ground: np.ndarray) -> float | None:
    """Median height of the mapping trajectory (base_link) above the ground cell under it."""
    from scipy.spatial import cKDTree

    if len(poses) == 0 or len(ground) == 0:
        return None
    dist, idx = cKDTree(ground[:, :2]).query(poses[:, :2], distance_upper_bound=0.3)
    ok = np.isfinite(dist)
    return float(np.median(poses[ok, 2] - ground[idx[ok], 2])) if ok.sum() >= 5 else None


def local_cloud(clouds_odom: list[np.ndarray], T_odom_base: np.ndarray, leaf: float, min_r: float, max_r: float,
                max_points: int, seed: int = 0) -> np.ndarray:
    """Accumulated odom clouds around the base, gravity-aligned and yawed with the base (base at origin)."""
    q = np.vstack(clouds_odom)
    q = q[np.isfinite(q).all(axis=1)] - T_odom_base[:3, 3]
    q = q @ rotz(yaw_of(T_odom_base[:3, :3]))
    r = np.hypot(q[:, 0], q[:, 1])
    q = voxel_down(q[(r > min_r) & (r < max_r)], leaf)
    if len(q) > max_points:
        q = q[np.random.default_rng(seed).choice(len(q), max_points, replace=False)]
    return q


def T_map_odom_from(cand: Candidate, T_odom_base: np.ndarray) -> np.ndarray:
    """map <- odom such that the base (gravity-aligned local frame) sits at ``cand``."""
    R = rotz(cand.yaw - yaw_of(T_odom_base[:3, :3]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cand.xyz() - R @ T_odom_base[:3, 3]
    return T


class MapSearch:
    """The map side: distance field, candidates, base height; plus the searches."""

    def __init__(self, prefix: Path, args, log=print) -> None:
        t0 = time.monotonic()
        self.args, self.log = args, log
        self.df = DistanceField(read_pcd_xyz(f"{prefix}_map.pcd"), args.df_res, args.dmax)
        if Path(f"{prefix}_traversable.pcd").is_file():
            cols = read_pcd(f"{prefix}_traversable.pcd", ("x", "y", "z", "intensity"))
            ground = cols[cols[:, 3] > args.lethal, :3]
        elif Path(f"{prefix}_ground.pcd").is_file():
            ground = read_pcd_xyz(f"{prefix}_ground.pcd")
        else:
            ground = np.zeros((0, 3))
        poses = read_pcd_xyz(f"{prefix}_poses.pcd") if Path(f"{prefix}_poses.pcd").is_file() else np.zeros((0, 3))
        self.ground = ground[np.isfinite(ground).all(axis=1)]
        self.poses = poses[np.isfinite(poses).all(axis=1)]
        self.base_height = base_height_from_poses(self.poses, self.ground)
        log(f"map {prefix.name}: distance field {self.df.shape} @ {self.df.res:.2f} m, ground {len(self.ground)} "
            f"pts, trajectory {len(poses)} poses, base height "
            f"{'unknown' if self.base_height is None else f'{self.base_height:.2f} m'} "
            f"({time.monotonic() - t0:.1f} s)")

    def candidates(self, base_height: float) -> np.ndarray:
        if len(self.ground):
            return grid_candidates(self.ground, base_height, self.args.step_xy, self.poses)
        return grid_candidates(self.poses, 0.0, self.args.step_xy)

    def search_global(self, pts: np.ndarray, local_base_height: float):
        a = self.args
        bh = self.base_height if self.base_height is not None else local_base_height
        cands = self.candidates(bh)
        yaws = np.arange(-np.pi, np.pi, math.radians(a.step_yaw_deg))
        t0 = time.monotonic()
        coarse = coarse_search(self.df, pts, cands, yaws, a.top_k)
        t1 = time.monotonic()
        refined = [refine(self.df, pts, c, a.inlier_d) for c in coarse]
        best, second, why = pick_unique(refined, a.uniqueness_ratio, a.min_inliers)
        self.log(f"global search: {len(cands)} positions x {len(yaws)} yaws x {len(pts)} pts, coarse "
                 f"{t1 - t0:.1f} s, refine {time.monotonic() - t1:.1f} s; best {min(refined, key=lambda c: c.cost) if refined else None}, "
                 f"runner-up {second}: {why}")
        return best, why

    def search_near(self, pts: np.ndarray, prior: Candidate):
        a = self.args
        r = np.arange(-a.prior_xy, a.prior_xy + 1e-6, 0.25)
        gx, gy = np.meshgrid(r, r, indexing="ij")
        cands = np.column_stack([prior.x + gx.ravel(), prior.y + gy.ravel(), np.full(gx.size, prior.z)])
        yaws = prior.yaw + np.radians(np.arange(-a.prior_yaw_deg, a.prior_yaw_deg + 1e-6, 5.0))
        coarse = coarse_search(self.df, pts, cands, yaws, 3, nms_xy=0.5, nms_yaw=math.radians(10))
        refined = [refine(self.df, pts, c, a.inlier_d) for c in coarse]
        best = min(refined, key=lambda c: c.cost) if refined else None
        ok = best is not None and best.inliers >= a.min_inliers
        self.log(f"last-pose search around {prior}: {best} -> {'ok' if ok else 'rejected'}")
        return (best if ok else None), ("ok" if ok else "last pose does not match")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map_prefix", required=True, help="<prefix>_map.pcd, _traversable.pcd, _poses.pcd")
    p.add_argument("--odom_topic", default="/lio/base/odom")
    p.add_argument("--cloud_topic", default="/lio/cloud_world", help="deskewed scans in odom")
    p.add_argument("--score_topic", default="/localize/icp_score", help="global_localization /icp_score (remapped)")
    p.add_argument("--status_topic", default="/localize/status", help="global_localization /loc_status (remapped)")
    p.add_argument("--accum_s", type=float, default=3.0, help="seconds of LIO cloud per search")
    p.add_argument("--max_range", type=float, default=20.0)
    p.add_argument("--max_points", type=int, default=800, help="local cloud points in the search")
    p.add_argument("--df_res", type=float, default=0.2, help="m; grown automatically for large maps")
    p.add_argument("--dmax", type=float, default=1.0, help="m; distance truncation")
    p.add_argument("--step_xy", type=float, default=0.5)
    p.add_argument("--step_yaw_deg", type=float, default=7.5)
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--uniqueness_ratio", type=float, default=0.85,
                   help="best cost must be <= ratio x the best distinct other one (1 disables)")
    p.add_argument("--min_inliers", type=float, default=0.5, help="fraction of points within --inlier_d")
    p.add_argument("--inlier_d", type=float, default=0.25)
    p.add_argument("--lethal", type=float, default=0.25, help="traversable intensity at or below = no candidate")
    p.add_argument("--prior_xy", type=float, default=1.5, help="m; last-pose search half window")
    p.add_argument("--prior_yaw_deg", type=float, default=30.0)
    p.add_argument("--fitness", type=float, default=0.8, help="global_localization fitness_threshold")
    p.add_argument("--rmse", type=float, default=0.3, help="global_localization rmse_threshold")
    p.add_argument("--retry_s", type=float, default=5.0)
    p.add_argument("--save_s", type=float, default=5.0, help="last-pose save period while localized")
    p.add_argument("--last_pose", default="", help="default <data>/loc/<map>_last_pose.json")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from scene_memory.mapping.scene_graph import pose_to_matrix
    from scene_memory.perception.lidar_depth import cloud_xyz_fast
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Empty, Float32MultiArray, String

    args = _parse_args(argv)
    prefix = Path(args.map_prefix)
    last_pose_path = Path(args.last_pose) if args.last_pose else DATA / "loc" / f"{prefix.name}_last_pose.json"

    rclpy.init()
    node = rclpy.create_node("auto_initial_pose")
    log = node.get_logger()
    search = MapSearch(prefix, args, log=log.info)

    lock = threading.Lock()
    clouds: deque = deque()
    st = {"T_odom_base": None, "T_map_odom": None, "state": "wait_data", "pub_t": 0.0, "next_t": 0.0,
          "score_t": 0.0, "score_ok": False, "busy": False, "use_prior": last_pose_path.is_file(), "saved_t": 0.0,
          "forced": False}
    pose_pub = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 1)
    status_pub = node.create_publisher(String, "/auto_initial_pose/status", 1)

    def status(text: str, warn: bool = False) -> None:
        # rclpy refuses two severities from one call site
        if warn:
            log.warning(text)
        else:
            log.info(text)
        status_pub.publish(String(data=text))

    def on_odom(msg) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        with lock:
            st["T_odom_base"] = pose_to_matrix((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))

    def on_cloud(msg) -> None:
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        try:
            xyz = cloud_xyz_fast(msg)
        except ValueError:
            return
        with lock:
            clouds.append((t, xyz.astype(np.float64)))
            while clouds and clouds[0][0] < t - args.accum_s:
                clouds.popleft()

    def on_score(msg) -> None:
        if len(msg.data) < 2:
            return
        fitness, rmse = float(msg.data[0]), float(msg.data[1])
        ok = fitness > args.fitness and rmse < args.rmse
        now = time.monotonic()
        verdict = None
        with lock:
            st["score_t"], st["score_ok"] = now, ok
            if st["state"] in ("wait_data", "searching", "failed") and ok and not st["forced"]:
                st["state"] = "localized"  # given by hand (or recovered by itself)
                verdict = f"global_localization already localized (fitness {fitness:.2f}, rmse {rmse:.3f})"
            elif st["state"] == "verify" and now > st["pub_t"] + 0.1:
                st["state"] = "localized" if ok else "failed"
                if not ok:
                    st["next_t"], st["use_prior"] = now + args.retry_s, False
                verdict = (f"localized (fitness {fitness:.2f}, rmse {rmse:.3f})" if ok else
                           f"ICP rejected the pose (fitness {fitness:.2f}, rmse {rmse:.3f}); retry in "
                           f"{args.retry_s:.0f} s")
        if verdict:
            status(verdict, warn=not ok)

    def on_status(msg) -> None:
        if msg.data == "localization_failed":
            with lock:
                st["state"], st["next_t"], st["use_prior"] = "failed", time.monotonic(), False
            status("global_localization lost the pose: global search again", warn=True)

    def on_trigger(_msg) -> None:
        with lock:
            st["state"], st["next_t"], st["use_prior"], st["forced"] = "failed", time.monotonic(), False, True
        status("search triggered")

    def on_map_to_odom(msg) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        with lock:
            st["T_map_odom"] = pose_to_matrix((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))

    def publish_pose(T_map_odom: np.ndarray) -> None:
        with lock:
            T = T_map_odom @ st["T_odom_base"]
        from scipy.spatial.transform import Rotation

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = (float(v) for v in T[:3, 3])
        qx, qy, qz, qw = Rotation.from_matrix(T[:3, :3]).as_quat()
        o = msg.pose.pose.orientation
        o.x, o.y, o.z, o.w = float(qx), float(qy), float(qz), float(qw)
        msg.pose.covariance[0] = msg.pose.covariance[7] = 0.05
        msg.pose.covariance[35] = 0.02
        pose_pub.publish(msg)

    def run_search() -> None:
        try:
            with lock:
                T_ob = st["T_odom_base"].copy()
                snap = [c for _, c in clouds]
                use_prior = st["use_prior"]
            pts = local_cloud(snap, T_ob, search.df.res, 0.6, args.max_range, args.max_points)
            if len(pts) < 100:
                with lock:
                    st["state"], st["next_t"] = "failed", time.monotonic() + 1.0
                status(f"only {len(pts)} local points, waiting for more LIO cloud", warn=True)
                return
            near = pts[np.hypot(pts[:, 0], pts[:, 1]) < 3.0]
            local_bh = float(-np.percentile(near[:, 2], 5)) if len(near) > 20 else 0.5
            cand, why = None, ""
            if use_prior:
                prior = json.loads(last_pose_path.read_text())
                cand, why = search.search_near(pts, Candidate(0, prior["x"], prior["y"], prior["z"], prior["yaw"]))
            if cand is None:
                cand, why = search.search_global(pts, local_bh)
            with lock:
                if st["state"] != "searching":
                    return  # localized meanwhile
                if cand is None:
                    st["state"], st["next_t"], st["use_prior"] = "failed", time.monotonic() + args.retry_s, False
                else:
                    st["state"], st["pub_t"], st["forced"] = "verify", time.monotonic(), False
            if cand is None:
                status(f"no reliable pose ({why}); retry in {args.retry_s:.0f} s - move the robot a little or give "
                       f"/initialpose by hand", warn=True)
                return
            publish_pose(T_map_odom_from(cand, T_ob))
            status(f"published /initialpose {cand}; waiting for global_localization's ICP")
        except Exception as exc:  # keep the node alive; next tick retries
            with lock:
                st["state"], st["next_t"] = "failed", time.monotonic() + args.retry_s
            status(f"search error: {exc!r}", warn=True)
        finally:
            with lock:
                st["busy"] = False

    def save_last_pose() -> None:
        with lock:
            if st["T_map_odom"] is None or st["T_odom_base"] is None:
                return
            T = st["T_map_odom"] @ st["T_odom_base"]
        last_pose_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = last_pose_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"map": str(prefix), "x": T[0, 3], "y": T[1, 3], "z": T[2, 3],
                                   "yaw": yaw_of(T[:3, :3]), "stamp": time.time()}))
        tmp.replace(last_pose_path)

    def tick() -> None:
        now = time.monotonic()
        with lock:
            state, busy = st["state"], st["busy"]
            have = st["T_odom_base"] is not None and clouds and clouds[-1][0] - clouds[0][0] > 0.8 * args.accum_s
            if state == "verify" and now > st["pub_t"] + 8.0:
                st["state"], st["next_t"], st["use_prior"] = "failed", now + args.retry_s, False
                state = "timeout"
            start = not busy and have and (state == "wait_data" or (state == "failed" and now >= st["next_t"]))
            if start:
                st["state"], st["busy"] = "searching", True
            save = state == "localized" and st["score_ok"] and now - st["score_t"] < 10.0 and \
                now - st["saved_t"] > args.save_s
            if save:
                st["saved_t"] = now
        if state == "timeout":
            status("no /icp_score within 8 s after /initialpose (global_localization running?); retrying", warn=True)
        if start:
            threading.Thread(target=run_search, daemon=True).start()
        if save:
            save_last_pose()

    node.create_subscription(Odometry, args.odom_topic, on_odom, 50)
    node.create_subscription(PointCloud2, args.cloud_topic, on_cloud, qos_profile_sensor_data)
    node.create_subscription(Float32MultiArray, args.score_topic, on_score, 10)
    node.create_subscription(String, args.status_topic, on_status, 10)
    node.create_subscription(Odometry, "/localize/map_to_odom", on_map_to_odom, 10)
    node.create_subscription(Empty, "/auto_initial_pose/trigger", on_trigger, 1)
    node.create_timer(0.5, tick)
    status(f"auto initial pose on {prefix.name}: last pose {last_pose_path if st['use_prior'] else 'none'}, "
           f"waiting {args.accum_s:.0f} s of {args.cloud_topic}")
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
