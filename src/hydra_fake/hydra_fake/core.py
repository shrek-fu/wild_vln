"""``HydraFakeMapper``: the topo mapper with Hydra's ideas that matter for object positions, without Hydra.

On top of ``TopoMapper`` (detections → lidar / triangulated centres → re-id, label fusion → places,
regions, floors):

1. objects live in a corrected world frame: every observation is kept as (keyframe, odometry-frame
   centre / lidar points); after each pose-graph optimisation (``keyframes.KeyframeGraph``: lidar loop
   closures, re-localisation after LIO resets) centres, boxes and places are re-derived from them, so the
   map deforms with the trajectory instead of keeping stale coordinates or splitting at a reset;
2. ``--hf_center voxel``: an object's centre is the middle of its fused multi-view lidar voxels
   (``InstancePoints`` box) rather than the median of per-frame front-surface centres;
3. the graph gets a free-space layer (``free_space``): traversable places with clearance and, per object,
   an ``approach`` pose (reachable cell facing it) and its free place, the navigation goal of the object.

The replay interface is the ``TopoMapper`` one, so ``remap_bag`` / ``topo_eval --mapper hydra_fake`` and
the live ``topo_map_node`` loop run it unchanged.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from scene_memory.mapping.centroid_lift import CentroidLifter, class_size, pixel_center
from scene_memory.mapping.scene_graph import T_odom_optical, pinhole_intrinsics
from scene_memory.mapping.topo_graph import PlaceLayer, VerticalTracker, build_graph
from scene_memory.nodes.topo_map_node import TopoMapper
from scene_memory.perception.lidar_depth import CloudWindow, mask_depth, project_points

from hydra_fake.free_space import approach, build_grid, free_places
from hydra_fake.keyframes import KeyframeGraph

_RANK = {"prior": 0, "tri": 1, "lidar": 2}


def add_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = p.add_argument_group("hydra_fake")
    g.add_argument("--hf_center", choices=["voxel", "median"], default="voxel",
                   help="object centre: fused lidar voxel box (voxel) or median of per-frame centres (median)")
    g.add_argument("--hf_kf_dist", type=float, default=1.0, help="m between keyframes")
    g.add_argument("--hf_kf_yaw", type=float, default=20.0, help="deg between keyframes")
    g.add_argument("--hf_loop_radius", type=float, default=4.0, help="m; loop candidates around a keyframe")
    g.add_argument("--hf_loop_min_dt", type=float, default=30.0, help="s; same-segment loops need this gap")
    g.add_argument("--hf_reloc_radius", type=float, default=8.0, help="m; search radius after an LIO reset")
    g.add_argument("--hf_min_fitness", type=float, default=0.55, help="ICP inlier share to accept a match")
    g.add_argument("--hf_no_loops", action="store_true", help="keyframes / reset stitching only, no loop closures")
    g.add_argument("--hf_reset_jump", type=float, default=1.0,
                   help="m; an odometry jump at least this large and faster than --hf_reset_speed is an LIO restart")
    g.add_argument("--hf_reset_speed", type=float, default=2.0, help="m/s; implied speed of a jump that is a restart")
    g.add_argument("--hf_reset_turn_rate", type=float, default=2.0,
                   help="rad/s; implied turn rate of an orientation jump (> 0.5 rad) that is a restart")
    g.add_argument("--hf_loop_every", type=int, default=3, help="keyframes between loop-closure attempts")
    g.add_argument("--hf_ground_tol", type=float, default=0.08,
                   help="m; returns this close to the floor are not object points (0 = keep them)")
    g.add_argument("--hf_grid_res", type=float, default=0.1, help="m; free-space grid")
    g.add_argument("--hf_robot_radius", type=float, default=0.35, help="m; clearance needed to pass")
    g.add_argument("--hf_sensor_height", type=float, default=0.0, help="m of the LIO frame above the floor (0 = estimate)")
    g.add_argument("--hf_place_spacing", type=float, default=1.0, help="m between free-space places")
    g.add_argument("--hf_obs_points", type=int, default=200, help="lidar points kept per observation for re-anchoring")
    return p


def hf_defaults() -> argparse.Namespace:
    return add_args(argparse.ArgumentParser(add_help=False)).parse_args([])


class _WorldMap:
    """World cloud of all keyframe submaps; rebuilt when the graph changes, else at most every ``max_age_s``
    (only the newest keyframe's scans can be stale)."""

    def __init__(self, kg: KeyframeGraph, voxel: float, max_age_s: float = 10.0) -> None:
        self.kg, self.voxel, self.max_age_s = kg, voxel, max_age_s
        self._key = None
        self._at = 0.0
        self._pts = np.zeros((0, 3), np.float32)

    def points(self) -> np.ndarray:
        key = (self.kg.version, len(self.kg.kfs))
        now = time.monotonic()
        if key != self._key or now - self._at > self.max_age_s:
            self._pts = self.kg.world_points(self.voxel)
            self._key, self._at = key, now
        return self._pts

    def add(self, xyz) -> None:
        pass


class HydraFakeMapper(TopoMapper):
    handles_reset = True

    def __init__(self, args: argparse.Namespace, env=None) -> None:
        for k, v in vars(hf_defaults()).items():
            if not hasattr(args, k):
                setattr(args, k, v)
        super().__init__(args, env=env)
        a = args
        self.kg = KeyframeGraph(kf_dist_m=a.hf_kf_dist, kf_yaw_deg=a.hf_kf_yaw, loop_radius_m=a.hf_loop_radius,
                                loop_min_dt_s=a.hf_loop_min_dt, reloc_radius_m=a.hf_reloc_radius,
                                min_fitness=a.hf_min_fitness, loop_every=a.hf_loop_every)
        if a.hf_no_loops:
            self.kg.loop_radius_m = 0.0
        self.map_cloud = _WorldMap(self.kg, max(a.map_voxel, 0.05))
        self.obs: dict[str, list[tuple]] = defaultdict(list)
        self.traj: list[tuple[float, int, np.ndarray, tuple]] = []
        self.env_events: list[tuple[float, str]] = []
        self._version = 0
        self._places_dirty = False
        self._last_odom: tuple[float, np.ndarray] | None = None
        self._sensor_h: float | None = None
        self._fs_pts: np.ndarray | None = None
        self._fs: tuple | None = None
        self.stats.update({"segments": 1, "reanchors": 0})
        absorb = self.objects.on_absorb

        def on_absorb(kept: str, removed: str) -> None:
            self.obs[kept].extend(self.obs.pop(removed, []))
            if absorb is not None:
                absorb(kept, removed)

        self.objects.on_absorb = on_absorb
        world_centers = self.objects.track_centers

        def track_centers() -> list[dict]:
            # The detector projects these with the live odometry, so hand them over in the current odom frame.
            if not self.kg.kfs:
                return world_centers()
            inv = np.linalg.inv(self.kg.correction(self.kg.kfs[-1].id))
            return [{**o, "center": _apply(inv, o["center"]).tolist()} for o in world_centers()]

        self.objects.track_centers = track_centers

    # --- odometry / resets -------------------------------------------------------------------------
    def on_reset(self) -> None:
        """Odometry restarted (``/lio/reset`` or a detected jump): keep the map, start a new odom frame."""
        self.kg.new_segment()
        self.stats["segments"] += 1
        self._odom_t, self._odom_T = [], []
        if self.cloud is not None:
            self.cloud = CloudWindow(self.args.cloud_window_s, keep_s=self.args.cloud_keep_s,
                                     voxel_m=self.args.cloud_voxel)
        self.lifter = CentroidLifter(min_rays=self.args.min_rays, min_parallax_deg=self.args.min_parallax_deg,
                                     min_prior_obs=self.args.min_prior_obs)
        self._last_odom = None

    def on_odom(self, t: float, T_odom_imu: np.ndarray, quat_xyzw=None) -> None:
        self.stats["odom"] += 1
        if self._odom_t and t <= self._odom_t[-1]:
            self.stats["odom_old"] += 1
            return
        if self._last_odom is not None:
            t0, T0 = self._last_odom
            dt = max(t - t0, 0.1)
            jump = float(np.linalg.norm(T_odom_imu[:3, 3] - T0[:3, 3]))
            R = T0[:3, :3].T @ T_odom_imu[:3, :3]
            turn = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
            if (jump > self.args.hf_reset_jump and jump / dt > self.args.hf_reset_speed) or \
                    (turn > 0.5 and turn / dt > self.args.hf_reset_turn_rate):
                self.on_reset()
        self._last_odom = (t, T_odom_imu)
        self._odom_t.append(t)
        self._odom_T.append(T_odom_imu)
        if len(self._odom_t) > 4000:
            del self._odom_t[:1000], self._odom_T[:1000]
        k = self.kg.on_odom(t, T_odom_imu)
        self.traj.append((t, k, T_odom_imu, quat_xyzw))
        W = self.kg.correction(k) @ T_odom_imu
        self.places.update(W[:3, 3], t, _quat(W))
        self._maybe_reanchor()

    def on_cloud(self, t: float, xyz_odom: np.ndarray) -> None:
        if self.cloud is not None:
            self.cloud.add(t, xyz_odom)
            self.stats["clouds"] += 1
        self.kg.on_cloud(xyz_odom)
        if not self.args.hf_sensor_height and self._odom_T and len(xyz_odom):
            # The MID-360 sees the floor only beyond a few metres (-7 deg): take the lowest returns of a ring.
            p = self._odom_T[-1][:3, 3]
            d = np.asarray(xyz_odom, np.float64) - p
            r = np.hypot(d[:, 0], d[:, 1])
            below = d[(r > 3.0) & (r < 10.0) & (d[:, 2] < -0.1), 2]
            if len(below) > 100:
                h = float(-np.percentile(below, 5))
                self._sensor_h = h if self._sensor_h is None else 0.95 * self._sensor_h + 0.05 * h

    def sensor_height(self) -> float | None:
        return self.args.hf_sensor_height or self._sensor_h

    def _drop_ground(self, cloud: np.ndarray, T_odom_imu: np.ndarray) -> np.ndarray:
        """Floor returns inside a mask would be the nearest cluster in front of low objects."""
        h = self.sensor_height()
        if h is None or not len(cloud):
            return cloud
        return cloud[cloud[:, 2] > T_odom_imu[2, 3] - h + self.args.hf_ground_tol]

    def on_equirect(self, t: float, bgr) -> str:
        before = self.places.environment
        env = super().on_equirect(t, bgr)
        if env != before:
            self.env_events.append((t, env))
        return env

    # --- detections --------------------------------------------------------------------------------
    def on_detections(self, body: dict) -> list[dict]:
        t = body["stamp"]["sec"] + 1e-9 * body["stamp"]["nanosec"]
        T_odom_imu = self.pose_at(t)
        k = self.kg.kf_at(t)
        if T_odom_imu is None or k is None:
            self.stats["no_pose"] += 1
            return []
        C = self.kg.correction(k)
        size = body.get("view_image") or body["image"]
        w, h = int(size["width"]), int(size["height"])
        cloud = self.cloud.points(t) if self.cloud is not None else None
        if cloud is not None and self.args.hf_ground_tol > 0:
            cloud = self._drop_ground(cloud, T_odom_imu)
        projected: dict[str, tuple] = {}
        lifted, raw = [], []
        for det in body.get("detections", []):
            tid, view = det.get("track_id"), self.views.get(det.get("view") or "")
            if tid is None or view is None:
                continue
            K = pinhole_intrinsics(w, h, view["hfov_deg"])
            T = T_odom_optical(T_odom_imu, self.T_imu_da360, view["R_body_optical"])
            center, source = self.lifter.observe(tid, det["name"], det["xyxy"], pixel_center(det), K, T, w, h, t)
            det = dict(det)
            pts = None
            if cloud is not None and len(cloud):
                if view["name"] not in projected:
                    projected[view["name"]] = project_points(cloud, T, K, w, h, z_max=self.args.lidar_max_range)
                hit = mask_depth(projected[view["name"]], w, h, det, min_points=self.args.lidar_min_points)
                if hit is not None:
                    center, source = hit["center"], "lidar"
                    det["lidar_depth"] = hit["depth"]
                    det["lidar_points"] = hit["n_points"]
                    pts = hit["points"]
            self.stats[source or "none"] += 1
            if center is None:
                continue
            center_odom = np.asarray(center, dtype=np.float64)
            det["cls_scores"] = {int(c): float(s) for c, s in (det.get("cls_scores") or {}).items()}
            det["obb"] = {"center": _apply(C, center_odom).tolist(), "size": list(class_size(det["name"])),
                          "source": source}
            lifted.append(det)
            raw.append((center_odom, source, pts))
        self.objects.update(lifted, t, w, h)
        for det, (center_odom, source, pts) in zip(lifted, raw):
            key = det.get("memory_key")
            obj = self.objects.objects.get(key or "")
            if obj is None:
                continue
            keep = None
            if pts is not None:
                n = self.args.hf_obs_points
                keep = pts[np.random.default_rng(len(self.obs[key])).choice(len(pts), n, replace=False)] if len(pts) > n else pts
                self.instances.add(key, _apply(C, pts), replace=self.objects.is_dynamic(obj))
            self.obs[key].append((k, center_odom, _RANK.get(source, 1), keep, t))
            self._refresh_box(obj)
        if self.args.miss_range > 0 and body.get("views"):
            self._negative_evidence(body, C @ T_odom_imu, w, h, t)
        self.objects.expire(t)
        self.lifter.forget(t)
        self.stats["frames"] += 1
        return lifted

    def _refresh_box(self, obj: dict) -> None:
        box = robust_box(self.instances, obj["key"])
        if box is None:
            return
        obj["bbox3d"] = {**box, "source": "lidar"}
        if self.args.hf_center == "voxel" and obj.get("center_rank", 0) >= _RANK["lidar"]:
            obj["center_median"] = np.asarray(obj["center"]).tolist()
            obj["center"] = np.asarray(box["center"], dtype=np.float64)

    # --- re-anchoring ------------------------------------------------------------------------------
    def _maybe_reanchor(self) -> None:
        if self.kg.version == self._version:
            return
        self._version = self.kg.version
        self.stats["reanchors"] += 1
        self._places_dirty = True
        window = self.objects.window
        for key, obj in self.objects.objects.items():
            obs = self.obs.get(key)
            if not obs:
                continue
            rank = obj.get("center_rank", 0)
            cs = [_apply(self.kg.correction(k), c) for k, c, r, _, _ in obs if r == rank][-window:]
            if cs:
                obj["centers"] = cs
                obj["center"] = np.median(np.stack(cs), axis=0)
            pts = [_apply(self.kg.correction(k), p) for k, _, _, p, _ in obs if p is not None and len(p)]
            if pts:
                self.instances.drop(key)
                for p in pts:
                    self.instances.add(key, p)
                self._refresh_box(obj)

    def _rebuild_places(self) -> None:
        a = self.args
        env = self.places.environment
        layer = PlaceLayer(spacing_m=a.place_spacing,
                           vertical=VerticalTracker(floor_dz=a.floor_min_dz, floor_height=a.floor_height))
        events = list(self.env_events)
        if a.environment != "auto":
            layer.set_environment(a.environment)
        for t, k, T, _ in self.traj:
            while events and events[0][0] <= t:
                layer.set_environment(events.pop(0)[1], t)
            W = self.kg.correction(k) @ T
            layer.update(W[:3, 3], t, _quat(W))
        layer.set_environment(env)
        self.places = layer
        self._places_dirty = False

    # --- export ------------------------------------------------------------------------------------
    def world_point(self, k: int, p_odom) -> np.ndarray:
        return _apply(self.kg.correction(k), p_odom)

    def world_pose_xyz(self, k: int, T_odom: np.ndarray) -> np.ndarray:
        return (self.kg.correction(k) @ T_odom)[:3, 3]

    def world_trajectory(self) -> np.ndarray:
        if not self.traj:
            return np.zeros((0, 3))
        return np.stack([(self.kg.correction(k) @ T)[:3, 3] for _, k, T, _ in self.traj])

    def graph(self, summaries: dict | None = None) -> dict:
        if self._places_dirty:
            self._rebuild_places()
        objs = self.objects.to_json()["objects"]
        graph = build_graph(objs, self.places, region_radius_m=self.args.region_radius,
                            door_radius_m=self.args.door_radius, min_region_places=self.args.min_region_places,
                            summaries=summaries)
        graph["mapper"] = "hydra_fake"
        graph["pose_graph"] = {**self.kg.stats, "keyframes": len(self.kg.kfs),
                               "events": self.kg.events[-50:]}
        try:
            self._free_space(graph, objs)
        except Exception as exc:  # noqa: BLE001 — the free-space layer is optional output
            graph["free_space_error"] = str(exc)
        return graph

    def _free_space(self, graph: dict, objs: list[dict]) -> None:
        a = self.args
        pts = self.map_cloud.points()
        if pts is not self._fs_pts:
            traj = self.world_trajectory()
            grid = build_grid(pts.astype(np.float64), traj[::5] if len(traj) > 5 else traj,
                              res=a.hf_grid_res, robot_radius=a.hf_robot_radius, sensor_height=self.sensor_height())
            self._fs_pts = pts
            self._fs = None if grid is None else (grid, *free_places(grid, a.hf_place_spacing))
        if self._fs is None:
            return
        grid, places, edges = self._fs
        self.grid = grid
        graph["layers"]["free_places"] = places
        graph["free_edges"] = [{"source": f"free{i}", "target": f"free{j}", "relation": "traversable"} for i, j in edges]
        graph["free_space"] = {"res": grid.res, "origin": grid.origin.tolist(), "shape": list(grid.free.shape),
                               "sensor_height": round(grid.sensor_height, 3),
                               "traversable_m2": round(float(grid.traversable.sum()) * grid.res ** 2, 1)}
        by_key = {o["key"]: o for o in objs}
        for node in graph["layers"]["objects"]:
            o = by_key.get(node["key"])
            if o is None:
                continue
            size = (o.get("bbox3d") or {}).get("size") or o.get("size")
            node["approach"] = approach(grid, node["center"], size, places)


def robust_box(instances, key: str, link_voxels: float = 2.5) -> dict | None:
    """AABB of the object's best-supported voxel cluster: voxels linked within ``link_voxels`` voxel sizes,
    the cluster with the most observation hits wins, so a few mis-associated frames elsewhere (occlusion,
    a neighbour's mask) cannot stretch the box and drag the centre."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    xyz, hits = instances.voxels(key)
    if len(xyz) < instances.min_voxels:
        return None
    strong = hits >= instances.min_hits
    if strong.sum() >= instances.min_voxels:
        xyz, hits = xyz[strong], hits[strong]
    pairs = cKDTree(xyz).query_pairs(link_voxels * instances.voxel_m, output_type="ndarray")
    n = len(xyz)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    _, labels = connected_components(graph, directed=False)
    best = np.argmax(np.bincount(labels, weights=hits.astype(np.float64)))
    pts = xyz[labels == best]
    if len(pts) < 3:
        return None
    lo, hi = np.percentile(pts, [instances.trim, 100.0 - instances.trim], axis=0)
    size = np.maximum(hi - lo, instances.voxel_m)
    return {"center": ((lo + hi) * 0.5).tolist(), "size": size.tolist(), "n_voxels": int(len(pts)),
            "cluster_share": round(float(hits[labels == best].sum() / max(hits.sum(), 1)), 3)}


def _apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return p @ T[:3, :3].T + T[:3, 3]


def _quat(T: np.ndarray) -> tuple[float, float, float, float]:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return float(x), float(y), float(z), float(w)
