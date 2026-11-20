"""2-D navigation grid on an exported scene's lidar map: what a map-based global planner (QiNav) works with.

Obstacles are ``map_cloud.npy`` returns ``band`` above the floor, the floor under a point being the nearest
trajectory pose (``agents`` layer) minus the sensor height (estimated from the returns under the trajectory),
after carving ``carve_m`` around the driven trajectory (people walking with the robot leave returns there).
A cell is traversable when it lies within ``reach_m`` of the driven trajectory (the mapped area) and no obstacle
is within ``robot_radius``. ``plan`` is 8-connected A* between two points, shortened by line of sight;
``distance_field`` is the Dijkstra cost-to-come from one point (oracle path lengths).
"""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import numpy as np

from scene_memory.navigation.plan_check import carve_driven

SQRT2 = math.sqrt(2.0)
NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def densify(track: np.ndarray, step: float) -> np.ndarray:
    out = [track[:1]]
    for a, b in zip(track, track[1:]):
        n = max(int(np.linalg.norm(b[:2] - a[:2]) / step), 1)
        out.append(a + np.linspace(0.0, 1.0, n + 1)[1:, None] * (b - a))
    return np.concatenate(out)


def estimate_sensor_height(cloud: np.ndarray, traj: np.ndarray, radius: float = 1.0, default: float = 0.5) -> float:
    """Height of the poses above the floor: the densest return height below the trajectory."""
    from scipy.spatial import cKDTree

    dist, idx = cKDTree(traj[:, :2]).query(cloud[:, :2], distance_upper_bound=radius)
    near = np.isfinite(dist)
    rel = cloud[near, 2] - traj[idx[near], 2]
    rel = rel[(rel > -1.5) & (rel < -0.15)]
    if len(rel) < 50:
        return default
    hist, edges = np.histogram(rel, bins=np.arange(-1.5, -0.149, 0.05))
    return float(-(edges[np.argmax(hist)] + 0.025))


class NavGrid:
    def __init__(self, cloud: np.ndarray, traj: np.ndarray, res: float = 0.1, robot_radius: float = 0.3,
                 band: tuple[float, float] = (0.15, 1.0), reach_m: float = 3.0, sensor_height: float | None = None,
                 carve_m: float | None = None):
        from scipy import ndimage
        from scipy.spatial import cKDTree

        self.res, self.robot_radius = res, robot_radius
        self.traj = densify(np.asarray(traj, dtype=np.float64), res)
        self.sensor_height = estimate_sensor_height(cloud, self.traj) if sensor_height is None else sensor_height
        carve = robot_radius if carve_m is None else carve_m
        cloud = carve_driven(cloud, [{"position": p} for p in traj], carve)
        self._tree = cKDTree(self.traj[:, :2])
        _, idx = self._tree.query(cloud[:, :2])
        rel = cloud[:, 2] - (self.traj[idx, 2] - self.sensor_height)
        self.obstacles = cloud[(rel > band[0]) & (rel < band[1])]

        lo = self.traj[:, :2].min(axis=0) - reach_m - 1.0
        hi = self.traj[:, :2].max(axis=0) + reach_m + 1.0
        self.origin = lo
        self.shape = tuple(np.ceil((hi - lo) / res).astype(int) + 1)
        occ = np.zeros(self.shape, dtype=bool)
        rc = self._cells(self.obstacles[:, :2])
        occ[rc[:, 0], rc[:, 1]] = True
        self.clearance = ndimage.distance_transform_edt(~occ) * res
        on_traj = np.zeros(self.shape, dtype=bool)
        rc = self._cells(self.traj[:, :2])
        on_traj[rc[:, 0], rc[:, 1]] = True
        self.reach = ndimage.distance_transform_edt(~on_traj) * res <= reach_m
        self.free = self.reach & (self.clearance >= robot_radius)
        self.free[rc[:, 0], rc[:, 1]] = True  # the robot drove there

    @classmethod
    def from_scene(cls, scene_dir: str | Path, **kw) -> "NavGrid":
        scene_dir = Path(scene_dir)
        graph = json.loads((scene_dir / "topo_graph.json").read_text(encoding="utf-8"))
        traj = np.array([a["position"] for a in graph["layers"]["agents"]], dtype=np.float64)
        cloud = np.load(scene_dir / "map_cloud.npy").astype(np.float64)[:, :3]
        return cls(cloud, traj, **kw)

    def _cells(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        xy = xy[:2].reshape(1, 2) if xy.ndim == 1 else xy[:, :2]
        rc = np.floor((xy - self.origin) / self.res).astype(int)
        return np.clip(rc, 0, np.array(self.shape) - 1)

    def cell(self, xy) -> tuple[int, int]:
        r, c = self._cells(xy)[0]
        return int(r), int(c)

    def xy(self, r: int, c: int) -> np.ndarray:
        return self.origin + (np.array([r, c], dtype=np.float64) + 0.5) * self.res

    def pose_z(self, xy) -> float:
        _, i = self._tree.query(np.asarray(xy, dtype=np.float64)[:2])
        return float(self.traj[i, 2])

    def clearance_at(self, xy) -> float:
        return float(self.clearance[self.cell(xy)])

    def snap(self, xy, max_m: float = 1.5) -> tuple[int, int] | None:
        """Nearest free cell to ``xy`` within ``max_m``."""
        r0, c0 = self.cell(xy)
        if self.free[r0, c0]:
            return r0, c0
        k = int(max_m / self.res)
        rs, cs = np.nonzero(self.free[max(r0 - k, 0):r0 + k + 1, max(c0 - k, 0):c0 + k + 1])
        if len(rs) == 0:
            return None
        rs, cs = rs + max(r0 - k, 0), cs + max(c0 - k, 0)
        i = int(np.argmin((rs - r0) ** 2 + (cs - c0) ** 2))
        return (int(rs[i]), int(cs[i])) if math.hypot(rs[i] - r0, cs[i] - c0) * self.res <= max_m else None

    def plan(self, a, b, snap_m: float = 1.5) -> np.ndarray | None:
        """Shortened A* path (N, 2) from ``a`` to ``b`` (both snapped to free cells), None when unreachable."""
        s, g = self.snap(a, snap_m), self.snap(b, snap_m)
        if s is None or g is None:
            return None
        h = lambda r, c: math.hypot(r - g[0], c - g[1])  # noqa: E731
        best = {s: 0.0}
        parent = {s: None}
        heap = [(h(*s), 0.0, s)]
        rows, cols = self.shape
        while heap:
            _, cost, cur = heapq.heappop(heap)
            if cur == g:
                break
            if cost > best.get(cur, math.inf):
                continue
            for dr, dc, w in NEIGHBOURS:
                n = (cur[0] + dr, cur[1] + dc)
                if not (0 <= n[0] < rows and 0 <= n[1] < cols) or not self.free[n]:
                    continue
                nc = cost + w
                if nc < best.get(n, math.inf):
                    best[n], parent[n] = nc, cur
                    heapq.heappush(heap, (nc + h(*n), nc, n))
        if g not in parent:
            return None
        cells = [g]
        while parent[cells[-1]] is not None:
            cells.append(parent[cells[-1]])
        cells.reverse()
        return np.array([self.xy(*c) for c in self._shorten(cells)])

    def _line_free(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        n = max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2 + 1
        rr = np.rint(np.linspace(a[0], b[0], n)).astype(int)
        cc = np.rint(np.linspace(a[1], b[1], n)).astype(int)
        return bool(self.free[rr, cc].all())

    def _shorten(self, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        out, i = [cells[0]], 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not self._line_free(cells[i], cells[j]):
                j -= 1
            out.append(cells[j])
            i = j
        return out

    def distance_field(self, a, snap_m: float = 1.5) -> np.ndarray:
        """Shortest free-space path length (m) from ``a`` to every cell (inf where unreachable)."""
        dist = np.full(self.shape, np.inf)
        s = self.snap(a, snap_m)
        if s is None:
            return dist
        dist[s] = 0.0
        heap = [(0.0, s)]
        rows, cols = self.shape
        while heap:
            cost, cur = heapq.heappop(heap)
            if cost > dist[cur]:
                continue
            for dr, dc, w in NEIGHBOURS:
                n = (cur[0] + dr, cur[1] + dc)
                if 0 <= n[0] < rows and 0 <= n[1] < cols and self.free[n]:
                    nc = cost + w * self.res
                    if nc < dist[n]:
                        dist[n] = nc
                        heapq.heappush(heap, (nc, n))
        return dist


def path_length(path: np.ndarray | None) -> float:
    if path is None or len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.asarray(path)[:, :2], axis=0), axis=1).sum())


def footprint_dist(xy, obj: dict) -> float:
    """Horizontal distance from ``xy`` to the object's axis-aligned footprint (centre ± size / 2)."""
    c = np.asarray(obj["center"][:2], dtype=np.float64)
    half = np.asarray((obj.get("size") or [0.0, 0.0])[:2], dtype=np.float64) / 2.0
    return float(np.linalg.norm(np.maximum(np.abs(np.asarray(xy[:2], dtype=np.float64) - c) - half, 0.0)))


_CACHE: dict[tuple[str, float], NavGrid] = {}


def scene_grid(scene_dir: str | Path, **kw) -> NavGrid | None:
    """``NavGrid.from_scene`` cached per scene and ``map_cloud.npy`` version; None without a map cloud."""
    f = Path(scene_dir) / "map_cloud.npy"
    if not f.exists() or not (Path(scene_dir) / "topo_graph.json").exists():
        return None
    key = (str(f.resolve()), f.stat().st_mtime)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = NavGrid.from_scene(scene_dir, **kw)
    return _CACHE[key]


def approach_pose(grid: NavGrid, obj: dict, from_xy, standoff: tuple[float, float] = (0.4, 1.2),
                  avoid=(), avoid_m: float = 0.8) -> dict | None:
    """Free cell ``standoff`` from the object's footprint with a clear line to it, nearest by free-space path
    from ``from_xy`` (the route's goal place), heading at the object. None when ``from_xy`` is already that close
    or nothing qualifies. ``avoid``: xy of approach poses found blocked, kept ``avoid_m`` away."""
    avoid = [np.asarray(a[:2], dtype=np.float64) for a in avoid]
    if not avoid and footprint_dist(from_xy, obj) <= standoff[1]:
        return None
    field = grid.distance_field(from_xy)
    c = np.asarray(obj["center"][:2], dtype=np.float64)
    reach = standoff[1] + float(max((obj.get("size") or [0.0, 0.0])[:2])) / 2.0
    (r0, c0), (r1, c1) = grid.cell(c - reach), grid.cell(c + reach)
    sub = field[r0:r1 + 1, c0:c1 + 1]
    rr, cc = np.nonzero(np.isfinite(sub))
    for i in np.argsort(sub[rr, cc]):
        r, col = int(rr[i] + r0), int(cc[i] + c0)
        xy = grid.xy(r, col)
        d = footprint_dist(xy, obj)
        if not standoff[0] <= d <= standoff[1]:
            continue
        if any(np.linalg.norm(xy - a) < avoid_m for a in avoid):
            continue
        # line of sight: no obstacle between the cell and the last 0.3 m before the object
        n = max(int(np.linalg.norm(c - xy) / grid.res), 1)
        pts = xy + np.linspace(0.0, 1.0, n + 1)[:, None] * (c - xy)
        pts = pts[[footprint_dist(p, obj) > 0.3 for p in pts]]
        if len(pts) and min(grid.clearance_at(p) for p in pts) < grid.res:
            continue
        return {"xy": [round(float(xy[0]), 3), round(float(xy[1]), 3)],
                "yaw": round(math.atan2(c[1] - xy[1], c[0] - xy[0]), 4),
                "standoff_m": round(d, 2), "extra_m": round(float(sub[rr[i], cc[i]]), 2)}
    return None


def edge_pose(grid: NavGrid, target_xy, from_xy, min_move_m: float = 0.5) -> dict | None:
    """``approach_pose``-like dict for the reachable cell nearest ``target_xy`` (a target off the map's free
    space), heading at it; None when that is within ``min_move_m`` of ``from_xy`` (no closer to get)."""
    field = grid.distance_field(from_xy)
    rr, cc = np.nonzero(np.isfinite(field))
    if len(rr) == 0:
        return None
    t = np.asarray(target_xy[:2], dtype=np.float64)
    xy = grid.origin + (np.stack([rr, cc], axis=1) + 0.5) * grid.res
    i = int(np.argmin(np.linalg.norm(xy - t, axis=1)))
    if np.linalg.norm(xy[i] - np.asarray(from_xy[:2], dtype=np.float64)) < min_move_m:
        return None
    return {"xy": [round(float(xy[i, 0]), 3), round(float(xy[i, 1]), 3)],
            "yaw": round(math.atan2(t[1] - xy[i, 1], t[0] - xy[i, 0]), 4),
            "standoff_m": round(float(np.linalg.norm(t - xy[i])), 2), "extra_m": round(float(field[rr[i], cc[i]]), 2)}


def approach_waypoint(grid: NavGrid, ap: dict) -> dict:
    """Route waypoint (position on the floor, yaw, quaternion) for an ``approach_pose`` result."""
    yaw = ap["yaw"]
    return {"place": "approach", "position": [ap["xy"][0], ap["xy"][1], round(grid.pose_z(ap["xy"]), 3)],
            "yaw": yaw, "orientation": [0.0, 0.0, round(math.sin(yaw / 2), 6), round(math.cos(yaw / 2), 6)]}


def with_approach(plan: dict, grid: NavGrid, obj: dict) -> dict:
    """``plan`` (``PlaceGraph.plan``) ending at an approach pose of ``obj`` instead of its goal place, when the
    last segment is a walk and an approach pose exists."""
    if "error" in plan or not plan.get("segments") or plan["segments"][-1]["mode"] != "walk":
        return plan
    goal = plan["goal_pose"]["position"]
    ap = approach_pose(grid, obj, goal)
    if ap is None:
        return plan
    wp = approach_waypoint(grid, ap)
    seg = plan["segments"][-1]
    seg["waypoints"] = seg["waypoints"] + [wp]
    seg["length_m"] = round(seg["length_m"] + ap["extra_m"], 1)
    return {**plan, "waypoints": plan["waypoints"] + [wp], "goal_pose": wp, "approach": ap,
            "path_length_m": round(plan["path_length_m"] + ap["extra_m"], 2),
            "goal_to_target_m": round(float(np.linalg.norm(np.asarray(ap["xy"]) - np.asarray(obj["center"][:2]))), 2)}
