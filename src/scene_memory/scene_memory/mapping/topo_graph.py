"""Layered semantic topological graph built without depth (DAAAM/Hydra layout).

Layers (bottom up):

* agents   — robot trajectory keyframes (stamp, pose, place).
* objects  — object centroids from the object memory, each attached to the
  nearest place it can be reached from.
* places   — nodes dropped along the VIO trajectory every ``spacing_m``; an edge
  joins two places the robot actually moved between, so every edge is known
  traversable.
* regions  — rooms: the place graph of one zone (floor / indoor-outdoor /
  stairs / elevator) cut at doors, then split into contiguous pieces no wider
  than ``region_radius_m``; small pieces merge into a neighbour. Regions sharing
  a traversable edge are ``adjacent``; across zones the link has a ``kind``
  (entrance / stairs / elevator / ramp).
* floors   — per building, from VIO height changes (``VerticalTracker``).
* buildings — indoor places connected without going outdoors (pieces with
  overlapping footprints merged), plus one ``outdoor`` area.

Object–object edges follow SuperMap's geometric predicates on centroids and
class-size boxes: ``on`` (resting on top) and ``near``. Temporal information is
the per-object ``first_seen``/``last_seen``/``history``, per-place visits and
the agent trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SUMMARY_KEYS = ("label", "description", "reasoning")


class VerticalTracker:
    """Floor changes from VIO height: stairs, elevator or ramp, without a map.

    Indoors the robot moves on flat floors, so a height change of at least ``floor_dz`` between two settled
    heights is a floor change, counted relative to the current floor (``level``), which keeps it immune to
    slow z drift (absorbed by a slow reference, ``drift_tau_s``). The horizontal distance covered while the
    height changed separates elevator (about none) from stairs (steep) and ramps. Outdoors height changes are
    terrain, never floors.
    """

    def __init__(self, start_dz: float = 0.8, floor_dz: float = 2.0, floor_height: float = 3.0,
                 settle_s: float = 3.0, settle_dz: float = 0.25, elevator_xy_m: float = 1.5,
                 stairs_slope: float = 0.35, drift_tau_s: float = 60.0, path_step_m: float = 0.2,
                 split_level_s: float = 20.0):
        self.start_dz, self.floor_dz, self.floor_height = start_dz, floor_dz, floor_height
        self.settle_s, self.settle_dz, self.split_level_s = settle_s, settle_dz, split_level_s
        self.elevator_xy_m, self.stairs_slope = elevator_xy_m, stairs_slope
        self.drift_tau_s, self.path_step_m = drift_tau_s, path_step_m
        self.level = 0
        self.ref_z: float | None = None
        self.transit: dict | None = None
        self._last_t: float | None = None

    def update(self, t: float, p, indoor: bool = True) -> dict | None:
        """Returns ``{"event": "start"}`` when the height starts changing, then ``{"event": "floor", kind,
        levels, dz, level}`` or ``{"event": "level"}`` (no floor change) once it settles."""
        p = np.asarray(p, dtype=np.float64)
        z = float(p[2])
        if self.ref_z is None:
            self.ref_z, self._last_t = z, t
            return None
        dt = max(t - self._last_t, 0.0)
        self._last_t = t
        if self.transit is None:
            if abs(z - self.ref_z) > self.start_dz:
                self.transit = {"t0": t, "z0": self.ref_z, "anchor": p[:2].copy(), "path": 0.0, "win": [(t, z)]}
                return {"event": "start"}
            self.ref_z += min(dt / self.drift_tau_s, 1.0) * (z - self.ref_z)
            return None
        tr = self.transit
        step = float(np.linalg.norm(p[:2] - tr["anchor"]))
        if step >= self.path_step_m:
            tr["path"] += step
            tr["anchor"] = p[:2].copy()
        tr["win"].append((t, z))
        while t - tr["win"][0][0] > self.settle_s:
            tr["win"].pop(0)
        zs = [v for _, v in tr["win"]]
        if t - tr["t0"] < self.settle_s or t - tr["win"][0][0] < 0.8 * self.settle_s or max(zs) - min(zs) > self.settle_dz:
            tr.pop("settled", None)
            return None
        dz = z - tr["z0"]
        # between floors (half landing, or a slow ramp that looks flat over settle_s): keep the transit open
        # unless the robot stays there (split level)
        if indoor and self.start_dz <= abs(dz) < self.floor_dz:
            if "settled" not in tr or abs(z - tr["settled"][1]) > self.settle_dz:
                tr["settled"] = (t, z)
            if t - tr["settled"][0] < self.split_level_s:
                return None
        self.transit, self.ref_z = None, z
        if not indoor or abs(dz) < self.floor_dz:
            return {"event": "level", "dz": dz}
        n = int(round(dz / self.floor_height)) or int(np.sign(dz))
        self.level += n
        slope = abs(dz) / max(tr["path"], 1e-3)
        kind = "elevator" if tr["path"] < self.elevator_xy_m else "stairs" if slope >= self.stairs_slope else "ramp"
        return {"event": "floor", "kind": kind, "levels": n, "dz": dz, "level": self.level}


class PlaceLayer:
    """Places dropped along the trajectory, tagged with ``floor`` (indoor level relative to where the robot
    started / entered the building; None outdoors), ``environment`` (indoor / outdoor) and ``connector``
    (stairs / elevator / ramp) for places created while changing floors. A place only merges with places of
    its own floor, so floors stacked above each other stay apart. ``edge_kind`` marks connector edges."""

    def __init__(self, spacing_m: float = 1.0, link_radius_m: float = 1.5,
                 keyframe_m: float = 0.25, keyframe_s: float = 1.0, vertical: VerticalTracker | None = None):
        self.spacing_m = float(spacing_m)
        self.link_radius_m = float(link_radius_m)
        self.keyframe_m = float(keyframe_m)
        self.keyframe_s = float(keyframe_s)
        self.places: list[dict] = []
        self.edges: set[tuple[int, int]] = set()
        self.edge_kind: dict[tuple[int, int], str] = {}
        self.trajectory: list[dict] = []
        self.transitions: list[dict] = []
        self._current: int | None = None
        self.vertical = vertical or VerticalTracker()
        self.environment = "indoor"
        self._transit_places: set[int] = set()
        self._transit_edges: set[tuple[int, int]] = set()

    def set_environment(self, environment: str, stamp_s: float = 0.0) -> None:
        if environment == self.environment:
            return
        if environment == "indoor":
            self.vertical.level = 0  # entering a building: count floors from the entrance floor
        self.transitions.append({"stamp": float(stamp_s), "kind": "entrance" if environment == "indoor" else "exit",
                                 "place": self._current})
        self.environment = environment

    def floor(self) -> int | None:
        return None if self.environment == "outdoor" else self.vertical.level

    def _on_vertical(self, ev: dict, stamp_s: float) -> None:
        if ev["event"] == "floor":
            for i in self._transit_places:
                self.places[i].update(connector=ev["kind"], floor=None,
                                      floors=sorted({ev["level"] - ev["levels"], ev["level"]}))
            for e in self._transit_edges:
                self.edge_kind[e] = ev["kind"]
            self.transitions.append({"stamp": float(stamp_s), "kind": ev["kind"], "levels": ev["levels"],
                                     "dz": round(ev["dz"], 2), "level": ev["level"], "place": self._current})
        if ev["event"] in ("floor", "level"):
            self._transit_places, self._transit_edges = set(), set()

    def _keyframe(self, p: np.ndarray, stamp_s: float, quat) -> None:
        if self.trajectory:
            last = self.trajectory[-1]
            if (np.linalg.norm(p - np.asarray(last["position"])) < self.keyframe_m
                    and stamp_s - last["stamp"] < self.keyframe_s):
                return
        self.trajectory.append({"stamp": float(stamp_s), "position": p.tolist(),
                                "orientation": [float(q) for q in quat] if quat is not None else [0.0, 0.0, 0.0, 1.0]})

    def _nearest(self, p: np.ndarray, same_floor: bool = False) -> tuple[int | None, float]:
        if not self.places:
            return None, float("inf")
        pos = np.array([pl["position"] for pl in self.places])
        dist = np.linalg.norm(pos - p, axis=1)
        if same_floor:
            floor = self.floor()
            ok = np.array([pl.get("floor", 0) == floor or bool(pl.get("connector")) for pl in self.places])
            dist = np.where(ok, dist, np.inf)
        i = int(dist.argmin())
        return i, float(dist[i])

    def update(self, position, stamp_s: float, quat_xyzw=None) -> int:
        """Robot position in odom → id of the place it is at (created if needed)."""
        p = np.asarray(position, dtype=np.float64)
        self._keyframe(p, stamp_s, quat_xyzw)
        ev = self.vertical.update(stamp_s, p, indoor=self.environment != "outdoor")
        if ev:
            self._on_vertical(ev, stamp_s)
        in_transit = self.vertical.transit is not None
        i, d = self._nearest(p, same_floor=not in_transit)
        created = i is None or d >= self.spacing_m
        if created:
            i = len(self.places)
            self.places.append({"id": i, "position": p.tolist(), "first_seen": float(stamp_s),
                                "last_seen": float(stamp_s), "visits": 1, "floor": self.floor(),
                                "environment": self.environment})
        elif d > self.link_radius_m:
            return self._current if self._current is not None else i
        if in_transit and created:
            self._transit_places.add(i)
        place = self.places[i]
        if i != self._current:
            if self._current is not None:
                e = (min(self._current, i), max(self._current, i))
                self.edges.add(e)
                if in_transit:
                    self._transit_edges.add(e)
                place["visits"] += int(stamp_s - place["last_seen"] > 10.0)
            self._current = i
        place["last_seen"] = float(stamp_s)
        return i

    def nearest(self, position, max_dist_m: float = float("inf")) -> int | None:
        i, d = self._nearest(np.asarray(position, dtype=np.float64))
        return i if d <= max_dist_m else None


def cluster_regions(places: list[dict], radius_m: float = 4.0) -> list[dict]:
    """Leader clustering of places in creation order; returns regions with member ids."""
    regions: list[dict] = []
    for place in places:
        p = np.asarray(place["position"])
        best, best_d = None, radius_m
        for region in regions:
            d = float(np.linalg.norm(np.asarray(region["seed"]) - p))
            if d < best_d:
                best, best_d = region, d
        if best is None:
            best = {"id": len(regions), "seed": p.tolist(), "places": []}
            regions.append(best)
        best["places"].append(place["id"])
        place["region"] = best["id"]
    for region in regions:
        pos = np.array([places[i]["position"] for i in region["places"]])
        region["center"] = pos.mean(axis=0).tolist()
    return regions


def _point_segment_xy(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(distance, unclipped projection parameter) of p onto segment ab in the XY plane."""
    ab = b[:2] - a[:2]
    t = float(np.dot(p[:2] - a[:2], ab) / max(float(np.dot(ab, ab)), 1e-9))
    return float(np.linalg.norm(p[:2] - (a[:2] + np.clip(t, 0.0, 1.0) * ab))), t


def _door_cuts(doors: list[np.ndarray], edges: set[tuple[int, int]], pos: np.ndarray, radius_m: float) -> set:
    """Edges a door lies across: its projection falls inside the edge, else the single closest edge."""
    cut = set()
    for d in doors:
        scored = [(e, *_point_segment_xy(d, pos[e[0]], pos[e[1]])) for e in edges]
        inside = {e for e, dist, t in scored if dist < radius_m and 0.0 < t < 1.0}
        if inside:
            cut |= inside
        elif scored:
            e, dist, _ = min(scored, key=lambda s: s[1])
            if dist < radius_m:
                cut.add(e)
    return cut


def _components(nodes, nbrs: dict[int, list[int]]) -> list[list[int]]:
    nodes, seen, comps = set(nodes), set(), []
    for root in sorted(nodes):
        if root in seen:
            continue
        seen.add(root)
        comp, queue = [], [root]
        while queue:
            i = queue.pop()
            comp.append(i)
            for j in nbrs[i]:
                if j in nodes and j not in seen:
                    seen.add(j)
                    queue.append(j)
        comps.append(sorted(comp))
    return comps


def filled_clearance(places: list[dict], nbrs: dict[int, list[int]]) -> np.ndarray | None:
    """``place["clearance"]`` with gaps filled from graph neighbours (else the median); None if no place has one."""
    n = len(places)
    clear = np.array([np.nan if p.get("clearance") is None else float(p["clearance"]) for p in places])
    if n == 0 or np.isnan(clear).all():
        return None
    for _ in range(n):
        gaps = [i for i in range(n) if np.isnan(clear[i]) and any(not np.isnan(clear[j]) for j in nbrs[i])]
        if not gaps:
            break
        clear[gaps] = [np.nanmean([clear[j] for j in nbrs[i]]) for i in gaps]
    clear[np.isnan(clear)] = np.nanmedian(clear)
    return clear


def clearance_filtration(clear: np.ndarray, nbrs: dict[int, list[int]], min_component: int = 4,
                         steps: int = 40, quantiles: tuple[float, float] = (0.0, 0.9)) -> dict:
    """Hydra RoomFinder filtration on the place graph: sweep a clearance threshold, drop places below it and
    count connected components with at least ``min_component`` places; the threshold range over which that
    count stays the same longest (Hydra's REPEATED mode) gives the room seeds. Thresholds are quantiles of
    the clearances, so the units (metres or camera heights) do not matter."""
    lo, hi = np.quantile(clear, quantiles)
    sweep = []
    for d in np.linspace(lo, hi, steps):
        comps = [c for c in _components(np.flatnonzero(clear >= d), nbrs) if len(c) >= min_component]
        sweep.append((float(d), comps))
    best, start = (0, 0, -1), 0
    for i in range(1, len(sweep) + 1):
        if i == len(sweep) or len(sweep[i][1]) != len(sweep[start][1]):
            count = len(sweep[start][1])
            if count and (i - start, count) > best[:2]:
                best = (i - start, count, start)
            start = i
    if best[2] < 0:
        return {"threshold": None, "seeds": [], "counts": [len(c) for _, c in sweep]}
    d, seeds = sweep[best[2]]
    return {"threshold": d, "seeds": seeds, "persistence": best[0], "counts": [len(c) for _, c in sweep]}


def segment_regions(
    places: list[dict],
    edges: set[tuple[int, int]] | list,
    doors: list | None = None,
    radius_m: float = 4.0,
    door_radius_m: float = 0.8,
    min_places: int = 3,
    min_room_places: int = 4,
    info: dict | None = None,
    same_room: list | None = None,
) -> tuple[list[dict], set[tuple[int, int]]]:
    """Rooms from the place graph: cut traversable edges passing a door, then

    * with ``place["room_pcd"]`` (room of the lidar map, ``mapping.pcd_rooms``): one region per room, no
      door cuts; places outside every room join the room they reach first in the graph;
    * with ``place["clearance"]`` (free space around the place, e.g. ``mapping.clearance``): Hydra-style —
      room seeds from ``clearance_filtration``, every other place joins the seed it reaches first in the
      graph (Hydra's NEIGHBORS clustering); pieces without a seed stay rooms of their own;
    * otherwise: split each connected piece into contiguous clusters within ``radius_m`` of their seed.

    Places joined by a ``same_room`` pair (one spot visited twice, ``mapping.revisit``) always end up in one
    region. Clusters smaller than ``min_places`` merge into the neighbour they share most edges with. Sets
    ``place["region"]``; returns (regions, adjacent region pairs). ``info`` receives the method used."""
    n = len(places)
    pos = np.array([p["position"] for p in places], dtype=np.float64).reshape(n, 3)
    edges = {(min(a, b), max(a, b)) for a, b in edges if a != b and a < n and b < n}
    rooms = [p.get("room_pcd") for p in places]
    use_rooms = any(r is not None for r in rooms)
    n_pcd = len({r for r in rooms if r is not None})
    door_xy = [np.asarray(d, dtype=np.float64) for d in doors or []] if not use_rooms else []
    cut = _door_cuts(door_xy, edges, pos, door_radius_m)
    nbrs: dict[int, list[int]] = {i: [] for i in range(n)}
    for a, b in edges - cut:
        nbrs[a].append(b)
        nbrs[b].append(a)
    info = info if info is not None else {}
    info["door_cuts"] = len(cut)

    label = [-1] * n
    seeds: list[int] = []
    clear = filled_clearance(places, nbrs) if not use_rooms else None
    filt = clearance_filtration(clear, nbrs, min_room_places) if clear is not None else None
    if use_rooms:
        ids = {r: k for k, r in enumerate(sorted({r for r in rooms if r is not None}))}
        info.update(method="pcd", rooms=len(ids), outside=sum(r is None for r in rooms))
        queue = []
        for i, r in enumerate(rooms):
            if r is not None:
                label[i] = ids[r]
                queue.append(i)
        seeds = [label.index(k) for k in range(len(ids))]
        while queue:
            nxt = []
            for i in queue:
                for j in nbrs[i]:
                    if label[j] < 0:
                        label[j] = label[i]
                        nxt.append(j)
            queue = nxt
    elif filt and filt["seeds"]:
        info.update(method="clearance", threshold=filt["threshold"], persistence=filt["persistence"],
                    seeds=len(filt["seeds"]), counts=filt["counts"])
        queue = []
        for c, comp in enumerate(filt["seeds"]):
            seeds.append(comp[0])
            for i in comp:
                label[i] = c
                queue.append(i)
        while queue:
            nxt = []
            for i in queue:
                for j in sorted(nbrs[i], key=lambda j: -clear[j]):
                    if label[j] < 0:
                        label[j] = label[i]
                        nxt.append(j)
            queue = nxt
        for comp in _components([i for i in range(n) if label[i] < 0], nbrs):
            for i in comp:
                label[i] = len(seeds)
            seeds.append(comp[0])
    else:
        info["method"] = "radius"
    for root in range(n):
        if label[root] >= 0:
            continue
        label[root] = len(seeds)
        seeds.append(root)
        queue = [root]
        while queue:
            i = queue.pop(0)
            for j in sorted(nbrs[i]):
                if label[j] >= 0:
                    continue
                cands = {label[k] for k in nbrs[j] if label[k] >= 0}
                best = min(cands, key=lambda c: np.linalg.norm(pos[j] - pos[seeds[c]]))
                if np.linalg.norm(pos[j] - pos[seeds[best]]) < radius_m:
                    label[j] = best
                else:
                    label[j] = len(seeds)
                    seeds.append(j)
                queue.append(j)

    for a, b in [] if use_rooms else same_room or []:
        if a < n and b < n and label[a] != label[b]:
            old, new = max(label[a], label[b]), min(label[a], label[b])
            label = [new if lab == old else lab for lab in label]

    changed = True
    while changed:
        changed = False
        sizes = np.bincount(label, minlength=len(seeds)) if n else np.zeros(0, int)
        # a lidar room stays a region even when the robot only passed it briefly
        for c in [c for c in range(len(seeds)) if 0 < sizes[c] < min_places and c >= n_pcd]:
            links: dict[int, int] = {}
            for a, b in edges:
                if (label[a] == c) != (label[b] == c):
                    other = label[b] if label[a] == c else label[a]
                    links[other] = links.get(other, 0) + 1
            if links:
                target = max(links, key=lambda k: (links[k], sizes[k]))
                label = [target if lab == c else lab for lab in label]
                changed = True
                break

    order = {old: new for new, old in enumerate(dict.fromkeys(label))}
    regions = [{"id": i, "places": []} for i in range(len(order))]
    for i, place in enumerate(places):
        place["region"] = order[label[i]]
        regions[place["region"]]["places"].append(place["id"])
    for region in regions:
        region["center"] = pos[[p for p in region["places"]]].mean(axis=0).tolist()
    adjacent = {(min(places[a]["region"], places[b]["region"]), max(places[a]["region"], places[b]["region"]))
                for a, b in edges if places[a]["region"] != places[b]["region"]}
    return regions, adjacent


def object_relations(objects: list[dict], near_m: float = 1.5, on_tol_m: float = 0.3, k_near: int = 3) -> list[dict]:
    """``on`` / ``near`` edges between objects from centroids and box sizes."""
    edges = []
    if len(objects) < 2:
        return edges
    c = np.array([o["center"] for o in objects], dtype=np.float64)
    s = np.array([o["size"] for o in objects], dtype=np.float64)
    dist = np.linalg.norm(c[:, None] - c[None], axis=2)
    horiz = np.linalg.norm(c[:, None, :2] - c[None, :, :2], axis=2)
    on_pairs = set()
    for a in range(len(objects)):
        for b in range(len(objects)):
            if a == b:
                continue
            gap = (c[a, 2] - 0.5 * s[a, 2]) - (c[b, 2] + 0.5 * s[b, 2])
            if horiz[a, b] < 0.5 * max(s[b, 0], s[b, 1]) and abs(gap) < on_tol_m and s[a, 2] < s[b, 2] * 1.5:
                on_pairs.add(frozenset((a, b)))
                edges.append({"source": objects[a]["key"], "target": objects[b]["key"], "relation": "on",
                              "distance": round(float(dist[a, b]), 3)})
    near_pairs = set()
    for a in range(len(objects)):
        for b in [int(b) for b in np.argsort(dist[a]) if b != a and dist[a, b] < near_m][:k_near]:
            near_pairs.add((min(a, b), max(a, b)))
    for a, b in sorted(near_pairs):
        if frozenset((a, b)) not in on_pairs:
            edges.append({"source": objects[a]["key"], "target": objects[b]["key"], "relation": "near",
                          "distance": round(float(dist[a, b]), 3)})
    return edges


def build_graph(
    objects: list[dict],
    place_layer: PlaceLayer,
    region_radius_m: float = 4.0,
    object_place_max_m: float = 6.0,
    near_m: float = 1.5,
    door_radius_m: float = 0.8,
    min_region_places: int = 3,
    summaries: dict | None = None,
) -> dict:
    return assemble_graph(objects, [dict(p) for p in place_layer.places], place_layer.edges, place_layer.trajectory,
                          region_radius_m=region_radius_m, object_place_max_m=object_place_max_m, near_m=near_m,
                          door_radius_m=door_radius_m, min_region_places=min_region_places, summaries=summaries,
                          edge_kinds=place_layer.edge_kind, transitions=place_layer.transitions)


def place_zone(p: dict) -> tuple:
    """Places of one zone can share a region: same floor + environment, or the same connector kind."""
    if p.get("connector"):
        return ("connector", p["connector"])
    env = p.get("environment", "indoor")
    return (env, None if env == "outdoor" else p.get("floor", 0))


def assign_buildings(places: list[dict], edges, margin_m: float = 0.5) -> list[list[int]]:
    """Indoor places (incl. stairs / elevators) connected without passing outdoors form a building; pieces
    whose XY footprints overlap (the same building entered twice) are merged. Sets ``place["building"]``."""
    parent = {p["id"]: p["id"] for p in places if p.get("environment", "indoor") != "outdoor"}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    comps: dict[int, list[int]] = {}
    for i in parent:
        comps.setdefault(find(i), []).append(i)
    groups = [sorted(c) for c in comps.values()]
    pos = {p["id"]: np.asarray(p["position"], dtype=np.float64) for p in places}
    merged = True
    while merged:
        merged = False
        boxes = [(np.min([pos[i][:2] for i in g], axis=0) - margin_m, np.max([pos[i][:2] for i in g], axis=0) + margin_m)
                 for g in groups]
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if np.all(boxes[a][0] <= boxes[b][1]) and np.all(boxes[b][0] <= boxes[a][1]):
                    groups[a] = sorted(groups[a] + groups.pop(b))
                    merged = True
                    break
            if merged:
                break
    groups.sort(key=lambda g: g[0])
    by_id = {p["id"]: p for p in places}
    for p in places:
        p["building"] = None
    for bid, g in enumerate(groups):
        for i in g:
            by_id[i]["building"] = bid
    return groups


def assemble_graph(
    objects: list[dict],
    places: list[dict],
    place_edges,
    trajectory: list[dict],
    region_radius_m: float = 4.0,
    object_place_max_m: float = 6.0,
    near_m: float = 1.5,
    door_radius_m: float = 0.8,
    min_region_places: int = 3,
    summaries: dict | None = None,
    edge_kinds: dict | None = None,
    transitions: list[dict] | None = None,
    vertical_weight: float = 2.0,
    min_room_places: int = 4,
    object_rooms: dict | None = None,
) -> dict:
    """All layers + edges from places/edges/trajectory and the object list.

    Regions never span zones (floor / indoor-outdoor / stairs / elevator); zones are linked by ``links``
    (``kind`` walk / entrance / stairs / elevator / ramp). Indoor regions are grouped into buildings and
    floors, outdoor regions into the ``outdoor`` area. ``summaries`` maps a region's sorted place ids to its
    ``SUMMARY_KEYS`` fields. ``object_rooms`` (object key -> ``room_pcd``) attaches an object to the nearest
    place of its own lidar room."""
    place_edges = sorted({(min(a, b), max(a, b)) for a, b in place_edges})
    edge_kinds = {(min(a, b), max(a, b)): k for (a, b), k in (edge_kinds or {}).items()}
    zone = {p["id"]: place_zone(p) for p in places}
    intra = [e for e in place_edges if zone[e[0]] == zone[e[1]]]
    doors = [o["center"] for o in objects if "door" in o.get("name", "")]
    seg_info: dict = {}
    same_room = [e for e in intra if edge_kinds.get(e) == "revisit"]
    regions, adjacent = segment_regions(places, intra, doors, region_radius_m, door_radius_m, min_region_places,
                                        min_room_places, seg_info, same_room)
    by_id = {p["id"]: p for p in places}
    links = {}
    for a, b in place_edges:
        ra, rb = by_id[a]["region"], by_id[b]["region"]
        if ra == rb:
            continue
        adjacent.add((min(ra, rb), max(ra, rb)))
        if zone[a] != zone[b]:
            kind = edge_kinds.get((a, b)) or next((z[1] for z in (zone[a], zone[b]) if z[0] == "connector"), None)
            if kind is None:
                kind = "entrance" if zone[a][0] != zone[b][0] else "walk"
            links[(min(ra, rb), max(ra, rb))] = kind
    groups = assign_buildings(places, place_edges)
    pos = np.array([p["position"] for p in places], dtype=np.float64).reshape(len(places), 3)
    scale = np.array([1.0, 1.0, vertical_weight])

    def nearest_place(xyz, max_d=float("inf"), weighted=False):
        if not len(pos):
            return None
        diff = pos - np.asarray(xyz, dtype=np.float64)
        d = np.linalg.norm(diff * scale if weighted else diff, axis=1)
        i = int(d.argmin())
        return i if d[i] <= max_d else None

    for r in regions:
        members = [by_id[i] for i in r["places"]]
        conn = members[0].get("connector")
        env = members[0].get("environment", "indoor")
        r["kind"] = conn or ("outdoor" if env == "outdoor" else "room")
        r["environment"] = env
        if conn:
            r["floor"], r["floors"] = None, sorted({f for m in members for f in m.get("floors") or []})
        elif env == "outdoor":
            r["floor"], r["floors"] = None, []
        else:
            levels = [m.get("floor", 0) for m in members]
            r["floor"] = max(set(levels), key=levels.count)
            r["floors"] = [r["floor"]]
        bids = [m["building"] for m in members if m.get("building") is not None]
        r["building"] = max(set(bids), key=bids.count) if bids else None

    buildings = []
    for bid, g in enumerate(groups):
        regs = [r for r in regions if r["building"] == bid]
        entrances = sorted({f"place{i}" for a, b in place_edges for i in (a, b)
                            if {by_id[a].get("environment", "indoor"), by_id[b].get("environment", "indoor")}
                            == {"indoor", "outdoor"} and by_id[i].get("building") == bid})
        buildings.append({"key": f"building{bid}", "id": bid, "kind": "building", "regions": [r["id"] for r in regs],
                          "floors": sorted({f for r in regs for f in r["floors"]}), "entrances": entrances,
                          "center": pos[g].mean(axis=0).tolist()})
    outdoor = [r for r in regions if r["kind"] == "outdoor"]
    if outdoor:
        idx = [i for r in outdoor for i in r["places"]]
        buildings.append({"key": "outdoor", "id": len(groups), "kind": "outdoor", "regions": [r["id"] for r in outdoor],
                          "floors": [], "entrances": [], "center": pos[idx].mean(axis=0).tolist()})
    if not buildings:
        buildings.append({"key": "building0", "id": 0, "kind": "building", "regions": [], "floors": [],
                          "entrances": [], "center": [0.0, 0.0, 0.0]})
    parent_key = {r["id"]: ("outdoor" if r["kind"] == "outdoor" else f"building{r['building']}")
                  for r in regions}
    floors_layer = [{"key": f"building{b['id']}_floor{f}", "building": b["key"], "floor": f,
                     "regions": [r["id"] for r in regions if r["building"] == b["id"] and r["floor"] == f]}
                    for b in buildings if b["kind"] == "building" for f in b["floors"]]

    edges = [{"source": f"place{a}", "target": f"place{b}", "relation": "traversable",
              **({"kind": edge_kinds[(a, b)]} if (a, b) in edge_kinds else {})} for a, b in place_edges]
    edges += [{"source": f"place{p['id']}", "target": f"region{p['region']}", "relation": "in"} for p in places]
    edges += [{"source": f"region{a}", "target": f"region{b}", "relation": "adjacent",
               **({"kind": links[(a, b)]} if (a, b) in links else {})} for a, b in sorted(adjacent)]
    edges += [{"source": f"region{r['id']}", "target": parent_key[r["id"]], "relation": "in"} for r in regions]
    nodes_obj = []
    for obj in objects:
        if obj.get("dynamic") or obj.get("status") == "rejected":
            continue
        node = {k: obj[k] for k in ("key", "name", "cls", "center", "size", "first_seen", "last_seen", "n_obs",
                                    "visible", "status", "existence", "events", "track_ids", "label_belief",
                                    "history", "bbox3d") if k in obj}
        node["center_source"] = obj.get("center_source", "tri")
        # vertical offsets count double so an object is not attached to the floor below / above
        pid = nearest_place(obj["center"], object_place_max_m, weighted=True)
        room = (object_rooms or {}).get(obj["key"])
        if room is not None:
            node["room_pcd"] = room
            same = [i for i, p in enumerate(places) if p.get("room_pcd") == room]
            if same:
                d = np.linalg.norm((pos[same] - np.asarray(obj["center"], dtype=np.float64)) * scale, axis=1)
                if d.min() <= object_place_max_m:
                    pid = same[int(d.argmin())]
        if pid is not None:
            node["place"] = pid
            node["region"] = places[pid]["region"]
            node["floor"] = places[pid].get("floor", 0)
            node["building"] = places[pid].get("building")
            edges.append({"source": obj["key"], "target": f"place{pid}", "relation": "reachable_from"})
        nodes_obj.append(node)
    edges += object_relations([o for o in nodes_obj if o.get("status", "present") == "present"], near_m=near_m)
    agents = [{**kf, "key": f"agent{i}", "place": nearest_place(kf["position"])} for i, kf in enumerate(trajectory)]
    summaries = summaries or {}
    region_nodes = []
    for r in regions:
        node = {"key": f"region{r['id']}", "id": r["id"], "center": r["center"], "places": r["places"],
                "kind": r["kind"], "environment": r["environment"], "floor": r["floor"], "floors": r["floors"],
                "building": parent_key[r["id"]],
                "objects": [o["key"] for o in nodes_obj if o.get("region") == r["id"]],
                "adjacent": sorted({b if a == r["id"] else a for a, b in adjacent if r["id"] in (a, b)}),
                "links": [{"region": b if a == r["id"] else a, "kind": k}
                          for (a, b), k in sorted(links.items()) if r["id"] in (a, b)]}
        node.update(summaries.get(tuple(sorted(r["places"])), {}))
        region_nodes.append(node)
    return {
        "frame_id": "odom",
        "layers": {
            "agents": agents,
            "objects": nodes_obj,
            "places": [{**p, "key": f"place{p['id']}"} for p in places],
            "regions": region_nodes,
            "floors": floors_layer,
            "buildings": buildings,
        },
        "transitions": list(transitions or []),
        "segmentation": {k: v for k, v in seg_info.items() if k != "counts"},
        "edges": edges,
    }


def regraph(graph: dict, objects: list[dict], **kwargs) -> dict:
    """Rebuild every derived layer of a saved graph (e.g. older files) from its places/edges/trajectory."""
    places = [{k: v for k, v in p.items() if k not in ("key", "region", "building")} for p in graph["layers"]["places"]]
    trav = [e for e in graph["edges"] if e["relation"] == "traversable"]
    place_edges = [(int(e["source"][5:]), int(e["target"][5:])) for e in trav]
    kwargs.setdefault("edge_kinds", {(int(e["source"][5:]), int(e["target"][5:])): e["kind"] for e in trav if e.get("kind")})
    kwargs.setdefault("transitions", graph.get("transitions", []))
    kwargs.setdefault("object_rooms", {o["key"]: o["room_pcd"] for o in graph["layers"].get("objects", [])
                                       if o.get("room_pcd") is not None})
    trajectory = [{k: a[k] for k in ("stamp", "position", "orientation")} for a in graph["layers"].get("agents", [])]
    if not trajectory:
        visits = sorted((t, p["id"]) for p in places for t in {p["first_seen"], p["last_seen"]})
        trajectory = [{"stamp": t, "position": places[i]["position"], "orientation": [0.0, 0.0, 0.0, 1.0]}
                      for t, i in visits]
    summaries = {tuple(sorted(r["places"])): {k: r[k] for k in SUMMARY_KEYS if k in r}
                 for r in graph["layers"].get("regions", []) if r.get("label")}
    out = assemble_graph(objects, places, place_edges, trajectory, summaries=summaries, **kwargs)
    out.update({k: graph[k] for k in ("revisits", "loop_corrected") if k in graph})
    return out


def save_graph(graph: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(graph, indent=1), encoding="utf-8")
    tmp.replace(path)
