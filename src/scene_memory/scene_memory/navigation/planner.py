"""Navigation goals on the topo scene graph, across floors, buildings and outdoors.

Places are keyframes of the robot's own trajectory, so every place and every traversable edge is space the
robot has already been through. A goal is planned as a Dijkstra path over places (Omniplanner-style) ending
at the place from which the target was best seen (ABot-N0), with the final heading facing the object.

Edges carry a ``kind`` (walk / stairs / elevator / ramp) with a cost factor and fixed penalty, so the route
prefers walking and can be told to avoid stairs (wheeled robot) or elevators. The route is cut into
``segments`` at every change of mode, floor, building or indoor/outdoor, because a 2D local planner (Nav2,
one map per floor) can only execute walking segments; stairs / elevator / ramp segments are for a higher-level
executor (or a human) and come with an ``instruction``.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

# kind: (length factor, fixed penalty in metres-equivalent)
EDGE_COST = {"walk": (1.0, 0.0), "ramp": (1.2, 0.0), "stairs": (3.0, 5.0), "elevator": (1.0, 25.0)}


def yaw_to_quat(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)]


class PlaceGraph:
    def __init__(self, graph: dict, allow_stairs: bool = True, allow_elevator: bool = True):
        layers = graph.get("layers", {})
        self.frame_id = graph.get("frame_id", "odom")
        self.places = {int(p["id"]): p for p in layers.get("places", [])}
        self.ids = list(self.places)
        self.pos = np.array([self.places[i]["position"] for i in self.ids], dtype=np.float64).reshape(-1, 3)
        self.walkable = {i for i in self.ids if not self._place_kind(i)}
        banned = {k for k, ok in (("stairs", allow_stairs), ("elevator", allow_elevator)) if not ok}
        self.G = nx.Graph()
        self.G.add_nodes_from(self.ids)
        for e in graph.get("edges", []):
            if e["relation"] != "traversable":
                continue
            a, b = int(e["source"][5:]), int(e["target"][5:])
            if a not in self.places or b not in self.places:
                continue
            kind = e.get("kind") or self._place_kind(a) or self._place_kind(b) or "walk"
            if kind in banned:
                continue
            d = float(np.linalg.norm(np.subtract(self.places[a]["position"], self.places[b]["position"])))
            factor, penalty = EDGE_COST.get(kind, EDGE_COST["walk"])
            # the fixed penalty is paid once per connector: split it over the edges entering it
            enters = (self._place_kind(a) is None) != (self._place_kind(b) is None)
            self.G.add_edge(a, b, weight=max(d, 1e-3) * factor + (penalty if enters else 0.0), length=d, kind=kind)

    def _place_kind(self, pid: int) -> str | None:
        return self.places[pid].get("connector")

    def _level(self, pid: int | None) -> tuple | None:
        """Floor of a building, or the outdoor area."""
        if pid is None:
            return None
        p = self.places[pid]
        if p.get("environment") == "outdoor":
            return ("outdoor",)
        return (p.get("building"), p.get("floor", 0))

    def _zone(self, pid: int) -> tuple:
        p = self.places[pid]
        if p.get("connector"):
            return (p["connector"], p.get("building"))
        return ("walk", p.get("building"), p.get("environment", "indoor"), p.get("floor", 0))

    def nearest(self, xyz, among: set[int] | None = None, floor=..., vertical_weight: float = 2.0) -> int | None:
        """Nearest place (vertical offsets weighted, so the floor below / above loses); ``floor`` restricts."""
        if not self.ids:
            return None
        diff = self.pos - np.asarray(xyz, dtype=np.float64)
        d = np.linalg.norm(diff * np.array([1.0, 1.0, vertical_weight]), axis=1)
        mask = np.ones(len(self.ids), bool)
        if among is not None:
            mask &= np.array([i in among for i in self.ids])
        if floor is not ...:
            mask &= np.array([self.places[i].get("floor", 0) == floor for i in self.ids])
        d = np.where(mask, d, np.inf)
        i = int(np.argmin(d))
        return self.ids[i] if np.isfinite(d[i]) else None

    def plan(self, start_xyz, target_xyz, seen_places: set[int] | None = None, standoff_m: float = 1.0,
             candidate_radius_m: float = 3.5, unseen_penalty_m: float = 1.5, detour_weight: float = 2.0,
             waypoint_spacing_m: float = 1.5, start_place: int | None = None) -> dict:
        """Route from the robot to a goal place near ``target_xyz``.

        Goal cost = route cost + ``detour_weight`` x (distance beyond ``standoff_m`` to the target) +
        ``unseen_penalty_m`` if the target was never observed from that place. Candidate places must be on the
        target's level (floor / outdoors of the place nearest to it) and within ``candidate_radius_m`` of it;
        places that observed it from further away count only when no place is that close. ``partial`` = no candidate is reachable (odometry gap, or the only way
        uses a banned stairs / elevator), so the route ends at the closest reachable place."""
        start = start_place if start_place in self.places else self.nearest(start_xyz)
        if start is None:
            return {"error": "map has no places"}
        cost, paths = nx.single_source_dijkstra(self.G, start)
        target = np.asarray(target_xyz, dtype=np.float64)
        seen = seen_places or set()
        level = self._level(self.nearest(target, among=self.walkable))
        # Places within candidate_radius_m first; a place that only saw the target from further away is a goal
        # only when nothing is that close (else "go to" could end where the robot stands, metres away).
        best = None
        for far_seen in (False, True):
            best_cost = math.inf
            for pid, c in cost.items():
                p = np.asarray(self.places[pid]["position"])
                gap = float(np.linalg.norm(p[:2] - target[:2]))
                if pid not in self.walkable or self._level(pid) != level:
                    continue
                if gap > candidate_radius_m and not (far_seen and pid in seen):
                    continue
                total = c + detour_weight * max(gap - standoff_m, 0.0) + (0.0 if pid in seen else unseen_penalty_m)
                if total < best_cost:
                    best, best_cost = pid, total
            if best is not None:
                break
        partial = best is None
        if partial:
            best = self.nearest(target, among=set(cost))
        path = paths[best]
        length = sum(self.G.edges[a, b]["length"] for a, b in zip(path, path[1:]))
        segments = self._segments(path, target, waypoint_spacing_m)
        goal_pose = segments[-1]["waypoints"][-1]
        waypoints = []
        for w in (w for s in segments for w in s["waypoints"]):
            if waypoints and waypoints[-1]["place"] == w["place"]:
                waypoints[-1] = w
            else:
                waypoints.append(w)
        return {
            "frame_id": self.frame_id,
            "start_place": f"place{start}",
            "goal_place": f"place{best}",
            "place_path": [f"place{p}" for p in path],
            "path_length_m": round(float(length), 2),
            "goal_to_target_m": round(float(np.linalg.norm(np.asarray(self.places[best]["position"])[:2] - target[:2])), 2),
            "target_seen_from_goal": best in seen,
            "partial": partial,
            "goal_floor": self.places[best].get("floor", 0),
            "goal_building": self.places[best].get("building"),
            "changes_floor": any(s["mode"] != "walk" for s in segments),
            "segments": segments,
            "instructions": [s["instruction"] for s in segments],
            "goal_pose": goal_pose,
            "waypoints": waypoints,
        }

    def _segments(self, path: list[int], target: np.ndarray, spacing: float) -> list[dict]:
        groups: list[list[int]] = [[path[0]]]
        for a, b in zip(path, path[1:]):
            if self._zone(a) != self._zone(b):
                groups.append([a])  # the boundary place starts the next segment too
            groups[-1].append(b)
        segments = []
        for gi, g in enumerate(groups):
            first, last = self.places[g[0]], self.places[g[-1]]
            kinds = {self._place_kind(i) for i in g[1:]} - {None}
            mode = next(iter(kinds)) if kinds else "walk"
            pts = [np.asarray(self.places[i]["position"], dtype=np.float64) for i in g]
            keep = list(range(len(pts)))
            if mode == "walk":
                keep = [0]
                for i in range(1, len(pts) - 1):
                    if np.linalg.norm(pts[i][:2] - pts[keep[-1]][:2]) >= spacing:
                        keep.append(i)
                if len(pts) > 1:
                    keep.append(len(pts) - 1)
            is_last = gi == len(groups) - 1
            waypoints = []
            for j, i in enumerate(keep):
                if j + 1 < len(keep):
                    nxt = pts[keep[j + 1]]
                    heading = math.atan2(nxt[1] - pts[i][1], nxt[0] - pts[i][0])
                elif is_last:
                    heading = math.atan2(target[1] - pts[i][1], target[0] - pts[i][0])
                else:
                    heading = waypoints[-1]["yaw"] if waypoints else 0.0
                waypoints.append({"place": f"place{g[i]}", "position": [round(float(v), 3) for v in pts[i]],
                                  "yaw": round(heading, 4), "orientation": [round(v, 6) for v in yaw_to_quat(heading)]})
            length = sum(float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(pts) - 1))
            seg = {"mode": mode, "building": last.get("building"), "environment": last.get("environment", "indoor"),
                   "floor": last.get("floor", 0) if mode == "walk" else None, "length_m": round(length, 1),
                   "waypoints": waypoints}
            if mode != "walk":
                nxt = self.places[groups[gi + 1][-1]] if gi + 1 < len(groups) else last
                seg.update(from_floor=first.get("floor", 0), to_floor=nxt.get("floor", 0))
            seg["instruction"] = self._instruction(seg)
            segments.append(seg)
        return segments

    @staticmethod
    def _instruction(seg: dict) -> str:
        where = "outdoors" if seg["environment"] == "outdoor" else f"building{seg['building']}"
        if seg["mode"] == "walk":
            floor = "" if seg["environment"] == "outdoor" else f" floor {seg['floor']}"
            return f"walk {seg['length_m']} m ({where}{floor})"
        a, b = seg.get("from_floor"), seg.get("to_floor")
        direction = "up" if (b or 0) > (a or 0) else "down"
        return f"take the {seg['mode']} {direction} from floor {a} to floor {b} ({where})"
