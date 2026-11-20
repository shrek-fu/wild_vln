"""Spatio-temporal scene QA and target grounding with GLM (Zhipu, OpenAI-compatible API).

GLM never sees the raw map. It plans with function calls over the topo scene; the
tool set mirrors DAAAM's scene-understanding agent:

=====================  ===========================================  ==================================
tool                   answers                                      DAAAM counterpart
=====================  ===========================================  ==================================
find_objects           open-vocabulary search (SigLIP), near/on     get_matching_subjects
objects_in_radius      what is around a point / object / the robot  get_objects_in_radius
objects_in_region      what is in region N (optionally matching)    get_objects_in_region
region_information     region labels, adjacency, visit timeline     get_region_information
robot_location         where the robot was at time t                get_robot_location
trajectory             path, distance and regions in [t0, t1]       get_agent_trajectory_information
objects_in_view        what the camera saw around time t            get_objects_in_view
object_info            one object's time spans, status, relations   —
route                  walking route to an object over places       (compute_path_in_scene_graph)
look_at                GLM-4V checks candidate crops                —
answer                 final reply (+ optional navigation target)   —
=====================  ===========================================  ==================================

Times are seconds since the start of the recording (``t``) plus wall clock; the
robot's "now" is the end of the recording (or the live pose). GLM only returns
object keys; ``answer.target`` is validated against the map and the geometry
(route over places, goal pose facing the object) is computed by
``scene_memory.navigation.planner``.

Example::

    export ZHIPUAI_API_KEY=...
    python3 -m scene_memory.query.glm_retriever data/scene4d/latest "去门旁边那个黑色垃圾桶"
    python3 -m scene_memory.query.glm_retriever data/scene4d/latest "刚开始经过的房间里有几把椅子？"
    python3 -m scene_memory.query.glm_retriever data/scene4d/latest -i      # interactive
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from scene_memory.query import semantic_index as si
from scene_memory.query.query_4d import fmt_time, load_scene, parse_time, seen_intervals

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "glm-4-plus"
DEFAULT_VISION_MODEL = "glm-4v-flash"

SYSTEM_PROMPT = """You are the scene-understanding module of a mobile robot. The robot built a 3D semantic map
while driving: objects (3D centroids, first/last seen times), places along its trajectory, regions (rooms) with
summaries, and its own timestamped trajectory. Answer the user's question or resolve their navigation instruction.

Rules:
- Ground every statement in tool results. Only use object keys returned by tools; never invent keys or numbers.
  Distances: quote from_robot_now_m / distance_m (straight-line metres) instead of computing them yourself.
- Times: tools report t = seconds since the recording started plus a clock time. "Now" is the end of the recording.
  Convert relative phrases ("two minutes ago", "at the beginning") to t before calling tools.
- find_objects works best with short ENGLISH noun phrases incl. attributes ("black trash can", "glass door").
  Translate Chinese and expand abstract requests ("something to sit on" -> "chair", "sofa"). Try a synonym if
  the best semantic score is low. Use relation/anchor for spatial phrases ("near the door").
  label_matches are further objects whose detector class matches the query; they are real map objects even
  when their image score is low (use look_at if unsure).
- For counting, list objects with objects_in_region / objects_in_radius / objects_in_view and count matching
  names; mention that counts come from the detector and may be approximate.
- If the question names an attribute (colour, material, state) or several candidates remain plausible,
  call look_at (max 6) and follow its visual verdict; detector labels can be coarse (a "tv" may be a monitor).
- If three phrasings all return found=false, the object is not in the map.
- status: objects without a status are present. status=absent means the robot later looked at that spot and the
  object was gone (removed or moved): say where it was last seen and prefer a present alternative as target.
  status=dynamic (people) means the position is only where it was at last_seen; never promise it is still there.
- When several candidates match equally well, prefer the one with the shortest route (call route) unless the
  user specified a location.
- "最远的X / 远处的X / farthest X": find_objects(query=X, order="farthest") lists every matching X in the map,
  farthest first; "最近的X": order="nearest". Pick from that list, not from the default semantic top-k.
- The map only contains the detector classes listed below. If find_objects returns a "note" that nothing is
  labelled like the query, say plainly that this kind of object is not in the map (e.g. "地图里没有检测到机器人，
  检测类别中不包含 robot") instead of describing unrelated candidates — unless search_views is available: then
  call search_views for it first and only say it is not in the map when that fails too.
- Ordinal / directional phrases ("第一排", "最前面", "第一个", "左边的"): rows carry ahead_m / left_m relative to
  the robot now (ahead = its last driving direction, left positive). "First row" = the objects with the smallest
  positive ahead_m (group items within ~1 m of each other as one row); "第一个X" = the nearest X unless the user
  means time ("最早看到的" -> first_seen). Use objects_in_region / objects_in_radius with order="ahead" or
  "distance", pick a concrete target and state your interpretation in the reply instead of refusing.
- Region labels may be missing (label=null); then identify a room by its contents (e.g. desks + chairs + laptops
  = office) via objects_in_region.
- Space hierarchy: places -> regions (kind room / stairs / elevator / outdoor) -> floors -> buildings
  (building0, building1, ...) and the "outdoor" area. Floors are relative levels counted from where the robot
  started or entered that building (0 = that floor, 1 = one floor up, -1 = one down); the real floor number is
  unknown unless a region label says so. Map "二楼 / upstairs / 楼下 / 另一栋楼 / 外面" onto these levels and
  buildings, say which level you assumed, and use objects_on_floor to search one floor. When a route changes
  floor or building, mention the stairs / elevator / entrance from route.instructions in the reply.
- Finish by calling answer exactly once. reply: one to three sentences in the SAME language as the user
  (Chinese question -> Chinese reply). Set target to an object key when the user wants to go to / find a
  specific object, else null.

Scene summary:
{summary}"""


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties,
                                                            "required": required}}}


_TIME = {"type": ["number", "string"], "description": "t seconds since start, negative = seconds before now, "
                                                      "'HH:MM:SS', or 'now'"}

TOOLS = [
    _tool("find_objects", "Open-vocabulary search over map objects; returns ranked candidates with semantic score.",
          {"query": {"type": "string", "description": "short English noun phrase for the target"},
           "relation": {"type": "string", "enum": ["none", "near", "on"]},
           "anchor": {"type": "string", "description": "English noun phrase of the reference object"},
           "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
           "order": {"type": "string", "enum": ["semantic", "nearest", "farthest"],
                     "description": "semantic (default); nearest / farthest: all equally matching objects in the "
                                    "whole map sorted by distance from the robot now (no relation)"}},
          ["query"]),
    _tool("objects_in_radius", "Objects within radius (m) of an object key, of the robot at a time, or of xyz.",
          {"center": {"type": ["string", "array"], "description": "object key, 'robot', or [x, y, z]"},
           "time": _TIME, "radius": {"type": "number"}}, ["center"]),
    _tool("objects_in_region", "Objects in a region (room), with class counts; optional semantic query to rank them.",
          {"region": {"type": ["integer", "string"]}, "query": {"type": "string"},
           "top_k": {"type": "integer", "minimum": 1, "maximum": 40},
           "order": {"type": "string", "enum": ["distance", "ahead", "first_seen"],
                     "description": "sort (without query): nearest first, front-to-back along the robot's "
                                    "driving direction, or earliest seen first"}}, ["region"]),
    _tool("region_information", "All regions: label, description, object counts, adjacency, when the robot "
                                "entered/left each; plus the current region and a visit timeline; buildings with "
                                "their floors, stairs/elevator regions and floor changes.", {}, []),
    _tool("objects_on_floor", "Objects on one floor of a building (or outdoors), optional semantic query to rank.",
          {"building": {"type": ["string", "integer"], "description": "building key (building0, ...) or 'outdoor'"},
           "floor": {"type": "integer", "description": "floor level as in region_information"},
           "query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 40}},
          ["building"]),
    _tool("robot_location", "Robot position, heading, place and region at a time.", {"time": _TIME}, ["time"]),
    _tool("trajectory", "Robot path between two times: sampled poses, distance travelled, regions passed.",
          {"start": _TIME, "end": _TIME}, ["start", "end"]),
    _tool("objects_in_view", "Objects the camera observed within +-window s of a time, nearest first.",
          {"time": _TIME, "window": {"type": "number"}}, ["time"]),
    _tool("object_info", "Details of one object: when it was seen, whether it moved, relations, description.",
          {"key": {"type": "string"}}, ["key"]),
    _tool("route", "Route from the robot's current position to an object over the mapped places: path length (m), "
                   "regions passed, and step instructions incl. stairs / elevator / leaving or entering buildings.",
          {"key": {"type": "string"}}, ["key"]),
    _tool("look_at", "Show the candidates' camera crops to a vision model and ask a question about them.",
          {"keys": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
           "question": {"type": "string"}}, ["keys", "question"]),
    _tool("search_views", "Find something that is NOT a detected object (not in the detector classes, or "
                          "find_objects found nothing fitting) in the mapped panorama views: retrieves views, the "
                          "vision model picks one and boxes the target, the map's point cloud gives its 3D position. "
                          "Returns a new key usable in route / answer.",
          {"query": {"type": "string", "description": "short ENGLISH noun phrase, e.g. 'fridge', 'red fire extinguisher'"},
           "request": {"type": "string", "description": "the user's request in their words (for the vision model)"},
           "top_k": {"type": "integer", "minimum": 2, "maximum": 9}},
          ["query"]),
    _tool("answer", "Final answer.",
          {"reply": {"type": "string", "description": "reply to the user, in the user's language"},
           "target": {"type": ["string", "null"], "description": "object key to navigate to / point at, else null"},
           "alternatives": {"type": "array", "items": {"type": "string"}}},
          ["reply"]),
]


class GlmClient:
    """Minimal chat-completions client (``requests``) for Zhipu or any OpenAI-compatible endpoint."""

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout_s: float = 60.0):
        import requests

        self.session = requests.Session()
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.timeout_s = timeout_s

    def chat(self, model: str, messages: list[dict], tools: list[dict] | None = None) -> dict:
        body = {"model": model, "messages": messages, "temperature": 0.1}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        r = self.session.post(self.url, headers=self.headers, json=body, timeout=self.timeout_s)
        if r.status_code != 200:
            raise RuntimeError(f"{model} HTTP {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]


def make_vision(model: str, base_url: str, api_key: str = "", timeout_s: float = 120.0
                ) -> Callable[[bytes, str], str] | None:
    """``vision(jpeg, prompt) -> text`` over an OpenAI-compatible endpoint (Zhipu, or Ollama's ``.../v1``)."""
    if not model:
        return None
    client = GlmClient(api_key or "none", base_url, timeout_s)

    def vision(image: bytes, text: str) -> str:
        url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
        msg = client.chat(model, [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]}])
        return msg.get("content") or ""

    return vision


def add_vision_args(parser) -> None:
    parser.add_argument("--vision_model", default=os.environ.get("VISION_MODEL", DEFAULT_VISION_MODEL),
                        help="'' disables look_at / the vision steps of search_views")
    parser.add_argument("--vision_url", default=os.environ.get("VISION_BASE_URL", ""),
                        help="separate OpenAI-compatible endpoint for the vision model, e.g. a local Ollama "
                             "http://127.0.0.1:11434/v1 (key from VISION_API_KEY, none needed for Ollama); "
                             "default: --base_url with the GLM key")


def agent_key(args) -> str | None:
    """Key for the text agent; a self-hosted ``--base_url`` (Ollama) needs none."""
    key = os.environ.get(args.api_key_env) or os.environ.get("GLM_API_KEY")
    return key or (None if args.base_url.rstrip("/") == DEFAULT_BASE_URL else "none")


def vision_from_args(args, api_key: str | None) -> Callable[[bytes, str], str] | None:
    if args.vision_url:
        return make_vision(args.vision_model, args.vision_url, os.environ.get("VISION_API_KEY", ""))
    return make_vision(args.vision_model, args.base_url, api_key) if api_key else None


def candidate_grid(scene_dir: Path, objects: list[dict], tile: int = 320, pad: float = 0.6) -> bytes | None:
    """Numbered grid of context crops (box drawn) → JPEG bytes."""
    from PIL import Image, ImageDraw

    tiles = []
    for n, obj in enumerate(objects, 1):
        views = obj.get("views") or []
        if not views:
            continue
        view = max(views, key=lambda v: v.get("quality", 0))
        path = scene_dir / view["image"]
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        ImageDraw.Draw(image).rectangle([float(v) for v in view["xyxy"]], outline=(255, 0, 0), width=3)
        image = si.crop(image, view["xyxy"], pad=pad, min_side=96)
        image.thumbnail((tile, tile))
        canvas = Image.new("RGB", (tile, tile), (0, 0, 0))
        canvas.paste(image, ((tile - image.width) // 2, (tile - image.height) // 2))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, 34, 26], fill=(255, 255, 0))
        draw.text((8, 6), str(n), fill=(0, 0, 0))
        tiles.append(canvas)
    if not tiles:
        return None
    cols = min(3, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    grid = Image.new("RGB", (cols * tile, rows * tile), (0, 0, 0))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % cols) * tile, (i // cols) * tile))
    buf = io.BytesIO()
    grid.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _yaw_deg(q) -> float:
    x, y, z, w = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class SceneTools:
    """``robot_pose`` (``{"position", "orientation"}``, live odometry) overrides the end of the trajectory as
    the robot's current pose for distances and route planning."""

    def __init__(self, scene_dir: str | Path, index, objects: list[dict], graph: dict | None = None,
                 vision: Callable[[bytes, str], str] | None = None, robot_pose: dict | None = None,
                 allow_stairs: bool = True, allow_elevator: bool = True):
        from scene_memory.navigation.planner import PlaceGraph

        self.scene_dir = Path(scene_dir)
        self.index = index
        self.objects = objects
        self.by_key = {o["key"]: o for o in objects}
        graph = graph or {}
        layers = graph.get("layers", {})
        self.graph_objects = {o["key"]: o for o in layers.get("objects", [])}
        self.places = {p["id"]: p for p in layers.get("places", [])}
        self.regions = {r.get("id", i): r for i, r in enumerate(layers.get("regions", []))}
        self.edges = [e for e in graph.get("edges", []) if e["relation"] in ("near", "on")]
        self.agents = sorted(layers.get("agents", []), key=lambda a: a["stamp"])
        self.vision = vision
        stamps = [a["stamp"] for a in self.agents] + [o["first_seen"] for o in objects] + [o["last_seen"] for o in objects]
        self.t0 = min(stamps) if stamps else 0.0
        self.t_end = max(stamps) if stamps else 0.0
        self._visits = self._region_visits()
        self.buildings = layers.get("buildings", [])
        self.floors = layers.get("floors", [])
        self.transitions = graph.get("transitions", [])
        # floor / building columns only when the map has more than one of them
        self.multi_level = (len(self.buildings) > 1 or any(len(b.get("floors", [])) > 1 for b in self.buildings))
        self.planner = PlaceGraph(graph, allow_stairs, allow_elevator) if self.places else None
        self._views = None
        self._found = 0
        self.robot_now = robot_pose or self._pose_at(self.t_end)
        self.heading = self._driving_heading()
        self.classes = sorted({o["name"] for o in objects})

    # ------------------------------------------------------------------ helpers
    def _t(self, stamp: float) -> dict:
        return {"t": round(stamp - self.t0, 1), "clock": fmt_time(stamp)}

    def _stamp(self, value) -> float:
        if value is None or (isinstance(value, str) and value.strip().lower() in ("now", "end", "")):
            return self.t_end
        if isinstance(value, str) and ":" in value:
            return parse_time(value, self.t_end)
        v = float(value)
        return self.t_end + v if v < 0 else self.t0 + v

    def _region_of_place(self, pid) -> int | None:
        return self.places.get(pid, {}).get("region")

    def _region_brief(self, rid) -> dict | None:
        r = self.regions.get(rid)
        if r is None:
            return None
        out = {"region": rid, "label": r.get("label")}
        if r.get("kind", "room") != "room":
            out["kind"] = r["kind"]
        return out

    def _where(self, pid) -> dict:
        """building / floor / indoor-outdoor of a place."""
        p = self.places.get(pid)
        if p is None:
            return {}
        if p.get("environment") == "outdoor":
            return {"building": "outdoor"}
        out = {"building": f"building{p['building']}" if p.get("building") is not None else None}
        if p.get("connector"):
            out["on"] = p["connector"]
        else:
            out["floor"] = p.get("floor", 0)
        return out

    def _pose_at(self, stamp: float) -> dict | None:
        if not self.agents:
            return None
        i = int(np.argmin([abs(a["stamp"] - stamp) for a in self.agents]))
        return self.agents[i]

    def _region_visits(self) -> list[dict]:
        visits = []
        for a in self.agents:
            rid = self._region_of_place(a.get("place"))
            if rid is None:
                continue
            if visits and visits[-1]["region"] == rid:
                visits[-1]["left"] = a["stamp"]
            else:
                visits.append({"region": rid, "entered": a["stamp"], "left": a["stamp"]})
        return visits

    def _from_robot_now(self, center) -> float | None:
        if self.robot_now is None:
            return None
        return round(float(np.linalg.norm(np.asarray(center, float) - np.asarray(self.robot_now["position"], float))), 1)

    def _driving_heading(self, min_m: float = 1.0) -> float | None:
        """Yaw of the robot's last driving direction (the IMU orientation need not point along the robot)."""
        if self.robot_now is None:
            return None
        end = np.asarray(self.robot_now["position"], float)[:2]
        for a in reversed(self.agents):
            d = end - np.asarray(a["position"], float)[:2]
            if np.linalg.norm(d) >= min_m:
                return float(math.atan2(d[1], d[0]))
        return None

    def _ego(self, center) -> dict:
        """Robot-centric offsets: ahead along the last driving direction, left positive."""
        if self.robot_now is None or self.heading is None:
            return {}
        d = np.asarray(center, float)[:2] - np.asarray(self.robot_now["position"], float)[:2]
        c, s = math.cos(self.heading), math.sin(self.heading)
        return {"ahead_m": round(float(c * d[0] + s * d[1]), 1), "left_m": round(float(-s * d[0] + c * d[1]), 1)}

    @staticmethod
    def _status(obj: dict) -> str:
        return "dynamic" if obj.get("dynamic") else obj.get("status", "present")

    def _obj_row(self, obj: dict, dist: float | None = None) -> dict:
        node = self.graph_objects.get(obj["key"], {})
        row = {"key": obj["key"], "name": obj["name"], "center": [round(float(v), 2) for v in obj["center"]],
               "region": node.get("region"), "first_seen": self._t(obj["first_seen"])["t"],
               "last_seen": self._t(obj["last_seen"])["t"]}
        if self._status(obj) != "present":
            row["status"] = self._status(obj)
        if self.multi_level:
            row.update(self._where(node.get("place")))
        if dist is not None:
            row["distance_m"] = round(dist, 2)
        else:
            row["from_robot_now_m"] = self._from_robot_now(obj["center"])
        row.update(self._ego(obj["center"]))
        return row

    def seen_places(self, obj: dict) -> set[int]:
        """Places the robot was at while the object was detected (line of sight was clear there)."""
        spans = seen_intervals(obj)
        return {a["place"] for a in self.agents
                if a.get("place") is not None and any(s - 1.0 <= a["stamp"] <= e + 1.0 for s, e in spans)}

    def plan_to(self, key: str) -> dict:
        obj = self.by_key.get(key)
        if obj is None:
            return {"error": f"unknown key {key}"}
        if self.planner is None or self.robot_now is None:
            return {"error": "no place graph / robot pose"}
        from scene_memory.navigation import grid_nav

        plan = self.planner.plan(self.robot_now["position"], obj["center"], self.seen_places(obj))
        grid = grid_nav.scene_grid(self.scene_dir)
        return grid_nav.with_approach(plan, grid, obj) if grid is not None else plan

    def summary(self) -> str:
        counts = Counter(o["name"] for o in self.objects)
        labels = ", ".join(f"R{r['id']}={r.get('label', '?')}"
                           + (f"[{r['kind']}]" if r.get("kind", "room") != "room" else "")
                           + (f"@{r.get('building')}/F{r.get('floor')}" if self.multi_level and r.get("kind") == "room"
                              else "")
                           for r in self.regions.values())
        spaces = "; ".join(f"{b['key']} floors {b.get('floors', [])}" if b.get("kind") != "outdoor" else "outdoor area"
                           for b in self.buildings)
        return (f"{len(self.objects)} objects, {len(self.places)} places, {len(self.regions)} regions; recording "
                f"t=0..{self.t_end - self.t0:.0f}s ({fmt_time(self.t0)}-{fmt_time(self.t_end)}), now = end.\n"
                f"Detected classes: {dict(counts.most_common())}.\n"
                f"Buildings: {spaces or 'single indoor space'}.\n"
                f"Regions: {labels or 'none'}.\n"
                f"Look-at (vision check): {'available' if self.vision else 'not available'}.\n"
                f"search_views (panorama keyframes): {'available' if self.views_available() else 'not available'}.")

    def views_available(self) -> bool:
        from scene_memory.query.view_search import ViewIndex

        return getattr(self.index, "encoder", None) is not None and ViewIndex.available(self.scene_dir)

    def search_views(self, query: str, request: str = "", top_k: int = 6) -> dict:
        """FSR-VLN slow path: something outside the detector classes, found in the panorama keyframes."""
        from scene_memory.query import view_search as vs

        if not self.views_available():
            return {"error": "no view layer for this map (mapping.view_layer) or no SigLIP encoder"}
        if self._views is None:
            self._views = vs.ViewIndex(self.scene_dir)
        q = self.index.encoder.texts([si.prompt(query)])[0]
        try:
            res = vs.search(self._views, q, request or query, query, self.vision, int(top_k))
        except Exception as exc:  # noqa: BLE001 — model / network errors go back to the planner
            return {"error": f"view search failed: {exc}"}
        if "error" in res:
            return res
        key = f"view{self._found}"
        self._found += 1
        center = [round(float(v), 3) for v in res["point"]]
        stamp = float(res["stamp"])
        self.by_key[key] = {"key": key, "name": query, "center": center, "size": [0.4, 0.4, 0.4], "status": "present",
                            "first_seen": stamp, "last_seen": stamp, "history": [[stamp, *center]], "views": [],
                            "source": "view_search"}
        pid = self.planner.nearest(center) if self.planner is not None else None
        region = self._region_of_place(pid) if pid is not None else None
        self.graph_objects[key] = {"key": key, "place": pid, "region": region}
        return {"key": key, "name": query, "center": center, "from_robot_now_m": self._from_robot_now(center),
                "region": self._region_brief(region) if region is not None else None,
                "seen_at": self._t(stamp)["t"], "view_score": res["score"], "verified": res["verified"],
                "vision_reason": res["reason"]}

    # -------------------------------------------------------------------- tools
    def find_objects(self, query: str, relation: str = "none", anchor: str = "", top_k: int = 5,
                     order: str = "semantic") -> dict:
        relation = "" if relation in (None, "none") else relation
        if order in ("nearest", "farthest") and not relation and self.index is not None:
            return self._find_by_distance(query, int(top_k), order == "farthest")
        hits = self.index.search(query, top_k=int(top_k), relation=relation, anchor=anchor if relation else "")
        rows = []
        for h in hits:
            row = si.hit_to_json(h, self.graph_objects, self.places)
            out = {k: row[k] for k in ("key", "name", "semantic", "score", "center", "region", "waypoint") if k in row}
            out["last_seen"] = self._t(row["last_seen"])["t"] if row.get("last_seen") else None
            out["from_robot_now_m"] = self._from_robot_now(row["center"])
            status = self._status(self.by_key.get(row["key"], {}))
            if status != "present":
                out["status"] = status
            if "anchor" in row:
                out["anchor"] = row["anchor"]
            rows.append(out)
        best = max((r["semantic"] for r in rows), default=0.0)
        q = query.lower().strip()
        listed = {r["key"] for r in rows}
        labelled = [self._obj_row(o) for o in self.objects
                    if q and o["key"] not in listed and (o["name"] in q or q in o["name"])][:int(top_k)]
        named = any(o["name"] in q or q in o["name"] for o in self.objects) if q else False
        out = {"query": query, "relation": relation or None, "anchor": anchor or None,
               "found": best >= 0.01 or bool(labelled), "candidates": rows}
        if labelled:
            out["label_matches"] = labelled
        if q and not named:
            out["note"] = (f"No map object is labelled '{query}'. The detector only knows: {', '.join(self.classes)}. "
                           "The candidates are image-similarity guesses and are probably a different thing; "
                           "unless look_at confirms one, tell the user this kind of object is not in the map "
                           "(not in the detector vocabulary).")
        return out

    def _find_by_distance(self, query: str, top_k: int, farthest: bool) -> dict:
        """All present objects matching ``query`` about as well as the best one, by distance from the robot now."""
        scores = np.asarray(self.index.semantic_scores(query), float)
        q = query.lower().strip()
        present = [i for i, o in enumerate(self.objects) if self._status(o) == "present"]
        named = [i for i in present if q and (self.objects[i]["name"] in q or q in self.objects[i]["name"])]
        if named:
            pool = named
        else:
            best = max((scores[i] for i in present), default=0.0)
            pool = [i for i in present if best > 0 and scores[i] >= 0.6 * best]
        # objects attached to no place are off the driven area (unreachable or misdetections): the extremes otherwise
        placed = [i for i in pool if self.graph_objects.get(self.objects[i]["key"], {}).get("place") is not None]
        pool = placed or pool
        rows = [{**self._obj_row(self.objects[i]), "semantic": round(float(scores[i]), 4)} for i in pool]
        rows.sort(key=lambda r: r.get("from_robot_now_m") or 0.0, reverse=farthest)
        return {"query": query, "order": "farthest" if farthest else "nearest", "num_matching": len(rows),
                "candidates": rows[:top_k]}

    def objects_in_radius(self, center, time=None, radius: float = 3.0) -> dict:
        if isinstance(center, str) and center in self.by_key:
            c, label = np.asarray(self.by_key[center]["center"], float), center
        elif isinstance(center, str) and center.lower() == "robot":
            pose = self._pose_at(self._stamp(time))
            if pose is None:
                return {"error": "no trajectory"}
            c, label = np.asarray(pose["position"], float), f"robot at t={self._t(pose['stamp'])['t']}"
        elif isinstance(center, (list, tuple)) and len(center) == 3:
            c, label = np.asarray(center, float), "point"
        else:
            return {"error": f"center must be an object key, 'robot' or [x,y,z], got {center!r}"}
        rows = sorted(((float(np.linalg.norm(np.asarray(o["center"]) - c)), o) for o in self.objects
                       if o["key"] != center), key=lambda r: r[0])
        hits = [self._obj_row(o, d) for d, o in rows if d <= float(radius)]
        return {"center": label, "radius_m": radius, "counts": dict(Counter(h["name"] for h in hits)),
                "objects": hits[:30]}

    def objects_in_region(self, region, query: str = "", top_k: int = 20, order: str = "") -> dict:
        try:
            rid = int(str(region).lstrip("Rregion"))
        except ValueError:
            return {"error": f"bad region {region!r}"}
        r = self.regions.get(rid)
        if r is None:
            return {"error": f"unknown region {rid}"}
        objs = [self.by_key[k] for k in r.get("objects", []) if k in self.by_key]
        if order and not query:
            sort_key = {"distance": lambda o: self._from_robot_now(o["center"]) or 0.0,
                        "ahead": lambda o: self._ego(o["center"]).get("ahead_m", 0.0),
                        "first_seen": lambda o: o["first_seen"]}.get(order)
            if sort_key is None:
                return {"error": f"order must be distance, ahead or first_seen, got {order!r}"}
            objs = sorted(objs, key=sort_key)
        if query and objs and self.index is not None:
            scores = self.index.semantic_scores(query)
            row = {o["key"]: i for i, o in enumerate(self.objects)}
            objs = sorted(objs, key=lambda o: -scores[row[o["key"]]])
            listed = [{**self._obj_row(o), "semantic": round(float(scores[row[o["key"]]]), 4)} for o in objs[:top_k]]
        else:
            listed = [self._obj_row(o) for o in objs[:top_k]]
        return {"region": rid, "label": r.get("label"), "description": r.get("description"),
                "num_objects": len(objs), "counts": dict(Counter(o["name"] for o in objs).most_common()),
                "objects": listed}

    def region_information(self) -> dict:
        now = self._pose_at(self.t_end)
        current = self._region_of_place(now.get("place")) if now else None
        neighbours = set(self.regions.get(current, {}).get("adjacent", [])) if current is not None else set()
        regions = []
        for rid, r in self.regions.items():
            row = {"region": rid, "label": r.get("label"), "description": r.get("description"),
                   "num_objects": len(r.get("objects", [])), "adjacent": r.get("adjacent", []),
                   "is_current": rid == current, "is_neighbor": rid in neighbours,
                   "visits": [{"entered": self._t(v["entered"])["t"], "left": self._t(v["left"])["t"]}
                              for v in self._visits if v["region"] == rid]}
            if self.multi_level or r.get("kind", "room") != "room":
                row.update(kind=r.get("kind", "room"), building=r.get("building"),
                           floor=r.get("floor") if r.get("kind") in ("room", None) else r.get("floors"))
                if r.get("links"):
                    row["links"] = r["links"]
            regions.append(row)
        timeline = " -> ".join(f"R{v['region']}({self._t(v['entered'])['t']:.0f}-{self._t(v['left'])['t']:.0f}s)"
                               for v in self._visits)
        out = {"current_region": current, "regions": regions, "visit_timeline": timeline}
        if now is not None:
            out["robot_now"] = self._where(now.get("place"))
        out["buildings"] = [{"key": b["key"], "kind": b.get("kind", "building"), "floors": b.get("floors", []),
                             "num_regions": len(b.get("regions", [])), "entrances": len(b.get("entrances", []))}
                            for b in self.buildings]
        if self.transitions:
            out["floor_changes"] = [{**self._t(tr["stamp"]), "kind": tr["kind"],
                                     **({"to_floor": tr["level"]} if "level" in tr else {})}
                                    for tr in self.transitions]
        return out

    def objects_on_floor(self, building=None, floor=None, query: str = "", top_k: int = 20) -> dict:
        """Objects of one building floor (building key or id, floor = level number), or outdoors."""
        b = str(building if building is not None else "building0")
        if b.isdigit():
            b = f"building{b}"
        if b == "outdoor":
            rids = [rid for rid, r in self.regions.items() if r.get("kind") == "outdoor"]
        else:
            fl = int(floor) if floor is not None else 0
            rids = [rid for rid, r in self.regions.items() if r.get("building") == b and r.get("floor") == fl]
        if not rids:
            return {"error": f"no regions on {b} floor {floor}", "buildings": [
                {"key": x["key"], "floors": x.get("floors", [])} for x in self.buildings]}
        keys = [k for rid in rids for k in self.regions[rid].get("objects", [])]
        objs = [self.by_key[k] for k in keys if k in self.by_key]
        if query and objs and self.index is not None:
            scores = self.index.semantic_scores(query)
            row = {o["key"]: i for i, o in enumerate(self.objects)}
            objs = sorted(objs, key=lambda o: -scores[row[o["key"]]])
        return {"building": b, "floor": None if b == "outdoor" else floor, "regions": rids,
                "counts": dict(Counter(o["name"] for o in objs).most_common()),
                "objects": [self._obj_row(o) for o in objs[:int(top_k)]]}

    def robot_location(self, time) -> dict:
        pose = self._pose_at(self._stamp(time))
        if pose is None:
            return {"error": "no trajectory"}
        rid = self._region_of_place(pose.get("place"))
        return {**self._t(pose["stamp"]), "position": [round(float(v), 2) for v in pose["position"]],
                "heading_deg": round(_yaw_deg(pose.get("orientation", [0, 0, 0, 1])), 1),
                "place": f"place{pose['place']}" if pose.get("place") is not None else None,
                "region": self._region_brief(rid), **self._where(pose.get("place"))}

    def trajectory(self, start, end, samples: int = 10) -> dict:
        a, b = sorted((self._stamp(start), self._stamp(end)))
        seg = [p for p in self.agents if a <= p["stamp"] <= b]
        if not seg:
            return {"error": "no trajectory in that interval"}
        pos = np.array([p["position"] for p in seg], float)
        dist = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()) if len(pos) > 1 else 0.0
        idx = np.unique(np.linspace(0, len(seg) - 1, min(samples, len(seg))).round().astype(int))
        passed = [{"region": v["region"], "label": self.regions.get(v["region"], {}).get("label"),
                   "entered": self._t(max(v["entered"], a))["t"], "left": self._t(min(v["left"], b))["t"]}
                  for v in self._visits if v["left"] >= a and v["entered"] <= b]
        return {"from": self._t(a), "to": self._t(b), "distance_m": round(dist, 1),
                "poses": [{**self._t(seg[i]["stamp"]), "position": [round(float(v), 2) for v in seg[i]["position"]]}
                          for i in idx],
                "regions_passed": passed}

    def objects_in_view(self, time, window: float = 3.0) -> dict:
        t = self._stamp(time)
        pose = self._pose_at(t)
        robot = np.asarray(pose["position"], float) if pose else None
        hits = []
        for obj in self.objects:
            if any(a - window <= t <= b + window for a, b in seen_intervals(obj)):
                d = float(np.linalg.norm(np.asarray(obj["center"]) - robot)) if robot is not None else None
                hits.append((d if d is not None else 0.0, self._obj_row(obj, d)))
        hits.sort(key=lambda h: h[0])
        return {**self._t(t), "window_s": window, "counts": dict(Counter(h["name"] for _, h in hits)),
                "objects": [h for _, h in hits][:30]}

    def object_info(self, key: str) -> dict:
        obj = self.by_key.get(key)
        if obj is None:
            return {"error": f"unknown key {key}"}
        hist = np.asarray([row[1:] for row in obj.get("history", [])])
        moved = float(np.linalg.norm(hist.max(0) - hist.min(0))) if len(hist) > 1 else 0.0
        node = self.graph_objects.get(key, {})
        rel = [{"relation": e["relation"], "other": e["target"] if e["source"] == key else e["source"],
                "role": "subject" if e["source"] == key else "object"}
               for e in self.edges if key in (e["source"], e["target"])]
        for r in rel:
            r["other_name"] = self.by_key.get(r["other"], {}).get("name")
        descs = obj.get("descriptions") or []
        return {"key": key, "name": obj["name"], "center": [round(float(v), 2) for v in obj["center"]],
                "from_robot_now_m": self._from_robot_now(obj["center"]), "label_belief": obj.get("label_belief"),
                "visible_now": obj.get("visible"), "status": self._status(obj), "existence": obj.get("existence"),
                "events": [f"t={self._t(s)['t']:.0f}s {ev}" for s, ev in obj.get("events", [])][-6:],
                "seen": [f"t={self._t(a)['t']:.0f}-{self._t(b)['t']:.0f}s" for a, b in seen_intervals(obj)],
                "moved_m": round(moved, 2), "region": self._region_brief(node.get("region")), "relations": rel[:12],
                "description": descs[-1]["text"] if descs else None}

    def look_at(self, keys: list[str], question: str) -> dict:
        if self.vision is None:
            return {"error": "vision model disabled"}
        objs = [self.by_key[k] for k in keys if k in self.by_key][:6]
        objs = [o for o in objs if o.get("views")]
        image = candidate_grid(self.scene_dir, objs)
        if image is None:
            return {"error": "none of these objects has a camera view"}
        legend = ", ".join(f"{n}={o['key']} ({o['name']})" for n, o in enumerate(objs, 1))
        prompt = (f"The image shows numbered crops; the red box marks the object in each crop. Tiles: {legend}.\n"
                  f"Question: {question}\nAnswer with the tile numbers/keys that match and a short reason.")
        try:
            return {"tiles": legend, "vision_answer": self.vision(image, prompt)}
        except Exception as exc:  # network/model errors go back to the planner instead of aborting
            return {"error": f"vision call failed: {exc}"}

    def verify(self, keys: list[str], request: str) -> list[str] | None:
        """Vision check: which of ``keys`` show what ``request`` asks for (one call, up to 6 crops).

        ``None`` = could not check (no vision model, no views, call failed), which callers treat as unverified."""
        if self.vision is None:
            return None
        objs = [o for o in (self.by_key.get(k) for k in keys) if o and o.get("views")][:6]
        image = candidate_grid(self.scene_dir, objs) if objs else None
        if image is None:
            return None
        legend = ", ".join(f"{n}={o['name']}" for n, o in enumerate(objs, 1))
        prompt = (f"Numbered crops from a robot camera; the red box marks the object in each tile ({legend}). "
                  f"The user asked for: \"{request}\". Judge only the object's type and visible attributes "
                  "(colour, material, state); its surroundings may be cut off. Which tiles show such an object? "
                  'Reply with JSON only: {"match": [tile numbers], "reason": "..."}')
        try:
            text = self.vision(image, prompt) or ""
        except Exception:  # noqa: BLE001 — a failed check must not block the answer
            return None
        m = re.search(r"\{.*\}", text, re.S)
        try:
            tiles = json.loads(m.group(0)).get("match", []) if m else None
        except json.JSONDecodeError:
            tiles = None
        if tiles is None:
            if re.search(r"\b(none|no tiles?)\b|没有|都不", text, re.I):
                tiles = []
            else:
                tiles = [int(d) for d in re.findall(r"\b([1-6])\b", text)]
        return [objs[int(n) - 1]["key"] for n in tiles if str(n).isdigit() and 1 <= int(n) <= len(objs)]

    def route(self, key: str) -> dict:
        plan = self.plan_to(key)
        if "error" in plan:
            return plan
        regions = []
        for p in plan["place_path"]:
            rid = self._region_of_place(int(p[5:]))
            if rid is not None and (not regions or regions[-1] != rid):
                regions.append(rid)
        return {"target": key, "status": self._status(self.by_key[key]), "path_length_m": plan["path_length_m"],
                "goal_place": plan["goal_place"], "goal_to_target_m": plan["goal_to_target_m"],
                "target_seen_from_goal": plan["target_seen_from_goal"], "partial": plan["partial"],
                "changes_floor": plan["changes_floor"], "instructions": plan["instructions"],
                "regions_on_route": [self._region_brief(r) for r in regions]}

    def call(self, name: str, args: dict) -> dict:
        fn = {"find_objects": self.find_objects, "objects_in_radius": self.objects_in_radius, "route": self.route,
              "objects_on_floor": self.objects_on_floor,
              "objects_in_region": self.objects_in_region, "region_information": self.region_information,
              "robot_location": self.robot_location, "trajectory": self.trajectory,
              "objects_in_view": self.objects_in_view, "object_info": self.object_info,
              "objects_near": lambda key, radius=2.0: self.objects_in_radius(key, radius=radius),
              "look_at": self.look_at, "search_views": self.search_views}.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except (TypeError, ValueError) as exc:
            return {"error": f"bad arguments: {exc}"}


def resolve(instruction: str, tools: SceneTools, chat: Callable[[list[dict]], dict], max_steps: int = 10,
            log: Callable[[str], None] | None = None, history: list[dict] | None = None,
            verify: bool = True, notes: list[str] | None = None, max_rejections: int = 2) -> dict:
    """Run the GLM tool loop; returns ``{reply, target, alternatives, waypoint, steps, verified}``.

    ``history`` (previous user/assistant turns) makes follow-up questions work. With ``verify`` and a vision
    model, a target the agent did not inspect with look_at is checked before the answer is accepted; a
    rejection goes back to the agent (at most ``max_rejections`` times). ``notes`` are extra context lines."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(summary=tools.summary())}]
    messages += history or []
    messages.append({"role": "user", "content": instruction})
    if notes:
        messages.append({"role": "system", "content": "\n".join(notes)})
    looked: set[str] = set()
    rejections = 0
    for step in range(max_steps):
        if step == max_steps - 1:
            messages.append({"role": "user", "content": "Step budget exhausted: call answer now with your best "
                                                        "answer (target=null if no specific object)."})
        msg = chat(messages)
        calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls} if calls
                        else {"role": "assistant", "content": msg.get("content") or ""})
        if not calls:
            messages.append({"role": "user", "content": "Call the answer tool to finish."})
            continue
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if log:
                log(f"[{step}] {name}({json.dumps(args, ensure_ascii=False)})")
            if name == "answer":
                result = _validate_answer(args, tools)
                if "error" not in result:
                    target = result["target"]
                    result["verified"] = None
                    if target and verify and tools.vision is not None and target not in looked:
                        keys = [target] + result["alternatives"][:3]
                        matches = tools.verify(keys, instruction)
                        looked.update(keys)
                        if matches is not None:
                            result["verified"] = target in matches
                            if target not in matches and rejections < max_rejections:
                                rejections += 1
                                result = {"error": f"visual check: {target} does not look like what the user asked "
                                                   f"for (matching among {keys}: {matches}). Answer again with a "
                                                   "matching key, another candidate, or target=null if none fits."}
                    if "error" not in result:
                        result["steps"] = step + 1
                        return result
            else:
                result = tools.call(name, args)
                if name == "look_at" and "error" not in result:
                    looked.update(args.get("keys") or [])
            if log:
                log(f"     -> {json.dumps(result, ensure_ascii=False)[:300]}")
            messages.append({"role": "tool", "tool_call_id": call.get("id", name),
                             "content": json.dumps(result, ensure_ascii=False)})
    return {"target": None, "alternatives": [], "reply": "未能在步数限制内得出答案 / no answer within step limit",
            "waypoint": None, "steps": max_steps, "verified": None}


def answer_instruction(instruction: str, tools: SceneTools, chat: Callable[[list[dict]], dict], fast: bool = True,
                       verify: bool = True, history: list[dict] | None = None,
                       log: Callable[[str], None] | None = None, **resolve_kwargs) -> dict:
    """Fast path (rules + SigLIP, optional one vision check) first, GLM tool agent if it declines.

    Adds ``path`` ("fast" / "slow"), ``fast_reason`` (why the fast path declined) and ``latency_s``."""
    from scene_memory.query import fast_path as fp

    t0 = time.monotonic()
    notes = None
    if fast:
        res = fp.fast_resolve(instruction, tools, verify=verify)
        if res.target is not None:
            out = _validate_answer({"target": res.target, "alternatives": [k for k in res.ranked if k != res.target][:3],
                                    "reply": "-"}, tools)
            out.update(reply=fp.reply_text(instruction, res, out, tools), steps=0, verified=res.verified,
                       path="fast", latency_s=round(time.monotonic() - t0, 3))
            if log:
                log(f"[fast] '{res.phrase}' -> class {res.cls}, level {res.level}, region {res.region}, "
                    f"ranked {res.ranked[:4]}, verified {res.verified}")
            return out
        if log:
            log(f"[fast] declined: {res.reason}")
        if res.ranked:
            notes = [f"Note: a quick visual check found that {res.ranked[:fp.VERIFY_MAX]} (class {res.cls}) do not "
                     f"look like '{res.phrase}'. Consider other classes or say it is not in the map."]
    out = resolve(instruction, tools, chat, log=log, history=history, verify=verify, notes=notes, **resolve_kwargs)
    out.update(path="slow", fast_reason=res.reason if fast else "disabled", latency_s=round(time.monotonic() - t0, 3))
    return out


def _validate_answer(args: dict, tools: SceneTools) -> dict:
    target = args.get("target") or None
    if target is not None and target not in tools.by_key:
        return {"error": f"{target} is not an object key in the map; choose a key returned by the tools or null"}
    if not str(args.get("reply") or "").strip():
        return {"error": "reply is empty; call answer again with a short reply to the user"}
    alts = [k for k in args.get("alternatives") or [] if k in tools.by_key and k != target]
    out = {"target": target, "alternatives": alts, "reply": args.get("reply", ""), "waypoint": None}
    if target is not None:
        obj = tools.by_key[target]
        node = tools.graph_objects.get(target, {})
        place = tools.places.get(node.get("place"))
        out.update(name=obj["name"], center=[round(float(v), 3) for v in obj["center"]], region=node.get("region"),
                   status=tools._status(obj), **tools._where(node.get("place")), place=place["key"] if place else None,
                   waypoint=[round(float(v), 3) for v in place["position"]] if place else None)
        plan = tools.plan_to(target)
        if "error" not in plan:
            out["plan"] = plan
    return out


def load_graph(scene_dir: Path, objects: list[dict]) -> dict | None:
    """``topo_graph.json``, upgraded in memory to the agents/regions layout if it predates it."""
    from scene_memory.mapping.topo_graph import regraph

    path = scene_dir / "topo_graph.json"
    if not path.exists():
        return None
    graph = json.loads(path.read_text(encoding="utf-8"))
    if "agents" not in graph.get("layers", {}):
        graph = regraph(graph, objects)
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="scene4d directory")
    parser.add_argument("instruction", nargs="*")
    parser.add_argument("-i", "--interactive", action="store_true", help="ask several questions (keeps context)")
    parser.add_argument("--model", default=os.environ.get("GLM_MODEL", DEFAULT_MODEL))
    add_vision_args(parser)
    parser.add_argument("--base_url", default=os.environ.get("GLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api_key_env", default="ZHIPUAI_API_KEY")
    parser.add_argument("--siglip", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--no_fast", action="store_true", help="always use the GLM agent (no rule/SigLIP fast path)")
    parser.add_argument("--no_verify", action="store_true", help="skip the vision check of the chosen target")
    parser.add_argument("--no_stairs", action="store_true", help="plan routes without stairs")
    parser.add_argument("--no_elevator", action="store_true", help="plan routes without elevators")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if not args.instruction and not args.interactive:
        parser.error("give an instruction or use -i")

    api_key = agent_key(args)
    if not api_key:
        parser.error(f"set {args.api_key_env} (Zhipu API key from https://open.bigmodel.cn) "
                     "or point GLM_BASE_URL / --base_url at a self-hosted model")
    scene_dir = Path(args.scene)
    objects = load_scene(scene_dir).get("objects", [])
    graph = load_graph(scene_dir, objects)
    index = si.SemanticIndex(scene_dir, objects, si.SiglipEncoder(args.siglip or si.DEFAULT_SIGLIP, device=args.device))
    client = GlmClient(api_key, args.base_url)
    vision = vision_from_args(args, api_key)
    tools = SceneTools(scene_dir, index, objects, graph, vision,
                       allow_stairs=not args.no_stairs, allow_elevator=not args.no_elevator)
    history: list[dict] = []

    def ask(question: str) -> None:
        result = answer_instruction(question, tools, lambda m: client.chat(args.model, m, TOOLS),
                                    fast=not args.no_fast, verify=not args.no_verify, history=history,
                                    log=print if args.verbose else None, max_steps=args.max_steps)
        history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": result["reply"]}])
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=1))
            return
        print(result["reply"])
        check = {True: "vision ok", False: "vision rejected", None: "not checked"}[result.get("verified")]
        print(f"  [{result['path']} {result['latency_s']} s, {check}"
              + (f", fast path declined: {result['fast_reason']}" if result["path"] == "slow" else "") + "]")
        if result["target"]:
            print(f"  target {result['target']} {result.get('name')} at {result.get('center')} "
                  f"region={result.get('region')} status={result.get('status')} "
                  f"goal={result.get('place')} {result.get('waypoint')}")
            plan = result.get("plan")
            if plan:
                print(f"  route {plan['path_length_m']} m, {len(plan['waypoints'])} waypoints, "
                      f"goal yaw {math.degrees(plan['goal_pose']['yaw']):.0f}°, "
                      f"{plan['goal_to_target_m']} m from target")
                if plan["changes_floor"] or len(plan["segments"]) > 1:
                    for i, step in enumerate(plan["instructions"], 1):
                        print(f"   {i}. {step}")
        if result["alternatives"]:
            print(f"  alternatives: {result['alternatives']}")

    if args.instruction:
        ask(" ".join(args.instruction))
    while args.interactive:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question in ("q", "quit", "exit"):
            break
        if question:
            ask(question)


if __name__ == "__main__":
    main()
