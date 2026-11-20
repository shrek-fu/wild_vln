"""Prompt → target object → navigation goal, live, on the map that ``topo_map_node`` keeps saving.

Send an instruction or question::

    ros2 topic pub --once /scene_memory/query std_msgs/String "{data: '去门旁边的垃圾桶'}"

The GLM tool agent (``scene_memory.query.glm_retriever``) answers it over the newest saved scene, with the
live odometry as "now". When the answer names a target the node publishes

* ``/scene_memory/answer``  std_msgs/String, JSON ``{reply, target, name, status, plan, ...}`` (always)
* ``/scene_memory/goal``    geometry_msgs/PoseStamped, final pose facing the object
* ``/scene_memory/path``    nav_msgs/Path, route over the place graph (already-driven, hence free, space)
* ``/scene_memory/goal_markers`` visualization_msgs/MarkerArray

* ``/scene_memory/route_leg``  std_msgs/String, JSON of the leg the robot should drive now

and, with ``--nav2 through_poses|pose``, forwards it to Nav2. With ``--qinav`` the walking legs go to the QiNav
nav_server instead (``/slam/navigation/execute``, api 1001, ``qi_nav_msgs/Waypoint`` x/y/z/yaw in its map
frame, so the scene must be mapped with ``odom:=qinav``); QiNav's ``/ql/nav/task_status`` is mirrored as JSON
on ``/scene_memory/nav_status``. With ``--p2p`` (``stvln.sh nav``) the last pose of each walking leg goes to
STVLN's own qi_move_base (``/p2p_move_base``); accepted / finished events go to ``nav_status`` too. When the
last goal ends within ``--near_goal_m`` of its pose without reaching it (someone stands there), that counts as
arrived (``arrived_near``); aborted farther out, other free approach poses around the target are tried
(``--goal_retries``), then ``--alt_targets`` alternative targets, else ``failed``. After arriving, the target
is looked for in the four Insta360 views and lidar-ranged (``scene_memory.navigation.arrival_check``):
``verified`` / ``seen`` (no lidar in its box) / ``far`` / ``not_seen`` (it may have moved: alternatives next).
A target the map lacks (navigation instructions only), or a dynamic / absent one, is first looked for around
the robot (``searching``); a sighting is driven to (``--search_moves``): a ranged one via an approach pose on
the map's free space, an unranged one ``--search_step_m`` along its bearing, then looked for again;
``not_found`` when nothing is in view. Off the map (``outside_map``) it turns on map_extender
(``/stvln/map_extend``) and steps over the growing planning ground (``mapground``) toward the target,
waiting up to ``--extend_wait_s`` for new ground when none leads closer (``--outside_moves`` steps).
In no view at all (and no alternatives left), it explores WildOS-style (``scene_memory.navigation.
frontier_explore``): frontiers of the planning ground, scored by the VLM's side-by-side rating of the four views
as directions to the target, cheapest ``path * (1 - k ln score)`` first (``exploring``), looking again at each
(``--explore_steps``); map_extender grows the map meanwhile; ``not_found`` when out of steps or frontiers.
These events go to
``nav_status`` and, with a reply text, to ``answer``. Only one GLM request runs at a time; a query that arrives while busy is
answered with ``{"error": "busy"}``. Needs ``ZHIPUAI_API_KEY`` in the environment.

Routes across floors are split into legs: walking legs (any number of rooms / buildings / outdoor on one
level) and ``stairs`` / ``elevator`` legs. Nav2 only gets walking legs (its costmap is one floor); a
stairs / elevator leg is published on ``route_leg`` for the robot's own behaviour (or an operator), which
then publishes ``std_msgs/Empty`` on ``/scene_memory/next_leg`` to continue with the next leg.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
import time
from pathlib import Path

from scene_memory import paths
from scene_memory.navigation import arrival_check as ac
from scene_memory.navigation import frontier_explore as fe
from scene_memory.navigation import grid_nav
from scene_memory.query import fast_path as fp
from scene_memory.query import glm_retriever as gr
from scene_memory.query import semantic_index as si
from scene_memory.query.query_4d import load_scene


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene_dir", default=str(paths.SCENES / "latest"),
                   help="scene4d directory (symlinks such as 'latest' are re-resolved on every query)")
    p.add_argument("--odom_topic", default="/equirect_vio/odometry")
    p.add_argument("--query_topic", default="/scene_memory/query")
    p.add_argument("--answer_topic", default="/scene_memory/answer")
    p.add_argument("--goal_topic", default="/scene_memory/goal")
    p.add_argument("--path_topic", default="/scene_memory/path")
    p.add_argument("--markers_topic", default="/scene_memory/goal_markers")
    p.add_argument("--leg_topic", default="/scene_memory/route_leg")
    p.add_argument("--next_leg_topic", default="/scene_memory/next_leg")
    p.add_argument("--nav2", choices=["none", "through_poses", "pose"], default="none")
    p.add_argument("--qinav", action="store_true", help="send walking legs to QiNav nav_server (api 1001)")
    p.add_argument("--qinav_service", default="/slam/navigation/execute")
    p.add_argument("--qinav_status_topic", default="/ql/nav/task_status")
    p.add_argument("--qinav_frame", default="map", help="frame QiNav waypoints are in (qinav_sim: the scene's frame)")
    p.add_argument("--qinav_route", action="store_true",
                   help="send the place-graph route as waypoints (default: only the goal; QiNav plans on its map)")
    p.add_argument("--qinav_route_spacing", type=float, default=3.0, help="m between route waypoints sent to QiNav")
    p.add_argument("--p2p", action="store_true",
                   help="send each walking leg's last pose to STVLN's qi_move_base (/p2p_move_base action)")
    p.add_argument("--p2p_action", default="p2p_move_base")
    p.add_argument("--goal_reached_m", type=float, default=0.5, help="qi_move_base goal_reached_dis")
    p.add_argument("--near_goal_m", type=float, default=1.5,
                   help="a p2p goal that ends (or aborts) this close counts as reached: its spot was occupied")
    p.add_argument("--goal_retries", type=int, default=2,
                   help="other approach poses around the target tried when the p2p goal aborts farther out")
    p.add_argument("--alt_targets", type=int, default=1, help="alternative targets tried after that")
    p.add_argument("--frame_id", default="",
                   help="frame the goals are published in (default: the scene graph's; 'map' for a scene filed "
                        "with a map, whose odom is that map's frame)")
    p.add_argument("--arrival_check", choices=["on", "off"], default="on",
                   help="after a goal ends: find the target in the Insta360 views (VLM box), lidar-range it, report "
                        "verified / far (walk closer once) / seen / not_seen")
    p.add_argument("--arrival_url", default=ac.DEFAULT_URL, help="OpenAI-compatible VLM endpoint (STVLN_VLM_URL)")
    p.add_argument("--arrival_model", default=ac.DEFAULT_MODEL)
    p.add_argument("--arrival_reach_m", type=float, default=2.0, help="target this close (lidar) counts as reached")
    p.add_argument("--search", choices=["on", "off"], default="on",
                   help="needs --arrival_check: look around for a target the semantic map lacks (navigation "
                        "instructions only) and for a dynamic / absent one before driving to its old spot")
    p.add_argument("--search_moves", type=int, default=3,
                   help="goals driven toward sightings per instruction (closer to a far one, or along the "
                        "bearing of an unranged one)")
    p.add_argument("--search_step_m", type=float, default=3.0, help="walk along an unranged sighting's bearing")
    p.add_argument("--extend_map", choices=["on", "off"], default="on",
                   help="off-map sightings: turn on map_extender (--extend_topic) and step toward them over the "
                        "growing planning ground (--ground_topic) instead of stopping at the map's edge")
    p.add_argument("--ground_topic", default="mapground", help="planning ground (traversable, intensity 9999 free)")
    p.add_argument("--extend_topic", default="/stvln/map_extend")
    p.add_argument("--outside_moves", type=int, default=10, help="goals driven toward an off-map sighting")
    p.add_argument("--explore", choices=["on", "off"], default="on",
                   help="target in no view (and no alternatives): WildOS-style frontier exploration over the "
                        "planning ground, frontiers scored by the VLM's per-view 'promise' ratings")
    p.add_argument("--explore_steps", type=int, default=8, help="frontiers visited per instruction")
    p.add_argument("--explore_score_factor", type=float, default=0.5,
                   help="cost = path_m * (1 - factor * ln(score)) (WildOS frontier_score_factor)")
    p.add_argument("--explore_default_score", type=float, default=0.3, help="unrated frontiers (WildOS: 0.3)")
    p.add_argument("--explore_rate_m", type=float, default=12.0, help="frontiers rated from a look (WildOS: 9)")
    p.add_argument("--extend_wait_s", type=float, default=8.0,
                   help="wait this long for the map to grow when no planning ground leads closer yet")
    p.add_argument("--arrival_settle_s", type=float, default=1.0, help="wait after stopping for a still cloud")
    p.add_argument("--arrival_views", default="front,left,rear,right")
    p.add_argument("--arrival_image_topic", default="/insta360/{view}/image/compressed")
    p.add_argument("--arrival_cloud_topic", default="/ql/nav/cloud_in_body", help="lidar cloud in --base_frame")
    p.add_argument("--base_frame", default="base_link")
    p.add_argument("--camera_frame", default="insta360")
    p.add_argument("--calib_dir", default=str(paths.CALIB), help="rig calibration (insta360_intrinsics.json)")
    p.add_argument("--nav_status_topic", default="/scene_memory/nav_status")
    p.add_argument("--no_fast", action="store_true", help="always use the GLM agent (no rule/SigLIP fast path)")
    p.add_argument("--no_verify", action="store_true", help="skip the vision check of the chosen target")
    p.add_argument("--no_stairs", action="store_true", help="never route over stairs")
    p.add_argument("--no_elevator", action="store_true", help="never route via elevators")
    p.add_argument("--odom_max_age_s", type=float, default=2.0, help="older odometry falls back to the map's last pose")
    p.add_argument("--model", default=os.environ.get("GLM_MODEL", gr.DEFAULT_MODEL))
    gr.add_vision_args(p)
    p.add_argument("--base_url", default=os.environ.get("GLM_BASE_URL", gr.DEFAULT_BASE_URL))
    p.add_argument("--api_key_env", default="ZHIPUAI_API_KEY")
    p.add_argument("--siglip", default=str(si.DEFAULT_SIGLIP))
    p.add_argument("--device", default=None)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--history_turns", type=int, default=4, help="previous Q/A pairs kept for follow-up questions")
    return p.parse_args(argv)


class SceneCache:
    """Newest saved scene + its semantic index, reloaded only when the files change."""

    FILES = ("objects_4d.json", "topo_graph.json")

    def __init__(self, scene_dir: str, encoder_factory):
        self.scene_arg = Path(scene_dir)
        self._encoder_factory = encoder_factory
        self._encoder = None
        self._sig = None
        self.scene_dir = None
        self.objects: list[dict] = []
        self.graph: dict | None = None
        self.index = None

    def get(self):
        scene_dir = self.scene_arg.resolve()
        files = [scene_dir / f for f in self.FILES]
        if not files[0].exists():
            raise FileNotFoundError(f"no {files[0]} yet (is topo_map running?)")
        sig = (scene_dir, *(f.stat().st_mtime_ns if f.exists() else 0 for f in files))
        if sig != self._sig:
            if self._encoder is None:
                self._encoder = self._encoder_factory()
            self.objects = load_scene(scene_dir).get("objects", [])
            self.graph = gr.load_graph(scene_dir, self.objects)
            self.index = si.SemanticIndex(scene_dir, self.objects, self._encoder)
            self.scene_dir, self._sig = scene_dir, sig
        return self.scene_dir, self.objects, self.graph, self.index


def route_legs(plan: dict) -> list[dict]:
    """Consecutive walking segments merged into one leg; each stairs / elevator segment is its own leg."""
    legs: list[dict] = []
    for seg in plan["segments"]:
        if seg["mode"] == "walk" and legs and legs[-1]["mode"] == "walk":
            leg = legs[-1]
            leg["waypoints"] += seg["waypoints"][1:] if leg["waypoints"] else seg["waypoints"]
            leg["instructions"].append(seg["instruction"])
        else:
            legs.append({"mode": seg["mode"], "waypoints": list(seg["waypoints"]),
                         "instructions": [seg["instruction"]],
                         **{k: seg[k] for k in ("from_floor", "to_floor", "building") if k in seg}})
    for i, leg in enumerate(legs):
        leg.update(index=i, count=len(legs))
    return legs


GOAL_SUCCEEDED, GOAL_ABORTED = 4, 6   # action_msgs/GoalStatus


def nav_outcome(status: int, robot_xy, goal_xy, goal_reached_m: float = 0.5, near_m: float = 1.5) -> str:
    """How a p2p goal ended: ``arrived``; ``arrived_near`` (stopped short within ``near_m``: the spot is occupied,
    qi_move_base's stall watchdog accepted it or gave up there); ``blocked`` (aborted farther out); ``other``."""
    if status not in (GOAL_SUCCEEDED, GOAL_ABORTED):
        return "other"
    d = math.dist(robot_xy[:2], goal_xy[:2]) if robot_xy is not None else None
    if status == GOAL_SUCCEEDED:
        return "arrived_near" if d is not None and d > goal_reached_m + 0.2 else "arrived"
    return "arrived_near" if d is not None and d <= near_m else "blocked"


def outcome_reply(event: str, name: str, zh: bool, dist_m: float | None = None, alt_name: str = "") -> str:
    near = (f"（离{name}约 {dist_m:.1f} 米）" if zh else f" ({dist_m:.1f} m from the {name})") if dist_m else ""
    return {
        "arrived_near": f"{name}跟前的位置被占了，我停在了附近{near}。" if zh else
                        f"The spot by the {name} is occupied; I stopped nearby{near}.",
        "retry": f"{name}跟前被挡住了，我换个位置靠近。" if zh else
                 f"The way to the {name} is blocked; trying another spot next to it.",
        "alternative": f"{name}那边过不去，改去另一个{alt_name or name}。" if zh else
                       f"Can't get to that {name}; going to another {alt_name or name} instead.",
        "failed": f"到不了{name}：路被挡住了{near}。" if zh else f"I can't reach the {name}: the way is blocked{near}.",
    }[event]


VIEW_ZH = {"front": "前方", "left": "左边", "rear": "后方", "right": "右边"}


NAV_INTENT = re.compile(r"去|到|找|走|带我|前往|过去|\b(go|find|take|walk|head|navigate|bring|move)\b", re.I)
QUESTION = re.compile(r"在哪|哪里|哪儿|吗|什么|几|多少|where|what|which|how|is there|\?|？", re.I)


def is_nav_instruction(text: str) -> bool:
    return bool(NAV_INTENT.search(text)) and not QUESTION.search(text)


def arrival_reply(verdict: str, name: str, zh: bool, res: dict, moving: bool = False, searching: bool = False,
                  outside: bool = False) -> str:
    """Spoken result of ``arrival_check``; ``moving``: walking toward the sighting; ``searching``: the robot is
    looking around for the target rather than standing where the map put it."""
    view = res.get("view", "")
    where = VIEW_ZH.get(view, view) if zh else view
    d = res.get("distance_m")
    edge = ("（它在地图外，我边走边扩展地图过去）" if zh else " (it is off the map; extending the map as I go)") \
        if outside else ""
    lead = "" if searching else ("到了，" if zh else "Arrived: ")
    if verdict == "verified":
        return f"{lead}{name}就在{where}约 {d:.1f} 米处。" if zh else f"{lead}the {name} is {d:.1f} m away ({where})."
    if verdict == "seen" and moving:
        return f"{where}方向好像有{name}，我往那边走走看。" if zh else \
            f"Something like the {name} is off to the {where}; heading that way."
    if verdict == "seen":
        return f"{lead}{where}看到了{name}。" if zh else f"{lead}I can see the {name} ({where})."
    if verdict == "far" and moving:
        return f"{where}约 {d:.1f} 米外看到了{name}，过去。{edge}" if zh else \
            f"I see the {name} {d:.1f} m away ({where}); going there.{edge}"
    if verdict == "far":
        return f"看到了{name}，但它还在{where}约 {d:.1f} 米外，过不去。" if zh else \
            f"I see the {name}, but it is still {d:.1f} m away ({where}) and I can't get closer."
    if verdict == "not_seen" and searching:
        return f"周围没看到{name}。" if zh else f"I can't see a {name} around here."
    if verdict == "not_seen":
        return f"到了{name}所在的位置，但没看到它，它可能被移走了。" if zh else \
            f"I'm where the {name} was, but I can't see it; it may have been moved."
    return f"到了{name}附近（视觉确认不可用）。" if zh else f"Arrived near the {name} (visual check unavailable)."


def ground_step(free_xyz, target_xy, pos, standoff_m: float, min_move_m: float = 0.5, min_gain_m: float = 0.5):
    """Waypoint on free planning ground (N, 3) toward ``target_xy``: nearest to it (no nearer than
    ``standoff_m`` counts), at least ``min_move_m`` from ``pos`` and ``min_gain_m`` closer than ``pos``, heading
    at the target; None when no such ground is known yet."""
    import numpy as np

    if free_xyz is None or len(free_xyz) == 0:
        return None
    t = np.asarray(target_xy[:2], dtype=np.float64)
    d_t = np.linalg.norm(free_xyz[:, :2] - t, axis=1)
    d_r = np.linalg.norm(free_xyz[:, :2] - np.asarray(pos[:2], dtype=np.float64), axis=1)
    ok = (d_r >= min_move_m) & (d_t <= math.dist(pos[:2], t) - min_gain_m)
    if not ok.any():
        return None
    idx = np.flatnonzero(ok)
    i = idx[np.lexsort((d_r[idx], np.maximum(d_t[idx], standoff_m)))[0]]
    x, y, z = (float(v) for v in free_xyz[i])
    yaw = math.atan2(t[1] - y, t[0] - x)
    return {"position": [x, y, z], "orientation": [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]}


def quat_matrix(x: float, y: float, z: float, w: float) -> list[list[float]]:
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def yaw_of(q) -> float:
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def qinav_waypoints(waypoints: list[dict], name: str, route: bool = False, spacing_m: float = 3.0) -> list[dict]:
    """Leg waypoints → QiNav Waypoint fields. Only the goal by default; with ``route`` also intermediate places
    at least ``spacing_m`` apart (and from the goal), each heading along the route."""
    if not waypoints:
        return []
    keep = [len(waypoints) - 1]
    if route:
        last = None
        picked = []
        for i, w in enumerate(waypoints[:-1]):
            p = w["position"]
            if last is None or math.dist(p[:2], last[:2]) >= spacing_m:
                picked.append(i)
                last = p
        goal = waypoints[-1]["position"]
        keep = [i for i in picked if math.dist(waypoints[i]["position"][:2], goal[:2]) >= spacing_m] + keep
        if keep[0] == 0 and len(keep) > 1:
            keep = keep[1:]  # the robot already stands at the first place
    tag = "".join(c if c.isalnum() else "_" for c in name) or "goal"
    out = []
    for n, i in enumerate(keep):
        p = waypoints[i]["position"]
        if i == len(waypoints) - 1:
            yaw = yaw_of(waypoints[i]["orientation"])
        else:
            q = waypoints[keep[n + 1]]["position"]
            yaw = math.atan2(q[1] - p[1], q[0] - p[0])
        out.append({"id": tag if i == len(waypoints) - 1 else f"{tag}_via{n}",
                    "x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "yaw": float(yaw)})
    return out


def main(argv=None) -> None:
    import rclpy
    from geometry_msgs.msg import Point, PoseStamped
    from nav_msgs.msg import Odometry, Path as PathMsg
    from rclpy.executors import ExternalShutdownException
    from std_msgs.msg import ColorRGBA, Empty, String
    from visualization_msgs.msg import Marker, MarkerArray

    args = _parse_args(argv)
    api_key = gr.agent_key(args)
    rclpy.init()
    node = rclpy.create_node("goal_server")
    log = node.get_logger()
    if not api_key:
        log.warn(f"{args.api_key_env} is not set: only fast-path instructions work (no GLM"
                 + (")" if args.vision_url else ", no vision check)"))

    client = gr.GlmClient(api_key or "", args.base_url)
    vision = gr.vision_from_args(args, api_key)

    cache = SceneCache(args.scene_dir, lambda: si.SiglipEncoder(args.siglip, device=args.device))
    pub_answer = node.create_publisher(String, args.answer_topic, 10)
    pub_goal = node.create_publisher(PoseStamped, args.goal_topic, 10)
    pub_path = node.create_publisher(PathMsg, args.path_topic, 10)
    pub_markers = node.create_publisher(MarkerArray, args.markers_topic, 1)
    pub_leg = node.create_publisher(String, args.leg_topic, 10)
    state = {"odom": None, "busy": False, "history": [], "legs": [], "leg": 0, "frame_id": "map",
             "target_name": "goal"}
    lock = threading.Lock()

    nav2 = None
    if args.nav2 != "none":
        from rclpy.action import ActionClient

        if args.nav2 == "through_poses":
            from nav2_msgs.action import NavigateThroughPoses as NavAction
            nav2 = (ActionClient(node, NavAction, "navigate_through_poses"), NavAction)
        else:
            from nav2_msgs.action import NavigateToPose as NavAction
            nav2 = (ActionClient(node, NavAction, "navigate_to_pose"), NavAction)

    pub_nav_status = node.create_publisher(String, args.nav_status_topic, 10)
    qinav = None
    if args.qinav:
        from qi_nav_msgs.msg import NavigationRequest, NavigationStatus, Waypoint
        from qi_nav_msgs.srv import NavigationExecute

        qinav = {"client": node.create_client(NavigationExecute, args.qinav_service), "status": None, "last": None}

        def on_qinav_status(msg) -> None:
            fields = ("status", "phase", "error_code", "error_message", "localization_ready", "navigation_ready",
                      "current_waypoint_id", "current_waypoint_index", "total_waypoints")
            status = {f: getattr(msg, f) for f in fields}
            qinav["status"] = status
            if status != qinav["last"]:
                qinav["last"] = status
                pub_nav_status.publish(String(data=json.dumps({"source": "qinav", **status}, ensure_ascii=False)))
                log.info(f"QiNav: {status['status']}/{status['phase']} wp {status['current_waypoint_index'] + 1}/"
                         f"{status['total_waypoints']} loc_ready={status['localization_ready']} "
                         f"{status['error_code']} {status['error_message']}".rstrip())

        node.create_subscription(NavigationStatus, args.qinav_status_topic, on_qinav_status, 10)

    def send_qinav(waypoints: list[dict], name: str) -> None:
        if state["frame_id"] != args.qinav_frame:
            msg = (f"route is in '{state['frame_id']}', QiNav needs '{args.qinav_frame}': map with odom:=qinav; "
                   "goal not sent")
            log.error(msg)
            pub_nav_status.publish(String(data=json.dumps({"source": "goal_server", "error": msg})))
            return
        status = qinav["status"]
        if status is not None and not status["localization_ready"]:
            log.warn(f"QiNav reports localization_ready=false (status {status['status']}); sending anyway")
        if not qinav["client"].wait_for_service(timeout_sec=2.0):
            msg = f"{args.qinav_service} not available (QiNav nav_server down?); goal only published on topics"
            log.warn(msg)
            pub_nav_status.publish(String(data=json.dumps({"source": "goal_server", "error": msg})))
            return
        wps = qinav_waypoints(waypoints, name, args.qinav_route, args.qinav_route_spacing)
        req = NavigationExecute.Request()
        req.request = NavigationRequest()
        req.request.request_id = time.time_ns() // 1_000_000
        req.request.api_id = 1001
        req.request.task_id = f"vln_{req.request.request_id}"
        for w in wps:
            wp = Waypoint()
            wp.waypoint_id = w["id"]
            wp.x, wp.y, wp.z, wp.yaw = w["x"], w["y"], w["z"], w["yaw"]
            req.request.waypoints.append(wp)
        future = qinav["client"].call_async(req)
        sent = {"source": "goal_server", "event": "sent", "task_id": req.request.task_id, "waypoints": wps}

        def done(fut) -> None:
            try:
                resp = fut.result().response
                sent.update(accepted=bool(resp.success), message=resp.message)
            except Exception as exc:  # noqa: BLE001
                sent.update(accepted=False, message=str(exc))
            pub_nav_status.publish(String(data=json.dumps(sent, ensure_ascii=False)))
            log.info(f"QiNav 1001 {len(wps)} waypoint(s) -> accepted={sent['accepted']} {sent['message']}")

        future.add_done_callback(done)

    p2p = None
    if args.p2p:
        from rclpy.action import ActionClient
        from dddmr_sys_core.action import PToPMoveBase

        p2p = ActionClient(node, PToPMoveBase, args.p2p_action)

    def send_p2p(pose: PoseStamped, name: str, on_done=None) -> bool:
        if not p2p.wait_for_server(timeout_sec=2.0):
            msg = f"/{args.p2p_action} not available (stvln.sh nav running?); goal only published on topics"
            log.warn(msg)
            pub_nav_status.publish(String(data=json.dumps({"source": "goal_server", "error": msg})))
            return False
        goal = PToPMoveBase.Goal()
        goal.target_pose = pose
        p = pose.pose.position
        sent = {"source": "p2p", "target": name, "goal": [p.x, p.y, p.z], "frame_id": pose.header.frame_id}

        def result(fut) -> None:
            res = fut.result()
            sent.update(event="finished", status=int(res.status))
            pub_nav_status.publish(String(data=json.dumps(sent, ensure_ascii=False)))
            log.info(f"p2p_move_base '{name}' finished, status {res.status}")
            if on_done is not None:
                threading.Thread(target=on_done, args=(int(res.status), [p.x, p.y, p.z]), daemon=True).start()

        def accepted(fut) -> None:
            handle = fut.result()
            sent.update(event="accepted" if handle.accepted else "rejected")
            pub_nav_status.publish(String(data=json.dumps(sent, ensure_ascii=False)))
            log.info(f"p2p_move_base '{name}' ({p.x:.2f}, {p.y:.2f}, {p.z:.2f}) -> {sent['event']}")
            if handle.accepted:
                handle.get_result_async().add_done_callback(result)

        p2p.send_goal_async(goal).add_done_callback(accepted)
        return True

    arrival = None
    if args.arrival_check == "on":
        from collections import deque

        import numpy as np
        import tf2_ros
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.time import Time
        from sensor_msgs.msg import CompressedImage, PointCloud2

        from scene_memory.perception.boxer_lift import load_pinhole_views
        from scene_memory.perception.lidar_depth import cloud_xyz_fast

        wanted = [v.strip() for v in args.arrival_views.split(",") if v.strip()]
        try:
            specs = {v["name"]: v for v in load_pinhole_views(args.calib_dir) if v["name"] in wanted}
        except (OSError, KeyError, ValueError) as exc:
            log.warn(f"arrival check off: no pinhole views in {args.calib_dir} ({exc})")
            specs = {}
        if specs:
            arrival = {"specs": specs, "frames": {}, "clouds": deque(), "tf": tf2_ros.Buffer()}
            arrival["tf_listener"] = tf2_ros.TransformListener(arrival["tf"], node)

            def on_view(name):
                def cb(msg) -> None:
                    arrival["frames"][name] = (time.monotonic(), bytes(msg.data))
                return cb

            for name in specs:
                node.create_subscription(CompressedImage, args.arrival_image_topic.format(view=name), on_view(name),
                                         qos_profile_sensor_data)

            def on_cloud(msg) -> None:
                now, clouds = time.monotonic(), arrival["clouds"]
                clouds.append((now, cloud_xyz_fast(msg)))
                while clouds and now - clouds[0][0] > 1.0:
                    clouds.popleft()

            node.create_subscription(PointCloud2, args.arrival_cloud_topic, on_cloud, qos_profile_sensor_data)
            log.info(f"arrival check: views {sorted(specs)}, VLM {args.arrival_model} @ {args.arrival_url}")

    ground = {"xyz": None, "free_mask": None, "free": None, "t": 0.0, "pub": None, "on": None}
    if arrival is not None and (args.extend_map == "on" or args.explore == "on"):
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import PointCloud2 as Cloud
        from std_msgs.msg import Bool

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        if args.extend_map == "on":
            ground["pub"] = node.create_publisher(Bool, args.extend_topic, latched)

        def on_ground(msg) -> None:
            off = {f.name: f.offset for f in msg.fields}
            if "intensity" not in off:
                return
            raw = np.frombuffer(bytes(msg.data), np.uint8).reshape(-1, msg.point_step)
            col = lambda k: raw[:, off[k]:off[k] + 4].copy().view(np.float32)[:, 0]  # noqa: E731
            xyz = np.stack([col("x"), col("y"), col("z")], axis=1).astype(np.float64)
            free = col("intensity") > 1.0
            ground.update(xyz=xyz, free_mask=free, free=xyz[free], t=time.monotonic())

        node.create_subscription(Cloud, args.ground_topic, on_ground, latched)

    def set_extend(on: bool) -> None:
        if ground["pub"] is not None and ground["on"] != on:
            ground["on"] = on
            ground["pub"].publish(Bool(data=on))
            log.info(f"map extension {'on' if on else 'off'}")

    def check_arrival() -> dict:
        now = time.monotonic()
        frames = {n: jpg for n, (t, jpg) in arrival["frames"].items() if now - t < 2.0}
        clouds = [xyz for t, xyz in arrival["clouds"] if now - t < 1.5]
        cloud = np.concatenate(clouds) if clouds else None
        T = None
        try:
            tr = arrival["tf"].lookup_transform(args.base_frame, args.camera_frame, Time()).transform
            q, t = tr.rotation, tr.translation
            T = np.eye(4)
            T[:3, :3] = quat_matrix(q.x, q.y, q.z, q.w)
            T[:3, 3] = (t.x, t.y, t.z)
        except Exception as exc:  # noqa: BLE001 — without TF the check is visual only
            log.warn(f"arrival check: no TF {args.base_frame} -> {args.camera_frame} ({exc})")
        obj = state.get("target_obj") or {}
        return ac.check_arrival(frames, arrival["specs"], cloud, T, str(obj.get("name") or state["target_name"]),
                                state.get("instruction", ""), args.arrival_reach_m, args.arrival_url,
                                args.arrival_model, with_promise=args.explore == "on")

    def toward(target_xy, pos, standoff_m: float):
        """Waypoint facing ``target_xy``: an approach pose on the scene's free space, else (target off the map)
        the reachable cell nearest it, else (no grid) straight at it stopping ``standoff_m`` short.
        ``(waypoint, outside)``; ``(None, _)`` when the robot is already as close as it can get."""
        grid = grid_nav.scene_grid(state["scene_dir"]) if state.get("scene_dir") else None
        tx, ty = target_xy
        if grid is not None:
            pseudo = {"center": [tx, ty, pos[2]], "size": [0.3, 0.3, 0.3]}
            if grid_nav.footprint_dist(pos, pseudo) <= standoff_m:
                return None, False
            ap = grid_nav.approach_pose(grid, pseudo, pos, standoff=(0.4, standoff_m))
            if ap is not None:
                return grid_nav.approach_waypoint(grid, ap), False
            if ground["pub"] is not None:  # off the scene: step over the (growing) planning ground instead
                return ground_step(ground["free"], target_xy, pos, standoff_m), True
            ap = grid_nav.edge_pose(grid, target_xy, pos)
            return (grid_nav.approach_waypoint(grid, ap), True) if ap is not None else (None, True)
        heading = math.atan2(ty - pos[1], tx - pos[0])
        back = min(standoff_m, math.hypot(tx - pos[0], ty - pos[1]))
        return {"position": [tx - back * math.cos(heading), ty - back * math.sin(heading), pos[2]],
                "orientation": [0.0, 0.0, math.sin(heading / 2), math.cos(heading / 2)]}, False

    def follow_sighting(verdict: str, res: dict) -> bool:
        """Drive toward a ``far`` (lidar-ranged) or ``seen`` (bearing only) sighting; True when a goal went out."""
        odom = state["odom"]
        if (verdict not in ("far", "seen") or p2p is None
                or state.get("moves", 0) >= args.search_moves + args.outside_moves
                or not odom or time.monotonic() - odom[0] > args.odom_max_age_s):
            return False
        pos, yaw = odom[1]["position"], yaw_of(odom[1]["orientation"])
        if verdict == "far":
            cx, cy = res["center"][0], res["center"][1]
            target = (pos[0] + math.cos(yaw) * cx - math.sin(yaw) * cy, pos[1] + math.sin(yaw) * cx + math.cos(yaw) * cy)
            standoff = 0.6 * args.arrival_reach_m
        elif "bearing_rad" in res:
            heading = yaw + res["bearing_rad"]
            target = (pos[0] + args.search_step_m * math.cos(heading), pos[1] + args.search_step_m * math.sin(heading))
            standoff = 0.5
        else:
            return False
        wp, outside = toward(target, pos, standoff)
        if outside and ground["pub"] is not None:
            set_extend(True)
            if wp is None:  # nothing known leads closer yet: let map_extender add what the lidar sees here
                since = time.monotonic()
                while time.monotonic() - since < args.extend_wait_s and wp is None:
                    time.sleep(0.5)
                    if ground["t"] > since:
                        wp, _ = toward(target, pos, standoff)
        res["outside_map"] = outside
        if wp is None:
            return False
        budget = args.outside_moves if outside else args.search_moves
        key = "outside_moves" if outside else "near_moves"
        if state.get(key, 0) >= budget:
            return False
        state[key] = state.get(key, 0) + 1
        state["moves"] = state.get("moves", 0) + 1
        res["goal"] = [round(float(v), 2) for v in wp["position"][:2]]
        return send_p2p(pose_msg(state["frame_id"], node.get_clock().now().to_msg(), wp), state["target_name"],
                        on_goal_done)

    memory = fe.FrontierMemory(default_score=args.explore_default_score)

    def explore_step(res: dict) -> bool | None:
        """Rate the frontiers in view with this look's per-view promise, walk to the cheapest one (WildOS
        cost). True: a goal went out (``exploring``); False: done (``not_found`` reported: out of steps or
        frontiers); None: cannot explore here (no planning ground / odometry / p2p), nothing reported."""
        name, zh = state["target_name"], state["zh"]
        odom = state["odom"]
        if ground["xyz"] is None or p2p is None or not odom or time.monotonic() - odom[0] > args.odom_max_age_s:
            return None
        state["exploring"] = True
        if state.get("explore_steps", 0) >= args.explore_steps:
            report("not_found", f"找了 {state['explore_steps']} 个地方，还是没找到{name}。" if zh else
                   f"Searched {state['explore_steps']} places and still no {name}.", explored=state["explore_steps"])
            return False
        pos, yaw = odom[1]["position"], yaw_of(odom[1]["orientation"])
        grid = fe.GroundGrid(ground["xyz"], ground["free_mask"])
        frontiers = fe.find_frontiers(grid)
        ratings, bearings = res.get("promise") or {}, res.get("view_bearing") or {}
        for f in frontiers:
            d = math.dist(pos[:2], f["xy"])
            if d <= args.explore_rate_m:
                s = fe.view_score(math.atan2(f["xy"][1] - pos[1], f["xy"][0] - pos[0]) - yaw, bearings, ratings)
                if s is not None:
                    memory.rate(f["xy"], s, d)
        pick = fe.pick_frontier(frontiers, grid.distance_from(pos), memory, args.explore_score_factor)
        if pick is None:
            report("not_found", f"能走到的地方都找过了，没找到{name}。" if zh else
                   f"I've covered everywhere I can reach; no {name}.", frontiers=len(frontiers))
            return False
        memory.remove(pick["xy"])
        b = pick["bearing"]
        wp = {"position": [pick["xy"][0], pick["xy"][1], pick["z"]],
              "orientation": [0.0, 0.0, math.sin(b / 2), math.cos(b / 2)]}
        if not send_p2p(pose_msg(state["frame_id"], node.get_clock().now().to_msg(), wp), name, on_goal_done):
            return None
        set_extend(True)
        state["explore_steps"] = state.get("explore_steps", 0) + 1
        rel = (math.atan2(pick["xy"][1] - pos[1], pick["xy"][0] - pos[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
        side = min(("front", 0.0), ("left", math.pi / 2), ("rear", math.pi), ("right", -math.pi / 2),
                   key=lambda kv: abs((rel - kv[1] + math.pi) % (2 * math.pi) - math.pi))[0]
        where = VIEW_ZH[side] if zh else side
        report("exploring", f"周围没看到{name}，去{where}约 {pick['path_m']:.0f} 米处找找（第 {state['explore_steps']} 处）。"
               if zh else f"No {name} in sight; checking {pick['path_m']:.0f} m to the {where} "
               f"(place {state['explore_steps']}).", frontier=pick["xy"], score=pick["score"], path_m=pick["path_m"],
               frontiers=len(frontiers), promise=ratings)
        return True

    def look_and_act(arrived: bool = True, report_miss: bool = True,
                     explore: bool = False) -> tuple[str, bool, dict]:
        """Arrival check here + follow a sighting, else (``explore``) the next frontier; reports the result (a
        miss only with ``report_miss``). ``arrived``: at the end of a goal (waits for a still cloud first).
        ``(verdict, moving, result)``."""
        if arrived:
            time.sleep(args.arrival_settle_s)
        t0 = time.monotonic()
        res = check_arrival()
        verdict = res.pop("verdict")
        res["check_s"] = round(time.monotonic() - t0, 1)
        moving = follow_sighting(verdict, res)
        if not moving and explore and verdict == "not_seen" and args.explore == "on":
            went = explore_step(res)
            if went is not None:
                if not went:
                    set_extend(False)
                return verdict, went, res
        if not moving:
            set_extend(False)
        searching = state.get("searching", False) or state.get("exploring", False)
        if not (report_miss or moving or verdict == "verified"):
            log.info(f"look for {state['target_name']}: {verdict} {res}")
            return verdict, moving, res
        event = {"error": "unverified"}.get(verdict, verdict)
        if searching and verdict == "not_seen":
            event = "not_found"
        report(event, arrival_reply(verdict, state["target_name"], state["zh"], res, moving,
                                    searching or not arrived, res.get("outside_map", False)), moving=moving, **res)
        return verdict, moving, res

    def search(phrase: str, why: str) -> None:
        """Look around for a target the semantic map does not have and go to it."""
        state.update(target_name=phrase, target_key=None, target_obj=None, legs=[], leg=0, alts=[], alt_used=0,
                     tried=[], retries=0, moves=0, near_moves=0, outside_moves=0, searching=True, exploring=False,
                     explore_steps=0)
        memory.reset()
        set_extend(False)
        zh = state["zh"]
        report("searching", f"语义地图里没有{phrase}，我在周围找找。" if zh else
               f"The {phrase} is not in my map; looking around for it.", reason=why)
        look_and_act(arrived=False, explore=True)

    def on_odom(msg) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        state["odom"] = (time.monotonic(), {"position": [p.x, p.y, p.z], "orientation": [q.x, q.y, q.z, q.w]})

    def pose_msg(frame_id: str, stamp, wp: dict) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id, ps.header.stamp = frame_id, stamp
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in wp["position"])
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = (float(v) for v in wp["orientation"])
        return ps

    def markers(plan: dict, center, stamp) -> MarkerArray:
        out = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        out.markers.append(clear)

        def mk(mid, mtype, rgba, scale):
            m = Marker()
            m.header.frame_id, m.header.stamp = plan["frame_id"], stamp
            m.ns, m.id, m.type, m.action = "goal", mid, mtype, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = scale
            m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
            return m

        line = mk(0, Marker.LINE_STRIP, (0.1, 0.9, 0.2, 0.9), (0.06, 0.06, 0.06))
        line.points = [Point(x=w["position"][0], y=w["position"][1], z=w["position"][2]) for w in plan["waypoints"]]
        target = mk(1, Marker.SPHERE, (1.0, 0.1, 0.1, 0.9), (0.45, 0.45, 0.45))
        target.pose.position = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
        arrow = mk(2, Marker.ARROW, (0.1, 0.4, 1.0, 1.0), (0.8, 0.12, 0.12))
        arrow.pose = pose_msg(plan["frame_id"], stamp, plan["goal_pose"]).pose
        out.markers += [line, target, arrow]
        return out

    def send_nav2(poses: list[PoseStamped]) -> None:
        action_client, NavAction = nav2
        if not action_client.wait_for_server(timeout_sec=2.0):
            log.warn("Nav2 action server not available; goal only published on topics")
            return
        goal = NavAction.Goal()
        if args.nav2 == "through_poses":
            goal.poses = poses
        else:
            goal.pose = poses[-1]
        action_client.send_goal_async(goal)
        log.info(f"sent {len(poses)} pose(s) to Nav2 ({args.nav2})")

    def publish_answer(body: dict) -> None:
        pub_answer.publish(String(data=json.dumps(body, ensure_ascii=False)))

    def start_leg(i: int) -> None:
        legs = state["legs"]
        if i >= len(legs):
            return
        state["leg"] = i
        leg = legs[i]
        pub_leg.publish(String(data=json.dumps(leg, ensure_ascii=False)))
        log.info(f"route leg {i + 1}/{len(legs)} [{leg['mode']}]: {' / '.join(leg['instructions'])}")
        if nav2 is not None and leg["mode"] == "walk":
            stamp = node.get_clock().now().to_msg()
            send_nav2([pose_msg(state["frame_id"], stamp, w) for w in leg["waypoints"]])
        last = i == len(legs) - 1
        name = state["target_name"] if last else f"{state['target_name']}_leg{i}"
        if qinav is not None and leg["mode"] == "walk":
            send_qinav(leg["waypoints"], name)
        if p2p is not None and leg["mode"] == "walk":
            send_p2p(pose_msg(state["frame_id"], node.get_clock().now().to_msg(), leg["waypoints"][-1]), name,
                     on_goal_done if last else None)

    def robot_xy():
        odom = state["odom"]
        return odom[1]["position"] if odom and time.monotonic() - odom[0] < args.odom_max_age_s else None

    def report(event: str, reply: str = "", **extra) -> None:
        body = {"source": "goal_server", "event": event, "target": state.get("target_key"),
                "name": state["target_name"], **extra}
        pub_nav_status.publish(String(data=json.dumps({**body, "reply": reply}, ensure_ascii=False)))
        if reply:
            publish_answer({"instruction": state.get("instruction", ""), "reply": reply, **body})
            log.info(f"[{event}] {reply}")

    def retry_approach(pos, goal_xy) -> bool:
        """Next free approach pose around the target, away from the ones found blocked."""
        if state["retries"] >= args.goal_retries or state.get("target_obj") is None:
            return False
        grid = grid_nav.scene_grid(state["scene_dir"])
        if grid is None:
            return False
        state["tried"].append(goal_xy)
        ap = grid_nav.approach_pose(grid, state["target_obj"], pos or goal_xy, avoid=state["tried"])
        if ap is None:
            return False
        state["retries"] += 1
        report("retry", outcome_reply("retry", state["target_name"], state["zh"]), goal=ap["xy"])
        wp = grid_nav.approach_waypoint(grid, ap)
        send_p2p(pose_msg(state["frame_id"], node.get_clock().now().to_msg(), wp), state["target_name"],
                 on_goal_done)
        return True

    def try_alternative() -> bool:
        while state["alts"] and state["alt_used"] < args.alt_targets:
            key = state["alts"].pop(0)
            scene_dir, objects, graph, index = cache.get()
            pos = robot_xy()
            tools = gr.SceneTools(scene_dir, index, objects, graph, None, allow_stairs=not args.no_stairs,
                                  allow_elevator=not args.no_elevator,
                                  robot_pose={"position": pos, "orientation": [0, 0, 0, 1]} if pos else None)
            if key not in tools.by_key:
                continue
            plan = tools.plan_to(key)
            if "error" in plan or plan.get("partial"):
                continue
            state["alt_used"] += 1
            obj = tools.by_key[key]
            report("alternative", outcome_reply("alternative", state["target_name"], state["zh"],
                                                alt_name=obj["name"]), alternative=key)
            start_route(plan, key, obj)
            return True
        return False

    def on_goal_done(status: int, goal_xyz) -> None:
        pos = robot_xy()
        outcome = nav_outcome(status, pos, goal_xyz, args.goal_reached_m, args.near_goal_m)
        obj = state.get("target_obj")
        dist = round(grid_nav.footprint_dist(pos, obj), 1) if pos and obj else None
        name, zh = state["target_name"], state["zh"]
        hunting = state.get("searching") or state.get("exploring")
        if arrival is not None and hunting and outcome in ("arrived", "arrived_near", "blocked"):
            look_and_act(explore=True)  # a frontier or sighting goal ended: look here, then go on
        elif outcome in ("arrived", "arrived_near"):
            if outcome == "arrived":
                report("arrived", distance_m=dist)
            else:
                report("arrived_near", outcome_reply("arrived_near", name, zh, dist), distance_m=dist)
            if arrival is None:
                return
            verdict, moving, res = look_and_act()
            if verdict == "not_seen" and not moving and not try_alternative() and args.explore == "on":
                explore_step(res)  # the map's target is gone and so are the alternatives: explore for it
        elif outcome == "blocked":
            if not (retry_approach(pos, goal_xyz) or try_alternative()):
                report("failed", outcome_reply("failed", name, zh, dist), distance_m=dist)
        else:
            report("ended", status=status)

    def start_route(plan: dict, key, obj: dict | None) -> None:
        if args.frame_id:
            plan["frame_id"] = args.frame_id
        stamp = node.get_clock().now().to_msg()
        poses = [pose_msg(plan["frame_id"], stamp, w) for w in plan["waypoints"]]
        path = PathMsg()
        path.header.frame_id, path.header.stamp, path.poses = plan["frame_id"], stamp, poses
        pub_path.publish(path)
        pub_goal.publish(poses[-1])
        if obj is not None:
            pub_markers.publish(markers(plan, obj["center"], stamp))
        state["legs"], state["frame_id"] = route_legs(plan), plan["frame_id"]
        state.update(target_key=key, target_obj=obj, tried=[], retries=0, moves=0, near_moves=0, outside_moves=0,
                     searching=False, exploring=False)
        set_extend(False)
        start_leg(0)

    def on_next_leg(_msg) -> None:
        if state["leg"] + 1 < len(state["legs"]):
            start_leg(state["leg"] + 1)
        else:
            log.info("route finished (no further legs)")

    def run(instruction: str) -> None:
        t_start = time.monotonic()
        try:
            scene_dir, objects, graph, index = cache.get()
            odom = state["odom"]
            live = odom[1] if odom and time.monotonic() - odom[0] < args.odom_max_age_s else None
            tools = gr.SceneTools(scene_dir, index, objects, graph, vision, robot_pose=live,
                                  allow_stairs=not args.no_stairs, allow_elevator=not args.no_elevator)
            def chat(messages):
                if not api_key:
                    raise RuntimeError(f"needs the GLM agent but {args.api_key_env} is not set")
                return client.chat(args.model, messages, gr.TOOLS)

            result = gr.answer_instruction(instruction, tools, chat,
                                           fast=not args.no_fast, verify=not args.no_verify,
                                           history=state["history"], log=lambda s: log.info(s),
                                           max_steps=args.max_steps)
            state["history"] = (state["history"] + [{"role": "user", "content": instruction},
                                                    {"role": "assistant", "content": result["reply"]}]
                                )[-2 * args.history_turns:]
            result.update(instruction=instruction, scene=str(scene_dir), robot_pose_live=live is not None,
                          latency_s=round(time.monotonic() - t_start, 2))
            publish_answer(result)
            log.info(f"'{instruction}' -> {result['reply']} target={result.get('target')} "
                     f"({result['latency_s']} s, {result.get('steps')} steps)")
            plan = result.get("plan")
            zh = any("\u4e00" <= c <= "\u9fff" for c in instruction)
            parsed = fp.parse(instruction, {})
            phrase = parsed["phrase"] if parsed["phrase"] and not parsed["slow"] else ""
            searchable = arrival is not None and args.search == "on"
            if plan:
                key = result.get("target")
                name = str(result.get("name") or key or "goal")
                if zh and phrase:  # the user's own word for spoken replies, not the English detector class
                    name = phrase
                state.update(target_name=name, instruction=instruction, zh=zh, scene_dir=scene_dir,
                             alts=list(result.get("alternatives") or []), alt_used=0, explore_steps=0,
                             exploring=False)
                memory.reset()
                if searchable and result.get("status") in ("dynamic", "absent"):
                    # its map position is stale: go straight to it when it is in view now
                    state.update(target_key=key, target_obj=tools.by_key.get(key), moves=0, near_moves=0,
                                 outside_moves=0, searching=False)
                    verdict, moving, _ = look_and_act(arrived=False, report_miss=False)
                    if verdict == "verified" or moving:
                        return
                start_route(plan, key, tools.by_key.get(key))
            elif searchable and not result.get("target") and is_nav_instruction(instruction):
                state.update(instruction=instruction, zh=zh, scene_dir=scene_dir)
                search(phrase or instruction, "not in the semantic map")
        except Exception as exc:  # noqa: BLE001 — a failed query must not kill the node
            log.error(f"query failed: {exc}")
            publish_answer({"instruction": instruction, "error": str(exc)})
            if arrival is not None and args.search == "on" and is_nav_instruction(instruction):
                parsed = fp.parse(instruction, {})
                state.update(instruction=instruction, zh=any("\u4e00" <= c <= "\u9fff" for c in instruction),
                             scene_dir=cache.scene_dir)
                search(parsed["phrase"] if parsed["phrase"] and not parsed["slow"] else instruction,
                       f"map query failed: {exc}")
        finally:
            with lock:
                state["busy"] = False

    def on_query(msg) -> None:
        text = msg.data.strip()
        if not text:
            return
        with lock:
            if state["busy"]:
                publish_answer({"instruction": text, "error": "busy"})
                return
            state["busy"] = True
        threading.Thread(target=run, args=(text,), daemon=True).start()

    node.create_subscription(Odometry, args.odom_topic, on_odom, 10)
    node.create_subscription(String, args.query_topic, on_query, 10)
    node.create_subscription(Empty, args.next_leg_topic, on_next_leg, 10)
    log.info(f"goal server: {args.query_topic} -> {args.answer_topic}, {args.goal_topic}, {args.path_topic} "
             f"(scene {args.scene_dir}, nav2={args.nav2}, qinav={args.qinav}, p2p={args.p2p})")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
