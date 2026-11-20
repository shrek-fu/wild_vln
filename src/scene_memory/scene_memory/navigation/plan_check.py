"""Offline check of goal routes on an exported scene, before they are handed to a navigation stack.

For every target (object names / keys, every object, or prompts through the goal server's fast path) the
route is planned exactly as the goal server does (``PlaceGraph.plan`` from the robot pose at the end of the
recording, or ``--start``) and checked:

* ``hop``: consecutive route places further apart than ``--max_hop_m`` (revisit / loop edges, odometry
  gaps): the local planner has to bridge that stretch on its own.
* ``blocked`` / ``tight``: with ``map_cloud.npy`` in the scene, obstacles (height band above the floor under
  the route) closer than ``--blocked_m`` / ``--robot_radius_m`` to the straight line between consecutive
  route places (the robot drove there, so this flags shortcuts the place graph takes: revisit links, corners).
  The accumulated cloud keeps people and other movers, so points within ``--carve_m`` of the driven
  trajectory (``agents`` layer) are dropped first: the robot's own body proved that space free.
* ``goal_far`` / ``unseen`` / ``partial``: goal place far from the target, target never observed from the
  goal place, or no candidate reachable.
* ``detour``: route length / straight-line distance above ``--max_detour``.

Writes ``report.json``, one top-view PNG per route and ``overview.png`` into ``--out``; ``--qinav`` also lists
the QiNav ``Waypoint`` fields the goal server would send.

  python -m scene_memory.navigation.plan_check data/scene4d/latest --targets tv,door
  python -m scene_memory.navigation.plan_check <scene> --all --start place12 --out /tmp/plans
  python -m scene_memory.navigation.plan_check <scene> --query 去电视那里 --query 找门 --qinav
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scene_memory.navigation.planner import PlaceGraph
from scene_memory.query.query_4d import load_scene


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("scene_dir")
    p.add_argument("--targets", default="", help="comma-separated object names or keys (nearest instance per name)")
    p.add_argument("--all", action="store_true", help="every present object")
    p.add_argument("--query", action="append", default=[], help="prompt through the fast path (SigLIP index)")
    p.add_argument("--start", default="robot", help="robot (end of recording) | place<N> | x,y[,z]")
    p.add_argument("--out", default="", help="default <scene_dir>/plan_check")
    p.add_argument("--max_hop_m", type=float, default=2.5)
    p.add_argument("--max_goal_m", type=float, default=3.0)
    p.add_argument("--max_detour", type=float, default=3.0)
    p.add_argument("--robot_radius_m", type=float, default=0.3, help="hop clearance below this: tight")
    p.add_argument("--blocked_m", type=float, default=0.1, help="hop clearance below this: blocked")
    p.add_argument("--sensor_height_m", type=float, default=0.52, help="pose height above the floor")
    p.add_argument("--obstacle_band_m", default="0.15,1.0", help="obstacle heights above the floor")
    p.add_argument("--min_obstacle_pts", type=int, default=5)
    p.add_argument("--carve_m", type=float, default=None,
                   help="drop cloud points this close to the driven trajectory (default --robot_radius_m, 0: off)")
    p.add_argument("--no_stairs", action="store_true")
    p.add_argument("--no_elevator", action="store_true")
    p.add_argument("--qinav", action="store_true", help="also list the QiNav waypoints the goal server would send")
    p.add_argument("--qinav_route", action="store_true")
    p.add_argument("--no_png", action="store_true")
    p.add_argument("--no_approach", action="store_true",
                   help="end at the goal place, not at an approach pose next to the object (as the goal server does)")
    return p.parse_args(argv)


def obstacle_cloud(scene_dir: Path) -> np.ndarray | None:
    f = scene_dir / "map_cloud.npy"
    return np.load(f).astype(np.float64)[:, :3] if f.exists() else None


def carve_driven(cloud: np.ndarray, agents: list[dict], radius: float, step: float = 0.1) -> np.ndarray:
    """``cloud`` without the points within ``radius`` (xy) of the trajectory polyline through ``agents``."""
    from scipy.spatial import cKDTree

    track = np.array([a["position"] for a in agents], dtype=np.float64)[:, :2]
    if len(track) == 0 or radius <= 0:
        return cloud
    dense = [track[:1]]
    for a, b in zip(track, track[1:]):
        n = max(int(np.linalg.norm(b - a) / step), 1)
        dense.append(a + np.linspace(0.0, 1.0, n + 1)[1:, None] * (b - a))
    dist, _ = cKDTree(np.concatenate(dense)).query(cloud[:, :2], distance_upper_bound=radius)
    return cloud[~np.isfinite(dist)]


def _place_xyz(pg: PlaceGraph, name: str) -> np.ndarray:
    return np.asarray(pg.places[int(name.removeprefix("place"))]["position"], dtype=np.float64)


def hop_clearance(route: list[np.ndarray], cloud: np.ndarray, sensor_h: float, band: tuple[float, float],
                  min_pts: int, reach: float = 1.0, step: float = 0.1) -> list[dict]:
    """Per hop between consecutive route points: the smallest distance from the straight line to an obstacle
    (``min_pts``-th nearest obstacle point in the height band above the floor, which is the pose height minus
    ``sensor_h``), capped at ``reach``, and where along the hop it occurs."""
    from scipy.spatial import cKDTree

    tree = cKDTree(cloud[:, :2])
    out = []
    for i, (a, b) in enumerate(zip(route, route[1:])):
        n = max(int(np.linalg.norm(b[:2] - a[:2]) / step), 1)
        best, at = reach, a
        for t in np.linspace(0.0, 1.0, n + 1):
            p = a + t * (b - a)
            idx = tree.query_ball_point(p[:2], reach)
            if len(idx) < min_pts:
                continue
            pts = cloud[idx]
            dz = pts[:, 2] - (p[2] - sensor_h)
            pts = pts[(dz > band[0]) & (dz < band[1])]
            if len(pts) < min_pts:
                continue
            d = np.sort(np.linalg.norm(pts[:, :2] - p[:2], axis=1))[min_pts - 1]
            if d < best:
                best, at = float(d), p
        out.append({"hop": i, "clearance_m": round(best, 2), "at": [round(float(v), 2) for v in at]})
    return out


def check_plan(plan: dict, pg: PlaceGraph, start_xyz: np.ndarray, cloud: np.ndarray | None, args) -> dict:
    route = [_place_xyz(pg, p) for p in plan["place_path"]]
    hops = [float(np.linalg.norm(b[:2] - a[:2])) for a, b in zip(route, route[1:])]
    straight = float(np.linalg.norm(route[-1][:2] - route[0][:2]))
    issues = []
    long_hops = [{"hop": i, "from": plan["place_path"][i], "to": plan["place_path"][i + 1], "m": round(h, 2)}
                 for i, h in enumerate(hops) if h > args.max_hop_m]
    if long_hops:
        issues.append("hop")
    clearance, blocked, tight = [], [], []
    if cloud is not None and len(route) > 1:
        band = tuple(float(v) for v in args.obstacle_band_m.split(","))
        clearance = hop_clearance(route, cloud, args.sensor_height_m, band, args.min_obstacle_pts)
        blocked = [h for h in clearance if h["clearance_m"] < args.blocked_m]
        tight = [h for h in clearance if args.blocked_m <= h["clearance_m"] < args.robot_radius_m]
        if blocked:
            issues.append("blocked")
        if tight:
            issues.append("tight")
    if plan["partial"]:
        issues.append("partial")
    if plan["goal_to_target_m"] > args.max_goal_m:
        issues.append("goal_far")
    if not plan["target_seen_from_goal"]:
        issues.append("unseen")
    detour = plan["path_length_m"] / straight if straight > 1.0 else 1.0
    if detour > args.max_detour:
        issues.append("detour")
    if clearance:
        clear = [h["clearance_m"] for h in clearance]
    else:  # the mapper's own per-place clearance, when it exports one
        clear = [pg.places[int(p.removeprefix("place"))].get("clearance") for p in plan["place_path"]]
        clear = [c for c in clear if c is not None]
    return {
        "start_snap_m": round(float(np.linalg.norm(route[0][:2] - start_xyz[:2])), 2),
        "places": len(route),
        "waypoints": len(plan["waypoints"]),
        "path_length_m": plan["path_length_m"],
        "straight_m": round(straight, 2),
        "detour": round(detour, 2),
        "max_hop_m": round(max(hops), 2) if hops else 0.0,
        "long_hops": long_hops,
        "blocked": blocked,
        "tight": tight,
        "min_clearance_m": round(min(clear), 2) if clear else None,
        "goal_to_target_m": plan["goal_to_target_m"],
        "target_seen_from_goal": plan["target_seen_from_goal"],
        "issues": issues,
    }


def render(path: Path, pg: PlaceGraph, objects: list[dict], cloud: np.ndarray | None, start_xyz, result: dict,
           sensor_h: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9))
    if cloud is not None:
        floor = float(np.median(pg.pos[:, 2])) - sensor_h if len(pg.pos) else 0.0
        c = cloud[(cloud[:, 2] > floor + 0.15) & (cloud[:, 2] < floor + 1.0)]
        if len(c) > 60000:
            c = c[np.random.default_rng(0).choice(len(c), 60000, replace=False)]
        ax.scatter(c[:, 0], c[:, 1], s=0.3, c="0.75", lw=0)
    for a, b, d in pg.G.edges(data=True):
        pa, pb = pg.places[a]["position"], pg.places[b]["position"]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="tab:orange" if d["kind"] == "revisit" else "0.55",
                lw=0.8, zorder=1)
    ax.scatter(pg.pos[:, 0], pg.pos[:, 1], s=6, c="0.4", zorder=2)
    for o in objects:
        ax.scatter(o["center"][0], o["center"][1], s=10, c="tab:blue", marker="s", zorder=3)
    plan, chk = result["plan"], result["check"]
    route = np.array([_place_xyz(pg, p) for p in plan["place_path"]])
    ax.plot(route[:, 0], route[:, 1], color="tab:green", lw=2.5, zorder=4, label="route")
    for h in chk["long_hops"]:
        seg = route[h["hop"]:h["hop"] + 2]
        ax.plot(seg[:, 0], seg[:, 1], color="tab:red", lw=3.5, zorder=5)
    for b in chk["blocked"]:
        ax.scatter(*b["at"][:2], s=120, facecolors="none", edgecolors="red", lw=2, zorder=6)
    for b in chk["tight"]:
        ax.scatter(*b["at"][:2], s=90, facecolors="none", edgecolors="darkorange", lw=1.5, zorder=6)
    wp = np.array([w["position"] for w in plan["waypoints"]])
    ax.scatter(wp[:, 0], wp[:, 1], s=25, c="tab:green", edgecolors="k", zorder=6)
    g = plan["goal_pose"]
    ax.arrow(g["position"][0], g["position"][1], 0.6 * math.cos(g["yaw"]), 0.6 * math.sin(g["yaw"]),
             width=0.06, color="tab:red", zorder=7)
    t = result["target_center"]
    ax.scatter(t[0], t[1], s=250, marker="*", c="gold", edgecolors="k", zorder=8, label=result["target"])
    ax.scatter(start_xyz[0], start_xyz[1], s=90, marker="o", c="lime", edgecolors="k", zorder=8, label="start")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{result['target']}: {plan['path_length_m']} m, goal {plan['goal_to_target_m']} m from target, "
                 f"issues: {','.join(chk['issues']) or 'none'}  [{plan['frame_id']}]", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _start(spec: str, pg: PlaceGraph, robot_now: dict | None) -> tuple[np.ndarray, int | None]:
    if spec == "robot":
        if robot_now is None:
            raise SystemExit("scene has no trajectory: pass --start place<N> or x,y")
        return np.asarray(robot_now["position"], float), robot_now.get("place")
    if spec.startswith("place"):
        return _place_xyz(pg, spec), int(spec[5:])
    v = [float(x) for x in spec.split(",")]
    return np.array(v + [0.0] * (3 - len(v))), None


def main(argv=None) -> None:
    from scene_memory.nodes.goal_server_node import qinav_waypoints
    from scene_memory.query import glm_retriever as gr

    args = _parse_args(argv)
    scene_dir = Path(args.scene_dir).resolve()
    out = Path(args.out) if args.out else scene_dir / "plan_check"
    out.mkdir(parents=True, exist_ok=True)
    objects = load_scene(scene_dir)["objects"]
    graph = json.loads((scene_dir / "topo_graph.json").read_text(encoding="utf-8"))
    index = None
    if args.query:
        from scene_memory.query import semantic_index as si

        index = si.SemanticIndex(scene_dir, objects, si.SiglipEncoder())
    tools = gr.SceneTools(scene_dir, index, objects, graph, allow_stairs=not args.no_stairs,
                          allow_elevator=not args.no_elevator)
    pg = tools.planner
    if pg is None:
        raise SystemExit("scene has no places")
    start_xyz, start_place = _start(args.start, pg, tools.robot_now)
    present = [o for o in objects if o.get("status", "present") == "present" and not o.get("dynamic")]

    jobs = []  # (label, object)
    if args.all:
        jobs += [(o["key"], o) for o in present]
    for name in filter(None, (s.strip() for s in args.targets.split(","))):
        if name in tools.by_key:
            jobs.append((name, tools.by_key[name]))
            continue
        hits = [o for o in present if o["name"] == name]
        if not hits:
            print(f"[plan_check] no object '{name}' (classes: {', '.join(tools.classes)})")
            continue
        o = min(hits, key=lambda o: np.linalg.norm(np.asarray(o["center"][:2]) - start_xyz[:2]))
        jobs.append((name, o))
    for q in args.query:
        def chat(_messages):
            raise RuntimeError("needs the GLM agent (fast path did not resolve it)")
        try:
            res = gr.answer_instruction(q, tools, chat, fast=True, verify=False, history=[], log=print, max_steps=1)
        except Exception as exc:  # noqa: BLE001
            print(f"[plan_check] '{q}': {exc}")
            continue
        key = res.get("target")
        if key in tools.by_key:
            jobs.append((q, tools.by_key[key]))
        else:
            print(f"[plan_check] '{q}' -> no target ({res.get('reply')})")

    cloud = obstacle_cloud(scene_dir)
    if cloud is None:
        print(f"[plan_check] no {scene_dir / 'map_cloud.npy'}: obstacle check skipped")
    else:
        carve = args.robot_radius_m if args.carve_m is None else args.carve_m
        n = len(cloud)
        cloud = carve_driven(cloud, graph.get("layers", {}).get("agents", []), carve)
        print(f"[plan_check] map cloud {n} points, {n - len(cloud)} within {carve} m of the driven trajectory dropped")
    from scene_memory.navigation import grid_nav

    grid = None if args.no_approach else grid_nav.scene_grid(scene_dir)
    n_comp = len(list(__import__("networkx").connected_components(pg.G)))
    results = []
    for label, obj in jobs:
        plan = pg.plan(start_xyz, obj["center"], tools.seen_places(obj), start_place=start_place)
        if grid is not None:
            plan = grid_nav.with_approach(plan, grid, obj)
        if "error" in plan:
            results.append({"target": label, "key": obj["key"], "error": plan["error"]})
            continue
        res = {"target": label, "key": obj["key"], "name": obj["name"],
               "target_center": [round(float(v), 3) for v in obj["center"]], "plan": plan,
               "check": check_plan(plan, pg, start_xyz, cloud, args)}
        if args.qinav:
            walk = [w for s in plan["segments"] if s["mode"] == "walk" for w in s["waypoints"]]
            res["qinav_waypoints"] = qinav_waypoints(walk, obj["name"], args.qinav_route)
        results.append(res)
        if not args.no_png:
            safe = "".join(c if c.isalnum() else "_" for c in label)
            render(out / f"route_{safe}_{obj['key']}.png", pg, present, cloud, start_xyz, res, args.sensor_height_m)

    summary = {"scene": str(scene_dir), "frame_id": pg.frame_id, "start": [round(float(v), 3) for v in start_xyz],
               "start_place": start_place, "places": len(pg.ids), "edges": pg.G.number_of_edges(),
               "components": n_comp, "map_cloud": cloud is not None, "routes": len(results),
               "with_issues": sum(1 for r in results if r.get("check", {}).get("issues") or "error" in r)}
    (out / "report.json").write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False,
                                                indent=1), encoding="utf-8")
    if not args.no_png:
        _overview(out / "overview.png", pg, present, cloud, start_xyz, results, args.sensor_height_m)

    print(f"scene {scene_dir.name}: {summary['places']} places, {summary['edges']} edges, "
          f"{n_comp} component(s), frame {pg.frame_id}, start {summary['start']} (place{start_place})")
    head = ["target", "key", "len_m", "detour", "max_hop", "min_clr", "goal_m", "seen", "wps", "issues"]
    rows = []
    for r in results:
        if "error" in r:
            rows.append([r["target"], r["key"], "-", "-", "-", "-", "-", "-", "-", "error: " + r["error"]])
            continue
        c = r["check"]
        rows.append([r["target"], r["key"], c["path_length_m"], c["detour"], c["max_hop_m"], c["min_clearance_m"],
                     c["goal_to_target_m"], "y" if c["target_seen_from_goal"] else "n", c["waypoints"],
                     ",".join(c["issues"]) or "ok"])
    w = [max(len(str(x)) for x in col) for col in zip(head, *rows)] if rows else []
    for row in [head, *rows] if rows else []:
        print("  ".join(str(x).ljust(n) for x, n in zip(row, w)))
    if args.qinav:
        for r in results:
            if "qinav_waypoints" in r:
                print(f"qinav {r['target']}: " + "; ".join(
                    f"{q['id']} ({q['x']:.2f}, {q['y']:.2f}, {q['z']:.2f}, yaw {q['yaw']:.2f})"
                    for q in r["qinav_waypoints"]))
    print(f"report: {out / 'report.json'}" + ("" if args.no_png else f", pictures: {out}/*.png"))


def _overview(path: Path, pg: PlaceGraph, objects, cloud, start_xyz, results, sensor_h: float) -> None:
    ok = [r for r in results if "plan" in r]
    if not ok:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    if cloud is not None:
        floor = float(np.median(pg.pos[:, 2])) - sensor_h
        c = cloud[(cloud[:, 2] > floor + 0.15) & (cloud[:, 2] < floor + 1.0)]
        if len(c) > 80000:
            c = c[np.random.default_rng(0).choice(len(c), 80000, replace=False)]
        ax.scatter(c[:, 0], c[:, 1], s=0.3, c="0.75", lw=0)
    for a, b, d in pg.G.edges(data=True):
        pa, pb = pg.places[a]["position"], pg.places[b]["position"]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="tab:orange" if d["kind"] == "revisit" else "0.6", lw=0.8)
    cmap = plt.get_cmap("tab20")
    for i, r in enumerate(ok):
        route = np.array([_place_xyz(pg, p) for p in r["plan"]["place_path"]])
        col = "red" if r["check"]["issues"] else cmap(i % 20)
        ax.plot(route[:, 0], route[:, 1], color=col, lw=1.6, alpha=0.8)
        t = r["target_center"]
        ax.scatter(t[0], t[1], s=40, marker="*", color=col, edgecolors="k", lw=0.4)
        ax.annotate(r["target"], t[:2], fontsize=6)
    ax.scatter(start_xyz[0], start_xyz[1], s=90, c="lime", edgecolors="k", zorder=5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(f"{len(ok)} routes (red = issues; orange edges = revisit links)  [{pg.frame_id}]", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
