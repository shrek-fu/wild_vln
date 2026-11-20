"""Offline VLN episodes: instruction → goal server → QiNav (qinav_sim) → where the robot ended up, scored.

Needs a running ``goal_server_node --qinav --odom_topic /ql/nav/localization`` and ``qinav_sim_node`` on the same
scene (``src/scripts/stvln.sh vln`` starts all three). Per episode the robot is reset to its start pose, the
instruction is published on ``/scene_memory/query``, the answer and the QiNav task result are collected, and

* ``chosen_ok``: the target the system picked is of an expected class;
* ``success``: the robot stopped within ``--success_m`` of the footprint of an expected-class object (ObjectNav
  style: any instance counts);
* ``spl``: success × oracle length / max(driven, oracle), the oracle being the shortest free-space path on the
  same grid to within ``--success_m`` of any expected-class object.

Episodes YAML::

    start: robot                      # robot | place<N> | x,y[,yaw], per episode overridable
    episodes:
      - {id: fridge, instruction: 去冰箱那里, expect: [refrigerator]}

Writes ``report.json``, one PNG per episode and ``overview.png`` into ``--out``; with ``--session`` (the
session the scene was mapped from) also ``<id>_view.jpg``: the recorded panorama nearest to where the robot
stopped, the target's bearing marked.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from scene_memory.navigation.grid_nav import NavGrid, footprint_dist
from scene_memory.navigation.qinav_sim_node import start_pose
from scene_memory.query.query_4d import load_scene


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("scene_dir")
    p.add_argument("--episodes", required=True)
    p.add_argument("--session", default="", help="session dir with bag/ for goal views")
    p.add_argument("--out", default="", help="default <scene_dir>/vln_eval")
    p.add_argument("--success_m", type=float, default=1.5)
    p.add_argument("--answer_timeout", type=float, default=120.0)
    p.add_argument("--nav_timeout", type=float, default=180.0)
    p.add_argument("--robot_radius", type=float, default=0.3)
    p.add_argument("--reach_m", type=float, default=3.0)
    p.add_argument("--only", default="", help="comma-separated episode ids")
    return p.parse_args(argv)


def spl(success: bool, oracle: float, driven: float, slack: float = 0.5) -> float:
    """success × oracle / max(driven, oracle); both within ``slack`` of zero (already there, stayed) counts 1."""
    if not success:
        return 0.0
    if not math.isfinite(oracle):  # goal region off the oracle grid (snapping): no better path known
        oracle = driven
    if max(driven, oracle) < slack:
        return 1.0
    return float(oracle / max(driven, oracle))


def oracle_length(grid: NavGrid, field: np.ndarray, objs: list[dict], success_m: float) -> float:
    """Shortest path length from the field's source to within ``success_m`` of any of ``objs``."""
    best = math.inf
    for o in objs:
        c = np.asarray(o["center"][:2])
        r = success_m + float(max(o.get("size", [0, 0])[:2])) / 2.0
        (r0, c0), (r1, c1) = grid.cell(c - r), grid.cell(c + r)
        sub = field[r0:r1 + 1, c0:c1 + 1]
        if not np.isfinite(sub).any():
            continue
        rr, cc = np.nonzero(np.isfinite(sub))
        for i in np.argsort(sub[rr, cc])[:200]:
            xy = grid.xy(rr[i] + r0, cc[i] + c0)
            if footprint_dist(xy, o) <= success_m:
                best = min(best, float(sub[rr[i], cc[i]]))
                break
    return best


class Client:
    def __init__(self):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from std_msgs.msg import String

        rclpy.init()
        self.rclpy, self.PoseStamped, self.String = rclpy, PoseStamped, String
        self.node = rclpy.create_node("vln_eval")
        self.inbox: dict[str, list] = {"answer": [], "result": [], "status": []}
        self.odom = None
        self.pub_query = self.node.create_publisher(String, "/scene_memory/query", 10)
        self.pub_reset = self.node.create_publisher(PoseStamped, "/qinav_sim/reset", 10)
        self.node.create_subscription(String, "/scene_memory/answer", lambda m: self.inbox["answer"].append(json.loads(m.data)), 10)
        self.node.create_subscription(String, "/qinav_sim/result", lambda m: self.inbox["result"].append(json.loads(m.data)), 10)
        self.node.create_subscription(String, "/scene_memory/nav_status", lambda m: self.inbox["status"].append(json.loads(m.data)), 10)
        self.node.create_subscription(Odometry, "/ql/nav/localization", self._on_odom, 10)

    def _on_odom(self, m) -> None:
        self.odom = (m.pose.pose.position.x, m.pose.pose.position.y)

    def spin_until(self, pred, timeout: float):
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            v = pred()
            if v:
                return v
        return None

    def wait_ready(self, timeout: float = 60.0) -> bool:
        def ready():
            return self.odom is not None and self.node.count_subscribers("/scene_memory/query") > 0
        return bool(self.spin_until(ready, timeout))

    def reset(self, x: float, y: float, yaw: float) -> None:
        m = self.PoseStamped()
        m.pose.position.x, m.pose.position.y = float(x), float(y)
        m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.pub_reset.publish(m)
        self.spin_until(lambda: self.odom is not None and math.hypot(self.odom[0] - x, self.odom[1] - y) < 0.05, 5.0)

    def run(self, instruction: str, answer_timeout: float, nav_timeout: float) -> dict:
        for box in self.inbox.values():
            box.clear()
        t0 = time.monotonic()
        self.pub_query.publish(self.String(data=instruction))
        answer = self.spin_until(lambda: next((a for a in self.inbox["answer"] if a.get("instruction") == instruction), None),
                                 answer_timeout)
        out = {"answer": answer or {"error": "no answer (timeout)"}, "answer_s": round(time.monotonic() - t0, 2)}
        if not answer or not answer.get("plan"):
            return out

        def sent():
            return next((s for s in self.inbox["status"] if s.get("event") == "sent" or s.get("error")), None)
        s = self.spin_until(sent, 10.0)
        out["sent"] = s
        if not s or s.get("error") or not s.get("accepted"):
            return out
        out["result"] = self.spin_until(lambda: next((r for r in self.inbox["result"] if r["task_id"] == s["task_id"]), None),
                                        nav_timeout)
        return out


def goal_views(session: Path, wanted: dict[str, float], out: Path) -> dict[str, Path]:
    """Nearest recorded panorama per episode stamp → ``<id>_view.jpg`` (raw JPEG bytes, marked later)."""
    import rosbag2_py

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(session / "bag"), storage_id=""), rosbag2_py.ConverterOptions("cdr", "cdr"))
    r.set_filter(rosbag2_py.StorageFilter(topics=["/insta360/equirect/image/compressed"]))
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage

    best = {k: (math.inf, None) for k in wanted}
    while r.has_next():
        _, data, t = r.read_next()
        ts = t * 1e-9
        for k, want in wanted.items():
            if abs(ts - want) < best[k][0]:
                best[k] = (abs(ts - want), data)
    paths = {}
    for k, (dt, data) in best.items():
        if data is None or dt > 1.0:
            continue
        p = out / f"{k}_view.jpg"
        p.write_bytes(bytes(deserialize_message(data, CompressedImage).data))
        paths[k] = p
    return paths


def mark_view(path: Path, bearing: float, label: str) -> None:
    """Crop ±70° around the target bearing (rad, left positive, 0 = camera forward) and mark it."""
    import cv2

    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    u = int(((math.pi - bearing) / (2 * math.pi)) * w) % w
    half = int(w * 70 / 360)
    img = np.roll(img, w // 2 - u, axis=1)[int(h * 0.2):int(h * 0.8), w // 2 - half:w // 2 + half].copy()
    cx = half
    cv2.line(img, (cx, 0), (cx, img.shape[0] - 1), (0, 0, 255), 3)
    cv2.putText(img, label, (cx + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    cv2.imwrite(str(path), cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))


def render(grid: NavGrid, ep: dict, objects: list[dict], path_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ext = [grid.origin[1], grid.origin[1] + grid.shape[1] * grid.res, grid.origin[0], grid.origin[0] + grid.shape[0] * grid.res]
    img = np.where(grid.free, 1.0, np.where(grid.reach, 0.75, 0.9))
    ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=1, extent=ext, alpha=0.6)
    ob = grid.obstacles[:: max(len(grid.obstacles) // 20000, 1)]
    ax.scatter(ob[:, 1], ob[:, 0], s=0.3, c="k", alpha=0.4)
    for o in objects:
        if o["name"] in ep["expect"]:
            ax.plot(o["center"][1], o["center"][0], "*", ms=12, mfc="gold", mec="k")
    if ep.get("target_center"):
        ax.plot(ep["target_center"][1], ep["target_center"][0], "*", ms=18, mfc="red" if not ep["chosen_ok"] else "lime", mec="k")
    if ep.get("driven"):
        d = np.array(ep["driven"])
        ax.plot(d[:, 1], d[:, 0], "-", c="tab:blue", lw=2)
    s = ep["start"]
    ax.plot(s[1], s[0], "o", ms=10, mfc="white", mec="k")
    if ep.get("final"):
        f = ep["final"]
        ax.arrow(f[1], f[0], 0.6 * math.sin(f[2]), 0.6 * math.cos(f[2]), width=0.08, color="tab:blue")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("x [m]")
    ax.invert_xaxis()
    ok = "SUCCESS" if ep["success"] else "FAIL"
    ax.set_title(f"{ep['id']}: {ep['instruction']}\n{ok}  chose {ep.get('target_name')} ({ep.get('target')})  "
                 f"d={ep.get('dist_expected_m')} m  SPL={ep['spl']:.2f}", fontsize=9, fontproperties=_cjk_font())
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path_png, dpi=110)
    plt.close(fig)


def _cjk_font():
    from matplotlib import font_manager

    for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "Droid Sans Fallback", "AR PL UMing CN"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return font_manager.FontProperties(family=name)
    return None


def main(argv=None) -> None:
    import yaml

    args = _parse_args(argv)
    scene_dir = Path(args.scene_dir).resolve()
    out = Path(args.out) if args.out else scene_dir / "vln_eval"
    out.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(Path(args.episodes).read_text(encoding="utf-8"))
    graph = json.loads((scene_dir / "topo_graph.json").read_text(encoding="utf-8"))
    objects = [o for o in load_scene(scene_dir)["objects"] if o.get("status", "present") == "present"]
    graph_objs = graph["layers"]["objects"]
    by_key = {o["key"]: o for o in (graph_objs.values() if isinstance(graph_objs, dict) else graph_objs)}
    by_key.update({o["key"]: {**by_key.get(o["key"], {}), **o} for o in objects})
    grid = NavGrid.from_scene(scene_dir, robot_radius=args.robot_radius, reach_m=args.reach_m)
    only = set(filter(None, args.only.split(",")))
    episodes = [e for e in spec["episodes"] if not only or e["id"] in only]

    cli = Client()
    if not cli.wait_ready():
        raise SystemExit("goal server / qinav_sim not up (no /ql/nav/localization or no /scene_memory/query subscriber)")
    fields: dict[tuple, np.ndarray] = {}
    results = []
    for e in episodes:
        sx, sy, syaw = start_pose(str(e.get("start", spec.get("start", "robot"))), graph)
        cli.reset(sx, sy, syaw)
        t0 = time.monotonic()
        r = cli.run(e["instruction"], args.answer_timeout, args.nav_timeout)
        ans, res = r["answer"], r.get("result")
        expect = [str(c) for c in e["expect"]]
        tgt = by_key.get(ans.get("target") or "")
        final = res["final"] if res else [sx, sy, syaw]
        cands = [o for o in objects if o["name"] in expect and not o.get("dynamic")]
        d_exp = min((footprint_dist(final, o) for o in cands), default=math.inf)
        if (sx, sy) not in fields:
            fields[(sx, sy)] = grid.distance_field((sx, sy))
        l_opt = oracle_length(grid, fields[(sx, sy)], cands, args.success_m)
        driven = res["driven_length_m"] if res else 0.0
        success = bool(res and res["status"] == "completed" and d_exp <= args.success_m)
        ep = {"id": e["id"], "instruction": e["instruction"], "expect": expect, "start": [sx, sy, syaw],
              "path": ans.get("path"), "fast_reason": ans.get("fast_reason"), "reply": ans.get("reply"),
              "error": ans.get("error") or (r.get("sent") or {}).get("error") or (res or {}).get("error") or None,
              "target": ans.get("target"), "target_name": (tgt or {}).get("name") or ans.get("name"),
              "target_center": (tgt or {}).get("center") or ans.get("center"),
              "chosen_ok": bool(tgt and tgt.get("name") in expect),
              "nav_status": (res or {}).get("status"), "final": final, "driven_length_m": driven,
              "oracle_length_m": None if math.isinf(l_opt) else round(l_opt, 2),
              "dist_expected_m": None if math.isinf(d_exp) else round(d_exp, 2),
              "dist_chosen_m": round(footprint_dist(final, tgt), 2) if tgt else None,
              "min_clearance_m": (res or {}).get("min_clearance_m"), "success": success,
              "spl": spl(success, l_opt, driven), "approach": (ans.get("plan") or {}).get("approach"),
              "answer_s": r["answer_s"], "episode_s": round(time.monotonic() - t0, 1),
              "driven": (res or {}).get("path")}
        results.append(ep)
        print(f"[vln] {ep['id']:12s} {'OK ' if success else 'FAIL'} chose={ep['target_name']}/{ep['target']} "
              f"path={ep['path']} nav={ep['nav_status']} d={ep['dist_expected_m']} m driven={driven} m "
              f"oracle={ep['oracle_length_m']} SPL={ep['spl']:.2f} ({ep['answer_s']} s answer)"
              + (f" error={ep['error']}" if ep["error"] else ""), flush=True)

    if args.session:
        wanted = {}
        agents = graph["layers"]["agents"]
        pos = np.array([a["position"][:2] for a in agents])
        for ep in results:
            if ep["nav_status"] == "completed":
                ep["view_agent"] = int(np.argmin(np.linalg.norm(pos - np.asarray(ep["final"][:2]), axis=1)))
                wanted[ep["id"]] = agents[ep["view_agent"]]["stamp"]
        views = goal_views(Path(args.session), wanted, out)
        for ep in results:
            if ep["id"] in views and ep.get("target_center"):
                a = agents[ep["view_agent"]]
                x, y, z, w = a["orientation"]
                yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                c = ep["target_center"]
                bearing = math.atan2(c[1] - a["position"][1], c[0] - a["position"][0]) - yaw
                mark_view(views[ep["id"]], (bearing + math.pi) % (2 * math.pi) - math.pi, str(ep["target_name"]))
                ep["view"] = str(views[ep["id"]])
                ep["view_offset_m"] = round(float(np.linalg.norm(np.asarray(a["position"][:2]) - np.asarray(ep["final"][:2]))), 2)

    for ep in results:
        render(grid, ep, objects, out / f"{ep['id']}.png")
    n = max(len(results), 1)
    summary = {"scene": str(scene_dir), "episodes": len(results), "success_m": args.success_m,
               "SR": round(sum(e["success"] for e in results) / n, 3),
               "SPL": round(sum(e["spl"] for e in results) / n, 3),
               "chosen_ok": round(sum(e["chosen_ok"] for e in results) / n, 3),
               "fast_path": sum(e["path"] == "fast" for e in results),
               "mean_answer_s": round(sum(e["answer_s"] for e in results) / n, 2),
               "sensor_height_m": round(grid.sensor_height, 2)}
    (out / "report.json").write_text(json.dumps({"summary": summary, "episodes": results}, ensure_ascii=False, indent=1))
    print(f"[vln] SR {summary['SR']}  SPL {summary['SPL']}  target class ok {summary['chosen_ok']}  "
          f"({summary['fast_path']}/{len(results)} fast path, {summary['mean_answer_s']} s/answer) -> {out}")
    cli.node.destroy_node()
    cli.rclpy.try_shutdown()


if __name__ == "__main__":
    main()
