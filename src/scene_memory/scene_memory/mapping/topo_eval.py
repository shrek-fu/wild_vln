"""Object localisation accuracy of the topo map (or ``hydra_fake``), replayed offline from a recorded bag.

The bag needs the mapper inputs recorded during a live run (``scripts/record_topo_eval.sh``):
``/detector/detections_json``, ``/lio/odom`` and ``/lio/cloud_world``. The mapper is rebuilt with
``remap_bag.remap`` (CPU, deterministic) while every lifted detection is logged, so each final object
keeps the per-frame centres it was fused from.

Without ground truth (always reported):

* ``spread``: median / p90 distance of an object's per-frame centres to its final centre;
* ``revisit_shift``: largest distance between the object's centres from separate passes (time gap
  ``--visit_gap_s``), i.e. odometry drift plus viewpoint bias, the error a goal on a revisit sees;
* ``view_span_deg``: range of bearings it was seen from (one-sided objects keep the front-surface bias);
* ``duplicates``: same-label objects closer than ``--dup_radius``; ``trajectory``: path length and the
  start-end gap (return to the start mark to read odometry drift).

With ``--gt gt_objects.yaml`` (object centres measured on site)::

    frame: room            # room: any floor frame, fitted to the map by SE(2); odom: LIO start frame
    objects:
      - {id: chair_a, label: chair, xy: [2.40, 1.10], z: 0.45}   # z optional (centre height)
      - {id: door_main, label: door, xy: [5.10, -0.30]}

objects are matched per label (Hungarian, ``--match_m``); errors are reported after the fit and
leave-one-out (each object's error with the frame fitted on the others), plus recall and the extra
same-label estimates near a GT object (split objects).

    python3 -m scene_memory.mapping.topo_eval <bag> --gt gt_objects.yaml --out_dir /tmp/topo_eval \
        [--mapper hydra_fake] [topo_map_node options, e.g. --cloud_window_s 1.0]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scene_memory.mapping.remap_bag import remap
from scene_memory.nodes.topo_map_node import TopoMapper, _parse_args as topo_args


class ObservationLog:
    """Hooks a mapper: lifted detections, merges and the trajectory, for evaluation."""

    def __init__(self, core) -> None:
        self.core = core
        self.obs: list[dict] = []
        self.alias: dict[str, str] = {}
        self.traj: list[tuple[float, float, float, float]] = []
        on_det, on_odom, absorb = core.on_detections, core.on_odom, core.objects.on_absorb

        def on_detections(body: dict) -> list[dict]:
            lifted = on_det(body)
            t = body["stamp"]["sec"] + 1e-9 * body["stamp"]["nanosec"]
            T = core.pose_at(t)
            robot = T[:3, 3] if T is not None else np.full(3, np.nan)
            for det in lifted:
                key = det.get("memory_key")
                if not key:
                    continue
                c = np.asarray(det["obb"]["center"], dtype=np.float64)
                self.obs.append({"t": t, "key": key, "name": det["name"], "xyz": c, "source": det["obb"]["source"],
                                 "robot": robot.copy(), "range": float(np.linalg.norm(c - robot)), "view": det.get("view")})
            return lifted

        def on_odom_hook(t: float, T: np.ndarray, quat=None) -> None:
            on_odom(t, T, quat)
            self.traj.append((t, *map(float, T[:3, 3])))

        def on_absorb(kept: str, removed: str) -> None:
            self.alias[removed] = kept
            if absorb is not None:
                absorb(kept, removed)

        core.on_detections = on_detections
        core.on_odom = on_odom_hook
        core.objects.on_absorb = on_absorb

    def resolve(self, key: str) -> str:
        seen = set()
        while key in self.alias and key not in seen:
            seen.add(key)
            key = self.alias[key]
        return key


def corrected_log(core, log: ObservationLog) -> None:
    """hydra_fake: per-frame centres and the trajectory in the final corrected world frame."""
    traj = [(t, *core.world_pose_xyz(k, T)) for t, k, T, _ in core.traj]
    times = np.array([q[0] for q in traj])
    names = {}
    for o in log.obs:
        names.setdefault(log.resolve(o["key"]), o["name"])
    sources = {0: "prior", 1: "tri", 2: "lidar"}
    obs = []
    for key, entries in core.obs.items():
        for k, c, rank, _, t in entries:
            xyz = core.world_point(k, c)
            i = int(np.clip(np.searchsorted(times, t), 0, len(times) - 1)) if len(times) else -1
            robot = np.array(traj[i][1:]) if i >= 0 else np.full(3, np.nan)
            obs.append({"t": t, "key": key, "name": names.get(key, ""), "xyz": xyz, "source": sources.get(rank, "tri"),
                        "robot": robot, "range": float(np.linalg.norm(xyz - robot))})
    log.obs, log.alias, log.traj = obs, {}, traj


def _visits(times: np.ndarray, gap_s: float) -> list[np.ndarray]:
    order = np.argsort(times)
    cuts = np.flatnonzero(np.diff(times[order]) > gap_s) + 1
    return np.split(order, cuts)


def object_stats(objects: list[dict], log: ObservationLog, args: argparse.Namespace) -> list[dict]:
    by_key: dict[str, list[dict]] = defaultdict(list)
    for o in log.obs:
        by_key[log.resolve(o["key"])].append(o)
    rows = []
    for obj in objects:
        obs = by_key.get(obj["key"], [])
        c = np.asarray(obj["center"], dtype=np.float64)
        row = {"key": obj["key"], "label": obj["name"], "center": c.round(3).tolist(), "n_obs": obj["n_obs"],
               "center_source": obj.get("center_source"), "status": obj.get("status"),
               "size": [round(float(v), 2) for v in obj["size"]], "bbox3d": obj.get("bbox3d")}
        if obs:
            xyz = np.stack([o["xyz"] for o in obs])
            t = np.array([o["t"] for o in obs])
            d = np.linalg.norm(xyz - c, axis=1)
            robot = np.stack([o["robot"] for o in obs])
            bearing = np.degrees(np.arctan2(robot[:, 1] - c[1], robot[:, 0] - c[0]))
            ok = np.isfinite(bearing)
            span = 0.0
            if ok.sum() > 1:
                b = np.sort(np.mod(bearing[ok], 360.0))
                span = 360.0 - float(np.max(np.diff(np.r_[b, b[0] + 360.0])))
            visits = [xyz[i] for i in _visits(t, args.visit_gap_s) if len(i) >= args.min_visit_obs]
            vc = [np.median(v, axis=0) for v in visits]
            shift = max((float(np.linalg.norm(a[:2] - b[:2])) for i, a in enumerate(vc) for b in vc[i + 1:]), default=None)
            row.update({
                "frames": len(obs), "sources": dict(Counter(o["source"] for o in obs)),
                "spread_med": float(np.median(d)), "spread_p90": float(np.percentile(d, 90)),
                "range_med": float(np.median([o["range"] for o in obs])), "view_span_deg": span,
                "visits": len(visits), "revisit_shift": shift,
                "visit_centers": [v.round(3).tolist() for v in vc],
            })
        rows.append(row)
    return rows


def duplicates(rows: list[dict], radius: float) -> list[dict]:
    out = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a["label"] == b["label"]:
                d = float(np.linalg.norm(np.subtract(a["center"][:2], b["center"][:2])))
                if d < radius:
                    out.append({"label": a["label"], "keys": [a["key"], b["key"]], "dist": round(d, 3)})
    return out


def trajectory_stats(traj: list[tuple]) -> dict:
    if len(traj) < 2:
        return {"poses": len(traj)}
    p = np.array([q[1:] for q in traj])
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return {"poses": len(traj), "duration_s": traj[-1][0] - traj[0][0], "path_m": float(step.sum()),
            "start_end_gap_m": float(np.linalg.norm(p[-1, :2] - p[0, :2])),
            "start_end_dz_m": float(p[-1, 2] - p[0, 2]), "max_step_m": float(step.max())}


def fit_se2(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 ``T`` with ``dst ≈ T @ src`` (xy, least squares, no scale)."""
    ms, md = src.mean(0), dst.mean(0)
    H = (src - ms).T @ (dst - md)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    T = np.eye(3)
    T[:2, :2], T[:2, 2] = R, md - R @ ms
    return T


def _apply(T: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return xy @ T[:2, :2].T + T[:2, 2]


def match(gt: list[dict], est: list[dict], T: np.ndarray, max_m: float) -> list[tuple[int, int]]:
    from scipy.optimize import linear_sum_assignment

    pairs = []
    for label in {g["label"] for g in gt}:
        gi = [i for i, g in enumerate(gt) if g["label"] == label]
        ei = [j for j, e in enumerate(est) if e["label"] == label]
        if not gi or not ei:
            continue
        g_xy = _apply(T, np.array([gt[i]["xy"] for i in gi], dtype=np.float64))
        e_xy = np.array([est[j]["center"][:2] for j in ei])
        cost = np.linalg.norm(g_xy[:, None] - e_xy[None], axis=2)
        r, c = linear_sum_assignment(np.where(cost < max_m, cost, 1e6))
        pairs += [(gi[a], ei[b]) for a, b in zip(r, c) if cost[a, b] < max_m]
    return pairs


def evaluate_gt(gt_doc: dict, est: list[dict], args: argparse.Namespace) -> dict:
    gt = gt_doc["objects"]
    for g in gt:
        g["xy"] = [float(v) for v in g["xy"]]
    frame = gt_doc.get("frame", "room")
    T = np.eye(3)
    if frame != "odom":
        # Seed with labels unique in the GT, each paired with that label's most observed estimate.
        seeds = []
        for label, n in Counter(g["label"] for g in gt).items():
            cands = [e for e in est if e["label"] == label]
            if n == 1 and cands:
                seeds.append((next(g for g in gt if g["label"] == label), max(cands, key=lambda e: e["n_obs"])))
        if len(seeds) < 2:
            return {"error": "frame=room needs >= 2 labels that occur once in the GT and in the map"}
        T = fit_se2(np.array([g["xy"] for g, _ in seeds]), np.array([e["center"][:2] for _, e in seeds]))
        for _ in range(5):
            pairs = match(gt, est, T, args.match_m)
            if len(pairs) < 2:
                break
            T = fit_se2(np.array([gt[i]["xy"] for i, _ in pairs]), np.array([est[j]["center"][:2] for _, j in pairs]))
    pairs = match(gt, est, T, args.match_m)
    rows = []
    for i, j in pairs:
        g, e = gt[i], est[j]
        g_xy = _apply(T, np.array([g["xy"]]))[0]
        row = {"id": g.get("id", g["label"]), "label": g["label"], "key": e["key"], "gt_xy_map": g_xy.round(3).tolist(),
               "est_xy": e["center"][:2], "err_xy": float(np.linalg.norm(g_xy - e["center"][:2])),
               "center_source": e.get("center_source"), "n_obs": e["n_obs"], "range_med": e.get("range_med"),
               "view_span_deg": e.get("view_span_deg")}
        if "z" in g:
            row["err_z"] = float(e["center"][2] - float(g["z"])) if frame == "odom" else None
        if frame != "odom" and len(pairs) >= 4:
            others = [(a, b) for a, b in pairs if a != i]
            T_loo = fit_se2(np.array([gt[a]["xy"] for a, _ in others]), np.array([est[b]["center"][:2] for _, b in others]))
            row["err_xy_loo"] = float(np.linalg.norm(_apply(T_loo, np.array([g["xy"]]))[0] - e["center"][:2]))
        rows.append(row)
    matched_est = {j for _, j in pairs}
    split = []
    for i, j in pairs:
        g_xy = _apply(T, np.array([gt[i]["xy"]]))[0]
        for k, e in enumerate(est):
            if k not in matched_est and e["label"] == gt[i]["label"] and np.linalg.norm(g_xy - e["center"][:2]) < args.match_m:
                split.append({"id": gt[i].get("id"), "extra_key": e["key"], "dist": float(np.linalg.norm(g_xy - e["center"][:2]))})

    def summary(key: str) -> dict:
        v = np.array([r[key] for r in rows if r.get(key) is not None])
        if not len(v):
            return {}
        return {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                "max": float(v.max()), "within_0.3m": float(np.mean(v <= 0.3)), "within_0.5m": float(np.mean(v <= 0.5)),
                "within_1m": float(np.mean(v <= 1.0))}

    return {"frame": frame, "T_map_gt": T.round(4).tolist(), "gt_objects": len(gt), "matched": len(pairs),
            "recall": len(pairs) / max(len(gt), 1), "missed": [g.get("id", g["label"]) for i, g in enumerate(gt)
                                                              if i not in {a for a, _ in pairs}],
            "err_xy": summary("err_xy"), "err_xy_loo": summary("err_xy_loo"),
            "split_objects": split, "objects": rows}


def plot(path: Path, core, log: ObservationLog, rows: list[dict], gt_eval: dict | None, gt_doc: dict | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    if core.map_cloud is not None:
        pts = core.map_cloud.points()
        if len(pts):
            pts = pts[(pts[:, 2] > -0.3) & (pts[:, 2] < 2.0)]
            ax.scatter(pts[:, 0], pts[:, 1], s=0.2, c="#bbbbbb", lw=0)
    if log.traj:
        tr = np.array([q[1:3] for q in log.traj])
        ax.plot(tr[:, 0], tr[:, 1], "-", c="#555555", lw=1, label="trajectory")
        ax.plot(*tr[0], "go", ms=8, label="start")
        ax.plot(*tr[-1], "rs", ms=8, label="end")
    keyset = {r["key"]: r for r in rows}
    for o in log.obs:
        k = log.resolve(o["key"])
        if k in keyset:
            ax.plot(o["xyz"][0], o["xyz"][1], ".", ms=2, c="#2e86c1" if o["source"] == "lidar" else "#f0ad4e", alpha=0.4)
    for r in rows:
        c = r["center"]
        ax.plot(c[0], c[1], "o", ms=7, mfc="none", mec="#d9534f" if r.get("center_source") != "lidar" else "#1f4e79", mew=2)
        a = r.get("approach")
        if a:
            ax.annotate("", xy=c[:2], xytext=a["xy"], arrowprops={"arrowstyle": "->", "color": "#27ae60", "lw": 1.2})
        ax.annotate(f"{r['label']}", c[:2], fontsize=7, xytext=(4, 4), textcoords="offset points")
    if gt_eval and gt_doc and "T_map_gt" in gt_eval:
        T = np.array(gt_eval["T_map_gt"])
        for g in gt_doc["objects"]:
            p = _apply(T, np.array([g["xy"]], dtype=np.float64))[0]
            ax.plot(p[0], p[1], "kx", ms=9, mew=2)
        for r in gt_eval["objects"]:
            ax.plot([r["gt_xy_map"][0], r["est_xy"][0]], [r["gt_xy_map"][1], r["est_xy"][1]], "k-", lw=1)
            ax.annotate(f"{r['err_xy']:.2f} m", r["gt_xy_map"], fontsize=7, color="k", xytext=(4, -10),
                        textcoords="offset points")
    ax.plot([], [], "o", mfc="none", mec="#1f4e79", mew=2, label="object (lidar centre)")
    ax.plot([], [], "o", mfc="none", mec="#d9534f", mew=2, label="object (tri / prior)")
    ax.plot([], [], ".", c="#2e86c1", label="per-frame centres")
    if gt_doc:
        ax.plot([], [], "kx", mew=2, label="ground truth")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("x odom [m]")
    ax.set_ylabel("y odom [m]")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _hydra_fake():
    import importlib
    import sys

    from scene_memory import paths

    if str(paths.SRC / "hydra_fake") not in sys.path:
        sys.path.insert(0, str(paths.SRC / "hydra_fake"))
    return importlib.import_module("hydra_fake.core")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     add_help=False)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--mapper", choices=["topo", "hydra_fake"], default="topo")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--match_m", type=float, default=2.0, help="m; max GT-estimate distance for a match")
    parser.add_argument("--dup_radius", type=float, default=1.5)
    parser.add_argument("--visit_gap_s", type=float, default=15.0, help="s without sightings that starts a new pass")
    parser.add_argument("--min_visit_obs", type=int, default=3)
    known, rest = parser.parse_known_args(argv)
    hf = None
    if known.mapper == "hydra_fake":
        hf_core = _hydra_fake()
        hf, rest = hf_core.add_args(argparse.ArgumentParser(add_help=False)).parse_known_args(rest)
    defaults = ["--pose_source", "lio", "--odom_topic", "/lio/odom", "--cloud_topic", "/lio/cloud_world",
                "--environment", "indoor", "--no_dsg", "--no_views"]
    args = topo_args(defaults + rest)
    for k, v in {**vars(known), **(vars(hf) if hf else {})}.items():
        setattr(args, k, v)
    out = known.out_dir or Path(f"/tmp/topo_eval_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    core = hf_core.HydraFakeMapper(args) if hf else TopoMapper(args)
    log = ObservationLog(core)
    remap(known.bag, args, core=core)
    raw_traj = trajectory_stats(log.traj)
    if hf:
        corrected_log(core, log)
    core.consolidate()
    graph = core.graph()
    from scene_memory.mapping.topo_graph import save_graph

    save_graph(graph, out / "topo_graph.json")
    # a scene dir for plan_check / the goal server
    core.objects.save(out / "objects_4d.json", extra={"mapper": known.mapper, "frame_id": graph.get("frame_id", "odom")})
    if core.map_cloud is not None:
        np.save(out / "map_cloud.npy", np.asarray(core.map_cloud.points(), dtype=np.float32)[:, :3])
    objects = [o for o in core.objects.to_json()["objects"] if not o["dynamic"]]
    rows = object_stats(objects, log, args)
    approach = {n["key"]: n.get("approach") for n in graph["layers"]["objects"]}
    for r in rows:
        r["approach"] = approach.get(r["key"])
    gt_doc = gt_eval = None
    if known.gt:
        import yaml

        gt_doc = yaml.safe_load(known.gt.read_text())
        gt_eval = evaluate_gt(gt_doc, rows, args)

    def med(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return float(np.median(v)) if v else None

    report = {
        "bag": str(known.bag), "mapper": known.mapper, "replay_s": time.perf_counter() - t0, "stats": dict(core.stats),
        "trajectory": trajectory_stats(log.traj), "trajectory_odom": raw_traj,
        "pose_graph": {**core.kg.stats, "keyframes": len(core.kg.kfs), "events": core.kg.events} if hf else None,
        "summary": {"objects": len(rows), "by_source": dict(Counter(r["center_source"] for r in rows)),
                    "spread_med": med("spread_med"), "spread_p90": med("spread_p90"),
                    "revisit_shift_med": med("revisit_shift"),
                    "revisited_objects": sum(r.get("revisit_shift") is not None for r in rows),
                    "view_span_med": med("view_span_deg"), "duplicates": duplicates(rows, known.dup_radius)},
        "gt": gt_eval, "objects": rows,
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    plot(out / "map.png", core, log, rows, gt_eval, gt_doc)

    s, tr = report["summary"], report["trajectory"]
    print(f"{out}: {report['replay_s']:.1f} s replay | path {tr.get('path_m', 0):.1f} m, "
          f"start-end gap {tr.get('start_end_gap_m', float('nan')):.2f} m")
    print(f"objects {s['objects']} {s['by_source']} | spread med {s['spread_med']} p90 {s['spread_p90']} | "
          f"revisit shift med {s['revisit_shift_med']} ({s['revisited_objects']} objects) | "
          f"duplicates {len(s['duplicates'])}")
    if gt_eval:
        if "error" in gt_eval:
            print("gt:", gt_eval["error"])
        else:
            e, loo = gt_eval["err_xy"], gt_eval["err_xy_loo"]
            print(f"gt: matched {gt_eval['matched']}/{gt_eval['gt_objects']} | err_xy median {e.get('median')} "
                  f"p90 {e.get('p90')} <=0.3m {e.get('within_0.3m')} | loo median {loo.get('median')} | "
                  f"split {len(gt_eval['split_objects'])} missed {gt_eval['missed']}")


if __name__ == "__main__":
    main()
