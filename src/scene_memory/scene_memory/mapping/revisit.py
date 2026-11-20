"""Revisits (loop closures) on the place graph, so one space walked twice is not split by VIO drift.

Places are dropped along the VIO trajectory and only merged when they land close together. If the robot
walks back through a room but VIO has drifted a few metres (or the return path does not overlap the first
pass), the two passes become separate place chains and therefore separate rooms. This module:

1. describes every equirect frame by SigLIP embeddings of ``tiles`` horizontal sectors; two frames are
   compared at every circular shift, so a pass in the opposite direction still matches;
2. keeps frame pairs that are ``min_dt`` apart in time, look alike (``sim_min``, best match of either), are within
   plausible drift (``gap_m`` + ``drift_rate`` x path walked in between) and are a real shortcut in the
   graph (not the same corridor seen twice in a row);
3. accepts a loop closure only when ``min_support`` neighbouring frame pairs agree on the offset
   (sequence + geometric consistency, against repeated-looking offices);
4. adds ``revisit`` traversable edges between the matched places (rooms then merge), and optionally
   corrects drift with a translation-only pose graph over places (odometry edges keep their shape, loop
   edges pull matched places together); agents and objects follow the correction of the trajectory at
   their stamps. Originals are kept as ``position_odom`` / ``center_odom``.

    python3 -m scene_memory.mapping.revisit data/sessions/<id>/scene4d/latest --plot /tmp/revisit.png [--save]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np

REVISIT = "revisit"


def pano_descriptors(session: Path, encoder=None, tiles: int = 8, cache: bool = True):
    """(stamps, files, descriptors tiles x D per frame) of ``<session>/equirect``; cached next to it."""
    path = session / f"equirect_siglip{tiles}.npz"
    rows = []
    with open(session / "equirect_stamps.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("file") and (session / "equirect" / row["file"]).exists():
                rows.append((float(row["ros_s"]), row["file"]))
    files = [f for _, f in rows]
    if cache and path.exists():
        data = np.load(path, allow_pickle=False)
        if list(data["files"]) == files:
            return data["stamps"], files, data["desc"]
    from PIL import Image

    if encoder is None:
        from scene_memory.query.semantic_index import SiglipEncoder

        encoder = SiglipEncoder()
    crops = []
    for _, name in rows:
        im = Image.open(session / "equirect" / name).convert("RGB")
        w, h = im.size
        crops += [im.crop((k * w // tiles, h // 4, (k + 1) * w // tiles, 3 * h // 4)) for k in range(tiles)]
    desc = encoder.images(crops).reshape(len(rows), tiles, -1).astype(np.float32)
    stamps = np.array([t for t, _ in rows])
    if cache:
        np.savez(path, stamps=stamps, files=np.array(files), desc=desc)
    return stamps, files, desc


def shifted_similarity(desc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best mean tile cosine over circular shifts (heading-invariant), and the shift in tiles."""
    n, k, _ = desc.shape
    sim, shift = np.full((n, n), -1.0), np.zeros((n, n), int)
    for s in range(k):
        c = np.einsum("akd,bkd->ab", desc, np.roll(desc, s, axis=1)) / k
        better = c > sim
        sim[better], shift[better] = c[better], s
    return sim, shift


def _trajectory(graph: dict):
    agents = sorted(graph["layers"]["agents"], key=lambda a: a["stamp"])
    stamps = np.array([a["stamp"] for a in agents], dtype=np.float64)
    pos = np.array([a["position"] for a in agents], dtype=np.float64).reshape(-1, 3)
    walked = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1))])
    return agents, stamps, pos, walked


def find_revisits(graph: dict, stamps: np.ndarray, desc: np.ndarray, min_dt: float = 30.0, sim_min: float = 0.86,
                  gap_m: float = 3.0, drift_rate: float = 0.1, max_gap_m: float = 15.0, shortcut: float = 2.5,
                  min_shortcut_m: float = 8.0, window: int = 4, offset_tol_m: float = 2.5,
                  min_support: int = 3, max_dt: float = 1.0) -> list[dict]:
    """Loop closures ``{"a", "b" (place ids), "pairs": [(frame i, frame j, sim)], "offset": xy}``."""
    import networkx as nx

    agents, t_ag, p_ag, walked = _trajectory(graph)
    if len(t_ag) < 2 or len(stamps) < 2:
        return []
    G = nx.Graph()
    places = {p["id"]: np.asarray(p["position"], dtype=np.float64) for p in graph["layers"]["places"]}
    for e in graph["edges"]:
        if e["relation"] == "traversable":
            a, b = int(e["source"][5:]), int(e["target"][5:])
            G.add_edge(a, b, w=float(np.linalg.norm(places[a] - places[b])))
    idx = np.clip(np.searchsorted(t_ag, stamps), 1, len(t_ag) - 1)
    idx = np.where(np.abs(t_ag[idx] - stamps) < np.abs(t_ag[idx - 1] - stamps), idx, idx - 1)
    ok = np.abs(t_ag[idx] - stamps) <= max_dt
    fplace = [agents[i].get("place") if good else None for i, good in zip(idx, ok)]
    fxy, fwalk = p_ag[idx, :2], walked[idx]

    sim, _ = shifted_similarity(desc)
    n = len(stamps)
    far = np.abs(stamps[:, None] - stamps[None, :]) >= min_dt
    cand = np.where(far & (sim >= sim_min), sim, -1.0)
    best = {(i, int(cand[i].argmax())) for i in range(n)} | {(int(cand[:, j].argmax()), j) for j in range(n)}
    pairs = []
    for i, j in sorted(best):
        if i >= j or cand[i, j] < 0:
            continue
        a, b = fplace[i], fplace[j]
        if a is None or b is None or a == b:
            continue
        d = float(np.linalg.norm(fxy[j] - fxy[i]))
        if d > min(max_gap_m, gap_m + drift_rate * abs(fwalk[j] - fwalk[i])):
            continue
        try:
            route = nx.shortest_path_length(G, a, b, weight="w")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            route = np.inf
        if route < max(min_shortcut_m, shortcut * d):
            continue
        pairs.append((i, j, float(sim[i, j]), fxy[j] - fxy[i]))

    groups: list[list] = []
    for pr in sorted(pairs):
        for g in groups:
            if any(abs(pr[0] - q[0]) <= window and abs(pr[1] - q[1]) <= window
                   and np.linalg.norm(pr[3] - q[3]) <= offset_tol_m for q in g):
                g.append(pr)
                break
        else:
            groups.append([pr])
    loops = []
    for g in groups:
        if len(g) < min_support:
            continue
        for i, j, s, off in g:
            loops.append({"a": int(fplace[i]), "b": int(fplace[j]), "frames": [i, j], "sim": round(s, 3),
                          "offset": [round(float(v), 2) for v in off], "support": len(g)})
    seen, out = set(), []
    for lp in loops:
        key = (min(lp["a"], lp["b"]), max(lp["a"], lp["b"]))
        if key not in seen:
            seen.add(key)
            out.append(lp)
    return out


def correct_positions(graph: dict, loops: list[dict], odom_sigma: float = 0.1, loop_sigma: float = 1.0) -> dict:
    """Translation-only pose graph over places (XY): odometry edges keep their original displacement
    (sigma ``odom_sigma`` per metre, min 0.05 m), loop edges want the matched places on top of each other
    (``loop_sigma``), the first place is fixed. Returns ``{place id: xy offset}``."""
    places = graph["layers"]["places"]
    ids = [p["id"] for p in places]
    col = {pid: k for k, pid in enumerate(ids)}
    o = np.array([p["position"][:2] for p in places], dtype=np.float64)
    rows, rhs, w = [], [], []
    for e in graph["edges"]:
        if e["relation"] != "traversable" or e.get("kind") == REVISIT:
            continue
        a, b = col[int(e["source"][5:])], col[int(e["target"][5:])]
        rows.append((a, b))
        rhs.append(o[b] - o[a])
        w.append(1.0 / max(0.05, odom_sigma * float(np.linalg.norm(o[b] - o[a]))))
    for lp in loops:
        rows.append((col[lp["a"]], col[lp["b"]]))
        rhs.append(np.zeros(2))
        w.append(1.0 / loop_sigma)
    m = len(rows) + 1
    A = np.zeros((m, len(ids)))
    B = np.zeros((m, 2))
    for r, ((a, b), y, wt) in enumerate(zip(rows, rhs, w)):
        A[r, a], A[r, b], B[r] = -wt, wt, wt * y
    A[-1, 0], B[-1] = 1e3, 1e3 * o[0]
    x = np.linalg.lstsq(A, B, rcond=None)[0]
    return {pid: x[k] - o[k] for pid, k in col.items()}


def apply_revisits(graph: dict, objects: list[dict], loops: list[dict], correct: bool = True) -> dict:
    """Add revisit edges (and the drift correction) to ``graph`` and ``objects`` in place; returns stats."""
    by_pair = {(min(int(e["source"][5:]), int(e["target"][5:])), max(int(e["source"][5:]), int(e["target"][5:])))
               for e in graph["edges"] if e["relation"] == "traversable"}
    added = 0
    for lp in loops:
        key = (min(lp["a"], lp["b"]), max(lp["a"], lp["b"]))
        if key not in by_pair:
            graph["edges"].append({"source": f"place{key[0]}", "target": f"place{key[1]}",
                                   "relation": "traversable", "kind": REVISIT})
            by_pair.add(key)
            added += 1
    graph["revisits"] = loops
    stats = {"loops": len(loops), "edges_added": added}
    if not (correct and loops):
        return stats
    offset = correct_positions(graph, loops)
    for p in graph["layers"]["places"]:
        p.setdefault("position_odom", list(p["position"]))
        p["position"] = [p["position_odom"][0] + float(offset[p["id"]][0]),
                         p["position_odom"][1] + float(offset[p["id"]][1]), p["position_odom"][2]]
    agents = sorted(graph["layers"]["agents"], key=lambda a: a["stamp"])
    t = np.array([a["stamp"] for a in agents])
    off = np.array([offset.get(a.get("place"), np.zeros(2)) for a in agents]).reshape(-1, 2)
    for a, d in zip(agents, off):
        a.setdefault("position_odom", list(a["position"]))
        a["position"] = [a["position_odom"][0] + float(d[0]), a["position_odom"][1] + float(d[1]), a["position_odom"][2]]

    def shift(ts):
        return np.stack([np.interp(ts, t, off[:, 0]), np.interp(ts, t, off[:, 1])], axis=-1)

    for obj in objects:
        hist = obj.get("history_odom") or obj.get("history") or []
        obj.setdefault("center_odom", list(obj["center"]))
        if hist:
            obj.setdefault("history_odom", [list(h) for h in hist])
            h = np.asarray(obj["history_odom"], dtype=np.float64)
            d = shift(h[:, 0])
            obj["history"] = [[float(r[0]), float(r[1] + dd[0]), float(r[2] + dd[1]), *map(float, r[3:])]
                              for r, dd in zip(h, d)]
            d_c = d.mean(axis=0)
        else:
            d_c = shift(np.array([0.5 * (obj.get("first_seen", t[0]) + obj.get("last_seen", t[0]))]))[0]
        obj["center"] = [obj["center_odom"][0] + float(d_c[0]), obj["center_odom"][1] + float(d_c[1]),
                         obj["center_odom"][2]]
    graph["loop_corrected"] = True
    moved = np.array([np.linalg.norm(v) for v in offset.values()])
    pos = {p["id"]: p["position"][:2] for p in graph["layers"]["places"]}
    loop_gap = [float(np.linalg.norm(np.subtract(pos[lp["a"]], pos[lp["b"]]))) for lp in loops]
    stats.update(max_shift_m=round(float(moved.max()), 2), mean_shift_m=round(float(moved.mean()), 2),
                 loop_gap_after_m=round(float(np.median(loop_gap)), 2))
    return stats


def duplicate_candidates(objects: list[dict], radius_m: float = 1.0, min_dt: float = 30.0) -> list[tuple]:
    """Same-class objects close together but observed at different times: likely one object seen twice."""
    obs = [o for o in objects if o.get("status", "present") == "present" and not o.get("dynamic")]
    out = []
    for i, a in enumerate(obs):
        for b in obs[i + 1:]:
            if a["name"] != b["name"]:
                continue
            if np.linalg.norm(np.subtract(a["center"][:2], b["center"][:2])) > radius_m:
                continue
            if max(a["first_seen"], b["first_seen"]) - min(a["last_seen"], b["last_seen"]) >= min_dt:
                out.append((a["key"], b["key"], a["name"]))
    return out


def _plot(before: dict, after: dict, loops: list[dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    cmap = plt.get_cmap("tab10")
    for ax, (title, g) in zip(axes, (("before (VIO odom)", before), ("after revisits", after))):
        xy = {p["id"]: np.asarray(p["position"][:2]) for p in g["layers"]["places"]}
        for e in g["edges"]:
            if e["relation"] == "traversable":
                a, b = xy[int(e["source"][5:])], xy[int(e["target"][5:])]
                rev = e.get("kind") == REVISIT
                ax.plot([a[0], b[0]], [a[1], b[1]], color="r" if rev else "0.75", lw=1.6 if rev else 0.8,
                        ls="--" if rev else "-", zorder=1)
        for r in g["layers"]["regions"]:
            pts = np.array([xy[i] for i in r["places"]])
            ax.scatter(pts[:, 0], pts[:, 1], s=26, color=cmap(r["id"] % 10), zorder=2)
            ax.text(*pts.mean(axis=0), f"region{r['id']}", fontsize=9, weight="bold", zorder=3)
        ax.set_title(f"{title}: {len(g['layers']['regions'])} regions")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def main() -> None:
    from scene_memory.mapping.topo_graph import regraph, save_graph

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="scene4d directory with topo_graph.json + objects_4d.json")
    ap.add_argument("--session", default=None, help="session dir with equirect/ (default: two levels up)")
    ap.add_argument("--sim_min", type=float, default=0.86)
    ap.add_argument("--min_support", type=int, default=3)
    ap.add_argument("--drift_rate", type=float, default=0.1, help="allowed drift per metre walked")
    ap.add_argument("--no_correct", action="store_true", help="only add revisit edges, keep VIO positions")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--save", action="store_true",
                    help="write topo_graph.json + objects_4d.json (originals kept as *_odom.json)")
    args = ap.parse_args()

    scene = Path(args.scene).resolve()
    session = Path(args.session) if args.session else scene.parents[1]
    graph = json.loads((scene / "topo_graph.json").read_text(encoding="utf-8"))
    if graph.get("revisits") or any(e.get("kind") == REVISIT for e in graph["edges"]):
        raise SystemExit("topo_graph.json already has revisits; restore topo_graph_odom.json first")
    raw = json.loads((scene / "objects_4d.json").read_text(encoding="utf-8"))
    objects = [o for o in raw["objects"] if o.get("status") != "rejected"]

    t = time.perf_counter()
    stamps, files, desc = pano_descriptors(session)
    t_desc = time.perf_counter() - t
    t = time.perf_counter()
    loops = find_revisits(graph, stamps, desc, sim_min=args.sim_min, min_support=args.min_support,
                          drift_rate=args.drift_rate)
    t_find = time.perf_counter() - t
    before = regraph(graph, objects)
    t = time.perf_counter()
    stats = apply_revisits(graph, raw["objects"], loops, correct=not args.no_correct)
    after = regraph(graph, objects)
    t_apply = time.perf_counter() - t

    print(f"{len(files)} frames; descriptors {t_desc:.1f} s, matching {1000 * t_find:.0f} ms, "
          f"correction + regraph {1000 * t_apply:.0f} ms")
    for lp in loops:
        print(f"  revisit place{lp['a']} <-> place{lp['b']}  frames {lp['frames']}  sim {lp['sim']}  "
              f"VIO offset {lp['offset']} m  support {lp['support']}")
    print(f"stats: {stats}")
    print(f"regions: {len(before['layers']['regions'])} -> {len(after['layers']['regions'])}  "
          f"sizes {sorted((len(r['places']) for r in after['layers']['regions']), reverse=True)}")
    dups = duplicate_candidates(objects)
    if dups:
        print(f"possible duplicate objects (same class, < 1 m, different passes): {len(dups)} e.g. {dups[:6]}")
    if args.plot:
        _plot(before, after, loops, args.plot)
        print(f"wrote {args.plot}")
    if args.save:
        for name in ("topo_graph.json", "objects_4d.json"):
            backup = scene / name.replace(".json", "_odom.json")
            if not backup.exists():
                shutil.copy2(scene / name, backup)
        if loops and not args.no_correct:
            raw["loop_corrected"] = True
        tmp = scene / "objects_4d.json.tmp"
        tmp.write_text(json.dumps(raw), encoding="utf-8")
        tmp.replace(scene / "objects_4d.json")
        save_graph(after, scene / "topo_graph.json")
        print(f"saved {scene}/topo_graph.json + objects_4d.json (originals: *_odom.json); "
              f"re-run dsg_export --summarize for region labels")


if __name__ == "__main__":
    main()
