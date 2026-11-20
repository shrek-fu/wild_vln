"""Per-place clearance from DA360 horizontal profiles, then Hydra-style rooms (``segment_regions``).

Input: ``equirect_profiles.json`` from ``src/depth/da360_depth/scripts/horizontal_profile.py`` (per frame,
nearest obstacle per azimuth in camera heights). Each frame is matched to the trajectory keyframe nearest
in time and that keyframe's place; a place's clearance is the median over its frames of a low percentile
of the profile, skipping the sector behind the camera where the operator / rig is. Without a metric
``--camera_height`` the clearance stays in camera heights, which the filtration does not mind.

    python3 -m scene_memory.mapping.clearance data/sessions/<id>/scene4d/latest \\
        --profiles data/sessions/<id>/equirect_profiles.json --plot /tmp/rooms.png [--save]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np


def frame_clearance(profile: list, ignore_sector: tuple[float, float] | None = (120.0, 240.0),
                    percentile: float = 10.0) -> float | None:
    prof = np.array([np.inf if v is None else float(v) for v in profile])
    az = (np.arange(len(prof)) + 0.5) * 360.0 / len(prof)
    keep = np.isfinite(prof)
    if ignore_sector:
        a, b = ignore_sector
        keep &= ~((az >= a) & (az <= b)) if a <= b else ~((az >= a) | (az <= b))
    if keep.sum() < len(prof) // 8:
        return None
    return float(np.percentile(prof[keep], percentile))


def place_clearances(graph: dict, frames: list[dict], max_dt: float = 1.0, radius_m: float = 1.2,
                     scale: float = 1.0, **kwargs) -> dict[int, float]:
    """Each frame is placed on the trajectory (interpolated at its stamp) and counts for every place within
    ``radius_m`` (frames are ~2 s apart, places ~1 m), weighted by proximity."""
    agents = sorted(graph["layers"]["agents"], key=lambda a: a["stamp"])
    stamps = np.array([a["stamp"] for a in agents], dtype=np.float64)
    traj = np.array([a["position"] for a in agents], dtype=np.float64).reshape(-1, 3)
    places = graph["layers"]["places"]
    ppos = np.array([p["position"] for p in places], dtype=np.float64).reshape(-1, 3)
    acc: dict[int, list[tuple[float, float]]] = {}
    for f in frames:
        if len(stamps) < 2:
            break
        i = int(np.clip(np.searchsorted(stamps, f["stamp"]), 1, len(stamps) - 1))
        if min(abs(stamps[i] - f["stamp"]), abs(stamps[i - 1] - f["stamp"])) > max_dt:
            continue
        w = np.clip((f["stamp"] - stamps[i - 1]) / max(stamps[i] - stamps[i - 1], 1e-9), 0.0, 1.0)
        xyz = (1 - w) * traj[i - 1] + w * traj[i]
        c = frame_clearance(f["profile"], **kwargs)
        if c is None:
            continue
        d = np.linalg.norm(ppos - xyz, axis=1)
        for j in np.flatnonzero(d < radius_m):
            acc.setdefault(int(places[j]["id"]), []).append((c * scale, 1.0 / (0.2 + d[j])))
    out = {}
    for pid, vw in acc.items():
        v, w = np.array(vw).T
        k = np.argsort(v)
        out[pid] = float(v[k][np.searchsorted(np.cumsum(w[k]), 0.5 * w.sum())])
    return out


def region_stats(graph: dict) -> str:
    sizes = sorted((len(r["places"]) for r in graph["layers"]["regions"]), reverse=True)
    return f"{len(sizes)} regions, places per region {sizes}"


def plot(graphs: list[tuple[str, dict]], path: str, clear: dict[int, float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(graphs) + 1, figsize=(6 * (len(graphs) + 1), 7))
    for ax, (title, g) in zip(axes, graphs):
        pl = g["layers"]["places"]
        xy = {p["id"]: np.asarray(p["position"][:2]) for p in pl}
        for e in g["edges"]:
            if e["relation"] == "traversable":
                a, b = xy[int(e["source"][5:])], xy[int(e["target"][5:])]
                ax.plot([a[0], b[0]], [a[1], b[1]], color="0.8", lw=0.8, zorder=1)
        cmap = plt.get_cmap("tab20")
        for r in g["layers"]["regions"]:
            pts = np.array([xy[i] for i in r["places"]])
            ax.scatter(pts[:, 0], pts[:, 1], s=28, color=cmap(r["id"] % 20), zorder=2)
            ax.text(*pts.mean(axis=0), str(r["id"]), fontsize=8, weight="bold", zorder=3)
        for o in g["layers"]["objects"]:
            if "door" in o.get("name", ""):
                ax.scatter(*o["center"][:2], marker="s", s=40, color="k", zorder=3)
        ax.set_title(f"{title}: {len(g['layers']['regions'])} regions")
        ax.set_aspect("equal")
    ax = axes[-1]
    pl = graphs[0][1]["layers"]["places"]
    pts = np.array([p["position"][:2] for p in pl if p["id"] in clear])
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=[clear[p["id"]] for p in pl if p["id"] in clear], cmap="viridis", s=28)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="clearance")
    ax.set_title("per-place clearance (DA360)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> None:
    from scene_memory.mapping.topo_graph import regraph, save_graph
    from scene_memory.query.query_4d import load_scene

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="scene4d directory with topo_graph.json + objects_4d.json")
    ap.add_argument("--profiles", required=True, help="equirect_profiles.json (horizontal_profile.py)")
    ap.add_argument("--camera_height", type=float, default=None, help="metres; else clearance in camera heights")
    ap.add_argument("--ignore_sector", type=float, nargs=2, default=(60.0, 240.0),
                    help="azimuth range (deg, 0 = camera forward, 90 = right) to skip; operator / rig")
    ap.add_argument("--percentile", type=float, default=30.0,
                    help="of the free distances around the place; low = Hydra-like min distance")
    ap.add_argument("--min_room_places", type=int, default=4)
    ap.add_argument("--plot", default=None, help="top-down PNG: radius regions | clearance regions | clearance")
    ap.add_argument("--save", action="store_true", help="write clearances + new regions into topo_graph.json")
    args = ap.parse_args()

    scene = Path(args.scene)
    t = time.perf_counter()
    graph = json.loads((scene / "topo_graph.json").read_text(encoding="utf-8"))
    objects = load_scene(scene)["objects"]
    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    t_load = time.perf_counter() - t

    t = time.perf_counter()
    clear = place_clearances(graph, profiles["frames"], scale=args.camera_height or 1.0,
                             ignore_sector=tuple(args.ignore_sector), percentile=args.percentile)
    t_map = time.perf_counter() - t
    for p in graph["layers"]["places"]:
        p.pop("clearance", None)
    t = time.perf_counter()
    before = regraph(graph, objects)
    t_radius = time.perf_counter() - t
    for p in graph["layers"]["places"]:
        p["clearance"] = clear.get(p["id"])
    t = time.perf_counter()
    after = regraph(graph, objects, min_room_places=args.min_room_places)
    t_clear = time.perf_counter() - t

    unit = "m" if args.camera_height else "camera heights"
    vals = np.array(list(clear.values()))
    print(f"profiles: {len(profiles['frames'])} frames ({profiles.get('model')}, timing {profiles.get('timing')})")
    print(f"clearance: {len(clear)}/{len(graph['layers']['places'])} places measured, "
          f"min {vals.min():.2f} median {np.median(vals):.2f} max {vals.max():.2f} {unit}")
    print(f"radius   : {region_stats(before)}")
    seg = after.get("segmentation", {})
    print(f"clearance: {region_stats(after)}  [{seg.get('method')}, threshold {seg.get('threshold', 0) or 0:.2f} "
          f"{unit}, stable over {seg.get('persistence')} sweep steps, {seg.get('seeds')} seeds, "
          f"{seg.get('door_cuts')} door cuts]")
    for r in after["layers"]["regions"]:
        cls = Counter(o["name"] for o in after["layers"]["objects"] if o.get("region") == r["id"])
        c = [clear[i] for i in r["places"] if i in clear]
        print(f"  region{r['id']:<3} {len(r['places']):>3} places  clearance median "
              f"{np.median(c) if c else float('nan'):.2f}  objects {dict(cls.most_common(5))}")
    print(f"timing: load {1000 * t_load:.0f} ms, frame->place clearance {1000 * t_map:.1f} ms, "
          f"regraph radius {1000 * t_radius:.1f} ms, regraph clearance {1000 * t_clear:.1f} ms")
    if args.plot:
        plot([("radius 4 m", before), ("clearance filtration", after)], args.plot, clear)
        print(f"wrote {args.plot}")
    if args.save:
        save_graph(after, scene / "topo_graph.json")
        print(f"saved {scene / 'topo_graph.json'} (re-run dsg_export --summarize for region labels)")


if __name__ == "__main__":
    main()
