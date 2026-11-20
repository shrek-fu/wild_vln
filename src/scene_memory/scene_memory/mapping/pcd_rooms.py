"""Rooms from the lidar point-cloud map (FSR-VLN / HOV-SG style), for a scene mapped by ``stvln.sh map``.

Per floor of the place graph:

1. floor and ceiling heights: the densest point-height bins below / above the places of that floor;
2. walls: cells with points in at least ``wall_cover`` of the 0.1 m slices of the wall band (``wall_lo``
   above the floor up to ``max_wall_hi`` / just under the ceiling); FSR-VLN thresholds a 2D histogram of a
   band instead, which turns desks and ceiling lamps into walls in a lidar map;
3. outside: everything outside the filled footprint of all points on that floor;
4. room seeds: distance transform of the free cells (neither wall nor outside), thresholded (Otsu, as in
   FSR-VLN, clipped to ``seed_min_m`` .. ``seed_max_m``); seeds smaller than ``min_seed_m2`` are dropped;
5. the seeds grow over the free cells at equal speed (FSR-VLN uses a watershed here, see ``grow``).

Each place / object takes the room of its cell (or of the nearest room cell within ``snap_m``) as
``room_pcd``; ``topo_graph.segment_regions`` then uses those rooms instead of door cuts + radius /
clearance, and objects join a place of their own room (not one behind a wall). The map and the scene must
share a frame: ``stvln.sh map`` / ``map_offline`` save both in Super-LIO's odom.

    python3 -m scene_memory.mapping.pcd_rooms data/maps/office/office_scene --pcd data/maps/office/office_map.pcd \
        --plot /tmp/rooms.png [--save]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from scene_memory import paths
from scene_memory.mapping.topo_graph import place_zone


def read_xyz(path: str | Path) -> np.ndarray:
    sys.path.insert(0, str(paths.WS_ROOT / "src" / "mapping" / "scripts"))
    from semantic_ground import read_pcd_xyz

    xyz = read_pcd_xyz(path)
    return xyz[np.isfinite(xyz).all(axis=1)]


def _mode(z: np.ndarray, bin_m: float = 0.05) -> float | None:
    if len(z) < 20:
        return None
    edges = np.arange(z.min(), z.max() + 2 * bin_m, bin_m)
    hist, edges = np.histogram(z, edges)
    i = int(hist.argmax())
    return float(0.5 * (edges[i] + edges[i + 1]))


def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, 128)
    p = hist / max(hist.sum(), 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    w = np.cumsum(p)
    mu = np.cumsum(p * mids)
    between = (mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return float(mids[int(np.nanargmax(between))])


def grow(markers: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Seeds grown over the free cells at equal speed (geodesic Voronoi), so two rooms split in the middle
    of the doorway; a watershed on the distance transform lets the first seed through a doorway flood the
    low-clearance cells along the other room's walls."""
    lab = np.where(free, markers, 0)
    while True:
        cand = np.zeros_like(lab)
        for src, dst in ((np.s_[1:, :], np.s_[:-1, :]), (np.s_[:-1, :], np.s_[1:, :]),
                         (np.s_[:, 1:], np.s_[:, :-1]), (np.s_[:, :-1], np.s_[:, 1:])):
            view = cand[dst]
            np.copyto(view, lab[src], where=view == 0)
        new = (lab == 0) & free & (cand > 0)
        if not new.any():
            return lab
        lab[new] = cand[new]


class FloorRooms:
    """Room label grid of one floor; ``label_at`` maps xy to a room id (offset by ``base``) or None."""

    def __init__(self, xyz: np.ndarray, place_xyz: np.ndarray, base: int = 0, res: float = 0.05,
                 wall_lo: float = 0.5, ceiling_margin: float = 0.3, max_wall_hi: float = 2.4,
                 wall_min_pts: int = 1, wall_cover: float = 0.5, slice_m: float = 0.1, margin_m: float = 8.0,
                 seed_min_m: float = 0.5, seed_max_m: float = 1.5, min_seed_m2: float = 1.0, snap_m: float = 1.0):
        import cv2
        from scipy.ndimage import distance_transform_edt
        from scipy.spatial import cKDTree

        self.res, self.snap_m, self.base = res, snap_m, base
        zp = float(np.median(place_xyz[:, 2]))
        tree = cKDTree(place_xyz[:, :2])
        near = tree.query(xyz[:, :2], distance_upper_bound=1.0)[0] < np.inf
        self.floor_z = _mode(xyz[near & (xyz[:, 2] < zp + 0.1) & (xyz[:, 2] > zp - 1.5), 2]) or zp - 0.4
        ceil = xyz[near & (xyz[:, 2] > self.floor_z + 1.8) & (xyz[:, 2] < self.floor_z + 5.0), 2]
        self.ceiling_z = _mode(ceil) if len(ceil) > 200 else None
        hi = min(self.floor_z + max_wall_hi, (self.ceiling_z or np.inf) - ceiling_margin)
        self.band = (self.floor_z + wall_lo, hi)

        lo_xy = place_xyz[:, :2].min(axis=0) - margin_m
        hi_xy = place_xyz[:, :2].max(axis=0) + margin_m
        on_floor = (xyz[:, 2] > self.floor_z - 0.3) & (xyz[:, 2] < hi) & \
            np.all((xyz[:, :2] >= lo_xy) & (xyz[:, :2] <= hi_xy), axis=1)
        pts = xyz[on_floor]
        self.origin = lo_xy
        shape = tuple(np.ceil((hi_xy - lo_xy) / res).astype(int) + 1)  # (nx, ny)
        self.shape = shape

        def hist(p):
            ij = ((p[:, :2] - lo_xy) / res).astype(int)
            h = np.zeros(shape, np.int32)
            np.add.at(h, (ij[:, 0], ij[:, 1]), 1)
            return h

        # wall = points over most of the band's height: desks fill only its bottom, lamps / ducts its top
        slices = np.arange(self.band[0], self.band[1], slice_m)
        cover = np.zeros(shape, np.int32)
        gaps = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # voxelised maps leave gaps between cells
        for z0 in slices:
            s = pts[(pts[:, 2] >= z0) & (pts[:, 2] < z0 + slice_m)]
            cover += cv2.dilate((hist(s) >= wall_min_pts).astype(np.uint8), gaps)
        walls = (cover >= max(1, int(np.ceil(wall_cover * len(slices))))).astype(np.uint8)
        walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        foot = (hist(pts) > 0).astype(np.uint8)
        foot = cv2.morphologyEx(foot, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                                iterations=3)
        contours, _ = cv2.findContours(foot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        inside = np.zeros_like(foot)
        cv2.drawContours(inside, contours, -1, 1, -1)
        free = (inside > 0) & (walls == 0)
        self.walls, self.free = walls > 0, free

        dt = distance_transform_edt(free) * res
        self.dt = dt
        thr = float(np.clip(_otsu(dt[free]) if free.any() else seed_min_m, seed_min_m, seed_max_m))
        self.seed_m = thr
        n, comp, stats, _ = cv2.connectedComponentsWithStats((dt >= thr).astype(np.uint8), connectivity=8)
        keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] * res * res >= min_seed_m2]
        markers = np.zeros(shape, np.int32)
        for k, i in enumerate(keep, start=1):
            markers[comp == i] = k
        self.labels = grow(markers, free)
        self.n_rooms = len(keep)
        _, idx = distance_transform_edt(self.labels == 0, return_indices=True)
        self._nearest = idx

    def label_at(self, xy) -> int | None:
        ij = np.floor((np.asarray(xy[:2], float) - self.origin) / self.res).astype(int)
        if not (0 <= ij[0] < self.shape[0] and 0 <= ij[1] < self.shape[1]):
            return None
        ni, nj = self._nearest[0][ij[0], ij[1]], self._nearest[1][ij[0], ij[1]]
        if self.labels[ni, nj] <= 0 or np.hypot(ni - ij[0], nj - ij[1]) * self.res > self.snap_m:
            return None
        return self.base + int(self.labels[ni, nj]) - 1


def segment(graph: dict, objects: list[dict], xyz: np.ndarray, **kwargs) -> tuple[dict, dict, list[FloorRooms]]:
    """``room_pcd`` per place id and per object key, plus the per-floor grids."""
    places = graph["layers"]["places"]
    floors: dict[tuple, list[dict]] = {}
    for p in places:
        zone = place_zone(p)
        if zone[0] == "indoor":
            floors.setdefault((p.get("building"), zone[1]), []).append(p)
    grids, place_room, base = [], {}, 0
    for key, members in sorted(floors.items(), key=lambda kv: str(kv[0])):
        pxyz = np.array([p["position"] for p in members], float)
        fr = FloorRooms(xyz, pxyz, base=base, **kwargs)
        fr.key = key
        grids.append(fr)
        for p in members:
            place_room[p["id"]] = fr.label_at(p["position"])
        base += fr.n_rooms
    object_room = {}
    for o in objects:
        if not grids:
            break
        c = np.asarray(o["center"], float)
        fr = min(grids, key=lambda g: abs(c[2] - (g.floor_z + 0.8)))
        if fr.floor_z - 0.5 <= c[2] <= (fr.ceiling_z or fr.floor_z + 3.0) + 0.2:
            object_room[o["key"]] = fr.label_at(c)
    return place_room, object_room, grids


def plot(grids: list[FloorRooms], graph: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(grids), figsize=(8 * len(grids), 8), squeeze=False)
    cmap = plt.get_cmap("tab20")
    for ax, fr in zip(axes[0], grids):
        img = np.ones((*fr.shape, 3))
        img[fr.free] = 0.92
        for k in range(fr.n_rooms):
            img[fr.labels == k + 1] = np.array(cmap(k % 20)[:3]) * 0.55 + 0.45
        img[fr.walls] = 0.15
        ext = [fr.origin[0], fr.origin[0] + fr.shape[0] * fr.res, fr.origin[1], fr.origin[1] + fr.shape[1] * fr.res]
        ax.imshow(img.transpose(1, 0, 2), origin="lower", extent=ext)
        regions = graph["layers"]["regions"]
        xy = {p["id"]: p["position"][:2] for p in graph["layers"]["places"]}
        for r in regions:
            pts = np.array([xy[i] for i in r["places"]])
            ax.scatter(pts[:, 0], pts[:, 1], s=14, color=cmap(r["id"] % 20), edgecolors="k", linewidths=0.4)
            ax.text(*pts.mean(axis=0), f"R{r['id']}", fontsize=9, weight="bold")
        ax.set_title(f"floor {fr.key}: {fr.n_rooms} rooms, z floor {fr.floor_z:.2f} ceiling "
                     f"{fr.ceiling_z if fr.ceiling_z is None else round(fr.ceiling_z, 2)}, wall band "
                     f"{fr.band[0]:.2f}-{fr.band[1]:.2f}, seed {fr.seed_m:.2f} m", fontsize=9)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def apply(graph: dict, place_room: dict, object_room: dict) -> None:
    for p in graph["layers"]["places"]:
        p["room_pcd"] = place_room.get(p["id"])
    by_key = {o["key"]: o for o in graph["layers"]["objects"]}
    for key, room in object_room.items():
        if key in by_key:
            by_key[key]["room_pcd"] = room


def main() -> None:
    from scene_memory.mapping.topo_graph import regraph, save_graph
    from scene_memory.query.query_4d import load_scene

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="scene dir with topo_graph.json + objects_4d.json (e.g. data/maps/<name>/<name>_scene)")
    ap.add_argument("--pcd", required=True, help="point-cloud map in the scene's frame (<name>_map.pcd)")
    ap.add_argument("--res", type=float, default=0.05, help="m per grid cell")
    ap.add_argument("--wall_lo", type=float, default=0.5, help="m above the floor where the wall band starts")
    ap.add_argument("--max_wall_hi", type=float, default=2.4, help="m above the floor where the wall band ends "
                                                                       "(at most ceiling - 0.3)")
    ap.add_argument("--wall_cover", type=float, default=0.5,
                    help="fraction of the band's 0.1 m slices a wall cell must have points in")
    ap.add_argument("--wall_min_pts", type=int, default=1, help="points that make a slice of a cell occupied")
    ap.add_argument("--seed_min_m", type=float, default=0.5, help="clip of the Otsu seed distance (m)")
    ap.add_argument("--seed_max_m", type=float, default=1.5)
    ap.add_argument("--min_seed_m2", type=float, default=1.0)
    ap.add_argument("--plot", default=None, help="top-down PNG of the room grid and the new regions")
    ap.add_argument("--save", action="store_true", help="write room_pcd + new regions into topo_graph.json")
    args = ap.parse_args()

    scene = Path(args.scene)
    t = time.perf_counter()
    graph = json.loads((scene / "topo_graph.json").read_text(encoding="utf-8"))
    objects = load_scene(scene)["objects"]
    xyz = read_xyz(args.pcd)
    t_load = time.perf_counter() - t
    before = len(graph["layers"]["regions"])
    t = time.perf_counter()
    place_room, object_room, grids = segment(graph, objects, xyz, res=args.res, wall_lo=args.wall_lo,
                                             max_wall_hi=args.max_wall_hi, wall_cover=args.wall_cover,
                                             wall_min_pts=args.wall_min_pts, seed_min_m=args.seed_min_m,
                                             seed_max_m=args.seed_max_m, min_seed_m2=args.min_seed_m2)
    t_seg = time.perf_counter() - t
    apply(graph, place_room, object_room)
    after = regraph(graph, objects)
    seg = after.get("segmentation", {})
    print(f"map: {len(xyz)} points, load {t_load:.1f} s, rooms {t_seg:.1f} s")
    for fr in grids:
        print(f"  floor {fr.key}: floor z {fr.floor_z:.2f}, ceiling {fr.ceiling_z}, wall band {fr.band[0]:.2f}-"
              f"{fr.band[1]:.2f}, seed distance {fr.seed_m:.2f} m, {fr.n_rooms} rooms")
    unassigned = sum(v is None for v in place_room.values())
    print(f"places: {len(place_room)} indoor, {unassigned} outside every room; objects: "
          f"{sum(v is not None for v in object_room.values())}/{len(objects)} in a room")
    print(f"regions: {before} -> {len(after['layers']['regions'])} [{seg.get('method')}]")
    for r in after["layers"]["regions"]:
        cls = Counter(o["name"] for o in after["layers"]["objects"] if o.get("region") == r["id"])
        print(f"  region{r['id']:<3} {len(r['places']):>3} places  objects {dict(cls.most_common(6))}")
    if args.plot:
        plot(grids, after, args.plot)
        print(f"wrote {args.plot}")
    if args.save:
        save_graph(after, scene / "topo_graph.json")
        print(f"saved {scene / 'topo_graph.json'} (region labels: dsg_export --summarize)")


if __name__ == "__main__":
    main()
