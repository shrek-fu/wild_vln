"""Offline traversable ground from a bag with ``/insta360/equirect/image/compressed``, ``/lio/odom`` and
``/lio/cloud_world``; compares segmenters against the lidar geometry.

Pinhole views are rendered from the equirect with the rig calibration (as the live ``insta360_views``
does). Each segmenter runs its own ``GroundMapper``. The score uses a reference independent of any
mask, valid for single-floor bags: one RANSAC plane over the returns of the whole bag below the sensor;
returns within 6 cm of it are ground, returns 15 cm..1.5 m above it are not. For every segmenter:

* precision = ground returns in the mask / labelled returns in the mask (does the mask leak onto walls,
  furniture, people?);
* recall = ground returns in the mask / all ground returns (does it miss floor?);
* traversable_on_obstacle = final traversable cells holding obstacle returns from anywhere in the bag
  (what is left of the leaks after the lidar veto).

    python3 -m ground_segmentation.bag_eval data/sessions/depthtest_20261110/raw_pano --out /tmp/ground_eval
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from scene_memory import paths
from scene_memory.common.calib_io import load_rig_calibration
from scene_memory.mapping.remap_bag import read_bag
from scene_memory.mapping.scene_graph import pose_frame_to_da360, pose_to_matrix
from scene_memory.perception.boxer_lift import equirect_remap, load_pinhole_views
from scene_memory.perception.lidar_depth import cloud_xyz_fast

from ground_segmentation.geometry import fit_ground_plane, heights
from ground_segmentation.mapper import GroundMapper, draw_overlay, grid_image
from ground_segmentation.segmenters import make_segmenter, terrain_classes


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", type=Path)
    p.add_argument("--out", type=Path, default=Path("/tmp/ground_eval"))
    p.add_argument("--calib_dir", default="", help="default: <bag>/../calib if present, else the workspace calib")
    p.add_argument("--image_topic", default="/insta360/equirect/image/compressed")
    p.add_argument("--odom_topic", default="/lio/odom")
    p.add_argument("--cloud_topic", default="/lio/cloud_world")
    p.add_argument("--views", default="front,left,rear,right")
    p.add_argument("--segmenters", default="detector", help="comma list of detector|gdino_sam2")
    p.add_argument("--classes", default=str(paths.CONFIG / "ground_classes.txt"),
                   help="ground classes file or inline spec (see node --classes)")
    p.add_argument("--max_step", type=float, default=0.18, help="m (see node --max_step)")
    p.add_argument("--max_slope_deg", type=float, default=35.0, help="see node --max_slope_deg")
    p.add_argument("--every", type=float, default=1.0, help="s between processed equirects")
    p.add_argument("--save_every", type=int, default=3, help="write overlays for every n-th processed frame")
    return p.parse_args(argv)


class Score:
    def __init__(self) -> None:
        self.tp = self.fp = self.fn = 0
        self.leaks_by_class: dict[str, int] = {}
        self.seg_s: list[float] = []
        self.plane_err: list[float] = []

    def add(self, res: dict, ref_ground: np.ndarray, ref_other: np.ndarray, terrain=()) -> None:
        """Precision/recall over flat classes; terrain classes legitimately hold returns above the plane."""
        lab = res["labels"][res["uv"][1], res["uv"][0]]
        flat = np.isin(lab, [i + 1 for i, c in enumerate(res["classes"]) if c not in terrain])
        self.tp += int((flat & ref_ground).sum())
        self.fp += int((flat & ref_other).sum())
        self.fn += int((~flat & ref_ground).sum())
        for i, c in enumerate(res["classes"]):
            n = int(((lab == i + 1) & ref_other).sum())
            if n:
                self.leaks_by_class[c] = self.leaks_by_class.get(c, 0) + n

    def summary(self) -> dict:
        prec = self.tp / max(self.tp + self.fp, 1)
        rec = self.tp / max(self.tp + self.fn, 1)
        return {"precision": round(prec, 3), "recall": round(rec, 3),
                "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3), "ground_hits": self.tp,
                "leaks": self.fp, "leaks_by_class": self.leaks_by_class, "misses": self.fn,
                "seg_ms_median": round(1e3 * float(np.median(self.seg_s)), 1) if self.seg_s else None,
                "plane_offset_err_cm": {q: round(100 * float(np.percentile(self.plane_err, p)), 1)
                                        for q, p in (("median", 50), ("p95", 95), ("max", 100))}
                if self.plane_err else None}


def _cell_keys(xyz: np.ndarray, res: float) -> np.ndarray:
    ij = np.floor(np.asarray(xyz, np.float64)[:, :2] / res).astype(np.int64)
    return (ij[:, 0] << 32) | (ij[:, 1] & 0xFFFFFFFF)


def reference_labels(pts: np.ndarray, plane):
    """``(ground, other)`` bool per projected return against the mask-free session plane."""
    hgt = heights(pts, plane)
    return np.abs(hgt) < 0.06, (hgt > 0.15) & (hgt < 1.5)


def main(argv=None) -> None:
    args = _parse_args(argv)
    calib_dir = Path(args.calib_dir) if args.calib_dir else (
        args.bag.parent / "calib" if (args.bag.parent / "calib" / "insta360_intrinsics.json").is_file() else paths.CALIB)
    calib = load_rig_calibration(calib_dir)
    T_child_da360 = pose_frame_to_da360("lio", paths.VIO_CONFIG, paths.LIO_CONFIG, calib)
    wanted = {v.strip() for v in args.views.split(",") if v.strip()}
    views = [v for v in load_pinhole_views(calib.source_dir) if v["name"] in wanted]
    kinds = [k.strip() for k in args.segmenters.split(",") if k.strip()]
    segs = {k: make_segmenter(k, args.classes) for k in kinds}
    mappers = {k: GroundMapper(segs[k], T_child_da360, terrain_classes=terrain_classes(args.classes), keep_s=1e9,
                            max_step=args.max_step, max_slope_deg=args.max_slope_deg)
               for k in kinds}
    scores = {k: Score() for k in kinds}
    print(f"calib {calib_dir}, views {[v['name'] for v in views]}, segmenters {kinds}")

    sensor_z, samples = [], []
    for topic, msg in read_bag(args.bag, [args.odom_topic, args.cloud_topic]):
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if topic == args.odom_topic:
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            T = pose_to_matrix((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
            sensor_z.append(p.z)
            for m in mappers.values():
                m.on_odom(t, T)
        else:
            xyz = cloud_xyz_fast(msg)
            samples.append(xyz[::4])
            for m in mappers.values():
                m.on_cloud(t, xyz)
    # Single-floor sessions only: one plane for the whole bag, fitted to returns well below the sensor.
    all_pts = np.concatenate(samples)
    low = all_pts[all_pts[:, 2] < np.median(sensor_z) - 0.2]
    ref_fit = fit_ground_plane(low[::5], dist=0.03, iters=500, min_inliers=200)
    if ref_fit is None:
        raise SystemExit("no reference ground plane in the bag")
    ref_plane = ref_fit[:2]
    hgt = heights(all_pts, ref_plane)
    obstacle_cells = set(_cell_keys(all_pts[(hgt > 0.15) & (hgt < 1.5)], 0.1).tolist())
    print(f"reference plane n={np.round(ref_plane[0], 4).tolist()} d={ref_plane[1]:.3f}, "
          f"{len(obstacle_cells)} obstacle cells from the whole bag")

    for k in kinds:
        (args.out / k).mkdir(parents=True, exist_ok=True)
    maps, last_t, n_frames = {}, -np.inf, 0
    t_start = time.perf_counter()
    for _, msg in read_bag(args.bag, [args.image_topic]):
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if t - last_t < args.every:
            continue
        eq = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if eq is None:
            continue
        last_t = t
        save = n_frames % max(args.save_every, 1) == 0
        for view in views:
            key = (view["name"], eq.shape[:2])
            if key not in maps:
                mx, my, *_ = equirect_remap(eq.shape[0], eq.shape[1], view["R_body_optical"], view["height"],
                                            view["width"], view["hfov_deg"])
                maps[key] = (mx, my)
            img = cv2.remap(eq, *maps[key], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            panels = [img]
            for k in kinds:
                res = mappers[k].process(view, img, t)
                if res is None:
                    break
                scores[k].seg_s.append(res["seg_s"])
                scores[k].add(res, *reference_labels(res["points"], ref_plane),
                              [mappers[k].classes[i - 1] for i in mappers[k].terrain_ids])
                if res["plane"] is not None:
                    scores[k].plane_err.append(abs(float(res["plane"][1]) - float(ref_plane[1])))
                panels.append(draw_overlay(img, res))
            if save and len(panels) == len(kinds) + 1:
                cv2.imwrite(str(args.out / f"{n_frames:04d}_{view['name']}.jpg"), np.hstack(panels))
        n_frames += 1

    report = {"bag": str(args.bag), "frames": n_frames, "views": [v["name"] for v in views],
              "elapsed_s": round(time.perf_counter() - t_start, 1), "segmenters": {}}
    for k in kinds:
        m = mappers[k]
        cv2.imwrite(str(args.out / k / "grid.png"), grid_image(m.grid, m.classes))
        trav, labels = m.grid.traversable(with_class=True)
        np.save(args.out / k / "traversable_xyz.npy", trav)
        np.save(args.out / k / "traversable_label.npy", labels)
        wrong = sum(1 for c in _cell_keys(trav, m.grid.res).tolist() if c in obstacle_cells)
        cell_m2 = m.grid.res ** 2
        report["segmenters"][k] = {**scores[k].summary(), "mapper": m.stats, "cells": len(m.grid),
                                   "traversable_cells": int(len(trav)), "traversable_on_obstacle": wrong,
                                   "traversable_m2": round(len(trav) * cell_m2, 1),
                                   "class_m2": {c: round(float((labels == i).sum()) * cell_m2, 1)
                                                for i, c in enumerate(m.classes, start=1)}}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
