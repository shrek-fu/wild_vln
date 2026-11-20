"""DepthART metric depth on a recorded bag, checked against lidar returns projected into the same pinhole view.

Same protocol as ``pano_depth_bag`` (DA360), so the numbers compare: ``--lidar_source raw`` (default) projects
the raw MID-360 scans of the last ``--window_s`` with the camera-lidar extrinsic alone (no odometry), drops
returns occluded from the camera's viewpoint, and holds out a random ``--holdout`` share as ground truth. Per
domain (needs the DepthART TensorRT engines on this device, ``scripts/depthart_trt.py``):

* ``raw``: DepthART's metric output as is (what Hydra gets with ``--depth_source depthart``);
* ``global``: times one per-frame scale (median lidar / prediction ratio on the non-held returns), i.e. the
  shape error once a lidar fixes the scale (DA360's ``global``).

The view comes from ``/insta360/<view>/image/compressed`` or, with ``--from_equirect``, is rendered from the
equirect. ``report.json`` holds the summary and per-frame values; ``--save_every`` frames are written as
``rgb | lidar | <domain> ...`` montages on one colour scale::

    python3 -m scene_memory.mapping.depthart_bag_eval <session>/raw_pano --from_equirect \
        --calib_dir <session>/calib --out_dir <session>/depthart_eval
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
from scene_memory.common.pose_buffer import PoseBuffer
from scene_memory.mapping.remap_bag import read_bag
from scene_memory.mapping.scene_graph import (T_da360_optical, pinhole_intrinsics, pose_frame_to_da360,
                                              pose_to_matrix)
from scene_memory.perception.boxer_lift import equirect_remap, load_pinhole_views
from scene_memory.perception.depthart_trt import DOMAINS, DepthArtMetric
from scene_memory.perception.lidar_depth import CloudWindow, cloud_xyz_fast, livox_custom_xyz, view_depth
from scene_memory.perception.pano_depth import range_bin_errors

BINS = (0.0, 5.0, 10.0, 30.0, np.inf)


def depth_color(depth: np.ndarray, max_range: float) -> np.ndarray:
    level = np.clip(np.sqrt(np.clip(depth, 0, None) / max_range) * 255.0, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(level, cv2.COLORMAP_TURBO)
    out[depth <= 0] = 0
    return out


def drop_occluded(depth: np.ndarray, px: int = 2, rel: float = 0.08) -> np.ndarray:
    """Lidar and camera see from different points: drop a return when one within ``px`` is nearer by more than
    ``rel`` of its depth (plus 0.1 m), as ``pano_depth.sparse_hemi_depth`` does."""
    d = np.where(depth > 0, depth, np.inf).astype(np.float32)
    k = 2 * px + 1
    near = cv2.erode(d, np.ones((k, k), np.uint8), borderType=cv2.BORDER_REPLICATE)
    d[d > near * (1.0 + rel) + 0.1] = np.inf
    d[~np.isfinite(d)] = 0.0
    return d


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    ok = (truth > 0) & (pred > 0)
    p, t = pred[ok], truth[ok]
    if not len(t):
        return {"n": 0}
    ratio = np.maximum(p / t, t / p)
    return {"n": int(len(t)), "abs_rel": float(np.mean(np.abs(p - t) / t)),
            "rmse": float(np.sqrt(np.mean((p - t) ** 2))), "delta1": float(np.mean(ratio < 1.25)),
            "bins": range_bin_errors(p, t, BINS)}


def label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", type=Path)
    p.add_argument("--calib_dir", default=str(paths.CALIB))
    p.add_argument("--lio_config", default=str(paths.LIO_CONFIG))
    p.add_argument("--view", default="front")
    p.add_argument("--from_equirect", action="store_true", help="render the view from the equirect topic")
    p.add_argument("--equirect_topic", default="/insta360/equirect/image/compressed")
    p.add_argument("--lidar_source", choices=["raw", "lio"], default="raw")
    p.add_argument("--lidar_topic", default="/livox/front_lidar/custom", help="raw: livox CustomMsg")
    p.add_argument("--blind", type=float, default=0.5, help="m; raw returns closer than this are the rig itself")
    p.add_argument("--odom_topic", default="/lio/odom")
    p.add_argument("--cloud_topic", default="/lio/cloud_world")
    p.add_argument("--domains", default=",".join(DOMAINS))
    p.add_argument("--window_s", type=float, default=None, help="scans stacked per image (raw 0.3 s, lio 1.0 s)")
    p.add_argument("--max_lag_s", type=float, default=10.0, help="how far images may trail the lidar in the bag")
    p.add_argument("--voxel", type=float, default=0.03)
    p.add_argument("--max_range", type=float, default=60.0)
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--every", type=int, default=2, help="use every n-th image")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=Path, default=Path("/tmp/depthart_eval"))
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    calib = load_rig_calibration(args.calib_dir)
    raw = args.lidar_source == "raw"
    window_s = args.window_s if args.window_s is not None else (0.3 if raw else 1.0)
    view = next(v for v in load_pinhole_views(calib.source_dir) if v["name"] == args.view)
    w, h, hfov = view["width"], view["height"], view["hfov_deg"]
    K = pinhole_intrinsics(w, h, hfov)
    T_da360_opt = T_da360_optical(view["R_body_optical"])
    T_child_da360 = pose_frame_to_da360("lio", paths.VIO_CONFIG, args.lio_config, calib)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    models = {d: DepthArtMetric(w, h, hfov, domain=d) for d in domains}
    rng = np.random.default_rng(args.seed)
    poses = PoseBuffer()
    cloud = CloudWindow(window_s, keep_s=window_s + args.max_lag_s, voxel_m=args.voxel)
    image_topic = args.equirect_topic if args.from_equirect else f"/insta360/{args.view}/image/compressed"
    maps: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    pending: list[tuple[float, object]] = []
    collected = {(d, k): ([], []) for d in domains for k in ("raw", "global")}
    per_frame: list[dict] = []
    n_image = 0

    def view_image(msg) -> np.ndarray | None:
        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or not args.from_equirect:
            return bgr if bgr is None or bgr.shape[:2] == (h, w) else cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        key = bgr.shape[:2]
        if key not in maps:
            maps[key] = equirect_remap(key[0], key[1], view["R_body_optical"], h, w, hfov)[:2]
        return cv2.remap(bgr, *maps[key], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def process(t: float, msg) -> None:
        if raw:
            T_ref_opt = calib.T_front_da360 @ T_da360_opt
        else:
            T_odom_child = poses.at(t)
            if T_odom_child is None:
                return
            T_ref_opt = T_odom_child @ T_child_da360 @ T_da360_opt
        xyz = cloud.points(t)
        if len(xyz) == 0:
            return
        bgr = view_image(msg)
        if bgr is None:
            return
        sparse = drop_occluded(view_depth(xyz, T_ref_opt, K, w, h, z_max=args.max_range))
        vv, uu = np.nonzero(sparse > 0)
        if len(vv) < 100:
            return
        held = rng.random(len(vv)) < args.holdout
        truth = np.zeros_like(sparse)
        truth[vv[held], uu[held]] = sparse[vv[held], uu[held]]
        train = sparse.copy()
        train[vv[held], uu[held]] = 0.0
        frame = {"t": t, "lidar_px": int(len(vv))}
        tiles = [label(bgr.copy(), args.view), label(depth_color(cv2.dilate(sparse, np.ones((3, 3))), 30.0), "lidar")]
        for d, model in models.items():
            t0 = time.perf_counter()
            pred = model.infer_bgr(bgr)
            frame[f"{d}_ms"] = 1e3 * (time.perf_counter() - t0)
            fit = (train > 0) & (pred > 0)
            scale = float(np.median(train[fit] / pred[fit])) if fit.any() else float("nan")
            frame[f"{d}_scale"] = scale
            m = (truth > 0) & (pred > 0)
            for key, p in (("raw", pred), ("global", pred * np.float32(scale))):
                collected[(d, key)][0].append(p[m])
                collected[(d, key)][1].append(truth[m])
            tiles.append(label(depth_color(pred, 30.0), f"DepthART {d}"))
        per_frame.append(frame)
        if len(per_frame) % args.save_every == 1:
            cv2.imwrite(str(args.out_dir / f"frame_{len(per_frame):04d}.jpg"), np.hstack(tiles),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

    topics = [image_topic] + ([args.lidar_topic] if raw else [args.odom_topic, args.cloud_topic])
    for topic, msg in read_bag(args.bag, topics, raw=frozenset([args.lidar_topic])):
        if topic == args.lidar_topic:
            cloud.add(*livox_custom_xyz(msg, args.blind))
            stamp = None
        else:
            stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if topic == args.odom_topic:
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            poses.add(stamp, pose_to_matrix((p.x, p.y, p.z), (q.x, q.y, q.z, q.w)))
        elif topic == args.cloud_topic:
            cloud.add(stamp, cloud_xyz_fast(msg))
        elif topic == image_topic:
            n_image += 1
            if n_image % args.every == 0:
                pending.append((stamp, msg))
        while pending and (raw or poses.last_t() >= pending[0][0]) and (cloud.last_t() or 0.0) >= pending[0][0]:
            process(*pending.pop(0))

    summary: dict = {"frames": len(per_frame), "view": f"{args.view} {w}x{h} hfov {hfov}",
                     "lidar_source": args.lidar_source, "window_s": window_s, "holdout": args.holdout}
    for (d, key), (preds, truths) in collected.items():
        if preds:
            summary[f"{d}_{key}"] = metrics(np.concatenate(preds), np.concatenate(truths))
    for d in domains:
        for k in ("scale", "ms"):
            vals = np.array([f[f"{d}_{k}"] for f in per_frame if np.isfinite(f[f"{d}_{k}"])])
            if len(vals):
                summary[f"{d}_{k}"] = {"median": float(np.median(vals)), "p10": float(np.percentile(vals, 10)),
                                       "p90": float(np.percentile(vals, 90))}
    if per_frame:
        summary["lidar_px_median"] = float(np.median([f["lidar_px"] for f in per_frame]))
    (args.out_dir / "report.json").write_text(json.dumps({"summary": summary, "frames": per_frame}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
