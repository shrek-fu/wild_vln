"""Per-stage latency of the depth paths on recorded frames, timed as the live nodes run them.

* DA360 (``live_ros.py``): JPEG decode, preprocess, network (fp32, and fp16 autocast for reference), postprocess
  to the equirect size, one global lidar scale (``estimate_scale_from_lidar``); then Hydra's per-view
  ``equirect_depth_to_pinhole``.
* DepthART (Hydra ``--depth_source depthart``): pinhole view rendered from the equirect, TensorRT inference.
* Lidar (``view_depth_node``, topo mapper): raw scans projected into the pinhole view.

Accuracy on the same pinhole view and the same held-out lidar returns (``depthart_bag_eval`` protocol):
``da360_live`` is exactly what Hydra samples (one equirect-wide scale), ``*_global`` refits one scale per view
on the non-held returns, ``depthart_<domain>_raw`` is the metric output as is.

Writes ``bench.json`` with median / p90 / max ms per stage and the accuracy summary::

    python3 -m scene_memory.mapping.depth_stage_bench <session>/raw_pano --calib_dir <session>/calib \
        --out_dir <session>/depth_bench
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from scene_memory import paths
from scene_memory.common.calib_io import load_rig_calibration
from scene_memory.mapping.depthart_bag_eval import drop_occluded, metrics
from scene_memory.mapping.pano_depth_bag import load_da360
from scene_memory.mapping.remap_bag import read_bag
from scene_memory.mapping.scene_graph import T_da360_optical, pinhole_intrinsics
from scene_memory.perception.boxer_lift import equirect_depth_to_pinhole, equirect_remap, load_pinhole_views
from scene_memory.perception.depthart_trt import DepthArtMetric
from scene_memory.perception.lidar_depth import CloudWindow, livox_custom_xyz, view_depth


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", type=Path)
    p.add_argument("--calib_dir", default=str(paths.CALIB))
    p.add_argument("--model_path", default=str(paths.DA360))
    p.add_argument("--view", default="front")
    p.add_argument("--domains", default="outdoor,indoor", help="first one is timed")
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--equirect_topic", default="/insta360/equirect/image/compressed")
    p.add_argument("--lidar_topic", default="/livox/front_lidar/custom")
    p.add_argument("--blind", type=float, default=0.5)
    p.add_argument("--window_s", type=float, default=0.3)
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--save_every", type=int, default=10, help="views saved as npz for figures")
    p.add_argument("--out_dir", type=Path, default=Path("/tmp/depth_bench"))
    return p.parse_args(argv)


def main(argv=None) -> None:
    import torch

    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = str(paths.DA360_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from insta_live.geometry import estimate_scale_from_lidar

    calib = load_rig_calibration(args.calib_dir)
    view = next(v for v in load_pinhole_views(calib.source_dir) if v["name"] == args.view)
    w, h, hfov = view["width"], view["height"], view["hfov_deg"]
    K = pinhole_intrinsics(w, h, hfov)
    T_front_opt = calib.T_front_da360 @ T_da360_optical(view["R_body_optical"])
    T_da360_front = np.linalg.inv(calib.T_front_da360)
    runner = load_da360(args.model_path)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    depthart = {d: DepthArtMetric(w, h, hfov, domain=d) for d in domains}
    rng = np.random.default_rng(args.seed)
    collected: dict[str, tuple[list, list]] = {}
    cloud = CloudWindow(args.window_s, keep_s=args.window_s + 10.0, voxel_m=0.03)
    cuda = runner.device.type == "cuda"
    times: dict[str, list[float]] = {}
    info: dict = {}
    maps: dict = {}
    n_image = 0
    n_done = 0

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    def timed(name: str, fn, *a, **kw):
        sync()
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        sync()
        times.setdefault(name, []).append(1e3 * (time.perf_counter() - t0))
        return out

    def net(tensor, fp16: bool):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            return runner.model(tensor)["pred_disp"]

    def preprocess(bgr):
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (runner.width, runner.height),
                         interpolation=cv2.INTER_CUBIC)
        return runner.normalize(runner.to_tensor(rgb)).unsqueeze(0).to(runner.device)

    def postprocess(disp, size):
        depth = 1.0 / np.maximum(disp.detach().float().cpu().numpy()[0, 0], 1e-6)
        depth = cv2.resize(depth / max(float(depth.min()), 1e-6), size, interpolation=cv2.INTER_LINEAR)
        return (depth / max(float(depth[depth > 0].min()), 1e-6)).astype(np.float32)

    def frame(t: float, msg) -> None:
        nonlocal n_done
        xyz = cloud.points(t)
        if len(xyz) == 0:
            return
        if n_done == args.warmup:
            times.clear()
        bgr = timed("decode_equirect", cv2.imdecode, np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        eh, ew = bgr.shape[:2]
        info.update(equirect=f"{ew}x{eh}", da360_input=f"{runner.width}x{runner.height}", view=f"{w}x{h} hfov {hfov}")
        tensor = timed("da360_preprocess", preprocess, bgr)
        timed("da360_net_fp16", net, tensor, True)
        disp = timed("da360_net_fp32", net, tensor, False)
        rel = timed("da360_postprocess", postprocess, disp, (ew, eh))
        pts = xyz @ T_da360_front[:3, :3].T + T_da360_front[:3, 3]
        scale = timed("da360_global_scale", estimate_scale_from_lidar, rel, pts, np.eye(4))
        metric = rel * np.float32(scale or 1.0)
        da360 = timed("da360_to_pinhole_view", equirect_depth_to_pinhole, metric, view["R_body_optical"], w, h, hfov)
        if (eh, ew) not in maps:
            maps[(eh, ew)] = equirect_remap(eh, ew, view["R_body_optical"], h, w, hfov)[:2]
        pin = timed("render_pinhole_rgb", cv2.remap, bgr, *maps[(eh, ew)], cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_WRAP)
        preds = {"da360_live": da360}
        for i, (d, model) in enumerate(depthart.items()):
            preds[f"depthart_{d}_raw"] = timed("depthart_trt", model.infer_bgr, pin) if i == 0 else model.infer_bgr(pin)
        lidar = timed("lidar_view_depth", view_depth, xyz, T_front_opt, K, w, h, z_max=60.0)
        n_done += 1
        sparse = drop_occluded(lidar)
        vv, uu = np.nonzero(sparse > 0)
        if len(vv) < 100:
            return
        held = rng.random(len(vv)) < args.holdout
        truth = np.zeros_like(sparse)
        truth[vv[held], uu[held]] = sparse[vv[held], uu[held]]
        train = sparse.copy()
        train[vv[held], uu[held]] = 0.0
        for name, pred in list(preds.items()):
            fit = (train > 0) & (pred > 0)
            if fit.any():
                preds[name.replace("_raw", "").replace("_live", "") + "_global"] = \
                    pred * np.float32(np.median(train[fit] / pred[fit]))
        if n_done % args.save_every == 1:
            np.savez_compressed(args.out_dir / f"view_{n_done:04d}.npz", rgb=pin, lidar=sparse,
                                **{k: v.astype(np.float16) for k, v in preds.items()})
        m = truth > 0
        for name, pred in preds.items():
            p_t = collected.setdefault(name, ([], []))
            p_t[0].append(pred[m])
            p_t[1].append(truth[m])

    for topic, msg in read_bag(args.bag, [args.equirect_topic, args.lidar_topic], raw=frozenset([args.lidar_topic])):
        if topic == args.lidar_topic:
            cloud.add(*livox_custom_xyz(msg, args.blind))
            continue
        n_image += 1
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if n_image % args.every == 0 and (cloud.last_t() or 0.0) >= t:
            frame(t, msg)
        if n_done >= args.frames + args.warmup:
            break

    stats = {k: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(np.max(v)),
                 "n": len(v)} for k, v in times.items()}
    accuracy = {k: metrics(np.concatenate(p), np.concatenate(t)) for k, (p, t) in collected.items()}
    report = {"info": info, "device": torch.cuda.get_device_name(0) if cuda else "cpu", "frames": n_done,
              "stages": stats, "accuracy": accuracy}
    (args.out_dir / "bench.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(info))
    for k, v in stats.items():
        print(f"{k:24s} median {v['median']:8.1f}  p90 {v['p90']:8.1f}  max {v['max']:8.1f}  n {v['n']}")
    for k, a in accuracy.items():
        if a["n"]:
            bins = "  ".join(f"{b['bin']}: {b.get('rel_mean', 0):.3f}/{b.get('abs_med', 0):.2f}m" for b in a["bins"])
            print(f"{k:24s} absrel {a['abs_rel']:.3f} rmse {a['rmse']:.2f} d1 {a['delta1']:.3f}  {bins}")


if __name__ == "__main__":
    main()
