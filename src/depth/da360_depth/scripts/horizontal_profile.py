#!/usr/bin/env python3
"""Per-frame horizontal free-space profile from DA360 equirect depth (input for clearance / room finding).

DA360 depth is scale-free, so each frame is normalised by the camera height above the floor: the floor plane
is fitted to the lower part of the panorama (this also gives gravity, so a tilted camera is handled), and
for each of ``--bins`` azimuth sectors the nearest obstacle between ``--band`` (in camera heights above the
floor) is stored as horizontal distance in camera heights. Points closer than ``--min_r`` (operator, rig)
are ignored. Output JSON: ``{"frames": [{"stamp", "file", "profile", "tilt_deg", "floor_rms", "ms"}], ...}``.

    .venv/bin/python scripts/horizontal_profile.py --session ../../../data/sessions/20261107_112406
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from insta_live.geometry import da360_depth_to_points  # noqa: E402
from insta_live.model_runner import DA360Runner  # noqa: E402


def fit_floor(pts: np.ndarray, below_deg: float = 35.0, iters: int = 3):
    """Floor normal (pointing down, camera frame y-down) and camera height, from rays well below the horizon."""
    r = np.linalg.norm(pts, axis=1)
    down = pts[:, 1] / np.maximum(r, 1e-9)
    cand = pts[down > np.sin(np.radians(below_deg))]
    if len(cand) < 200:
        return None
    n = np.array([0.0, 1.0, 0.0])
    h = float(np.median(cand @ n))
    for _ in range(iters):
        keep = cand[np.abs(cand @ n - h) < 0.12 * h]
        if len(keep) < 100:
            return None
        c = keep.mean(axis=0)
        _, _, vt = np.linalg.svd(keep - c, full_matrices=False)
        n = vt[2] * np.sign(vt[2][1] or 1.0)
        h = float(np.median(keep @ n))
    rms = float(np.sqrt(np.mean((keep @ n - h) ** 2)) / h)
    return n, h, rms


def profile(depth: np.ndarray, bins: int, band: tuple[float, float], min_r: float):
    pts = da360_depth_to_points(depth)
    pts = pts[np.isfinite(pts).all(axis=1)]
    floor = fit_floor(pts)
    if floor is None:
        return None
    n, h, rms = floor
    a = pts @ n                                  # downward component
    height = (h - a) / h                         # above the floor, camera heights
    horiz = pts - a[:, None] * n[None, :]
    e1 = np.array([0.0, 0.0, 1.0]) - n[2] * n    # camera forward projected onto the floor plane
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    x, y = horiz @ e1, horiz @ e2
    rad = np.hypot(x, y) / h
    sel = (height > band[0]) & (height < band[1]) & (rad > min_r)
    az = ((np.degrees(np.arctan2(y[sel], x[sel])) + 360.0) % 360.0 * bins / 360.0).astype(int) % bins
    prof = np.full(bins, np.inf)
    np.minimum.at(prof, az, rad[sel])
    tilt = float(np.degrees(np.arccos(abs(n[1]))))
    return prof, tilt, rms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="session dir with equirect/ and equirect_stamps.csv")
    ap.add_argument("--model_path", default=str(ROOT / "checkpoints" / "DA360_base.pth"))
    ap.add_argument("--out", default="", help="default <session>/equirect_profiles.json")
    ap.add_argument("--bins", type=int, default=360)
    ap.add_argument("--band", type=float, nargs=2, default=(0.25, 1.4), help="obstacle heights, camera heights")
    ap.add_argument("--min_r", type=float, default=0.6, help="ignore closer points (camera heights)")
    ap.add_argument("--step", type=int, default=2, help="depth pixel subsampling for the profile")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    session = Path(args.session)
    rows = []
    with open(session / "equirect_stamps.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("file") and (session / "equirect" / row["file"]).exists():
                rows.append((float(row["ros_s"]), row["file"]))
    t0 = time.time()
    runner = DA360Runner(args.model_path, device=args.device)
    load_s = time.time() - t0
    frames, times = [], {"decode": [], "da360": [], "profile": []}
    for stamp, name in rows:
        t = time.time()
        bgr = cv2.imread(str(session / "equirect" / name))
        times["decode"].append(time.time() - t)
        t = time.time()
        depth, _ = runner.predict_depth(bgr, is_bgr=True)
        if runner.device.type == "cuda":
            import torch
            torch.cuda.synchronize()
        times["da360"].append(time.time() - t)
        t = time.time()
        res = profile(depth[::args.step, ::args.step], args.bins, tuple(args.band), args.min_r)
        times["profile"].append(time.time() - t)
        if res is None:
            continue
        prof, tilt, rms = res
        frames.append({"stamp": stamp, "file": name, "tilt_deg": round(tilt, 2), "floor_rms": round(rms, 4),
                       "profile": [None if not np.isfinite(v) else round(float(v), 3) for v in prof]})
    stats = {k: {"mean_ms": round(1000 * float(np.mean(v[1:] or v)), 1), "n": len(v)} for k, v in times.items() if v}
    out = Path(args.out) if args.out else session / "equirect_profiles.json"
    out.write_text(json.dumps({"model": Path(args.model_path).name, "network_hw": [runner.height, runner.width],
                               "band": list(args.band), "min_r": args.min_r, "load_s": round(load_s, 2),
                               "timing": stats, "frames": frames}), encoding="utf-8")
    print(f"{out}: {len(frames)}/{len(rows)} frames, model {Path(args.model_path).name} "
          f"{runner.height}x{runner.width} on {runner.device}, load {load_s:.1f} s, timing {stats}")
    if frames:
        tilts = [f["tilt_deg"] for f in frames]
        print(f"floor tilt median {np.median(tilts):.1f} deg, max {max(tilts):.1f}; "
              f"floor rms median {np.median([f['floor_rms'] for f in frames]):.3f} camera heights")


if __name__ == "__main__":
    main()
