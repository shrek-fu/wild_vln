#!/usr/bin/env python3
"""Fit our dual-fisheye -> equirect model against MediaSDK output of a recorded session.

The session keeps the raw Camera SDK stream (raw/seg_000/stream_0.h264 + camera.json) and the
MediaSDK-stitched equirect frames (equirect/<ros_s>.jpg), so both views of the same frame exist.

  fit_stitch.py extract <session> [--n 6]        decode matching fisheye frames into <session>/stitch_fit/
  fit_stitch.py search  <session>                axis-convention search per lens (NCC vs MediaSDK)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

EQ_W, EQ_H = 960, 480


def parse_v3(calibration: list[str]) -> dict:
    """Third Camera SDK offset string: per lens xi fx fy cx cy yaw pitch roll tx ty tz k1 k2 k3 p1 p2 W H id."""
    for s in calibration:
        v = [float(t) for t in s.split("_") if t]
        if len(v) >= 39 and int(v[0]) == 2:
            lenses = []
            for i in range(2):
                b = 1 + 19 * i
                names = "xi fx fy cx cy yaw pitch roll tx ty tz k1 k2 k3 p1 p2 W H id".split()
                lenses.append(dict(zip(names, v[b:b + 19])))
            return {"lenses": lenses}
    raise ValueError("no v3 calibration string")


def fit_dir(session: Path) -> Path:
    d = session / "stitch_fit"
    d.mkdir(exist_ok=True)
    return d


def cmd_extract(session: Path, n: int) -> None:
    raw = session / "raw" / "seg_000"
    seq_by_ts = {r["device_ts"]: int(r["seq"]) for r in csv.DictReader(open(raw / "video_index.csv"))}
    ts_by_ros = {f"{float(r['ros_s']):.6f}": r["device_ts"] for r in csv.DictReader(open(raw / "frames.csv"))}
    jpgs = sorted((session / "equirect").glob("*.jpg"))
    pick = [jpgs[i] for i in np.linspace(0, len(jpgs) - 1, n).astype(int)]
    out = fit_dir(session)
    pairs = []
    for j in pick:
        ts = ts_by_ros.get(j.stem)
        if ts is None or ts not in seq_by_ts:
            print("no frame for", j.name)
            continue
        seq = seq_by_ts[ts]
        dst = out / f"fish_{seq:05d}.png"
        if not dst.exists():
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "h264", "-i", str(raw / "stream_0.h264"),
                            "-vf", f"select=eq(n\\,{seq})", "-frames:v", "1", str(dst)], check=True)
        pairs.append({"seq": seq, "fisheye": dst.name, "equirect": j.name})
        print(j.name, "->", dst.name)
    json.dump(pairs, open(out / "pairs.json", "w"), indent=2)


def load_pairs(session: Path):
    out = fit_dir(session)
    pairs = json.load(open(out / "pairs.json"))
    data = []
    for p in pairs:
        fish = cv2.imread(str(out / p["fisheye"]))
        eq = cv2.imread(str(session / "equirect" / Path(p["equirect"]).name))
        eq = cv2.resize(eq, (EQ_W, EQ_H), interpolation=cv2.INTER_AREA)
        data.append((fish, eq))
    return data


def eq_dirs(w: int, h: int) -> np.ndarray:
    lon = ((np.arange(w) + 0.5) / w) * 2 * np.pi - np.pi
    lat = np.pi / 2 - ((np.arange(h) + 0.5) / h) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    # x forward (image centre), y left, z up; longitude grows to the right, i.e. clockwise from above.
    return np.stack([np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def mei_project(p: np.ndarray, lens: dict, k3: bool = True):
    """Unified omnidirectional (Mei) model with radial k1 k2 k3 and tangential p1 p2; p in lens frame, z = axis."""
    n = p / np.linalg.norm(p, axis=-1, keepdims=True)
    z = n[..., 2] + lens["xi"]
    x, y = n[..., 0] / z, n[..., 1] / z
    r2 = x * x + y * y
    rad = 1 + lens["k1"] * r2 + lens["k2"] * r2 ** 2 + (lens["k3"] * r2 ** 3 if k3 else 0)
    p1, p2 = lens["p1"], lens["p2"]
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = lens["fx"] * xd + lens["cx"]
    v = lens["fy"] * yd + lens["cy"]
    return u, v, n[..., 2]


def rot(axis: str, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def axis_rotations() -> list[np.ndarray]:
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            m = np.zeros((3, 3))
            for r, (c, s) in enumerate(zip(perm, signs)):
                m[r, c] = s
            if np.linalg.det(m) > 0:
                mats.append(m)
    return mats


def sample(fish: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return cv2.remap(fish, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    a = a[mask].astype(np.float64)
    b = b[mask].astype(np.float64)
    if a.size < 100:
        return -1.0
    a -= a.mean()
    b -= b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def to_stream(u, v, lens_idx: int, scale: float, crop: float):
    """Calibration pixels (15360x7680, two 7680 squares) -> stream pixels (two half-width squares)."""
    half = 7680.0
    u_local = u - lens_idx * half
    return (u_local - crop) * scale, (v - crop) * scale


def score_lens(data, dirs, R, lens, lens_idx, scale, crop, max_angle_deg=80.0):
    p = dirs @ R  # world -> lens: p_lens = R^T d
    u, v, cz = mei_project(p, lens)
    us, vs = to_stream(u, v, lens_idx, scale, crop)
    half_w = data[0][0].shape[1] / 2
    us = us + lens_idx * half_w
    mask = cz > np.cos(np.radians(max_angle_deg))
    vals = []
    for fish, eq in data:
        g = cv2.cvtColor(fish, cv2.COLOR_BGR2GRAY)
        s = sample(g, us, vs)
        e = cv2.cvtColor(eq, cv2.COLOR_BGR2GRAY)
        vals.append(ncc(s, e, mask & (s > 0)))
    return float(np.mean(vals))


def cmd_search(session: Path) -> None:
    calib = json.load(open(session / "raw" / "seg_000" / "camera.json"))["calibration"]
    lenses = parse_v3(calib)["lenses"]
    data = load_pairs(session)
    dirs = eq_dirs(EQ_W, EQ_H)
    W = data[0][0].shape[1]
    scales = {"full": (W / 2) / 7680.0, "crop7424": (W / 2) / 7424.0}
    for li, lens in enumerate(lenses):
        results = []
        for base in axis_rotations():
            for sign in (1, -1):
                R = base @ rot("z", sign * lens["roll"])
                for sname, sc in scales.items():
                    crop = 0.0 if sname == "full" else 128.0
                    results.append((score_lens(data[:2], dirs, R, lens, li, sc, crop), sname, sign, base))
        results.sort(key=lambda r: -r[0])
        print(f"lens {li}: top candidates")
        for sc, sname, sign, base in results[:5]:
            print(f"  ncc={sc:.3f} scale={sname} roll_sign={sign} base={base.astype(int).tolist()}")


BASES = {
    0: np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float),
    1: np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], float),
}
CROP = 128.0  # (7680 - 7424) / 2
BLEND_BAND = 0.05  # cos-difference half width of the seam feather (~±3 deg at the seam)


def rodrigues(w: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(w, float).reshape(3, 1))
    return R


PITCH_SIGN = {0: 1.0, 1: -1.0}


def lens_rotation(lens: dict, li: int, correction=np.zeros(3)) -> np.ndarray:
    """Columns: lens x, y, z (optical axis) in the rig frame (x forward, y left, z up)."""
    return (BASES[li] @ rot("y", -lens["yaw"]) @ rot("x", PITCH_SIGN[li] * lens["pitch"])
            @ rot("z", lens["roll"]) @ rodrigues(correction))


def lens_maps(dirs, lens, li, W, params):
    """params: correction rotvec(3), log-scale, du, dv (stream px)."""
    R = lens_rotation(lens, li, params[:3])
    p = dirs @ R
    u, v, cz = mei_project(p, lens)
    sc = (W / 2) / (7680.0 - 2 * CROP) * np.exp(params[3])
    us, vs = to_stream(u, v, li, sc, CROP)
    return us + li * (W / 2) + params[4], vs + params[5], cz


def grad(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def cmd_refine(session: Path) -> None:
    from scipy.optimize import minimize

    calib = json.load(open(session / "raw" / "seg_000" / "camera.json"))["calibration"]
    lenses = parse_v3(calib)["lenses"]
    data = load_pairs(session)
    dirs = eq_dirs(EQ_W, EQ_H)
    W = data[0][0].shape[1]
    fish_g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f, _ in data]
    eq_grad = [grad(e) for _, e in data]
    result = {}
    for li, lens in enumerate(lenses):
        def cost(x):
            us, vs, cz = lens_maps(dirs, lens, li, W, x)
            mask = cz > np.cos(np.radians(80))
            vals = []
            for g, eg in zip(fish_g, eq_grad):
                s = sample(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), us, vs)
                vals.append(ncc(grad(s), eg, mask & (s[..., 0] > 0)))
            return -float(np.mean(vals))

        x0 = np.zeros(6)
        print(f"lens {li}: start cost {cost(x0):.3f}")
        best = minimize(cost, x0, method="Powell",
                        options={"xtol": 1e-4, "ftol": 1e-4, "maxfev": 1500})
        x = best.x
        print(f"lens {li}: cost {best.fun:.3f} rotvec_deg {np.degrees(x[:3]).round(3).tolist()} "
              f"scale x{np.exp(x[3]):.4f} du {x[4]:.2f} dv {x[5]:.2f}")
        result[li] = x.tolist()
    json.dump(result, open(fit_dir(session) / "refine.json", "w"), indent=2)
    cmd_render(session)


def cmd_render(session: Path) -> None:
    calib = json.load(open(session / "raw" / "seg_000" / "camera.json"))["calibration"]
    lenses = parse_v3(calib)["lenses"]
    data = load_pairs(session)
    path = fit_dir(session) / "refine.json"
    params = {int(k): np.array(v) for k, v in json.load(open(path)).items()} if path.exists() else \
        {0: np.zeros(6), 1: np.zeros(6)}
    dirs = eq_dirs(EQ_W, EQ_H)
    W = data[0][0].shape[1]
    maps = [lens_maps(dirs, lens, li, W, params[li]) for li, lens in enumerate(lenses)]
    w0 = np.clip(0.5 + (maps[0][2] - maps[1][2]) / (2 * BLEND_BAND), 0, 1)[..., None]
    scores = []
    for k, (fish, eq) in enumerate(data):
        s0, s1 = (sample(fish, m[0], m[1]).astype(np.float32) for m in maps)
        out = (w0 * s0 + (1 - w0) * s1).round().clip(0, 255).astype(np.uint8)
        g_out, g_eq = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (out, eq))
        full = np.ones(g_out.shape, bool)
        scores.append((ncc(g_out, g_eq, full), float(np.abs(g_out.astype(int) - g_eq).mean())))
        if k == 0:
            diff = cv2.absdiff(out, eq)
            cv2.imwrite(str(fit_dir(session) / "render_ours.jpg"), out)
            cv2.imwrite(str(fit_dir(session) / "render_compare.jpg"), np.vstack([out, eq, diff]))
    for k, (c, mad) in enumerate(scores):
        print(f"frame {k}: full-image NCC {c:.4f}  mean |diff| {mad:.2f} grey levels")
    print("wrote", fit_dir(session) / "render_compare.jpg")


def cmd_compare(session: Path, ours: Path) -> None:
    """Full-resolution check of stitch_segment output (eq_<seq>.ppm, RGB) against Media SDK."""
    pairs = json.load(open(fit_dir(session) / "pairs.json"))
    for p in pairs:
        f = ours / f"eq_{p['seq']}.ppm"
        if not f.exists():
            continue
        mine = cv2.imread(str(f))  # cv2 swaps PPM RGB to BGR
        ref = cv2.imread(str(session / "equirect" / Path(p["equirect"]).name))
        g0, g1 = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (mine, ref))
        full = np.ones(g0.shape, bool)
        print(f"seq {p['seq']:5d}: NCC {ncc(g0, g1, full):.4f}  grad-NCC {ncc(grad(mine), grad(ref), full):.4f}  "
              f"mean |diff| {np.abs(g0.astype(int) - g1).mean():.2f}")
        if p is pairs[0]:
            cv2.imwrite(str(ours / "compare.jpg"), np.vstack([mine, ref, cv2.absdiff(mine, ref) * 4]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "search", "refine", "render", "compare"])
    ap.add_argument("session", type=Path)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--ours", type=Path, help="compare: directory with eq_<seq>.ppm")
    a = ap.parse_args()
    {"extract": lambda: cmd_extract(a.session, a.n), "search": lambda: cmd_search(a.session),
     "refine": lambda: cmd_refine(a.session), "render": lambda: cmd_render(a.session),
     "compare": lambda: cmd_compare(a.session, a.ours)}[a.cmd]()


if __name__ == "__main__":
    main()
