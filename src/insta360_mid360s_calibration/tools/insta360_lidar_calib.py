#!/usr/bin/env python3
"""Insta360 panorama to front lidar calibration around direct_visual_lidar_calibration.

  initial <run_dir>   write results.init_T_lidar_camera into <run_dir>/preprocessed/calib.json
  import  <run_dir>   read results.T_lidar_camera, write results/insta360_extrinsic.json and overlays

Frames:
  front_lidar   points of the calibration cloud (the primary lidar)
  insta360      camera body of the stitched panorama: X forward = panorama centre, Y left, Z up
  equirect      the solver's camera frame: z forward = panorama centre, x right, y down
  <view>        optical frames of the four views, see results/insta360_intrinsics.json

The solver estimates T_front_equirect (its T_lidar_camera, p_front = T * p_equirect).
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

import extrinsic_io
import write_insta360_intrinsics as intr

ROOT = Path(__file__).resolve().parents[1]
RESULT_FILE = "insta360_extrinsic.json"
SCHEMA = "insta360_multi_mid360s/insta360_extrinsic/v1"
SEARCH_WIDTH = 720
# A 720x360 panorama has 259k pixels; more points per scene only slow the search down.
SEARCH_POINTS = 300_000


def T_from_R(R: np.ndarray, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def Rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


T_BODY_EQUIRECT = T_from_R(intr.R_body_optical(0.0))


def tum_from_T(T: np.ndarray) -> list:
    q = rot_to_quat(T[:3, :3])
    return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), *q]


def T_from_tum(v) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = [float(a) for a in v]
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    return T_from_R(R, (x, y, z))


def rot_to_quat(R: np.ndarray) -> list:
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw, qx, qy, qz = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return [float(qx), float(qy), float(qz), float(qw)]


def rotation_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))))


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """xyz and intensity from the solver's binary little-endian PLY."""
    with path.open("rb") as f:
        props, count = [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property"):
                _, typ, name = line.split()
                props.append((name, {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}[typ]))
            elif line == "end_header":
                break
        data = np.fromfile(f, dtype=np.dtype(props), count=count)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    inten = data["intensity"].astype(np.float64) if "intensity" in data.dtype.names else np.ones(count)
    return xyz, inten


def load_level(root: Path) -> tuple[np.ndarray, np.ndarray | None]:
    doc = json.loads((root / "results" / extrinsic_io.EXTRINSIC_FILE).read_text(encoding="utf-8"))
    rear = doc.get("T_level_rear")
    return np.array(doc["T_level_front"], dtype=float), None if rear is None else np.array(rear, dtype=float)


def camera_seed(root: Path, T_level_rear: np.ndarray | None) -> tuple[np.ndarray, str]:
    """Panorama centre in lidar_level before the search, and how it was obtained.

    Dual rig: above the midpoint of the two lidars by geometry.camera_dz_mm. Single lidar:
    geometry.camera_from_lidar_mm, the camera centre relative to the lidar in lidar_level
    (x = lidar forward after leveling, z up)."""
    geo = yaml.safe_load((root / "config" / "rig.yaml").read_text(encoding="utf-8")).get("geometry", {})
    if T_level_rear is None:
        off = np.asarray(geo.get("camera_from_lidar_mm", [0.0, 0.0, 0.0]), dtype=float) / 1000.0
        return off, f"lidar + camera_from_lidar_mm {np.round(off * 1000).tolist()}"
    return 0.5 * T_level_rear[:3, 3] + np.array([0.0, 0.0, float(geo["camera_dz_mm"]) / 1000.0]), \
        "lidar midpoint + camera_dz_mm"


def seed_candidates(c0: np.ndarray, xy_span: float, dz_span: float, ring: bool) -> list:
    """Grid around the seed (xy 50 mm, z 30 mm steps); with ``ring`` also the seed's horizontal offset
    turned to 12 headings, for a lidar whose yaw relative to the camera is unknown."""
    xy = np.arange(-xy_span, xy_span + 1e-9, 0.05) if xy_span > 0 else [0.0]
    dz = np.arange(-dz_span, dz_span + 1e-9, 0.03) if dz_span > 0 else [0.0]
    out = [c0 + np.array([x, y, z]) for x in xy for y in xy for z in dz]
    r = float(np.hypot(c0[0], c0[1]))
    if ring and r > 0.02:
        for a in np.radians(np.arange(0, 360, 30)):
            out += [np.array([r * np.cos(a), r * np.sin(a), c0[2] + z]) for z in dz]
    return out


def equirect_project(p_body: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coordinates of body-frame points (X forward, Y left, Z up)."""
    r = np.linalg.norm(p_body, axis=1)
    lon = np.arctan2(p_body[:, 1], p_body[:, 0])
    lat = np.arcsin(np.clip(p_body[:, 2] / np.maximum(r, 1e-9), -1.0, 1.0))
    u = width * (0.5 - lon / (2 * math.pi))
    v = height * (0.5 - lat / math.pi)
    return u, v, r


def render_lidar(p_body: np.ndarray, inten: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-point intensity panorama and its valid mask."""
    u, v, r = equirect_project(p_body, width, height)
    ui = np.clip(u.astype(int), 0, width - 1)
    vi = np.clip(v.astype(int), 0, height - 1)
    order = np.argsort(-r)
    img = np.zeros((height, width), np.float32)
    img[vi[order], ui[order]] = inten[order]
    valid = np.zeros((height, width), np.uint8)
    valid[vi, ui] = 1
    kernel = np.ones((3, 3), np.uint8)
    filled = cv2.dilate(img, kernel)
    img = np.where(valid > 0, img, filled)
    valid = cv2.dilate(valid, kernel)
    return img, valid.astype(bool)


def edges(img: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(img.astype(np.float32), (5, 5), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def yaw_scores(cam_gray: np.ndarray, lidar_img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Masked edge correlation for every horizontal shift, via FFT along columns."""
    ec = edges(cam_gray)
    el = edges(lidar_img)
    m = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    el = el * m
    ec = (ec - ec.mean()) / (ec.std() + 1e-9)
    el = np.where(m > 0, (el - el[m > 0].mean()) / (el[m > 0].std() + 1e-9), 0.0)
    # score[s] = sum ec(x) * el(x - s): the lidar image rolled right by s
    F = np.fft.rfft(ec, axis=1) * np.conj(np.fft.rfft(el, axis=1))
    corr = np.fft.irfft(F, n=ec.shape[1], axis=1).sum(axis=0)
    return corr / max(m.sum(), 1.0)


def preprocessed(run: Path) -> Path:
    return run / "preprocessed"


def scenes(run: Path) -> list:
    meta = json.loads((preprocessed(run) / "calib.json").read_text(encoding="utf-8"))
    return list(meta["meta"]["bag_names"])


def cmd_initial(run: Path, dz_span_mm: float, xy_span_mm: float | None, ring: bool = False) -> int:
    pre = preprocessed(run)
    calib = json.loads((pre / "calib.json").read_text(encoding="utf-8"))
    T_level_front, T_level_rear = load_level(ROOT)
    T_front_level = np.linalg.inv(T_level_front)
    single = T_level_rear is None
    c0, how = camera_seed(ROOT, T_level_rear)
    if xy_span_mm is None:
        xy_span_mm = 100.0 if single else 0.0
    W, H = SEARCH_WIDTH, SEARCH_WIDTH // 2
    data = []
    for name in scenes(run):
        cam = cv2.imread(str(pre / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_AREA)
        xyz, inten = read_ply(pre / f"{name}.ply")
        if len(xyz) > SEARCH_POINTS:
            keep = np.random.default_rng(0).choice(len(xyz), SEARCH_POINTS, replace=False)
            xyz, inten = xyz[keep], inten[keep]
        p_level = xyz @ T_level_front[:3, :3].T + T_level_front[:3, 3]
        lo, hi = np.percentile(inten, [2, 98])
        data.append((name, cam, p_level, np.clip((inten - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0))
    best = None
    candidates = seed_candidates(c0, xy_span_mm / 1000.0, dz_span_mm / 1000.0, ring=single and ring)
    print(f"initial guess search: seed {how}, {len(candidates)} camera positions x {len(data)} scene(s)")
    for i, c in enumerate(candidates):
        if i % 50 == 0:
            print(f"  {i}/{len(candidates)}", flush=True)
        total = np.zeros(W)
        for _, cam, p_level, inten in data:
            img, valid = render_lidar(p_level - c, inten, W, H)
            total += yaw_scores(cam, img, valid)
        s = int(np.argmax(total))
        if best is None or total[s] > best[1]:
            best = (c, total[s], s, total.copy())
    c, peak, s, total = best
    yaw = (360.0 * s / W + 180.0) % 360.0 - 180.0
    away = np.abs(((np.arange(W) - s + W / 2) % W) - W / 2) > W * 15 / 360
    second = float(total[away].max())
    T_level_body = T_from_R(Rz(yaw), c)
    T_front_eq = T_front_level @ T_level_body @ T_BODY_EQUIRECT
    calib.setdefault("results", {})["init_T_lidar_camera"] = tum_from_T(T_front_eq)
    (pre / "calib.json").write_text(json.dumps(calib, indent=2) + "\n", encoding="utf-8")
    print(f"initial guess: camera at {np.round(c * 1000).tolist()} mm in lidar_level ({how} searched), "
          f"yaw {yaw:.1f} deg (body X in lidar_level)")
    print(f"yaw search peak {peak:.3f}, best other peak {second:.3f} (ratio {peak / max(second, 1e-9):.2f})")
    if peak < 1.2 * second:
        print("WARNING: the yaw peak is not distinct. Check the overlay, or run ./calibrate.sh cam-initial-manual")
    pose = extrinsic_io.pose(T_front_eq @ np.linalg.inv(T_BODY_EQUIRECT))
    print(f"init T_front_insta360 rpy_deg={pose['rpy_deg']} t_mm={pose['t_mm']}")
    for name, *_ in data:
        overlay_panorama(pre, name, T_front_eq, run / "overlay" / f"{name}_initial.jpg")
    print(f"overlays: {run / 'overlay'}")
    return 0


def overlay_panorama(pre: Path, name: str, T_front_eq: np.ndarray, out: Path) -> None:
    img = cv2.imread(str(pre / f"{name}.png"), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    xyz, _ = read_ply(pre / f"{name}.ply")
    T_body_front = np.linalg.inv(T_front_eq @ np.linalg.inv(T_BODY_EQUIRECT))
    p = xyz @ T_body_front[:3, :3].T + T_body_front[:3, 3]
    draw_depth(img, *equirect_project(p, W, H))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def draw_depth(img: np.ndarray, u: np.ndarray, v: np.ndarray, d: np.ndarray, max_points: int = 120000) -> None:
    H, W = img.shape[:2]
    keep = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (d > 0.3)
    idx = np.flatnonzero(keep)
    if idx.size > max_points:
        idx = np.random.default_rng(0).choice(idx, max_points, replace=False)
    color = cv2.applyColorMap(np.clip(255 * (1 - np.log(d[idx]) / np.log(15.0)), 0, 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)
    img[v[idx].astype(int), u[idx].astype(int)] = color.reshape(-1, 3)


def first_images(bag_dir: Path, topics: list) -> dict:
    """First CompressedImage per topic, decoded."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=""), rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    out = {}
    while reader.has_next() and len(out) < len(topics):
        topic, raw, _ = reader.read_next()
        if topic in out:
            continue
        msg = deserialize_message(raw, CompressedImage)
        out[topic] = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    return out


def cmd_import(run: Path) -> int:
    pre = preprocessed(run)
    calib = json.loads((pre / "calib.json").read_text(encoding="utf-8"))
    res = calib.get("results", {})
    if "T_lidar_camera" not in res:
        raise SystemExit(f"{pre / 'calib.json'} has no results.T_lidar_camera; the solver did not finish")
    T_front_eq = T_from_tum(res["T_lidar_camera"])
    T_front_eq_init = T_from_tum(res["init_T_lidar_camera"]) if "init_T_lidar_camera" in res else None
    T_level_front, _ = load_level(ROOT)
    T_front_body = T_front_eq @ np.linalg.inv(T_BODY_EQUIRECT)
    T_level_body = T_level_front @ T_front_body
    t_nid_level = T_level_body[:3, 3].copy()
    translation = translation_source(ROOT)
    if translation == "geometry":
        # Far scenes leave the panorama-lidar translation weakly observed: keep the NID rotation
        # and take the translation from the rig geometry (lidar_level frame).
        T_level_body[:3, 3] = np.asarray(rig_geometry(ROOT)["camera_from_lidar_mm"], dtype=float) / 1000.0
        T_front_body = np.linalg.inv(T_level_front) @ T_level_body
        T_front_eq = T_front_body @ T_BODY_EQUIRECT
    intr_doc = json.loads((ROOT / "results" / intr.INTRINSICS_FILE).read_text(encoding="utf-8"))
    views = {}
    for name, v in intr_doc["views"].items():
        T_front_view = T_front_body @ T_from_R(np.array(v["R_body_optical"], dtype=float))
        views[name] = {
            "frame_id": v["frame_id"],
            "T_front_view": {**extrinsic_io.pose(T_front_view), "meaning": f"{name} view optical frame in the front lidar frame"},
            "matrix": extrinsic_io.matrix(T_front_view),
        }
    run_meta = json.loads((run / "run.json").read_text(encoding="utf-8")) if (run / "run.json").is_file() else {}
    doc = {
        "schema": SCHEMA,
        "status": "nid_calibrated",
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "direct_visual_lidar_calibration (NID) on the stitched equirectangular panorama; "
                  "initial guess from rig geometry and a yaw search"
                  + ("; rotation from NID, translation from config/rig.yaml geometry.camera_from_lidar_mm"
                     if translation == "geometry" else ""),
        "translation_source": translation,
        "reference": "front_lidar (primary)",
        "convention": {
            "equation": "p_front = T_front_X * p_X",
            "insta360": "camera body of the stitched panorama: X forward = panorama centre, Y left, Z up",
            "views": "optical frames of results/insta360_intrinsics.json: x right, y down, z forward",
            "rpy_deg": extrinsic_io.CONVENTION["rpy_deg"],
            "t_mm": extrinsic_io.CONVENTION["t_mm"],
        },
        "result": {
            "T_front_insta360": {**extrinsic_io.pose(T_front_body), "meaning": "camera body in the front lidar frame"},
            "T_level_insta360": {**extrinsic_io.pose(T_level_body), "meaning": "camera body in lidar_level (gravity-aligned)"},
            "nid_translation_level_mm": [extrinsic_io._r(1000 * x, 2) for x in t_nid_level],
        },
        "calibration": {
            "run": run.name,
            "scenes": scenes(run),
            "cloud_topic": calib["meta"].get("points_topic", ""),
            "image_topic": calib["meta"].get("image_topic", ""),
            "stitch_type": run_meta.get("stitch_type", ""),
            "overlays": str((run / "overlay").relative_to(ROOT)),
        },
        "views": views,
        "T_front_insta360": extrinsic_io.matrix(T_front_body),
        "T_level_insta360": extrinsic_io.matrix(T_level_body),
    }
    if T_front_eq_init is not None:
        T_front_body_init = T_front_eq_init @ np.linalg.inv(T_BODY_EQUIRECT)
        d = np.linalg.inv(T_front_body_init) @ T_front_body
        doc["result"]["initial_guess"] = {**extrinsic_io.pose(T_front_body_init), "meaning": "T_front_insta360 before NID"}
        doc["result"]["refined_minus_initial"] = {
            "rotation_deg": extrinsic_io._r(rotation_deg(d[:3, :3]), 3),
            "t_mm": [extrinsic_io._r(1000 * x, 2) for x in (T_front_body[:3, 3] - T_front_body_init[:3, 3])],
        }
    path = ROOT / "results" / RESULT_FILE
    extrinsic_io.write(path, doc)
    p = doc["result"]["T_front_insta360"]
    print(f"wrote {path}")
    print(f"T_front_insta360 rpy_deg={p['rpy_deg']} t_mm={p['t_mm']}")
    if "refined_minus_initial" in doc["result"]:
        r = doc["result"]["refined_minus_initial"]
        print(f"refined minus initial: {r['rotation_deg']} deg, t_mm={r['t_mm']}")
    print(f"translation from {translation}: T_level_insta360 t_mm={doc['result']['T_level_insta360']['t_mm']} "
          f"(NID alone: {doc['result']['nid_translation_level_mm']})")
    write_overlays(run, T_front_eq, T_front_body, intr_doc)
    write_lidar_check(run, T_front_body)
    return 0


def rig_geometry(root: Path) -> dict:
    return yaml.safe_load((root / "config" / "rig.yaml").read_text(encoding="utf-8")).get("geometry", {})


def translation_source(root: Path) -> str:
    """config/rig.yaml camera_lidar.translation: nid (default) or geometry."""
    cfg = yaml.safe_load((root / "config" / "rig.yaml").read_text(encoding="utf-8")).get("camera_lidar", {})
    src = str(cfg.get("translation", "nid"))
    if src not in ("nid", "geometry"):
        raise SystemExit(f"camera_lidar.translation must be nid or geometry, got {src}")
    return src


def write_lidar_check(run: Path, T_front_body: np.ndarray) -> None:
    """Mark the lidar origin in each panorama. The lidar is in view below the panorama centre, so
    the cross should sit inside its glass dome; this checks the translation, which NID observes weakly."""
    pre = preprocessed(run)
    p = np.linalg.inv(T_front_body)[:3, 3][None, :]
    for name in scenes(run):
        img = cv2.imread(str(pre / f"{name}.png"))
        if img is None:
            continue
        h, w = img.shape[:2]
        u, v, _ = equirect_project(p, w, h)
        u, v = int(u[0]), int(v[0])
        cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
        y0, x0 = max(v - 150, 0), max(u - 250, 0)
        cv2.imwrite(str(run / "overlay" / f"{name}_lidar_origin.jpg"), img[y0:v + 150, x0:u + 250])


def write_overlays(run: Path, T_front_eq: np.ndarray, T_front_body: np.ndarray, intr_doc: dict) -> None:
    pre = preprocessed(run)
    out_dir = run / "overlay"
    for name in scenes(run):
        overlay_panorama(pre, name, T_front_eq, out_dir / f"{name}_equirect.jpg")
        bag = run / "bags" / name
        if not bag.exists():
            continue
        topics = {v: d["image_topic"] for v, d in intr_doc["views"].items()}
        images = first_images(bag, list(topics.values()))
        xyz, _ = read_ply(pre / f"{name}.ply")
        for view, d in intr_doc["views"].items():
            img = images.get(d["image_topic"])
            if img is None:
                continue
            T_view_front = np.linalg.inv(T_front_body @ T_from_R(np.array(d["R_body_optical"], dtype=float)))
            p = xyz @ T_view_front[:3, :3].T + T_view_front[:3, 3]
            z = p[:, 2]
            front = z > 0.3
            u = d["fx"] * p[front, 0] / z[front] + d["cx"] + 0.5
            v = d["fy"] * p[front, 1] / z[front] + d["cy"] + 0.5
            draw_depth(img, u, v, np.linalg.norm(p[front], axis=1))
            cv2.imwrite(str(out_dir / f"{name}_{view}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"overlays: {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("initial")
    a.add_argument("run", type=Path)
    a.add_argument("--dz-span-mm", type=float, default=90.0, help="search camera height +- this around the seed")
    a.add_argument("--xy-span-mm", type=float, default=None,
                   help="search camera x/y +- this around the seed (default 100 single lidar, 0 dual)")
    a.add_argument("--ring", action="store_true",
                   help="single lidar: also try the seed's horizontal offset in 12 headings (lidar yaw unknown)")
    b = sub.add_parser("import")
    b.add_argument("run", type=Path)
    args = ap.parse_args()
    if args.cmd == "initial":
        return cmd_initial(args.run.resolve(), args.dz_span_mm, args.xy_span_mm, args.ring)
    return cmd_import(args.run.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
