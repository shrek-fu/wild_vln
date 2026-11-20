#!/usr/bin/env python3
"""Offline PromptDA on recorded office_loop8: equirect keyframes + map_cloud.npy.

Uses keyframe ``T_odom_da360`` and the saved map cloud (odom), crops the front
pinhole view from the equirect, builds a Livox-style lidar prompt, runs PromptDA.

  src/depth/da360_depth/.venv/bin/python src/depth/promptda_livox/run_office_loop8.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[2]
sys.path[:0] = [str(ROOT), str(WS / "src" / "scene_memory"), str(ROOT.parent / "da360_depth")]

from prompt_depth import ViewGeom, build_prompt_depth  # noqa: E402
from runner import PromptDARunner, default_ckpt  # noqa: E402
from scene_memory.perception.boxer_lift import equirect_remap, load_pinhole_views  # noqa: E402
from scene_memory.common.calib_io import R_BODY_DA360  # noqa: E402


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = Path("/media/foo/50B3920F3AB07555/stvln_data")
    p.add_argument("--scene", default=str(data / "maps/office_loop8_off/office_loop8_off_scene"))
    p.add_argument("--cloud", default=str(data / "maps/office_loop8_off/office_loop8_off_map_cloud.npy"))
    p.add_argument("--calib_dir", default=str(ROOT.parent / "da360_depth" / "calib"))
    p.add_argument("--view", default="front")
    p.add_argument("--ids", default="0,10,30", help="keyframe ids, comma list")
    p.add_argument("--save", default="/tmp/promptda_office_loop8")
    p.add_argument("--encoder", default="vits")
    p.add_argument("--ckpt", default="")
    p.add_argument("--voxel", type=float, default=0.1, help="m; downsample map cloud")
    p.add_argument("--device", default="")
    return p.parse_args()


def _voxel(xyz: np.ndarray, leaf: float) -> np.ndarray:
    if leaf <= 0 or len(xyz) == 0:
        return xyz
    keys = np.floor(xyz / leaf).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[np.sort(idx)]


def _T_odom_optical(T_odom_da360: np.ndarray, R_body_optical: np.ndarray) -> np.ndarray:
    """p_odom = T_odom_optical @ p_optical."""
    R_da360_optical = R_BODY_DA360.T @ np.asarray(R_body_optical, np.float64)
    T = np.eye(4)
    T[:3, :3] = R_da360_optical
    return np.asarray(T_odom_da360, np.float64) @ T


def _colorize(d: np.ndarray) -> np.ndarray:
    m = d > 0
    x = np.zeros_like(d, np.float32)
    if m.any():
        lo, hi = np.percentile(d[m], [2, 98])
        x[m] = np.clip((d[m] - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def main() -> None:
    args = _parse()
    scene = Path(args.scene)
    meta = json.loads((scene / "keyframes.json").read_text())
    keyframes = {int(k["id"]): k for k in meta["keyframes"]}
    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    views = {v["name"]: v for v in load_pinhole_views(args.calib_dir)}
    view = views[args.view]
    wh = int(view.get("width", 640)), int(view.get("height", 640))
    w, h = wh
    hfov = float(view["hfov_deg"])
    R = np.asarray(view["R_body_optical"], np.float64)

    cloud = _voxel(np.load(args.cloud).astype(np.float64).reshape(-1, 3), args.voxel)
    print(f"[office8] cloud {len(cloud)} pts after voxel {args.voxel} m; view {args.view} {w}x{h}")

    ckpt = Path(args.ckpt) if args.ckpt else default_ckpt(args.encoder)
    runner = PromptDARunner(ckpt, encoder=args.encoder, device=args.device)
    out = Path(args.save)
    out.mkdir(parents=True, exist_ok=True)

    for kid in ids:
        kf = keyframes[kid]
        equi = cv2.imread(str(scene / kf["file"]))
        if equi is None:
            raise SystemExit(f"missing {kf['file']}")
        eh, ew = equi.shape[:2]
        map_x, map_y, fx, fy, cx, cy = equirect_remap(eh, ew, R, h, w, hfov)
        bgr = cv2.remap(equi, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_WRAP)
        T_odom_optical = _T_odom_optical(np.asarray(kf["T_odom_da360"], np.float64), R)
        # ViewGeom expects T_level_optical with p_level = T @ p_optical; here level≡odom for this run.
        geom = ViewGeom(args.view, w, h, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
                        T_odom_optical)
        prompt, stats = build_prompt_depth(cloud, geom, project_scale=1 / 7, out_size=(w, h))
        print(f"[office8] kf{kid:04d} stamp={kf['stamp']:.2f} prompt {stats}")
        if stats["hits"] < 20:
            print(f"[office8] kf{kid:04d}: too few lidar hits, skip")
            continue
        depth = runner.predict(bgr, prompt)
        stem = f"kf{kid:04d}_{args.view}"
        cv2.imwrite(str(out / f"{stem}_rgb.jpg"), bgr)
        cv2.imwrite(str(out / f"{stem}_prompt.jpg"), _colorize(prompt))
        cv2.imwrite(str(out / f"{stem}_depth.jpg"), _colorize(depth))
        # side-by-side
        panel = np.hstack([bgr, _colorize(prompt), _colorize(depth)])
        cv2.imwrite(str(out / f"{stem}_panel.jpg"), panel)
        np.save(out / f"{stem}_depth.npy", depth)
        np.save(out / f"{stem}_prompt.npy", prompt)
        d = depth[depth > 0]
        print(f"[office8] kf{kid:04d} depth {d.min():.2f}..{d.max():.2f} m -> {out}/{stem}_*")

    print(f"[office8] done; open {out}/*_panel.jpg")


if __name__ == "__main__":
    main()
