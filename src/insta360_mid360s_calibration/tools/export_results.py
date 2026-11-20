#!/usr/bin/env python3
"""Copy the calibration results into results/export/ for other projects.

lidar_extrinsic.initial.json  lidar installation seed
lidar_extrinsic.json          latest lidar extrinsic
insta360_intrinsics.json      four view intrinsics
insta360_extrinsic.json       camera to front lidar extrinsic
insta360_projection/          one scene of the camera calibration: lidar points
                              projected into the panorama and the four views
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import extrinsic_io

FILES = (
    (extrinsic_io.INITIAL_FILE, "./calibrate.sh initial"),
    (extrinsic_io.EXTRINSIC_FILE, "./calibrate.sh initial"),
    ("insta360_intrinsics.json", "./calibrate.sh intrinsics"),
    ("insta360_extrinsic.json", "./calibrate.sh cam-calib"),
)
PROJECTION_DIR = "insta360_projection"
VIEWS = ("front", "left", "rear", "right")


def export_projection(root: Path, out: Path, scene: str | None) -> None:
    dst = out / PROJECTION_DIR
    src_doc = root / "results" / "insta360_extrinsic.json"
    if not src_doc.is_file():
        return
    calib = json.loads(src_doc.read_text(encoding="utf-8"))["calibration"]
    scenes = list(calib["scenes"])
    scene = scene or scenes[0]
    if scene not in scenes:
        raise SystemExit(f"scene {scene} is not in run {calib['run']}: {' '.join(scenes)}")
    overlay = root / "data" / "cam_calib" / calib["run"] / "overlay"
    names = ["equirect", *VIEWS]
    srcs = [overlay / f"{scene}_{n}.jpg" for n in names]
    missing = [p for p in srcs if not p.is_file()]
    if missing:
        print(f"skipped {PROJECTION_DIR}: missing {missing[0]}")
        return

    import cv2
    import numpy as np

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for name, src in zip(names, srcs):
        shutil.copy2(src, dst / f"{scene}_{name}.jpg")

    pano = cv2.imread(str(srcs[0]))
    w = pano.shape[1]
    tile = w // len(VIEWS)
    row = np.hstack([cv2.resize(cv2.imread(str(p)), (tile, tile), interpolation=cv2.INTER_AREA)
                     for p in srcs[1:]])

    def label(img, text, x, y, scale):
        for color, thick in (((0, 0, 0), 5), ((255, 255, 255), 2)):
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    label(pano, f"{scene}  equirect {pano.shape[1]}x{pano.shape[0]}  run {calib['run']}", 16, 40, 1.0)
    for i, v in enumerate(VIEWS):
        label(row, f"{v}  640x640  hfov 90", i * tile + 12, 34, 0.8)
    cv2.imwrite(str(dst / f"{scene}_overview.jpg"), np.vstack([pano, row]),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"exported {dst} ({scene}: overview, equirect, {', '.join(VIEWS)})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", help="camera calibration scene for insta360_projection/ (default: first scene)")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "export"
    out.mkdir(parents=True, exist_ok=True)
    kept = set()
    single = extrinsic_io.single(extrinsic_io.load_rig(root))
    for name, how in FILES:
        if single and name == extrinsic_io.INITIAL_FILE:
            continue
        src = root / "results" / name
        if not src.is_file():
            print(f"skipped {name}: not produced yet ({how})")
            continue
        shutil.copy2(src, out / name)
        kept.add(name)
        print(f"exported {out / name}")
    for path in out.iterdir():
        if path.is_file() and path.name not in kept:
            path.unlink()
            print(f"removed {path}")
    export_projection(root, out, args.scene)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
