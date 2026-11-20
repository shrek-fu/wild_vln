#!/usr/bin/env python3
"""Write results/insta360_intrinsics.json for the four Insta360 views.

The views are ideal pinhole images resampled from the stitched equirectangular
panorama by tools/insta360_views, so the intrinsics follow from
config/insta360_views.yaml and match the published camera_info exactly.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

import extrinsic_io

INTRINSICS_FILE = "insta360_intrinsics.json"
SCHEMA = "insta360_multi_mid360s/insta360_intrinsics/v1"
# Same order and yaw offsets as tools/insta360_views/insta360_views.cpp.
VIEWS = (("front", 0.0), ("left", 90.0), ("rear", 180.0), ("right", -90.0))


def load_params(root: Path) -> dict:
    cfg = yaml.safe_load((root / "config" / "insta360_views.yaml").read_text(encoding="utf-8"))
    return cfg["insta360_views"]["ros__parameters"]


def R_body_optical(yaw_deg: float) -> np.ndarray:
    """Columns are the optical x (right), y (down), z (forward) axes in the camera body frame."""
    y = math.radians(yaw_deg)
    x_opt = [math.sin(y), -math.cos(y), 0.0]
    y_opt = [0.0, 0.0, -1.0]
    z_opt = [math.cos(y), math.sin(y), 0.0]
    return np.array([x_opt, y_opt, z_opt]).T


def build(root: Path) -> dict:
    p = load_params(root)
    size = int(p["view_size"])
    hfov = float(p["view_hfov_deg"])
    front_yaw = float(p["front_yaw_deg"])
    prefix = str(p["frame_prefix"])
    f = 0.5 * size / math.tan(math.radians(hfov) / 2.0)
    c = 0.5 * size - 0.5
    views = {}
    for name, offset in VIEWS:
        yaw = front_yaw + offset
        R = R_body_optical(yaw)
        views[name] = {
            "frame_id": f"{prefix}_{name}_optical",
            "image_topic": f"/{prefix}/{name}/image/compressed",
            "camera_info_topic": f"/{prefix}/{name}/camera_info",
            "width": size,
            "height": size,
            "hfov_deg": hfov,
            "vfov_deg": hfov,
            "fx": extrinsic_io._r(f, 4),
            "fy": extrinsic_io._r(f, 4),
            "cx": c,
            "cy": c,
            "K": [[extrinsic_io._r(f, 4), 0.0, c], [0.0, extrinsic_io._r(f, 4), c], [0.0, 0.0, 1.0]],
            "distortion_model": "plumb_bob",
            "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            "yaw_deg": extrinsic_io._r(yaw, 3),
            "R_body_optical": [[extrinsic_io._r(v, 9) for v in row] for row in R],
        }
    return {
        "schema": SCHEMA,
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "camera": "Insta360 X4 Air",
        "source": (
            "ideal pinhole views resampled from the Media SDK real-time stitched equirectangular panorama; "
            "no lens distortion remains. Derived from config/insta360_views.yaml, not measured."
        ),
        "convention": {
            "body_frame": (
                f"{prefix}: camera body frame of the stitched panorama (FlowState off). "
                "X forward = panorama centre, Y left, Z up"
            ),
            "optical_frame": "x right, y down, z forward (ROS optical)",
            "pixel": "principal point in pixel-centre coordinates, (0, 0) is the centre of the top-left pixel",
            "yaw_deg": "view direction about body Z, positive to the left; includes front_yaw_deg",
            "R_body_optical": "p_body = R_body_optical * p_optical",
            "extrinsic": "camera to lidar extrinsic is not calibrated yet",
        },
        "stream": {
            "stream_resolution": str(p["stream_resolution"]),
            "stitch_type": str(p["stitch_type"]),
            "equirect_width": int(p["equirect_width"]),
            "equirect_height": int(p["equirect_width"]) // 2,
            "front_yaw_deg": front_yaw,
        },
        "views": views,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    doc = build(root)
    path = root / "results" / INTRINSICS_FILE
    extrinsic_io.write(path, doc)
    v = doc["views"]["front"]
    print(f"wrote {path}  {v['width']}x{v['height']} hfov {v['hfov_deg']} deg  fx=fy={v['fx']}  cx=cy={v['cx']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
