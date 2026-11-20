#!/usr/bin/env python3
"""Installation seed for the front and rear MID-360S from doc/mid360s_2_insta360.png.

The drawing fixes only what lies between the two heads once each is leveled
with its own IMU: the rear lidar is yawed 180 deg about Z (mirrored) and sits
247.5 mm from the front lidar along -X. Roll and pitch of each lidar are not
taken from the drawing; refine-auto measures them from the IMU.

Writes:
    results/lidar_extrinsic.initial.json   installation seed, never touched by refine
    results/lidar_extrinsic.json           working extrinsic, reset to the seed
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

import extrinsic_io


def rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--results", type=Path, default=root / "results")
    args = ap.parse_args()
    cfg = yaml.safe_load((root / "config" / "rig.yaml").read_text(encoding="utf-8"))
    rig = extrinsic_io.load_rig(root)
    baseline_m = float(cfg["geometry"]["baseline_mm"]) * 1e-3
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    T_seed = np.eye(4)
    T_seed[:3, :3] = rz(180.0)
    T_seed[:3, 3] = [-baseline_m, 0.0, 0.0]

    args.results.mkdir(parents=True, exist_ok=True)
    initial_doc = extrinsic_io.installation_doc(rig, T_seed, now)
    initial_path = args.results / extrinsic_io.INITIAL_FILE
    extrinsic_io.write(initial_path, initial_doc)

    eye = np.eye(4)
    working_doc = {
        "schema": extrinsic_io.SCHEMA,
        "status": "installation_unleveled",
        "updated_utc": now,
        "method": "installation seed only; run refine-auto to level with the IMU and register",
        "lidars": extrinsic_io.lidars_block(rig),
        "convention": extrinsic_io.CONVENTION,
        "result": {
            "T_front_rear": {
                **extrinsic_io.pose(T_seed),
                "meaning": "Not leveled yet: roll and pitch of each lidar are missing until refine-auto runs.",
            },
            "installation_seed": extrinsic_io.pose(T_seed),
        },
        "T_front_rear": extrinsic_io.matrix(T_seed),
        "T_level_front": extrinsic_io.matrix(eye),
        "T_level_rear": extrinsic_io.matrix(T_seed),
        "T_front_rear_level": extrinsic_io.matrix(T_seed),
        "T_front_rear_level_seed": extrinsic_io.matrix(T_seed),
        "R_level_rear": extrinsic_io.matrix(eye),
    }
    working_path = args.results / extrinsic_io.EXTRINSIC_FILE
    extrinsic_io.write(working_path, working_doc)
    seed = initial_doc["installation_seed"]
    print(f"wrote {initial_path}  installation_seed rpy_deg={seed['rpy_deg']}  t_mm={seed['t_mm']}")
    print(f"wrote {working_path}  status=installation_unleveled; run refine-auto next")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
