"""Lidar naming and the layout of results/lidar_extrinsic*.json.

Lidars are named by mounting position (front_lidar, rear_lidar), never by IP.
The IP appears only in config/rig.yaml. The Livox driver names its topics by
IP; ./calibrate.sh bringup remaps them to the position names below, so every
other tool and every result file is IP-free.

Each result file starts with a human-readable block (Euler angles in degrees,
translation in millimeters) and ends with the 4x4 matrices the tools load.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

SCHEMA = "insta360_multi_mid360s/lidar_extrinsic/v3"


def capture_root(package_root: Path | None = None) -> Path:
    """Workspace capture tree: ``<ws>/data/calib_capture`` (override with ``CALIB_DATA_ROOT``).

    Layout: ``lidar/{bags,clouds}``, ``camera/{bags,runs,stills}``. Not under ``src/``.
    """
    env = os.environ.get("CALIB_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parents[1]
    return (root.parents[1] / "data" / "calib_capture").resolve()


def clouds_dir(package_root: Path | None = None) -> Path:
    return capture_root(package_root) / "lidar" / "clouds"
EXTRINSIC_FILE = "lidar_extrinsic.json"
INITIAL_FILE = "lidar_extrinsic.initial.json"
LEVEL_FRAME = "lidar_level"

CONVENTION = {
    "equation": "p_front = T_front_rear * p_rear",
    "level_frame": (
        f"{LEVEL_FRAME}: origin at the front lidar, Z against gravity, yaw of the front lidar. "
        "p_level = T_level_front * p_front = T_level_rear * p_rear"
    ),
    "rpy_deg": "[roll, pitch, yaw] in degrees, R = Rz(yaw) * Ry(pitch) * Rx(roll)",
    "t_mm": "translation in millimeters; the matrices are in meters",
}


def load_rig(root: Path) -> Dict[str, Dict[str, str]]:
    """Return {"front": {...}} (single lidar) or {"front": {...}, "rear": {...}} from config/rig.yaml."""
    cfg = yaml.safe_load((root / "config" / "rig.yaml").read_text(encoding="utf-8"))
    lidars = cfg["lidars"]
    if len(lidars) == 1:
        lidars = [{**lidars[0], "primary": True}]
    out: Dict[str, Dict[str, str]] = {}
    for lidar in lidars:
        role = "front" if lidar.get("primary") else "rear"
        if role in out:
            raise SystemExit("config/rig.yaml needs one primary (front) lidar and at most one rear lidar")
        name = str(lidar["id"])
        ip = str(lidar["ip"]).replace(".", "_")
        out[role] = {
            "id": name,
            "ip": str(lidar["ip"]),
            "frame": name,
            "points": f"/livox/{name}/points",
            "imu": f"/livox/{name}/imu",
            "driver_points": f"/livox/lidar_{ip}",
            "driver_imu": f"/livox/imu_{ip}",
        }
    if "front" not in out:
        raise SystemExit("config/rig.yaml needs one primary (front) lidar")
    return out


def roles(rig: Dict[str, Dict[str, str]]) -> tuple:
    return tuple(r for r in ("front", "rear") if r in rig)


def single(rig: Dict[str, Dict[str, str]]) -> bool:
    return "rear" not in rig


def lidars_block(rig: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    return {
        role: {"id": rig[role]["id"], "frame": rig[role]["frame"], "primary": role == "front"}
        for role in roles(rig)
    }


def rpy_deg(R: np.ndarray) -> list:
    R = np.asarray(R, dtype=float)[:3, :3]
    pitch = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) < 1e-8:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    return [_r(math.degrees(v), 3) for v in (roll, pitch, yaw)]


def _r(v: float, digits: int) -> float:
    return round(float(v), digits) + 0.0


def pose(T: np.ndarray) -> Dict[str, Any]:
    T = np.asarray(T, dtype=float)
    return {
        "rpy_deg": rpy_deg(T),
        "t_mm": [_r(v * 1000.0, 2) for v in T[:3, 3]],
    }


def matrix(T: np.ndarray) -> list:
    return [[_r(v, 9) for v in row] for row in np.asarray(T, dtype=float)]


def dumps(doc: Dict[str, Any]) -> str:
    """JSON with every list of plain numbers kept on one line."""
    text = json.dumps(doc, indent=2, ensure_ascii=False)
    number_list = re.compile(r"\[\s*(-?[\d.eE+-]+(?:\s*,\s*-?[\d.eE+-]+)*)\s*\]")
    return number_list.sub(lambda m: "[" + ", ".join(p.strip() for p in m.group(1).split(",")) + "]", text) + "\n"


def write(path: Path, doc: Dict[str, Any]) -> None:
    path.write_text(dumps(doc), encoding="utf-8")


def installation_doc(rig: Dict[str, Dict[str, str]], T_seed: np.ndarray, created_utc: str) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "installation",
        "created_utc": created_utc,
        "source": "doc/mid360s_2_insta360.png, doc/install_action.jpg",
        "lidars": lidars_block(rig),
        "convention": CONVENTION,
        "installation_seed": {
            **pose(T_seed),
            "meaning": (
                "Rear lidar relative to the front lidar after each one is leveled with its own IMU: "
                "yaw 180 deg (mirrored heads) and the 247.5 mm baseline along -X. "
                "Roll and pitch are not part of the seed."
            ),
        },
        "note": "refine-auto reads this file as the seed and never overwrites it. ./calibrate.sh initial rewrites it.",
        "T_front_rear_level_seed": matrix(T_seed),
    }
