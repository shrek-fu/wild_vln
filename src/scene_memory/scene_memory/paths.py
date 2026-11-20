"""Workspace layout. Every default model/data path comes from here.

::

    <ws>/
      src/       ROS 2 packages by layer: sensors/ localization/ depth/ scene_memory/ tools/
      models/    all network weights (models/README.md lists them)
      insta360_driver/  Camera SDK + Media SDK (outside src/; shared)
      data/
        calib/     rig calibration JSON (insta360_*, lidar_*, rig_calib)
        calib_capture/  bags/clouds for the calibration package
        sessions/  collection_data recordings; offline runs write <session>/scene4d/<time>
        scene4d/   live mapping runs (objects_4d.json, topo_graph.json, views/, ...)
      refer/     upstream reference repos (DAAAM, boxer, ByteTrack, hydra_ws, ...)

The root is found by walking up from this file (works from ``src/`` and from
``<ws>/install/``), then from the working directory, until a directory holds both
``src`` and ``models``.
``STVLN_ROOT`` overrides it; ``STVLN_MODELS`` / ``STVLN_DATA`` / ``STVLN_REFER``
move single trees.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_root() -> Path:
    env = os.environ.get("STVLN_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for start in (here, Path.cwd().resolve() / "_"):
        for parent in start.parents:
            if (parent / "src").is_dir() and (parent / "models").is_dir():
                return parent
    return here.parents[3]


def _env_dir(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


WS_ROOT = _find_root()
SRC = WS_ROOT / "src"
PKG = SRC / "scene_memory"
CONFIG = PKG / "config"
MODELS = _env_dir("STVLN_MODELS", WS_ROOT / "models")
DATA = _env_dir("STVLN_DATA", WS_ROOT / "data")
REFER = _env_dir("STVLN_REFER", WS_ROOT / "refer")

CALIB = DATA / "calib"
SESSIONS = DATA / "sessions"
SCENES = DATA / "scene4d"

YOLOE = MODELS / "yoloe" / "yoloe-26s-seg.pt"
MONO_DEPTH = MODELS / "depth_anything_v2" / "depth_anything_v2_metric_hypersim_vits.pth"
DA360 = MODELS / "da360" / "DA360_large.pth"
DEPTHART = MODELS / "depthart"
GDINO = MODELS / "grounding-dino-tiny"
SAM2 = MODELS / "sam2" / "sam2.1_t.pt"
SAM3 = MODELS / "sam3" / "sam3.pt"
SIGLIP = MODELS / "siglip-base-multilingual"
DAM = MODELS / "DAM-3B"

VIEWS_CONFIG = SRC / "sensors" / "insta360_views" / "config" / "insta360_views.yaml"
INSTA360_DRIVER = WS_ROOT / "insta360_driver"
MEDIA_SDK_MODELS = INSTA360_DRIVER / "insta360_media_sdk" / "bin" / "models"
VIO_WS = SRC / "localization" / "equirect_vio" / "ros2_ws"
VIO_CONFIG = VIO_WS / "src" / "equirect_vio" / "config" / "x4_air_equirect_1920.yaml"
LIO_CONFIG = CONFIG / "super_lio_mid360.yaml"
RIG_CALIB_REPO = SRC / "insta360_mid360s_calibration"
DA360_ROOT = SRC / "depth" / "da360_depth"
DEPTHART_ROOT = REFER / "DepthART"
# Interpreter for the GPU nodes: <ws>/.venv (Jetson: JetPack torch wheels, --system-site-packages for rclpy).
PYTHON = str(WS_ROOT / ".venv" / "bin" / "python") if (WS_ROOT / ".venv" / "bin" / "python").is_file() else "python3"
JETSON = Path("/etc/nv_tegra_release").is_file()

REPLAY = SRC / "tools" / "collection_data" / "collection_data" / "replay_session.py"
