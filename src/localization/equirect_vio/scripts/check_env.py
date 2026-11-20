#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ok = True
    print(f"EquirectangularVIO root: {ROOT}")

    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(name)
        print(f"  {name}: {path or 'MISSING'}")
        ok = ok and bool(path)

    vendored = ROOT / "third_party" / "openvio" / "CMakeLists.txt"
    print(f"  vendored openvio: {vendored.parent} {'OK' if vendored.is_file() else 'MISSING'}")
    ok = ok and vendored.is_file()

    openvio = ROOT / "build" / "openvio" / "bin" / "vio"
    print(f"  openvio bin: {openvio} {'OK' if openvio.is_file() else 'MISSING (run ./scripts/build_openvio.sh)'}")
    if not openvio.is_file():
        ok = False

    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        import yaml  # noqa: F401

        print("  python deps: ok")
    except ImportError as e:
        print(f"  python deps: MISSING ({e})")
        ok = False

    cfg = ROOT / "config" / "x4_air_equirect_1440.yaml"
    print(f"  config: {cfg} {'OK' if cfg.is_file() else 'MISSING'}")
    ok = ok and cfg.is_file()

    if openvio.is_file():
        r = subprocess.run([str(openvio)], capture_output=True, text=True)
        print(f"  openvio usage exit={r.returncode}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())