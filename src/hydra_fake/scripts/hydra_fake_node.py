#!/usr/bin/env python3
"""Run ``hydra_fake.node`` from the source tree (no colcon install needed)."""

import sys
from pathlib import Path

src = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(src / "hydra_fake"), str(src / "scene_memory")]

from hydra_fake.node import main  # noqa: E402

if __name__ == "__main__":
    main()
