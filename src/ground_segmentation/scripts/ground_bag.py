#!/usr/bin/env python3
"""Run ``ground_segmentation.bag_eval`` from the source tree (no colcon install needed)."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SRC / "ground_segmentation"), str(SRC / "scene_memory")]

from ground_segmentation.bag_eval import main  # noqa: E402

if __name__ == "__main__":
    main()
