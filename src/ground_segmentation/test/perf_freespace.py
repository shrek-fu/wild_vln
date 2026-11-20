"""Timing of LocalFreespace on a full 30 m window and three MID-360 scans: python3 test/perf_freespace.py"""

import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SRC / "ground_segmentation"), str(SRC / "scene_memory")]

from ground_segmentation.freespace import LocalFreespace, obstacle_mask  # noqa: E402

rng = np.random.default_rng(0)
xs, ys = np.meshgrid(np.arange(-15, 15, 0.1), np.arange(-15, 15, 0.1))
cells = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
free = LocalFreespace()
t = time.perf_counter()
free.set_cells(cells)
print(f"set_cells {len(cells)} cells: {1e3 * (time.perf_counter() - t):.0f} ms")
pts = np.column_stack([rng.uniform(-15, 15, (60000, 2)), rng.uniform(-0.1, 2, 60000)])
for _ in range(3):
    t = time.perf_counter()
    h, _ = free.heights(pts, free.ground_under((0, 0, 0), 0.0))
    mask = obstacle_mask(pts, h, (0, 0), h_min=0.15, h_max=1.2, clear_radius=0.45, half_m=15)
    print(f"{len(pts)} points: {1e3 * (time.perf_counter() - t):.0f} ms, {mask.sum()} obstacles")
