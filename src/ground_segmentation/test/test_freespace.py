import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SRC / "ground_segmentation"), str(SRC / "scene_memory")]

from ground_segmentation.freespace import LocalFreespace, obstacle_mask  # noqa: E402
from ground_segmentation.grid import TraversableGrid  # noqa: E402

RES = 0.1


def _cells(x=(0.0, 8.0), y=(-2.0, 2.0), z=lambda x: 0.0 * x):
    xs, ys = np.meshgrid(np.arange(*x, RES) + RES / 2, np.arange(*y, RES) + RES / 2)
    xs, ys = xs.ravel(), ys.ravel()
    return np.column_stack([xs, ys, z(xs)])


def _obstacles(free, pts, robot=(0.0, 0.0, 0.0), **kw):
    h, known = free.heights(pts, free.ground_under(robot, robot[2]))
    opts = dict(h_min=0.15, h_max=1.2, clear_radius=0.45, half_m=15.0) | kw
    return obstacle_mask(pts, h, robot, **opts), h, known


def test_ramp_surface_is_free_box_on_it_is_not():
    ramp = lambda x: np.where(x > 3.0, 0.15 * (x - 3.0), 0.0)  # noqa: E731 — 8.5 deg ramp from x = 3 m
    free = LocalFreespace()
    free.set_cells(_cells(z=ramp))
    surface = np.column_stack([np.linspace(1, 7.5, 50), np.zeros(50), ramp(np.linspace(1, 7.5, 50)) + 0.02])
    box = np.array([[6.0, 0.5, ramp(6.0) + 0.3], [6.0, 0.5, ramp(6.0) + 0.6]])
    mask, h, known = _obstacles(free, np.vstack([surface, box]))
    assert known.all()
    assert not mask[:50].any(), "ramp floor must not be an obstacle (a flat robot-height threshold marks it)"
    assert mask[50:].all()
    # the same ramp against the robot's flat ground: the far end of the ramp would look like a wall
    assert (surface[:, 2] > 0.15).sum() > 10


def test_low_object_and_overhead():
    free = LocalFreespace()
    free.set_cells(_cells())
    pts = np.array([[2.0, 0.0, 0.05], [2.0, 0.3, 0.2], [2.0, -0.3, 1.0], [2.0, 0.0, 1.6]])
    mask, _, _ = _obstacles(free, pts)
    assert mask.tolist() == [False, True, True, False]


def test_unseen_area_uses_the_ground_under_the_robot():
    free = LocalFreespace(search_m=1.0)
    free.set_cells(_cells(x=(-1.0, 1.0), y=(-1.0, 1.0), z=lambda x: 0.0 * x + 0.1))
    pts = np.array([[10.0, 0.0, 0.12], [10.0, 0.5, 0.5]])
    mask, h, known = _obstacles(free, pts)
    assert not known.any()
    assert np.allclose(h, [0.02, 0.4])
    assert mask.tolist() == [False, True]


def test_no_freespace_falls_back_to_base_link_on_the_ground():
    free = LocalFreespace()
    mask, h, _ = _obstacles(free, np.array([[2.0, 0.0, 0.02], [2.0, 0.0, 0.4]]), robot=(0.0, 0.0, 0.0))
    assert mask.tolist() == [False, True]


def test_window_and_clear_radius():
    free = LocalFreespace()
    free.set_cells(_cells(x=(-20.0, 20.0), y=(-1.0, 1.0)))
    pts = np.array([[0.2, 0.0, 0.5], [14.0, 0.0, 0.5], [16.0, 0.0, 0.5]])
    mask, _, _ = _obstacles(free, pts)
    assert mask.tolist() == [False, True, False]


def test_blocked_cells():
    cells = _cells(x=(0.0, 2.0), y=(0.0, 1.0))
    labels = np.where(cells[:, 0] > 1.0, 2, 1)
    free = LocalFreespace(blocked_labels=[2])
    free.set_cells(cells, labels)
    assert len(free.blocked_xyz) == (labels == 2).sum() and (free.blocked_xyz[:, 0] > 1.0).all()


def test_grid_crop_keeps_the_window():
    grid = TraversableGrid(RES)
    grid.update(_cells(x=(-20.0, 20.0), y=(-0.5, 0.5)), np.zeros((0, 3)))
    grid.update(_cells(x=(-20.0, 20.0), y=(-0.5, 0.5)), np.zeros((0, 3)))
    n = len(grid)
    dropped = grid.crop((0.0, 0.0), 15.0)
    x = grid.traversable()[:, 0]
    assert dropped == n - len(grid) and len(x) == len(grid)
    assert -15.0 <= x.min() < -14.8 and 14.8 < x.max() <= 15.0
