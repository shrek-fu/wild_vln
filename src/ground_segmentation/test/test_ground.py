import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SRC / "ground_segmentation"), str(SRC / "scene_memory")]

from ground_segmentation.geometry import fit_ground_plane, heights, ray_plane_fill  # noqa: E402
from ground_segmentation.grid import TraversableGrid  # noqa: E402
from ground_segmentation.mapper import GroundMapper  # noqa: E402

rng = np.random.default_rng(1)


def _patch(z, n, x=(0, 6), y=(-3, 3), noise=0.01):
    return np.column_stack([rng.uniform(*x, n), rng.uniform(*y, n), z + rng.normal(0, noise, n)])


# Optical frame looking along odom +x, 0.5 m above the floor (optical x right = -y, y down = -z, z fwd = +x).
R_ODOM_OPTICAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)


def _camera(z=0.5):
    T = np.eye(4)
    T[:3, :3] = R_ODOM_OPTICAL
    T[2, 3] = z
    return T


def test_plane_prefers_the_lowest_large_surface():
    floor = _patch(0.0, 600)
    desk = _patch(0.75, 900, x=(2, 4), y=(-1, 1))
    n, d, inliers = fit_ground_plane(np.vstack([floor, desk]))
    assert n[2] > 0.999 and abs(d) < 0.02
    assert inliers[:600].mean() > 0.95 and inliers[600:].mean() < 0.01


def test_plane_rejects_walls_and_respects_prior():
    floor = _patch(-0.3, 300)
    wall = np.column_stack([np.full(2000, 3.0), rng.uniform(-3, 3, 2000), rng.uniform(-0.3, 2.5, 2000)])
    n, d, _ = fit_ground_plane(np.vstack([floor, wall]))
    assert abs(-d / n[2] + 0.3) < 0.02
    # A tilted slab near the prior height is refused rather than replacing it.
    tilted = _patch(0.0, 500)
    tilted[:, 2] = 0.2 * tilted[:, 0] - 0.3
    assert fit_ground_plane(tilted, prior=(np.array([0, 0, 1.0]), 0.3)) is None


def test_ray_fill_hits_the_floor_and_is_blocked_by_nearer_returns():
    K = (100.0, 100.0, 49.5, 49.5)
    mask = np.zeros((100, 100), bool)
    mask[60:, :] = True
    plane = (np.array([0, 0, 1.0]), 0.0)
    pts = ray_plane_fill(mask, K, _camera(), plane, stride=4, max_range=20)
    assert len(pts) > 100 and np.allclose(pts[:, 2], 0.0, atol=1e-9) and (pts[:, 0] > 0).all()
    blocked = np.zeros((100, 100), np.float32)
    blocked[:, :50] = 0.5
    kept = ray_plane_fill(mask, K, _camera(), plane, stride=4, max_range=20, sparse_depth=blocked)
    # Image left (u < 50, odom +y) has a return 0.5 m away in front of the floor: only the right half stays.
    assert 0 < len(kept) < len(pts) and (kept[:, 1] < 0).all()
    near = ray_plane_fill(mask, K, _camera(), plane, stride=4, max_range=1.5)
    assert np.hypot(near[:, 0], near[:, 1]).max() <= 1.5


def test_grid_votes_obstacles_win_and_occupancy_layout():
    g = TraversableGrid(0.5, hit=0.6, obstacle=-1.2, free_thresh=1.0, occ_thresh=-1.0)
    ground = np.array([[0.1, 0.1, 0.0], [0.2, 0.2, 0.02], [1.1, 0.1, 0.1]])
    for _ in range(2):
        g.update(ground, np.array([[1.2, 0.2, 0.5]]))
    trav = g.traversable()
    assert trav.shape == (1, 3) and np.allclose(trav[0], [0.25, 0.25, 0.01])
    (ox, oy), w, h, data = g.occupancy()
    assert (ox, oy, w, h) == (0.0, 0.0, 3, 1)
    assert data.tolist() == [0, -1, 100]
    g.update(np.array([[-0.7, -0.3, 0.0]]), np.zeros((0, 3)))
    (ox, oy), w, h, data = g.occupancy()
    assert (ox, oy, w, h) == (-1.0, -0.5, 5, 2)
    assert data.reshape(h, w)[1, 2] == 0 and data.reshape(h, w)[1, 4] == 100


def test_outlines_find_region_and_enclosed_obstacle():
    g = TraversableGrid(0.1)
    xs, ys = np.meshgrid(np.arange(0.05, 3.0, 0.1), np.arange(0.05, 2.0, 0.1))
    floor = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 0.2)])
    box = (np.abs(floor[:, 0] - 1.5) < 0.5) & (np.abs(floor[:, 1] - 1.0) < 0.5)
    for _ in range(3):
        g.update(floor[~box], floor[box])
    rings = g.outlines()
    outer = [xyz for xyz, hole in rings if not hole]
    inner = [xyz for xyz, hole in rings if hole]
    assert len(outer) == 1 and len(inner) == 1
    assert np.allclose(outer[0][:, 2], 0.2)
    assert outer[0][:, 0].min() < 0.1 and outer[0][:, 0].max() > 2.9
    assert 0.9 < inner[0][:, 0].min() < 1.1 and 1.9 < inner[0][:, 0].max() < 2.1
    assert len(g.obstacles()) == int(box.sum())


def test_parse_ground_classes():
    from scene_memory.perception.gdino_sam2_detector import DEFAULT_GROUND_CLASSES, parse_ground_classes

    assert parse_ground_classes("floor:floor|ground ; stairs:stairs;grass") == [
        ("floor", ["floor", "ground"]), ("stairs", ["stairs"]), ("grass", ["grass"])]
    assert [c for c, _ in parse_ground_classes(DEFAULT_GROUND_CLASSES)] == ["floor", "grass", "stairs", "step"]


def test_ground_classes_file_and_flags(tmp_path):
    from scene_memory import paths
    from scene_memory.perception.gdino_sam2_detector import parse_ground_classes

    from ground_segmentation.segmenters import blocked_classes, class_names, terrain_classes

    f = tmp_path / "g.txt"
    f.write_text("# comment\nfloor: floor | tile floor\n\nramp: ramp   @terrain  # inline comment\n"
                 "stairs: stairs @terrain @blocked\n", encoding="utf-8")
    assert parse_ground_classes(str(f)) == [("floor", ["floor", "tile floor"]), ("ramp", ["ramp"]),
                                            ("stairs", ["stairs"])]
    assert terrain_classes(str(f)) == ["ramp", "stairs"] and terrain_classes(str(f), "ramp") == ["ramp"]
    assert blocked_classes(str(f), "ramp,stairs") == ["stairs", "ramp"]
    assert terrain_classes("floor:floor;stairs:stairs") == ["stairs", "step"]  # no flags: defaults
    assert terrain_classes("floor:floor;ramp:ramp @terrain") == ["ramp"]
    shipped = paths.CONFIG / "ground_classes.txt"
    assert class_names(str(shipped))[0] == "floor" and "stairs" in terrain_classes(str(shipped))
    with pytest.raises(FileNotFoundError):
        parse_ground_classes(str(tmp_path / "missing.txt"))


def test_grid_classes_majority_blocked_and_outlines():
    g = TraversableGrid(0.1, n_classes=3)
    xs, ys = np.meshgrid(np.arange(0.05, 2.0, 0.1), np.arange(0.05, 1.0, 0.1))
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    stairs = pts[:, 0] > 1.0
    pts[stairs, 2] = 0.15 * np.floor((pts[stairs, 0] - 1.0) / 0.3)
    labels = np.where(stairs, 3, 1)
    for i in range(3):
        noisy = labels.copy()
        if i == 0:
            noisy[:5] = 2  # one minority vote does not flip a cell
        g.update(pts, np.zeros((0, 3)), noisy)
    xyz, cls = g.traversable(with_class=True)
    assert set(cls.tolist()) == {1, 3}
    assert (xyz[cls == 3, 2] > 0.1).any() and np.allclose(xyz[cls == 1, 2], 0.0)
    (_, _), w, h, data = g.class_occupancy()
    assert set(np.unique(data).tolist()) <= {1, 3}
    _, _, _, nav = g.occupancy(blocked=(3,))
    assert (nav == 100).sum() == (cls == 3).sum() and (nav == 0).sum() == (cls == 1).sum()
    floor_rings = g.outlines(1)
    stair_rings = g.outlines(3)
    assert len(floor_rings) == 1 and len(stair_rings) == 1
    assert floor_rings[0][0][:, 0].max() <= 1.1 and stair_rings[0][0][:, 0].min() >= 0.95


def test_mapper_keeps_stairs_at_lidar_height_and_not_obstacles():
    class StairsLeft:
        classes = ["floor", "grass", "stairs"]

        def __call__(self, bgr):
            h, w = bgr.shape[:2]
            lab = np.zeros((h, w), np.uint8)
            lab[h // 2 + 2:, w // 2:] = 1
            lab[:, : w // 2 - 2] = 3
            return lab

    from scene_memory.mapping.scene_graph import T_da360_optical

    view = {"name": "front", "hfov_deg": 90.0, "R_body_optical": np.eye(3)}
    m = GroundMapper(StairsLeft(), np.eye(4), fill_range=8.0, min_ground_points=20)
    T_child = _camera() @ np.linalg.inv(T_da360_optical(np.eye(3)))
    for t in np.arange(0.0, 1.0, 0.05):
        m.on_odom(float(t), T_child)
    floor = _patch(0.0, 3000, x=(2.5, 10), y=(-4, -0.3))
    stairs = _patch(0.0, 3000, x=(3, 6), y=(0.5, 2.5), noise=0.005)
    stairs[:, 2] = 0.15 * np.floor((stairs[:, 0] - 3.0) / 0.3)
    table = _patch(0.75, 400, x=(7.0, 8.0), y=(0.6, 1.4), noise=0.005)  # also inside the "stairs" mask
    m.on_cloud(0.5, np.vstack([floor, stairs, table]).astype(np.float32))
    for _ in range(2):
        res = m.process(view, np.zeros((200, 200, 3), np.uint8), 0.5)
    assert res["plane_src"] == "fit" and abs(res["plane"][1]) < 0.03
    assert len(res["obstacle"]) > 0 and (np.abs(res["obstacle"][:, 2] - 0.75) < 0.05).all()
    xyz, cls = m.grid.traversable(with_class=True)
    on_stairs = cls == 3
    assert on_stairs.sum() > 50 and xyz[on_stairs, 2].max() > 1.0
    assert (xyz[on_stairs, 0] < 6.2).all()
    assert (xyz[on_stairs, 1] > 0).all() and (cls == 1).sum() > 50 and (xyz[cls == 1, 1] < 0).all()


def test_step_reachable_rejects_tall_jumps():
    from ground_segmentation.geometry import step_reachable

    xs = np.arange(0.0, 3.0, 0.05)
    stair_h = 0.17 * np.floor(xs / 0.3)
    table_x = np.arange(4.0, 4.8, 0.05)
    xy = np.column_stack([np.concatenate([xs, table_x, [1.0]]), np.zeros(len(xs) + len(table_x) + 1)])
    h = np.concatenate([stair_h, np.full(len(table_x), 0.75), [stair_h[20] + 0.6]])
    keep = step_reachable(xy, h)
    assert keep[: len(xs)].all()
    assert not keep[len(xs):].any()


def _surface(fn, x=(0.0, 3.0), y=(0.0, 1.0), step=0.04):
    xs, ys = np.meshgrid(np.arange(*x, step), np.arange(*y, step))
    xy = np.column_stack([xs.ravel(), ys.ravel()])
    return xy, fn(xy[:, 0])


@pytest.mark.parametrize("name, fn, max_step, max_slope, ok", [
    ("ramp 12 deg", lambda x: np.tan(np.radians(12)) * x, 0.18, 35, True),
    ("ramp 45 deg", lambda x: np.minimum(x, 1.5), 0.18, 35, False),
    ("stairs 0.16/0.30 (28 deg)", lambda x: 0.16 * np.floor(x / 0.3), 0.18, 35, True),
    ("stairs 0.17/0.20 (40 deg)", lambda x: 0.17 * np.floor(x / 0.2), 0.18, 35, False),
    ("stairs 0.16 for a 0.10 m robot", lambda x: 0.16 * np.floor(x / 0.3), 0.10, 35, False),
    ("low step 0.08 for a 0.10 m robot", lambda x: 0.08 * (x > 1.5), 0.10, 20, True),
])
def test_step_reachable_robot_limits(name, fn, max_step, max_slope, ok):
    from ground_segmentation.geometry import step_reachable

    xy, h = _surface(fn)
    keep = step_reachable(xy, h, max_step=max_step, max_slope_deg=max_slope)
    climbed = keep[h > 0.5].mean() if (h > 0.5).any() else keep[h > 0.05].mean()
    assert (climbed > 0.9) if ok else (climbed < 0.1), (name, climbed)


def test_mapper_lifts_mask_and_vetoes_obstacles():
    class LowerHalf:
        def __call__(self, bgr):
            m = np.zeros(bgr.shape[:2], bool)
            m[bgr.shape[0] // 2 + 2:] = True
            return m

    view = {"name": "front", "hfov_deg": 90.0, "R_body_optical": np.eye(3)}
    m = GroundMapper(LowerHalf(), np.eye(4), fill_range=8.0, min_ground_points=20)
    # T_odom_optical = T_odom_child @ T_child_da360 @ T_da360_optical: put the camera pose in T_odom_child.
    from scene_memory.mapping.scene_graph import T_da360_optical

    T_child = _camera() @ np.linalg.inv(T_da360_optical(np.eye(3)))
    for t in np.arange(0.0, 1.0, 0.05):
        m.on_odom(float(t), T_child)
    floor = _patch(0.0, 3000, x=(2.5, 10), y=(-4, 4))
    box = np.column_stack([rng.uniform(4, 4.3, 400), rng.uniform(-0.2, 0.2, 400), rng.uniform(0.05, 0.6, 400)])
    m.on_cloud(0.5, np.vstack([floor, box]).astype(np.float32))
    res = m.process(view, np.zeros((200, 200, 3), np.uint8), 0.5)
    assert res["plane_src"] == "fit" and abs(res["plane"][1]) < 0.03
    assert len(res["ground"]) > 100 and len(res["obstacle"]) > 50 and len(res["fill"]) > 50
    assert (np.abs(heights(res["fill"], res["plane"])) < 1e-6).all()
    m.process(view, np.zeros((200, 200, 3), np.uint8), 0.5)
    trav = m.grid.traversable()
    assert len(trav) > 100
    on_box = (trav[:, 0] > 4.0) & (trav[:, 0] < 4.3) & (np.abs(trav[:, 1]) < 0.2)
    assert not on_box.any()
