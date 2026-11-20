import numpy as np

from scene_memory.common.geometry import project_level_points_to_equirect
from scene_memory.perception.pano_depth import (SOURCE_ALIGNED, SOURCE_COARSE, SOURCE_LIDAR, align_local, crop_front,
                                                hemi_project, range_bin_errors, sparse_hemi_depth)


def _random_points(n, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d * rng.uniform(1.0, 20.0, size=(n, 1))


def test_hemi_project_matches_full_equirect_crop():
    T = np.eye(4)
    T[:3, 3] = [0.3, -0.2, 0.1]
    pts = _random_points(5000)
    W, H = 400, 200
    u_full, v_full, r_full = project_level_points_to_equirect(pts, T, H, W)
    u, v, r = hemi_project(pts, T, W // 2, H, r_min=0.0, r_max=1e9)
    front = np.rint(u_full).astype(int) - W // 4
    keep = (front >= 0) & (front < W // 2) & (np.rint(v_full) >= 0) & (np.rint(v_full) < H)
    assert len(u) == keep.sum()
    np.testing.assert_array_equal(np.sort(u), np.sort(front[keep]))
    np.testing.assert_allclose(np.sort(r), np.sort(r_full[keep]), rtol=1e-5)


def test_crop_front_takes_middle_half():
    img = np.arange(8)[None, :].repeat(2, axis=0)
    np.testing.assert_array_equal(crop_front(img)[0], [2, 3, 4, 5])


def test_sparse_depth_drops_background_behind_near_return():
    # A near return at 2 m next to a far one at 10 m one pixel apart: the far one is leak and goes.
    T = np.eye(4)
    near = np.array([[0.0, 0.0, 2.0]])
    far = np.array([[np.tan(np.deg2rad(1.0)) * 10.0, 0.0, 10.0]])
    depth = sparse_hemi_depth(np.vstack([near, far]), T, 180, 180, occlusion_px=2)
    assert np.isclose(depth.max(), 2.0, atol=1e-3)
    assert (depth > 0).sum() == 1


def test_align_local_recovers_regional_scales():
    h, w = 120, 120
    truth = np.full((h, w), 8.0, np.float32)
    truth[:, w // 2:] = 4.0
    # Relative map with a different wrong scale on each half.
    rel = truth.copy()
    rel[:, : w // 2] /= 2.0
    rel[:, w // 2:] /= 3.0
    rng = np.random.default_rng(1)
    sparse = np.zeros_like(truth)
    idx = rng.choice(h * w, 600, replace=False)
    sparse.flat[idx] = truth.flat[idx]
    depth, source, stats = align_local(rel, sparse, cell_px=8, sigma_cells=1.0)
    assert stats["inliers"] > 500
    inner = np.zeros_like(truth, bool)
    inner[:, 10: w // 2 - 10] = True
    inner[:, w // 2 + 10: -10] = True
    err = np.abs(depth - truth)[inner & (source == SOURCE_ALIGNED)]
    assert len(err) > 0.5 * inner.sum()
    assert np.median(err) < 0.05 * 6.0
    assert (source == SOURCE_LIDAR).sum() >= 0.8 * 600


def test_align_local_marks_far_and_unsupported_as_coarse():
    h, w = 64, 64
    rel = np.full((h, w), 1.0, np.float32)
    rel[:, 48:] = 20.0
    sparse = np.zeros_like(rel)
    sparse[::4, :24:4] = 2.0
    depth, source, stats = align_local(rel, sparse, near_m=30.0, min_points=10)
    assert np.isclose(stats["scale"], 2.0)
    assert (source[:, 50:] == SOURCE_COARSE).all()
    assert np.isclose(depth[0, 60], 40.0)


def test_align_local_needs_points():
    depth, source, stats = align_local(np.ones((10, 10)), np.zeros((10, 10)))
    assert depth is None and source is None and stats["points"] == 0


def test_range_bin_errors():
    truth = np.array([1.0, 6.0, 20.0, 50.0])
    pred = truth + np.array([0.1, 0.2, 1.0, 5.0])
    bins = range_bin_errors(pred, truth)
    assert [b["n"] for b in bins] == [1, 1, 1, 1]
    assert np.isclose(bins[2]["abs_med"], 1.0)
