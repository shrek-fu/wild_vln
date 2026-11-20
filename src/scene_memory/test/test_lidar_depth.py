import numpy as np

from scene_memory import paths
from scene_memory.common.pose_buffer import PoseBuffer, interpolate_pose
from scene_memory.mapping.scene_graph import load_lio_imu_from_lidar, pinhole_intrinsics
from scene_memory.perception.lidar_depth import (
    CloudWindow,
    mask_depth,
    nearest_cluster,
    project_points,
    scale_relative_depth,
    view_depth,
)

W = H = 640
K = pinhole_intrinsics(W, H, 90.0)


def _scene() -> np.ndarray:
    """Optical frame = odom: a wall 6 m ahead and a 0.4 m cube 3 m ahead, left of centre."""
    g = np.linspace(-3.0, 3.0, 121)
    wall = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    wall = np.c_[wall, np.full(len(wall), 6.0)]
    c = np.linspace(-0.2, 0.2, 9)
    cube = np.stack(np.meshgrid(c, c), -1).reshape(-1, 2)
    cube = np.c_[cube[:, 0] - 0.5, cube[:, 1], np.full(len(cube), 2.8)]
    hidden = (np.abs(wall[:, 0] * 2.8 / 6.0 + 0.5) < 0.2) & (np.abs(wall[:, 1] * 2.8 / 6.0) < 0.2)
    return np.r_[wall[~hidden], cube].astype(np.float32)


def _box(center, half, z):
    fx, fy, cx, cy = K
    x0, y0 = fx * (center[0] - half) / z + cx, fy * (center[1] - half) / z + cy
    x1, y1 = fx * (center[0] + half) / z + cx, fy * (center[1] + half) / z + cy
    return {"xyxy": [x0 - 15, y0 - 15, x1 + 15, y1 + 15]}


def test_mask_depth_takes_the_object_not_the_wall():
    projected = project_points(_scene(), np.eye(4), K, W, H)
    hit = mask_depth(projected, W, H, _box((-0.5, 0.0), 0.2, 2.8))
    assert hit is not None
    assert abs(hit["depth"] - 2.8) < 1e-3
    np.testing.assert_allclose(hit["center"], [-0.5, 0.0, 2.8], atol=0.06)


def test_mask_depth_needs_returns():
    projected = project_points(np.zeros((0, 3), np.float32), np.eye(4), K, W, H)
    assert mask_depth(projected, W, H, {"xyxy": [0, 0, 100, 100]}) is None


def test_nearest_cluster_skips_sparse_foreground():
    z = np.r_[np.full(3, 1.0), np.full(40, 4.0), np.full(60, 8.0)]
    idx = nearest_cluster(z, min_points=12)
    assert np.allclose(z[idx], 4.0) and len(idx) == 40


def test_view_depth_keeps_the_nearest_return():
    depth = view_depth(_scene(), np.eye(4), K, W, H, splat=1)
    fx, _, cx, cy = K
    u = int(round(fx * -0.5 / 2.8 + cx))
    assert abs(depth[int(round(cy)), u] - 2.8) < 1e-3
    assert abs(depth[int(round(cy)), W - 5] - 6.0) < 0.2 or depth[int(round(cy)), W - 5] == 0.0


def test_scale_relative_depth_rejects_outliers():
    rng = np.random.default_rng(0)
    rel = rng.uniform(0.5, 2.0, (60, 80)).astype(np.float32)
    sparse = np.zeros_like(rel)
    sel = rng.random(rel.shape) < 0.2
    sparse[sel] = 3.0 * rel[sel]
    bad = sel & (rng.random(rel.shape) < 0.1)
    sparse[bad] = 0.3 * rel[bad]
    depth, stats = scale_relative_depth(rel, sparse)
    assert depth is not None
    assert abs(stats["scale"] - 3.0) < 1e-3
    assert stats["inliers"] < stats["points"]


def test_cloud_window_time_range():
    w = CloudWindow(window_s=1.0, keep_s=3.0)
    for i in range(30):
        w.add(i * 0.1, np.full((2, 3), i, np.float32))
    pts = w.points(2.0)
    assert pts[:, 0].min() >= 10 and pts[:, 0].max() <= 21


def test_pose_interpolation():
    T0, T1 = np.eye(4), np.eye(4)
    T1[:3, 3] = [2.0, 0.0, 0.0]
    T1[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    T = interpolate_pose(T0, T1, 0.5)
    np.testing.assert_allclose(T[:3, 3], [1.0, 0.0, 0.0])
    yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    assert abs(yaw - 45.0) < 1e-6
    buf = PoseBuffer()
    buf.add(0.0, T0)
    buf.add(0.1, T1)
    np.testing.assert_allclose(buf.at(0.05)[:3, 3], [1.0, 0.0, 0.0])
    assert buf.at(5.0) is None


def test_lio_extrinsic_from_config():
    T = load_lio_imu_from_lidar(paths.LIO_CONFIG)
    np.testing.assert_allclose(T[:3, 3], [-0.011, -0.02329, 0.04412])
    np.testing.assert_allclose(T[:3, :3], np.eye(3))
