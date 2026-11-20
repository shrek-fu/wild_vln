import numpy as np

from scene_memory.mapping.instance_points import InstancePoints, MapCloud
from scene_memory.mapping.pano_overlay import box_edge_points, draw_boxes, view_to_equirect
from scene_memory.common.geometry import project_level_points_to_equirect

rng = np.random.default_rng(0)


def _chair(n=400, center=(2.0, 1.0, 0.45), size=(0.5, 0.5, 0.9)):
    return np.asarray(center) + (rng.random((n, 3)) - 0.5) * np.asarray(size)


def test_box_from_repeated_observations_ignores_one_off_outliers():
    inst = InstancePoints(voxel_m=0.05)
    for _ in range(3):
        inst.add("obj1", _chair())
    inst.add("obj1", np.array([[5.0, 5.0, 5.0], [5.0, 5.1, 5.0]]))  # background seen once
    box = inst.box("obj1")
    np.testing.assert_allclose(box["center"], [2.0, 1.0, 0.45], atol=0.05)
    np.testing.assert_allclose(box["size"], [0.5, 0.5, 0.9], atol=0.08)


def test_box_needs_enough_voxels():
    inst = InstancePoints(voxel_m=0.05, min_voxels=10)
    inst.add("a", np.array([[0.0, 0.0, 0.0]]))
    assert inst.box("a") is None
    assert inst.box("missing") is None


def test_merge_and_replace():
    inst = InstancePoints(voxel_m=0.05)
    inst.add("a", _chair(center=(0, 0, 0.45)))
    inst.add("b", _chair(center=(0.1, 0, 0.45)))
    inst.merge("a", "b")
    assert "b" not in inst and len(inst.points("a", supported_only=False)) > 0
    inst.add("p", _chair(center=(0, 0, 0.9)), replace=True)
    inst.add("p", _chair(center=(3, 0, 0.9)), replace=True)
    assert abs(inst.box("p")["center"][0] - 3.0) < 0.1


def test_map_cloud_voxelises():
    m = MapCloud(voxel_m=0.1)
    m.add(np.array([[0.01, 0.01, 0.01], [0.02, 0.03, 0.04], [1.0, -1.0, 2.0]]))
    m.add(np.array([[0.05, 0.05, 0.05]]))
    pts = m.points()
    assert len(pts) == 2
    np.testing.assert_allclose(sorted(pts[:, 0]), [0.05, 1.05], atol=1e-5)


def test_view_and_box_projection_agree():
    # Front view looking along DA360 +Z: a point on a view pixel's ray lands on the same panorama pixel.
    W, H, vw, vh = 1920, 960, 640, 640
    K = (320.0, 320.0, 319.5, 319.5)
    R = np.eye(3)
    u, v = view_to_equirect(np.array([400.0]), np.array([200.0]), K, R, W, H)
    ray = np.array([(400.0 - K[2]) / K[0], (200.0 - K[3]) / K[1], 1.0])
    pu, pv, _ = project_level_points_to_equirect(3.0 * ray[None], np.eye(4), H, W)
    np.testing.assert_allclose([u[0], v[0]], [pu[0], pv[0]], atol=1e-6)


def test_draw_boxes_marks_the_image():
    img = np.zeros((480, 960, 3), np.uint8)
    obj = {"key": "obj1", "name": "chair", "status": "present", "center": np.zeros(3), "size": np.ones(3),
           "bbox3d": {"center": [0.0, 0.0, 3.0], "size": [0.5, 0.5, 0.9], "source": "lidar"}}
    assert draw_boxes(img, [obj], np.eye(4)) == 1
    assert img.any()
    assert box_edge_points([0, 0, 0], [1, 1, 1], samples=2).shape == (12, 2, 3)
