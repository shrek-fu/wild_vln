import numpy as np

from scene_memory.mapping.box_limits import clamp_box, limit_object
from scene_memory.mapping.centroid_lift import class_max_size
from scene_memory.mapping.instance_points import InstancePoints


def test_instance_box_ignores_background_outside_the_class_limit():
    rng = np.random.default_rng(0)
    chair = rng.uniform([-0.25, -0.25, 0.0], [0.25, 0.25, 0.9], (400, 3))
    wall = np.column_stack([np.full(300, 2.5), rng.uniform(-3, 3, 300), rng.uniform(0, 2.5, 300)])  # mask edge
    inst = InstancePoints(min_hits=1)
    inst.add("c", np.vstack([chair, wall]))
    loose = inst.box("c")
    assert max(loose["size"]) > 3.0
    tight = inst.box("c", anchor=[0.0, 0.0, 0.45], max_size=class_max_size("chair"))
    assert all(s <= m + 1e-9 for s, m in zip(tight["size"], class_max_size("chair")))
    assert np.allclose(tight["center"][:2], [0.0, 0.0], atol=0.1)


def test_limit_object_clamps_saved_boxes_around_the_anchor():
    m = class_max_size("table")
    obj = {"name": "table", "center": [1.0, 2.0, 0.4], "size": [1.2, 0.7, 0.75],
           "bbox3d": {"center": [4.0, 2.0, 0.4], "size": [10.0, 1.0, 0.8], "n_voxels": 99, "source": "lidar"}}
    limit_object(obj)
    b = obj["bbox3d"]
    assert b["size"][0] <= m[0] + 1e-9 and abs(b["center"][0] - 1.0) <= m[0] / 2 and b["n_voxels"] == 99
    assert b["size"][1] == 1.0 and b["size"][2] == 0.8          # within limits: unchanged
    apart = clamp_box({"center": [9.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0]}, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
    assert apart["center"][0] == 0.0 and apart["size"][0] == 1.0  # no overlap: anchor-centred
    big = limit_object({"name": "person", "center": [0, 0, 0.8], "size": [3.0, 3.0, 5.0]})
    assert big["size"] == [round(v, 3) for v in class_max_size("person")]
