from pathlib import Path

import numpy as np
import pytest

from scene_memory import paths
from scene_memory.common.calib_io import rotation_to_quaternion_xyzw
from scene_memory.mapping.scene_graph import quat_xyzw_to_matrix
from scene_memory.nodes.camera_tf_node import transforms

needs_calib = pytest.mark.skipif(not Path(paths.CALIB).is_dir(), reason="no calibration")


@needs_calib
def test_view_frames_look_where_they_are_named():
    tf = {child: R for _, child, R in transforms(str(paths.CALIB))}
    forward = {"front": [1, 0, 0], "left": [0, 1, 0], "rear": [-1, 0, 0], "right": [0, -1, 0]}
    for view, axis in forward.items():
        z_body = tf[f"insta360_{view}_optical"] @ np.array([0.0, 0.0, 1.0])
        assert np.dot(z_body, axis) > 0.9, view
        y_body = tf[f"insta360_{view}_optical"] @ np.array([0.0, 1.0, 0.0])
        assert y_body[2] < -0.9, f"{view}: optical +y should point down"
    np.testing.assert_allclose(tf["insta360_equirect"] @ [0, 0, 1], [1, 0, 0], atol=1e-9)


@needs_calib
def test_quaternions_round_trip():
    for _, _, R in transforms(str(paths.CALIB)):
        np.testing.assert_allclose(quat_xyzw_to_matrix(rotation_to_quaternion_xyzw(np.asarray(R))), R, atol=1e-6)
