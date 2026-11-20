"""Pose, metric depth and label image for one virtual pinhole."""

import unittest

import numpy as np

from scene_memory import paths
from scene_memory.perception.boxer_lift import load_pinhole_views
from scene_memory.common.calib_io import load_rig_calibration
from scene_memory.mapping.scene_graph import (
    HydraStepper,
    T_odom_optical,
    attach_track_masks,
    depth_from_level_points,
    lift_track_obb_from_depth,
    load_imu_from_camera,
    paint_label_image,
    pinhole_intrinsics,
    pose7_xyzw,
    pose_to_matrix,
)

CALIB = paths.CALIB
VIO = paths.VIO_CONFIG


class SceneGraphTest(unittest.TestCase):
    def test_pose_chain_keeps_a_known_translation(self):
        T_odom_imu = pose_to_matrix((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
        T_imu_da360 = load_imu_from_camera(VIO)
        front = next(view for view in load_pinhole_views(CALIB) if view["name"] == "front")
        pose = T_odom_optical(T_odom_imu, T_imu_da360, front["R_body_optical"])
        # Camera and IMU origins coincide in this calibration, so the view origin stays at the IMU.
        np.testing.assert_allclose(pose[:3, 3], [1.0, 2.0, 3.0], atol=1e-6)
        stored = pose7_xyzw(pose)
        self.assertEqual(stored.shape, (7,))
        np.testing.assert_allclose(stored[:3], [1.0, 2.0, 3.0], atol=1e-6)

    def test_optical_axis_point_lands_on_the_principal_point(self):
        rig = load_rig_calibration(CALIB)
        front = next(view for view in load_pinhole_views(CALIB) if view["name"] == "front")
        # A point 2 m along the front optical axis, expressed in lidar_level.
        from scene_memory.mapping.scene_graph import T_da360_optical

        T_level_opt = rig.T_level_da360 @ T_da360_optical(front["R_body_optical"])
        point = (T_level_opt @ np.array([0.0, 0.0, 2.0, 1.0]))[:3]
        depth = depth_from_level_points(
            point.reshape(1, 3), rig.T_level_da360, front["R_body_optical"], 640, 640, front["hfov_deg"]
        )
        cx = int(np.rint((640 - 1) * 0.5))
        self.assertAlmostEqual(float(depth[cx, cx]), 2.0, places=2)
        self.assertEqual(int(np.count_nonzero(depth)), 1)

    def test_mask_paints_class_and_background_stays_zero(self):
        tracks = attach_track_masks(
            [{"xyxy": (10, 10, 30, 40), "score": 0.9, "cls": 2, "name": "chair"}],
            [{"xyxy": (10, 10, 30, 40), "score": 0.9, "cls": 2, "mask": [(10, 10), (30, 10), (30, 40), (10, 40)]}],
        )
        labels = paint_label_image(64, 64, tracks)
        self.assertEqual(int(labels[20, 20]), 3)
        self.assertEqual(int(labels[0, 0]), 0)
        stepper = HydraStepper("")
        # Empty config is fine: packaged habitat/ade20k_mp3d configs are used.
        # Unavailable only when hydra_python cannot be imported.
        if stepper.available:
            self.assertEqual(stepper.reason, "")
        else:
            self.assertIn("hydra", stepper.reason.lower())

    def test_depth_mask_lifts_to_odom_obb(self):
        depth = np.zeros((64, 64), dtype=np.float32)
        depth[20:40, 20:40] = 2.0
        fx, fy, cx, cy = pinhole_intrinsics(64, 64, 90.0)
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = 1.0
        track = {
            "xyxy": (20, 20, 40, 40),
            "mask": [(20, 20), (40, 20), (40, 40), (20, 40)],
            "cls": 0,
            "name": "person",
            "score": 0.9,
            "track_id": 7,
            "view": "front",
        }
        obb = lift_track_obb_from_depth(track, depth, fx, fy, cx, cy, T, min_pixels=8)
        self.assertIsNotNone(obb)
        # Box is slightly left of the principal point; odom x ≈ 1 + optical_x.
        self.assertAlmostEqual(obb["center_odom"][0], 0.875, places=2)
        self.assertAlmostEqual(obb["center_odom"][2], 2.0, places=1)
        self.assertEqual(obb["track_id"], 7)


if __name__ == "__main__":
    unittest.main()
