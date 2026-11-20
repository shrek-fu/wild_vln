"""Geometry, seam NMS and calibration checks. No GPU and no YOLO weights."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from scene_memory.common.calib_io import R_BODY_DA360, load_rig_calibration  # noqa: E402
from scene_memory.common.geometry import equirect_uv_to_da360_rays, rays_to_level  # noqa: E402
from scene_memory.common.tiles import (  # noqa: E402
    circular_iou,
    equirect_tiles,
    nms_circular,
    split_wrapped_box,
)

CALIB = Path("/home/foo/00WS/00qiRobotics/07DepthEnhancementPack/src/DA360/calib")
DA360 = Path("/home/foo/00WS/00qiRobotics/07DepthEnhancementPack/src/DA360")


class EquirectGeometryTest(unittest.TestCase):
    def test_panorama_centre_is_forward(self) -> None:
        width, height = 1920, 960
        u = np.array([width / 2.0 - 0.5])
        v = np.array([height / 2.0 - 0.5])
        ray = equirect_uv_to_da360_rays(u, v, width, height)[0]
        np.testing.assert_allclose(ray, [0.0, 0.0, 1.0], atol=1e-6)
        body = R_BODY_DA360 @ ray
        np.testing.assert_allclose(body, [1.0, 0.0, 0.0], atol=1e-6)

    def test_matches_da360_unprojection(self) -> None:
        sys.path.insert(0, str(DA360))
        from insta_live.geometry import da360_depth_to_points

        height, width = 48, 96
        depth = np.ones((height, width), dtype=np.float64)
        expected = da360_depth_to_points(depth).reshape(height, width, 3)
        vv, uu = np.mgrid[0:height, 0:width]
        got = equirect_uv_to_da360_rays(uu.astype(np.float64), vv.astype(np.float64), width, height)
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_level_ray_uses_extrinsic(self) -> None:
        calib = load_rig_calibration(CALIB)
        ray = np.array([[0.0, 0.0, 1.0]])
        level = rays_to_level(ray, calib.T_level_da360)[0]
        expected = calib.T_level_insta360[:3, :3] @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(level, expected, atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(level)), 1.0, places=6)


class TileAndNmsTest(unittest.TestCase):
    def test_tiles_cover_every_column(self) -> None:
        image = np.zeros((960, 1920, 3), dtype=np.uint8)
        covered = np.zeros(1920, dtype=bool)
        tiles = equirect_tiles(image, overlap=0.25)
        self.assertGreaterEqual(len(tiles), 3)
        for origin, crop in tiles:
            self.assertEqual(crop.shape[0], 960)
            self.assertEqual(crop.shape[1], 960)
            for du in range(crop.shape[1]):
                covered[(origin + du) % 1920] = True
        self.assertTrue(bool(covered.all()))

    def test_tiles_copy_across_the_seam(self) -> None:
        image = np.arange(64, dtype=np.uint8)[None, :, None].repeat(32, 0).repeat(3, 2)
        for origin, crop in equirect_tiles(image, overlap=0.25):
            for du in range(crop.shape[1]):
                self.assertEqual(int(crop[0, du, 0]), int(image[0, (origin + du) % 64, 0]))

    def test_seam_duplicates_collapse(self) -> None:
        width = 1920
        first = {"xyxy": (-10.0, 20.0, 40.0, 80.0), "score": 0.9, "cls": 0, "name": "person"}
        second = {"xyxy": (width - 10.0, 20.0, width + 40.0, 80.0), "score": 0.4, "cls": 0, "name": "person"}
        self.assertGreater(circular_iou(first["xyxy"], second["xyxy"], width), 0.9)
        kept = nms_circular([first, second], 0.5, width)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0]["score"], 0.9)

    def test_wrapped_box_splits_at_seam(self) -> None:
        parts = split_wrapped_box((-10.0, 5.0, 30.0, 40.0), 100, 50)
        self.assertEqual(parts, [(90.0, 5.0, 100.0, 40.0), (0.0, 5.0, 30.0, 40.0)])


if __name__ == "__main__":
    unittest.main()
