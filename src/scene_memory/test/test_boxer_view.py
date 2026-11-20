"""Pinhole views cut from the equirect, without loading BoxerNet."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_memory.perception.boxer_lift import equirect_remap, load_pinhole_views, project_equirect_box
from scene_memory.common.geometry import equirect_uv_to_da360_rays

CALIB = Path("/home/foo/00WS/00qiRobotics/07DepthEnhancementPack/src/DA360/calib")


class EquirectPinholeTest(unittest.TestCase):
    def test_front_center_is_panorama_center(self) -> None:
        views = {view["name"]: view for view in load_pinhole_views(CALIB)}
        width, height = 1920, 960
        side = 64
        map_x, map_y, _, _, cx, cy = equirect_remap(
            height, width, views["front"]["R_body_optical"], side, side, views["front"]["hfov_deg"]
        )
        # The principal point sits between pixels. The four neighbours bracket the panorama centre.
        iu, iv = int(np.floor(cx)), int(np.floor(cy))
        self.assertAlmostEqual(float(map_x[iv : iv + 2, iu : iu + 2].mean()), width / 2 - 0.5, delta=0.5)
        self.assertAlmostEqual(float(map_y[iv : iv + 2, iu : iu + 2].mean()), height / 2 - 0.5, delta=0.5)

    def test_center_box_lands_in_front_view(self) -> None:
        views = {view["name"]: view for view in load_pinhole_views(CALIB)}
        width, height = 1920, 960
        side = 320
        _, _, fx, fy, cx, cy = equirect_remap(
            height, width, views["front"]["R_body_optical"], side, side, views["front"]["hfov_deg"]
        )
        box = project_equirect_box(
            [(width / 2 - 40, height / 2 - 40, width / 2 + 40, height / 2 + 40)],
            width,
            height,
            views["front"]["R_body_optical"],
            fx,
            fy,
            cx,
            cy,
            side,
            side,
        )
        self.assertIsNotNone(box)
        x1, x2, y1, y2 = box
        self.assertLess(x1, cx)
        self.assertGreater(x2, cx)
        self.assertLess(y1, cy)
        self.assertGreater(y2, cy)
        ray = equirect_uv_to_da360_rays(np.array([width / 2]), np.array([height / 2]), width, height)[0]
        self.assertGreater(float(ray[2]), 0.9)


if __name__ == "__main__":
    unittest.main()
