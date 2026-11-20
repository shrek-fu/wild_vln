"""Gravity OBB fit and equirect point association."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_memory.common.geometry import equirect_uv_to_da360_rays, project_level_points_to_equirect
from scene_memory.perception.lift import points_in_detection
from scene_memory.common.obb import fit_gravity_obb


class GravityObbTest(unittest.TestCase):
    def test_axis_aligned_box(self) -> None:
        rng = np.random.default_rng(0)
        pts = np.column_stack(
            [
                rng.uniform(1.0, 3.0, 80),
                rng.uniform(-0.4, 0.6, 80),
                rng.uniform(0.2, 1.2, 80),
            ]
        )
        obb = fit_gravity_obb(pts)
        self.assertIsNotNone(obb)
        np.testing.assert_allclose(obb["center"], [2.0, 0.1, 0.7], atol=0.05)
        np.testing.assert_allclose(sorted(obb["size"][:2]), [1.0, 2.0], atol=0.08)
        self.assertAlmostEqual(obb["size"][2], 1.0, delta=0.05)

    def test_yawed_box_stays_gravity_aligned(self) -> None:
        yaw = 0.45
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        rng = np.random.default_rng(1)
        local = np.column_stack(
            [
                rng.uniform(-1.0, 1.0, 300),
                rng.uniform(-0.25, 0.25, 300),
                rng.uniform(0.0, 1.0, 300),
            ]
        )
        world = local @ rotation.T + np.array([3.0, 1.0, 0.5])
        obb = fit_gravity_obb(world)
        self.assertIsNotNone(obb)
        np.testing.assert_allclose(obb["center"], [3.0, 1.0, 1.0], atol=0.15)
        self.assertAlmostEqual(obb["size"][2], 1.0, delta=0.1)
        self.assertAlmostEqual(float(np.max(obb["size"][:2])), 2.0, delta=0.25)
        self.assertAlmostEqual(float(np.min(obb["size"][:2])), 0.5, delta=0.2)


class EquirectLiftTest(unittest.TestCase):
    def test_projection_roundtrip(self) -> None:
        width, height = 96, 48
        u = np.array([10.0, 40.0, 80.0])
        v = np.array([8.0, 24.0, 40.0])
        points = equirect_uv_to_da360_rays(u, v, width, height) * 3.0
        got_u, got_v, depth = project_level_points_to_equirect(points, np.eye(4), height, width)
        np.testing.assert_allclose(got_u, u, atol=1e-6)
        np.testing.assert_allclose(got_v, v, atol=1e-6)
        np.testing.assert_allclose(depth, 3.0, atol=1e-6)

    def test_depth_slab_drops_the_wall_behind(self) -> None:
        width, height = 64, 32
        rng = np.random.default_rng(2)
        near = np.column_stack(
            [
                rng.uniform(-0.15, 0.15, 50),
                rng.uniform(-0.15, 0.15, 50),
                rng.uniform(2.7, 3.2, 50),
            ]
        )
        wall = np.column_stack(
            [
                rng.uniform(-0.4, 0.4, 50),
                rng.uniform(-0.4, 0.4, 50),
                np.full(50, 9.0),
            ]
        )
        kept = points_in_detection(
            np.vstack([near, wall]),
            np.eye(4),
            [(0.0, 0.0, float(width), float(height))],
            width,
            height,
            depth_extent=1.5,
            max_range=20.0,
        )
        self.assertGreaterEqual(kept.shape[0], 20)
        self.assertLess(float(kept[:, 2].max()), 5.0)


if __name__ == "__main__":
    unittest.main()
