"""DAAAM-aligned label/color images and depth clipping."""

import unittest

import numpy as np

from scene_memory.mapping.daaam_mapping import clip_depth_meters, create_label_and_color
from scene_memory.common.labelspace import load_labelspace
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / "config" / "prompts.txt"


class DaaamMappingTest(unittest.TestCase):
    def test_label_image_uses_semantic_id_not_track_id(self):
        space = load_labelspace(PROMPTS)
        tracks = [
            {
                "track_id": 42,
                "cls": 2,  # chair → semantic 3
                "name": "chair",
                "score": 0.9,
                "mask": [(10, 10), (40, 10), (40, 40), (10, 40)],
            }
        ]
        labels, color = create_label_and_color(64, 64, tracks, space)
        self.assertEqual(int(labels[20, 20]), 3)
        self.assertEqual(int(labels[0, 0]), 0)
        self.assertEqual(color.shape, (64, 64, 3))
        self.assertTrue(int(color[20, 20].sum()) > 0)

    def test_depth_clip_matches_daaam_bounds(self):
        depth = np.array([[0.1, 1.0, 25.0]], dtype=np.float32)
        clipped = clip_depth_meters(depth, 0.25, 20.0)
        self.assertEqual(float(clipped[0, 0]), 0.0)
        self.assertEqual(float(clipped[0, 1]), 1.0)
        self.assertEqual(float(clipped[0, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
