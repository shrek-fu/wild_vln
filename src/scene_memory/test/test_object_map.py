"""SuperMap-style persistent object memory."""

import unittest

import numpy as np

from scene_memory.mapping.object_map import SuperMapMemory, map_payload


class ObjectMapTest(unittest.TestCase):
    def test_same_track_keeps_id_and_gains_relations(self):
        memory = SuperMapMemory(match_dist=1.0, stale_sec=8.0)
        a = {
            "track_id": 1,
            "view": "front",
            "name": "chair",
            "cls": 2,
            "score": 0.9,
            "center_odom": [1.0, 0.0, 0.5],
            "size": [0.5, 0.5, 0.9],
            "yaw_rad": 0.1,
            "points_odom": np.array([[1.0, 0.0, 0.5], [1.1, 0.0, 0.5]], dtype=np.float32),
        }
        b = {
            "track_id": 2,
            "view": "front",
            "name": "table",
            "cls": 3,
            "score": 0.8,
            "center_odom": [1.6, 0.0, 0.4],
            "size": [0.8, 0.8, 0.7],
            "yaw_rad": 0.0,
            "points_odom": np.array([[1.6, 0.0, 0.4]], dtype=np.float32),
        }
        rows = memory.update([a], 1.0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], 1)
        self.assertEqual(rows[0]["status"], "active")
        rows = memory.update([{**a, "center_odom": [1.05, 0.0, 0.5]}, b], 1.2)
        self.assertEqual(len(memory.objects), 2)
        chair = next(r for r in rows if r["label"] == "chair")
        self.assertEqual(chair["id"], 1)
        self.assertTrue(any(rel["relation"] == "beside" for rel in chair["spatial_relations"]))
        payload = map_payload(rows, 1.2)
        self.assertIn("bbox3d", payload)
        self.assertIn("spatial_relations", payload)


if __name__ == "__main__":
    unittest.main()
