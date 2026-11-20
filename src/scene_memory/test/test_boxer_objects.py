"""Depth/label cleanup before Hydra and the per-track Boxer OBB map."""

import unittest

import numpy as np

from scene_memory.mapping.boxer_objects import BoxerObjectMap, backproject_depth
from scene_memory.mapping.daaam_mapping import depth_edge_mask, erode_label_edges


class CleanupTest(unittest.TestCase):
    def test_depth_edges_only_on_discontinuity(self):
        depth = np.full((40, 40), 5.0, dtype=np.float32)
        depth[10:30, 10:30] = 2.0
        edges = depth_edge_mask(depth, ratio=0.08)
        self.assertTrue(edges[10, 15] and edges[9, 15])
        self.assertFalse(edges[20, 20] or edges[0, 0])

    def test_label_erosion_shrinks_mask(self):
        labels = np.zeros((40, 40), dtype=np.int32)
        labels[10:30, 10:30] = 3
        eroded = erode_label_edges(labels, radius=3)
        self.assertEqual(int(np.count_nonzero(eroded)), 14 * 14)
        self.assertTrue(np.all(eroded[13:27, 13:27] == 3))


class BoxerObjectMapTest(unittest.TestCase):
    def test_backproject_centre_pixel(self):
        depth = np.zeros((9, 9), dtype=np.float32)
        depth[4, 4] = 3.0
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 0.0]
        points = backproject_depth(depth, 10.0, 10.0, 4.0, 4.0, T, stride=1)
        np.testing.assert_allclose(points, [[1.0, 2.0, 3.0]], atol=1e-6)

    def test_median_size_and_hidden_not_deleted(self):
        objects = BoxerObjectMap(ttl_s=1.0)
        det = {"track_id": 7, "cls": 1, "name": "chair", "obb": {"center": [0, 0, 0], "size": [1, 1, 1]}}
        objects.update([det], 0.0)
        det["obb"] = {"center": [0.2, 0, 0], "size": [3, 1, 1]}
        objects.update([det], 0.5)
        det["obb"] = {"center": [0.1, 0, 0], "size": [1.2, 1, 1]}
        objects.update([det], 0.6)
        box = objects.boxes()[0]
        np.testing.assert_allclose(box["center"], [0.1, 0, 0])
        np.testing.assert_allclose(box["corners"].max(0) - box["corners"].min(0), [1.2, 1, 1])
        objects.expire(5.0)
        hidden = objects.boxes()
        self.assertEqual(len(hidden), 1)
        self.assertFalse(hidden[0]["visible"])
        self.assertEqual(objects.boxes(include_hidden=False), [])

    def test_truncated_box_keeps_size(self):
        objects = BoxerObjectMap()
        full = {"track_id": 1, "cls": 0, "xyxy": [100, 100, 200, 200], "obb": {"center": [0, 0, 0], "size": [1, 1, 1]}}
        cut = {"track_id": 1, "cls": 0, "xyxy": [500, 100, 639, 200], "obb": {"center": [0, 0, 0], "size": [0.3, 1, 1]}}
        objects.update([full], 0.0, 640, 640)
        objects.update([cut], 0.5, 640, 640)
        np.testing.assert_allclose(objects.boxes()[0]["size"], [1, 1, 1])

    def test_new_track_reattaches_to_hidden_object(self):
        objects = BoxerObjectMap(ttl_s=1.0)
        objects.update([{"track_id": 1, "cls": 2, "name": "tv", "obb": {"center": [1, 0, 0], "size": [1, 1, 1]}}], 0.0)
        objects.expire(3.0)
        objects.update([{"track_id": 9, "cls": 2, "name": "tv", "obb": {"center": [1.2, 0, 0], "size": [1, 1, 1]}}], 4.0)
        objects.update([{"track_id": 10, "cls": 3, "name": "cup", "obb": {"center": [1.2, 0, 0], "size": [1, 1, 1]}}], 4.0)
        saved = objects.to_json()["objects"]
        self.assertEqual(len(saved), 2)
        tv = [o for o in saved if o["name"] == "tv"][0]
        self.assertEqual(tv["track_ids"], [1, 9])
        self.assertTrue(tv["visible"])
        self.assertEqual(tv["first_seen"], 0.0)


class KeyViewStoreTest(unittest.TestCase):
    def test_best_view_per_visit(self):
        import tempfile
        from pathlib import Path

        from scene_memory.mapping.object_views import KeyViewStore

        with tempfile.TemporaryDirectory() as tmp:
            objects = BoxerObjectMap(ttl_s=1.0)
            store = KeyViewStore(tmp)
            bgr = np.zeros((640, 640, 3), dtype=np.uint8)
            small = {"track_id": 1, "cls": 0, "name": "cup", "score": 0.9, "xyxy": [300, 300, 360, 360],
                     "obb": {"center": [0, 0, 0], "size": [0.1, 0.1, 0.1]}}
            big = dict(small, xyxy=[200, 200, 400, 400])
            for det, t in ((small, 0.0), (big, 0.5), (dict(small), 0.6)):
                objects.update([det], t, 640, 640)
                store.offer(objects.objects[det["memory_key"]], det, bgr, t)
            obj = objects.objects["obj0001"]
            self.assertEqual(len(obj["views"]), 1)
            self.assertEqual(obj["views"][0]["stamp"], 0.5)
            objects.expire(5.0)
            again = dict(small)
            objects.update([again], 6.0, 640, 640)
            store.offer(obj, again, bgr, 6.0)
            self.assertEqual([v["visit"] for v in obj["views"]], [0, 1])
            for view in obj["views"]:
                self.assertTrue((Path(tmp) / view["image"]).is_file())
                self.assertTrue((Path(tmp) / view["mask"]).is_file())


if __name__ == "__main__":
    unittest.main()
