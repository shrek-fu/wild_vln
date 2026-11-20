"""ByteTrack keeps an id on a moving YOLOE box and does not mix classes."""

import unittest

from scene_memory import paths
from scene_memory.perception.byte_track import ViewByteTracker


ROOT = paths.REFER / "ByteTrack"


def box(cls, name, xyxy, score=0.9):
    return {"cls": cls, "name": name, "score": score, "xyxy": xyxy}


class ByteTrackTests(unittest.TestCase):
    def tracker(self):
        return ViewByteTracker(ROOT, ["person", "chair"], track_thresh=0.25, frame_rate=10)

    def test_same_box_keeps_id_and_class_does_not_cross(self):
        tracker = self.tracker()
        first = tracker.update(
            "front",
            [box(0, "person", (10, 10, 50, 80)), box(1, "chair", (12, 12, 48, 78))],
            640,
            640,
        )
        self.assertEqual(len(first), 2)
        ids = {det["name"]: det["track_id"] for det in first}
        self.assertNotEqual(ids["person"], ids["chair"])
        second = tracker.update(
            "front",
            [box(0, "person", (14, 12, 54, 82)), box(1, "chair", (16, 14, 52, 80))],
            640,
            640,
        )
        kept = {det["name"]: det["track_id"] for det in second}
        self.assertEqual(kept, ids)

    def test_gap_inside_buffer_reuses_id(self):
        tracker = self.tracker()
        first = tracker.update("left", [box(0, "person", (100, 100, 160, 220))], 640, 640)
        person_id = first[0]["track_id"]
        tracker.update("left", [], 640, 640)
        again = tracker.update("left", [box(0, "person", (104, 102, 164, 222))], 640, 640)
        self.assertEqual(again[0]["track_id"], person_id)

    def test_views_do_not_share_a_track(self):
        tracker = self.tracker()
        front = tracker.update("front", [box(0, "person", (10, 10, 40, 80))], 640, 640)
        rear = tracker.update("rear", [box(0, "person", (10, 10, 40, 80))], 640, 640)
        self.assertNotEqual(front[0]["track_id"], rear[0]["track_id"])


if __name__ == "__main__":
    unittest.main()
