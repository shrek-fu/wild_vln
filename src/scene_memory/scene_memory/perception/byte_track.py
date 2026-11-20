"""Run the vendored ByteTrack on YOLOE boxes and publish only confirmed tracks.

One tracker is kept per camera view and class, so a person cannot inherit a
chair's id. Boxes are already in the view's pixel frame; ByteTrack is told the
image and the network size are the same, which keeps its internal resize at 1.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np


class _TrackArgs:
    def __init__(self, track_thresh: float, track_buffer: int, match_thresh: float) -> None:
        self.track_thresh = float(track_thresh)
        self.track_buffer = int(track_buffer)
        self.match_thresh = float(match_thresh)
        self.mot20 = False


def _bbox_overlaps(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """IoU of tlbr boxes. Same result ByteTrack expects from cython_bbox."""
    boxes = np.asarray(boxes, dtype=np.float64)
    query = np.asarray(query_boxes, dtype=np.float64)
    if boxes.size == 0 or query.size == 0:
        return np.zeros((len(boxes), len(query)), dtype=np.float64)
    lt = np.maximum(boxes[:, None, :2], query[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], query[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0.0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    area_b = np.clip(query[:, 2] - query[:, 0], 0.0, None) * np.clip(query[:, 3] - query[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0)


def _lapjv(cost: np.ndarray, extend_cost: bool = False, cost_limit: float | None = None):
    """Jonker-Volgenant assignment with the cost limit ByteTrack passes to lap."""
    del extend_cost
    from scipy.optimize import linear_sum_assignment

    cost = np.asarray(cost, dtype=np.float64)
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return 0.0, np.full(rows, -1, dtype=int), np.full(cols, -1, dtype=int)
    work = cost.copy()
    if cost_limit is not None:
        work[work > cost_limit] = 1.0e6
    assigned_rows, assigned_cols = linear_sum_assignment(work)
    row_to_col = np.full(rows, -1, dtype=int)
    col_to_row = np.full(cols, -1, dtype=int)
    for row, col in zip(assigned_rows, assigned_cols):
        if cost_limit is not None and cost[row, col] > cost_limit:
            continue
        row_to_col[row] = int(col)
        col_to_row[col] = int(row)
    return 0.0, row_to_col, col_to_row


def _load_byte_tracker(root: Path):
    """Import BYTETracker from the reference tree without its training package."""
    if not hasattr(np, "float"):
        np.float = np.float64  # type: ignore[attr-defined]
    root = root.expanduser().resolve()
    tracker_dir = root / "yolox" / "tracker"
    if not (tracker_dir / "byte_tracker.py").is_file():
        raise FileNotFoundError(f"ByteTrack tracker not found under {root}")
    if "cython_bbox" not in sys.modules:
        try:
            importlib.import_module("cython_bbox")
        except ImportError:
            shim = types.ModuleType("cython_bbox")
            shim.bbox_overlaps = _bbox_overlaps
            sys.modules["cython_bbox"] = shim
    if "lap" not in sys.modules:
        try:
            importlib.import_module("lap")
        except ImportError:
            shim = types.ModuleType("lap")
            shim.lapjv = _lapjv
            sys.modules["lap"] = shim
    yolox_dir = root / "yolox"
    if "yolox" not in sys.modules:
        package = types.ModuleType("yolox")
        package.__path__ = [str(yolox_dir)]  # type: ignore[attr-defined]
        sys.modules["yolox"] = package
    if "yolox.tracker" not in sys.modules:
        package = types.ModuleType("yolox.tracker")
        package.__path__ = [str(tracker_dir)]  # type: ignore[attr-defined]
        sys.modules["yolox.tracker"] = package
        sys.modules["yolox"].tracker = package
    module = importlib.import_module("yolox.tracker.byte_tracker")
    return module.BYTETracker


class ViewByteTracker:
    """Confirmed ByteTrack targets for every camera view."""

    def __init__(
        self,
        bytetrack_root: str | Path,
        names: list[str],
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: float = 10.0,
    ) -> None:
        self._byte_tracker = _load_byte_tracker(Path(bytetrack_root))
        self.names = [str(name) for name in names]
        self.args = _TrackArgs(track_thresh, track_buffer, match_thresh)
        self.frame_rate = max(1, int(round(frame_rate)))
        self._trackers: dict[tuple[str, int], object] = {}

    def update(self, view: str, detections: list[dict], width: int, height: int) -> list[dict]:
        """Return activated tracks. A new id is kept across later IoU matches."""
        by_class: dict[int, list[dict]] = {}
        for det in detections:
            by_class.setdefault(int(det["cls"]), []).append(det)
        classes = set(by_class)
        classes.update(cls for tracked_view, cls in self._trackers if tracked_view == view)
        targets: list[dict] = []
        image_info = (int(height), int(width))
        for cls in sorted(classes):
            tracker = self._trackers.get((view, cls))
            if tracker is None:
                tracker = self._byte_tracker(self.args, frame_rate=self.frame_rate)
                self._trackers[(view, cls)] = tracker
            rows = self._rows(by_class.get(cls, []))
            online = tracker.update(rows, image_info, image_info)
            name = self.names[cls] if 0 <= cls < len(self.names) else str(cls)
            for track in online:
                target = self._target(view, cls, name, track, width, height)
                if target is not None:
                    targets.append(target)
        targets.sort(key=lambda det: -float(det["score"]))
        return targets

    @staticmethod
    def _rows(detections: list[dict]) -> np.ndarray:
        if not detections:
            return np.zeros((0, 5), dtype=np.float32)
        rows = np.empty((len(detections), 5), dtype=np.float32)
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["xyxy"]
            rows[i] = (x1, y1, x2, y2, float(det["score"]))
        return rows

    @staticmethod
    def _target(view: str, cls: int, name: str, track, width: int, height: int) -> dict | None:
        x1, y1, x2, y2 = [float(v) for v in track.tlbr]
        x1 = min(max(x1, 0.0), float(width - 1))
        y1 = min(max(y1, 0.0), float(height - 1))
        x2 = min(max(x2, 0.0), float(width - 1))
        y2 = min(max(y2, 0.0), float(height - 1))
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            return None
        return {
            "xyxy": (x1, y1, x2, y2),
            "score": float(track.score),
            "cls": int(cls),
            "name": name,
            "view": view,
            "track_id": int(track.track_id),
            "parts": [(x1, y1, x2, y2)],
        }
