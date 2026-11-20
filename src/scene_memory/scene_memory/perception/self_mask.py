"""Robot self-occlusion mask for the pinhole views (the robot's own head / body seen by the camera).

``config/self_mask.yaml`` holds, per view, polygons in normalised image coordinates (x, y in 0..1, origin top
left), so they do not depend on the view size. The camera is rigid on the robot, so the regions are fixed as long
as the view layout (``insta360_views.yaml``: view_hfov_deg, front_yaw_deg) does not change.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

FILL = (114, 114, 114)


class SelfMask:
    def __init__(self, path: str | Path, dilate_px: int = 8) -> None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        self.polygons = {str(view): [np.asarray(p, dtype=np.float64) for p in polys or []]
                         for view, polys in data.items()}
        self.dilate_px = int(dilate_px)
        self._cache: dict[tuple[str, int, int], np.ndarray | None] = {}

    def mask(self, view: str, height: int, width: int) -> np.ndarray | None:
        """bool HxW, True on the robot; None when the view has no region."""
        key = (view, height, width)
        if key not in self._cache:
            polys = self.polygons.get(view)
            if not polys:
                self._cache[key] = None
            else:
                m = np.zeros((height, width), np.uint8)
                scale = np.array([width - 1, height - 1])
                cv2.fillPoly(m, [np.round(p * scale).astype(np.int32) for p in polys], 1)
                if self.dilate_px > 0:
                    k = 2 * self.dilate_px + 1
                    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
                self._cache[key] = m.astype(bool)
        return self._cache[key]

    def paint(self, view: str, bgr: np.ndarray) -> np.ndarray:
        """Copy of ``bgr`` with the robot region filled flat grey (the detector input)."""
        m = self.mask(view, *bgr.shape[:2])
        if m is None:
            return bgr
        out = bgr.copy()
        out[m] = FILL
        return out

    def overlap(self, view: str, xyxy, height: int, width: int) -> float:
        """Fraction of the box inside the robot region."""
        m = self.mask(view, height, width)
        if m is None:
            return 0.0
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, width), min(y2, height)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return float(m[y1:y2, x1:x2].mean())
