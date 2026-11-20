"""Draw seam-aware boxes on an equirectangular BGR image."""

from __future__ import annotations

import zlib

import cv2
import numpy as np

from scene_memory.common.tiles import split_wrapped_box


def color_bgr(name: str) -> tuple[int, int, int]:
    h = zlib.crc32(name.encode("utf-8"))
    return (int(h & 255), int((h >> 8) & 255), int((h >> 16) & 255))


def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for det in detections:
        color = color_bgr(str(det["name"]))
        parts = det.get("parts")
        if not parts:
            parts = split_wrapped_box(det["xyxy"], width, height)
        for x1, y1, x2, y2 in parts:
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            cv2.rectangle(canvas, p1, p2, color, 2)
        track_id = det.get("track_id")
        prefix = f"#{int(track_id)} " if track_id is not None else ""
        label = f"{prefix}{det['name']} {float(det['score']):.2f}"
        anchor = parts[0] if parts else None
        if anchor is None:
            continue
        tx, ty = int(round(anchor[0])), max(0, int(round(anchor[1])) - 4)
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas
