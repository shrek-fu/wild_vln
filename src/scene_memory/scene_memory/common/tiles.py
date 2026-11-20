"""Overlapping square windows on a 2:1 equirect, plus seam-aware NMS."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def equirect_tiles(image: np.ndarray, overlap: float = 0.25) -> list[tuple[int, np.ndarray]]:
    """Full-height square crops of an equirect, padded so the seam is covered.

    Each item is ``(origin_u, crop)``. ``origin_u`` is the crop's left edge in the
    original image and may be negative (content wrapped from the right edge).
    """
    if image.ndim != 3:
        raise ValueError(f"expected HxWxC, got {image.shape}")
    height, width = image.shape[:2]
    tile_w = min(height, width)
    if tile_w >= width:
        return [(0, image)]

    overlap = float(np.clip(overlap, 0.0, 0.9))
    stride = max(1, int(round(tile_w * (1.0 - overlap))))
    pad = tile_w // 2
    padded = np.concatenate([image[:, width - pad :], image, image[:, :pad]], axis=1)
    tiles: list[tuple[int, np.ndarray]] = []
    u = 0
    padded_w = padded.shape[1]
    while u + tile_w <= padded_w:
        origin = u - pad
        tiles.append((origin, padded[:, u : u + tile_w]))
        if origin + stride >= width:
            break
        u += stride
    if not tiles:
        tiles.append((0, image[:, :tile_w]))
    return tiles


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def circular_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    width: int,
) -> float:
    """IoU on an equirect, allowing the pair to meet across the left/right seam."""
    shifted = (
        (b[0] - width, b[1], b[2] - width, b[3]),
        b,
        (b[0] + width, b[1], b[2] + width, b[3]),
    )
    return max(box_iou(a, s) for s in shifted)


def nms_circular(dets: Iterable[dict], iou_threshold: float, width: int, agnostic: bool = False) -> list[dict]:
    """Greedy NMS. Boxes use unwrapped ``xyxy`` that may lie outside ``[0, width)``."""
    ordered = sorted(dets, key=lambda d: -float(d["score"]))
    kept: list[dict] = []
    for det in ordered:
        duplicate = False
        for prev in kept:
            if not agnostic and int(det["cls"]) != int(prev["cls"]):
                continue
            if circular_iou(det["xyxy"], prev["xyxy"], width) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def split_wrapped_box(
    xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
) -> list[tuple[float, float, float, float]]:
    """Fold one equirect box into one or two axis-aligned pieces inside the image."""
    x1, y1, x2, y2 = xyxy
    y1 = float(np.clip(y1, 0.0, height))
    y2 = float(np.clip(y2, 0.0, height))
    if y2 <= y1:
        return []
    if x2 < x1:
        x2 += width
    span = x2 - x1
    if span >= width:
        return [(0.0, y1, float(width), y2)]
    x1m = x1 % width
    x2m = x2 % width
    if abs(x2m) < 1e-6 and span > 0:
        x2m = float(width)
    if x1m <= x2m:
        return [(float(x1m), y1, float(x2m), y2)]
    return [(float(x1m), y1, float(width), y2), (0.0, y1, float(x2m), y2)]


def box_center_uv(xyxy: tuple[float, float, float, float], width: int) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    if x2 < x1:
        x2 += width
    return ((x1 + x2) * 0.5) % width, (y1 + y2) * 0.5
