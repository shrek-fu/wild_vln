"""SuperMap-style panorama overlay: 2D detections (top) and the 3D object boxes projected back (bottom).

ROS-free; ``topo_map_node --pano_overlay`` feeds it the equirect frame of each detector frame.
Colours come from the class name, so the same object has the same colour here and in the 3D markers.
"""

from __future__ import annotations

import colorsys
import zlib

import cv2
import numpy as np

from scene_memory.common.geometry import project_level_points_to_equirect

_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
_SIGNS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                   [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)


def class_rgb(name: str) -> tuple[float, float, float]:
    """Bright colour per class name, 0-1 RGB."""
    hue = (zlib.crc32(str(name).lower().encode()) % 997) / 997.0
    return colorsys.hsv_to_rgb(hue, 0.85, 1.0)


def class_bgr(name: str) -> tuple[int, int, int]:
    r, g, b = class_rgb(name)
    return int(255 * b), int(255 * g), int(255 * r)


def box_corners(center, size) -> np.ndarray:
    return np.asarray(center, dtype=np.float64) + 0.5 * _SIGNS * np.asarray(size, dtype=np.float64)


def box_edge_points(center, size, samples: int = 24) -> np.ndarray:
    """``(12, samples, 3)`` points along the AABB edges."""
    c = box_corners(center, size)
    s = np.linspace(0.0, 1.0, samples)[None, :, None]
    a = np.stack([c[i] for i, _ in _EDGES])[:, None, :]
    b = np.stack([c[j] for _, j in _EDGES])[:, None, :]
    return a + s * (b - a)


def view_to_equirect(u, v, K, R_da360_optical: np.ndarray, width: int, height: int):
    """Pinhole view pixels to equirect pixels (rotation only; both share the camera centre)."""
    fx, fy, cx, cy = K
    rays = np.stack([(np.asarray(u, float) - cx) / fx, (np.asarray(v, float) - cy) / fy, np.ones(np.shape(u))], -1)
    rays = rays @ np.asarray(R_da360_optical, dtype=np.float64).T
    r = np.linalg.norm(rays, axis=-1)
    theta = np.arccos(np.clip(rays[..., 1] / r, -1.0, 1.0))
    phi = np.arctan2(rays[..., 0], rays[..., 2])
    return (phi + np.pi) * width / (2.0 * np.pi) - 0.5, (np.pi - theta) * height / np.pi - 0.5


def _polyline(img, u, v, color, thickness: int) -> None:
    """Draws ``u, v`` and breaks the line where it wraps around the panorama seam."""
    w = img.shape[1]
    pts = np.stack([u, v], -1)
    cut = np.flatnonzero(np.abs(np.diff(u)) > 0.5 * w) + 1
    for seg in np.split(pts, cut):
        if len(seg) >= 2:
            cv2.polylines(img, [np.rint(seg).astype(np.int32)], False, color, thickness, cv2.LINE_AA)


def _label(img, text: str, x: float, y: float, color, scale: float) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(np.clip(x, 0, img.shape[1] - tw - 2))
    y = int(np.clip(y, th + 4, img.shape[0] - 1))
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_detections(img, detections: list[dict], views: dict, view_size: tuple[int, int], R_da360_optical: dict,
                    K_of) -> None:
    """Detection boxes of the pinhole views, bent onto the panorama."""
    h, w = img.shape[:2]
    vw, vh = view_size
    scale = max(0.4, w / 3200.0)
    t = np.linspace(0.0, 1.0, 16)
    for det in detections:
        name = det.get("view")
        if name not in views:
            continue
        x1, y1, x2, y2 = (float(c) for c in det["xyxy"])
        u = np.concatenate([x1 + (x2 - x1) * t, np.full(16, x2), x2 - (x2 - x1) * t, np.full(16, x1)])
        v = np.concatenate([np.full(16, y1), y1 + (y2 - y1) * t, np.full(16, y2), y2 - (y2 - y1) * t])
        pu, pv = view_to_equirect(u, v, K_of(name), R_da360_optical[name], w, h)
        color = class_bgr(det.get("name", ""))
        _polyline(img, pu, pv, color, 2)
        tid = det.get("track_id")
        text = f"{det.get('name', '')} {tid if tid is not None else ''} {float(det.get('score', 0)):.2f}".strip()
        _label(img, text, pu[0], pv[0] - 2, color, scale)


def draw_boxes(img, objects: list[dict], T_odom_da360: np.ndarray, max_range: float = 15.0) -> int:
    """Each object's 3D box (``bbox3d``, else its centre / class size) projected with the frame's pose."""
    h, w = img.shape[:2]
    scale = max(0.4, w / 3200.0)
    cam = np.asarray(T_odom_da360, dtype=np.float64)[:3, 3]
    drawn = 0
    for obj in objects:
        if obj.get("status", "present") != "present":
            continue
        box = obj.get("bbox3d") or {"center": obj["center"], "size": obj["size"], "source": "prior"}
        if np.linalg.norm(np.asarray(box["center"]) - cam) > max_range:
            continue
        pts = box_edge_points(box["center"], box["size"]).reshape(-1, 3)
        u, v, _ = project_level_points_to_equirect(pts, T_odom_da360, h, w)
        if len(u) != len(pts):
            continue
        u, v = u.reshape(12, -1), v.reshape(12, -1)
        color = class_bgr(obj.get("name", ""))
        thick = 2 if box.get("source") != "prior" else 1
        for e in range(12):
            _polyline(img, u[e], v[e], color, thick)
        i = int(np.argmin(v))
        _label(img, f"{obj.get('name', '')} {obj.get('key', '')}", u.flat[i], v.flat[i] - 2, color, scale)
        drawn += 1
    return drawn


def render(equirect_bgr: np.ndarray, detections: list[dict], objects: list[dict], T_odom_da360: np.ndarray,
           views: dict, view_size: tuple[int, int], R_da360_optical: dict, K_of, out_width: int = 1280) -> np.ndarray:
    """Top: 2D detections. Bottom: 3D boxes. Both ``out_width`` wide."""
    top, bottom = equirect_bgr.copy(), equirect_bgr.copy()
    draw_detections(top, detections, views, view_size, R_da360_optical, K_of)
    draw_boxes(bottom, objects, T_odom_da360)
    out = np.vstack([top, bottom])
    if out.shape[1] != out_width:
        out = cv2.resize(out, (out_width, out.shape[0] * out_width // out.shape[1]), interpolation=cv2.INTER_AREA)
    return out
