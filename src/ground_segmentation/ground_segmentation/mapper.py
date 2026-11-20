"""ROS-free core: image ground labels + LIO cloud → 3D traversable ground by class, accumulated in an odom grid.

Ground classes come in two kinds:

* flat (floor, grass, ...): lie on one ground plane;
* terrain (``terrain_classes``, default stairs and step): not planar, so they are taken from the lidar alone.

Per view image at stamp ``t``:

1. the segmenter gives a label image (class index + 1 per pixel); each kind's mask is eroded so edge pixels
   do not catch walls or obstacle feet;
2. the LIO scans around ``t`` (already in ``odom``) are projected with the pose interpolated at ``t``;
3. returns inside the flat mask are ground candidates; a RANSAC plane (near horizontal) is fitted to them,
   or the last plane is reused for ``plane_hold_s`` when too few returns land in the mask;
4. flat ground = candidates within ``ground_tol`` of the plane; terrain ground = returns inside the terrain
   mask at their own height that the robot can climb from the plane (hops of at most ``max_step``, local
   slope at most ``max_slope_deg``; see ``step_reachable``); obstacles = any other return in the view
   ``obstacle_min``..``obstacle_max`` above the plane (geometry overrides semantics);
5. flat mask pixels without returns (the MID-360 sees no floor within a few metres) are filled by
   intersecting their camera rays with the plane, unless a nearer lidar return blocks the ray;
6. ground points (with their pixel's class) and obstacle points vote in the ``TraversableGrid``.

The image decides *what* the ground is, the lidar decides *where* it is.
"""

from __future__ import annotations

import time
import zlib

import cv2
import numpy as np

from scene_memory.common.pose_buffer import PoseBuffer
from scene_memory.mapping.scene_graph import T_odom_optical, pinhole_intrinsics
from scene_memory.perception.lidar_depth import CloudWindow, project_points

from ground_segmentation.geometry import (erode_mask, fit_ground_plane, heights, min_depth_image, ray_plane_fill,
                                          step_reachable)
from ground_segmentation.grid import TraversableGrid

TERRAIN_CLASSES = ("stairs", "step")
CLASS_RGB = {"floor": (0.2, 0.85, 0.3), "grass": (0.65, 0.95, 0.15), "stairs": (0.65, 0.3, 1.0),
             "step": (1.0, 0.75, 0.1), "road": (0.55, 0.6, 0.65), "ramp": (0.1, 0.8, 0.9)}


def class_rgb(name: str) -> tuple[float, float, float]:
    if name in CLASS_RGB:
        return CLASS_RGB[name]
    h = zlib.crc32(name.encode())
    return tuple(0.3 + 0.7 * ((h >> s) & 0xFF) / 255.0 for s in (0, 8, 16))


def class_bgr(name: str) -> tuple[int, int, int]:
    r, g, b = class_rgb(name)
    return int(255 * b), int(255 * g), int(255 * r)


class GroundMapper:
    def __init__(self, segmenter, T_child_da360: np.ndarray, *, classes=("floor",), terrain_classes=TERRAIN_CLASSES,
                 window_s: float = 1.0, keep_s: float = 10.0, voxel: float = 0.03, max_range: float = 15.0,
                 erode_px: int = 4, ground_tol: float = 0.08, obstacle_min: float = 0.15, obstacle_max: float = 1.5,
                 terrain_min: float = -3.0, max_step: float = 0.18, max_slope_deg: float | None = 35.0,
                 fill_range: float = 6.0, fill_stride: int = 8, plane_hold_s: float = 5.0, min_ground_points: int = 40, resolution: float = 0.1) -> None:
        self.segmenter = segmenter
        self.classes = list(getattr(segmenter, "classes", None) or classes)
        self.terrain_ids = [i + 1 for i, c in enumerate(self.classes) if c in set(terrain_classes)]
        self.flat_ids = [i + 1 for i in range(len(self.classes)) if i + 1 not in self.terrain_ids]
        self.T_child_da360 = np.asarray(T_child_da360, np.float64)
        self.poses = PoseBuffer()
        self.cloud = CloudWindow(window_s, keep_s=keep_s, voxel_m=voxel)
        self.grid = TraversableGrid(resolution, n_classes=len(self.classes))
        self.max_range = float(max_range)
        self.erode_px = int(erode_px)
        self.ground_tol = float(ground_tol)
        self.obstacle_min, self.obstacle_max = float(obstacle_min), float(obstacle_max)
        self.terrain_min = float(terrain_min)
        self.max_step = float(max_step)
        self.max_slope_deg = None if max_slope_deg is None or max_slope_deg <= 0 else float(max_slope_deg)
        self.fill_range, self.fill_stride = float(fill_range), int(fill_stride)
        self.plane_hold_s = float(plane_hold_s)
        self.min_ground_points = int(min_ground_points)
        self.plane = None
        self.plane_t = -np.inf
        self.stats = {"frames": 0, "no_pose": 0, "no_cloud": 0, "no_plane": 0, "plane_fit": 0, "plane_held": 0}

    def on_odom(self, t: float, T_odom_child: np.ndarray) -> None:
        self.poses.add(t, T_odom_child)

    def on_cloud(self, t: float, xyz_odom: np.ndarray) -> None:
        self.cloud.add(t, xyz_odom)

    def _plane(self, xyz: np.ndarray, t: float):
        recent = self.plane is not None and abs(t - self.plane_t) <= self.plane_hold_s
        fit = fit_ground_plane(xyz, dist=0.5 * self.ground_tol, min_inliers=self.min_ground_points,
                               prior=self.plane if recent else None)
        if fit is not None:
            self.plane, self.plane_t = (fit[0], fit[1]), t
            self.stats["plane_fit"] += 1
            return self.plane, "fit"
        if recent:
            self.stats["plane_held"] += 1
            return self.plane, "held"
        self.stats["no_plane"] += 1
        return None, "none"

    def process(self, view: dict, bgr: np.ndarray | None, t: float, labels: np.ndarray | None = None) -> dict | None:
        """Segment ``bgr`` (or take ``labels``: uint8 class index + 1, or a bool mask = first class), lift and
        vote; None without pose or cloud."""
        T_odom_child = self.poses.at(t)
        if T_odom_child is None:
            self.stats["no_pose"] += 1
            return None
        xyz = self.cloud.points(t)
        if len(xyz) == 0:
            self.stats["no_cloud"] += 1
            return None
        h, w = (bgr if labels is None else labels).shape[:2]
        t0 = time.perf_counter()
        lab = self.segmenter(bgr) if labels is None else labels
        lab = lab.astype(np.uint8)
        t_seg = time.perf_counter() - t0
        flat = erode_mask(np.isin(lab, self.flat_ids), self.erode_px)
        terrain = erode_mask(np.isin(lab, self.terrain_ids), self.erode_px) if self.terrain_ids else \
            np.zeros_like(flat)
        K = pinhole_intrinsics(w, h, view["hfov_deg"])
        T = T_odom_optical(T_odom_child, self.T_child_da360, view["R_body_optical"])
        u, v, z, pts = project_points(xyz, T, K, w, h, z_max=self.max_range)
        in_flat, in_terrain = flat[v, u], terrain[v, u]
        plane, plane_src = self._plane(pts[in_flat], t)
        point_label = np.zeros(len(u), np.uint8)
        obstacle = np.zeros(len(u), bool)
        result = {"t": t, "view": view["name"], "labels": lab, "mask": lab > 0, "classes": self.classes, "K": K,
                  "T_odom_optical": T, "plane": plane, "plane_src": plane_src, "uv": (u, v), "points": pts,
                  "seg_s": t_seg, "n_in_mask": int(in_flat.sum() + in_terrain.sum()), "point_label": point_label,
                  "obstacle_mask": obstacle}
        self.stats["frames"] += 1
        if plane is None:
            result.update(ground=np.zeros((0, 3)), obstacle=np.zeros((0, 3)), fill=np.zeros((0, 3)),
                          fill_label=np.zeros(0, np.uint8))
            return result
        hgt = heights(pts, plane)
        terrain_pt = in_terrain & (hgt > self.terrain_min) & (hgt < self.obstacle_max)
        idx = np.flatnonzero(terrain_pt)
        terrain_pt[idx] = step_reachable(pts[idx], hgt[idx], self.grid.res, self.max_step,
                                         max_slope_deg=self.max_slope_deg)
        ground = (in_flat & (np.abs(hgt) < self.ground_tol)) | terrain_pt
        obstacle[:] = (hgt > self.obstacle_min) & (hgt < self.obstacle_max) & ~terrain_pt
        point_label[ground] = lab[v[ground], u[ground]]
        fill, (fu, fv) = ray_plane_fill(flat, K, T, plane, stride=self.fill_stride, max_range=self.fill_range,
                                        sparse_depth=min_depth_image(u, v, z, w, h), return_pixels=True)
        fill_label = lab[fv, fu]
        self.grid.update(np.vstack([pts[ground], fill]), pts[obstacle],
                         np.concatenate([point_label[ground], fill_label]))
        result.update(ground=pts[ground], obstacle=pts[obstacle], fill=fill, fill_label=fill_label,
                      lift_s=time.perf_counter() - t0 - t_seg)
        return result


def draw_overlay(bgr: np.ndarray, result: dict) -> np.ndarray:
    """Each ground class tinted in its colour with a solid edge; lidar ground returns in the class colour,
    obstacles red, ray-filled ground small dots."""
    out = bgr.copy()
    lab = result["labels"]
    classes = result["classes"]
    for i, name in enumerate(classes, start=1):
        m = lab == i
        if not m.any():
            continue
        color = np.array(class_bgr(name))
        out[m] = (0.55 * out[m] + 0.45 * color).astype(np.uint8)
        edges = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
        out[edges] = color
    fill = result["fill"]
    if len(fill):
        T = np.linalg.inv(result["T_odom_optical"])
        cam = fill @ T[:3, :3].T + T[:3, 3]
        fx, fy, cx, cy = result["K"]
        for x, y in zip(fx * cam[:, 0] / cam[:, 2] + cx, fy * cam[:, 1] / cam[:, 2] + cy):
            cv2.circle(out, (int(round(x)), int(round(y))), 1, (0, 220, 255), -1)
    u, v = result["uv"]
    for i, name in enumerate(classes, start=1):
        color = class_bgr(name)
        for x, y in zip(*(a[result["point_label"] == i] for a in (u, v))):
            cv2.circle(out, (int(x), int(y)), 2, color, -1)
    for x, y in zip(u[result["obstacle_mask"]], v[result["obstacle_mask"]]):
        cv2.circle(out, (int(x), int(y)), 2, (0, 0, 255), -1)
    plane = result["plane"]
    text = f"{result['view']} plane:{result['plane_src']}"
    if plane is not None:
        text += f" n_z={plane[0][2]:.3f} d={plane[1]:.2f}"
    text += f" ground={len(result['ground'])} obst={len(result['obstacle'])} fill={len(fill)}"
    y = 22
    for line in [text, " ".join(f"{name}={int((lab == i).mean() * 100)}%" for i, name in enumerate(classes, 1))]:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return out


def grid_image(grid: TraversableGrid, classes=(), scale: int = 4) -> np.ndarray:
    """Top view: traversable cells in their class colour, red obstacle, dark unknown or undecided; x right, y up."""
    _, w, h, data = grid.class_occupancy()
    if w == 0:
        return np.zeros((10, 10, 3), np.uint8)
    data = data.reshape(h, w)
    img = np.full((h, w, 3), 40, np.uint8)
    img[data == 0] = (60, 200, 60)
    for i, name in enumerate(classes, start=1):
        img[data == i] = class_bgr(name)
    img[data == 100] = (40, 40, 220)
    return cv2.resize(img[::-1], (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
