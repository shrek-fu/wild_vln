"""Hydra inputs for the four Insta360 views.

Each view needs three things Hydra's ``pipeline.step`` consumes:

* ``T_odom_optical`` from equirect_vio's ``odom`` to ``imu`` pose
* a metric depth image aligned with that view
* a label image painted from the tracked masks

The OpenVIO equirect camera is the DA360 frame (x right, y down, z forward).
Virtual pinhole views share that optical centre and differ by ``R_body_optical``.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scene_memory.perception.boxer_lift import _da360_from_optical
from scene_memory.common.calib_io import rotation_to_quaternion_xyzw


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(position, quaternion_xyzw) -> np.ndarray:
    """``T_parent_child`` from a ``nav_msgs/Odometry`` pose."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_xyzw_to_matrix(quaternion_xyzw)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def pose7_xyzw(transform: np.ndarray) -> np.ndarray:
    """``[x, y, z, qx, qy, qz, qw]``. Hydra's adapter swaps this to scalar-first."""
    quaternion = rotation_to_quaternion_xyzw(transform[:3, :3])
    return np.array(
        [
            transform[0, 3],
            transform[1, 3],
            transform[2, 3],
            quaternion[0],
            quaternion[1],
            quaternion[2],
            quaternion[3],
        ],
        dtype=np.float64,
    )


def load_imu_from_camera(path: str | Path) -> np.ndarray:
    """``T_imu_da360`` from an OpenVIO YAML (``p_imu = R p_camera + t``)."""
    rows: list[list[float]] = []
    seen = False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("imu_from_camera:"):
            seen = True
            continue
        if not seen:
            continue
        if stripped.startswith("- ["):
            body = stripped[stripped.find("[") + 1 : stripped.rfind("]")]
            rows.append([float(part) for part in body.split(",")])
            if len(rows) == 4:
                break
            continue
        if rows:
            break
    if len(rows) != 4:
        raise ValueError(f"imu_from_camera not found in {path}")
    transform = np.asarray(rows, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"imu_from_camera must be 4x4, got {transform.shape}")
    return transform


def load_lio_imu_from_lidar(path: str | Path) -> np.ndarray:
    """``T_imu_lidar`` from a Super-LIO YAML (``lio.extrinsic.lidar_imu`` = t then R).

    Super-LIO builds the rotation with Eigen's column-major ``M3(ptr)``; read it the same way.
    """
    import yaml

    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    params = next(iter(doc.values()))["ros__parameters"]
    values = [float(v) for v in params["lio.extrinsic.lidar_imu"]]
    if len(values) != 12:
        raise ValueError(f"lio.extrinsic.lidar_imu in {path} needs 12 values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(values[3:], dtype=np.float64).reshape(3, 3).T
    transform[:3, 3] = values[:3]
    return transform


def pose_frame_to_da360(pose_source: str, vio_config: str | Path, lio_config: str | Path, calib) -> np.ndarray:
    """``T_child_da360`` for the odometry child frame.

    ``vio``: equirect_vio ``imu`` (OpenVIO ``imu_from_camera``).
    ``lio``: Super-LIO MID-360 IMU (``livox_frame``) → front lidar → camera body (rig calibration).
    """
    if pose_source == "vio":
        return load_imu_from_camera(vio_config)
    if pose_source == "lio":
        return load_lio_imu_from_lidar(lio_config) @ np.asarray(calib.T_front_da360, dtype=np.float64)
    raise ValueError(f"unknown pose source {pose_source!r}")


def T_da360_optical(R_body_optical: np.ndarray) -> np.ndarray:
    """Virtual pinhole shares the equirect centre. Rotation only."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _da360_from_optical(np.asarray(R_body_optical, dtype=np.float64))
    return transform


def T_odom_optical(T_odom_imu: np.ndarray, T_imu_da360: np.ndarray, R_body_optical: np.ndarray) -> np.ndarray:
    """World-to-view pose. ``odom`` is the VIO world frame."""
    return np.asarray(T_odom_imu, dtype=np.float64) @ np.asarray(T_imu_da360, dtype=np.float64) @ T_da360_optical(
        R_body_optical
    )


def T_optical_level(T_level_da360: np.ndarray, R_body_optical: np.ndarray) -> np.ndarray:
    """``p_optical = T @ p_level`` for a cloud already in ``lidar_level``."""
    level_from_da360 = np.asarray(T_level_da360, dtype=np.float64)
    return np.linalg.inv(level_from_da360 @ T_da360_optical(R_body_optical))


def pinhole_intrinsics(width: int, height: int, hfov_deg: float) -> tuple[float, float, float, float]:
    fx = 0.5 * width / np.tan(np.deg2rad(hfov_deg) * 0.5)
    fy = 0.5 * height / np.tan(np.deg2rad(hfov_deg) * 0.5)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    return float(fx), float(fy), float(cx), float(cy)


def points_to_level(points: np.ndarray, frame_id: str, transforms: dict[str, np.ndarray]) -> np.ndarray:
    """Map a cloud into ``lidar_level``. Unknown frames are left unchanged."""
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    name = frame_id or "lidar_level"
    if name == "lidar_level":
        return xyz
    transform = transforms.get(name)
    if transform is None:
        return xyz
    homo = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
    return (np.asarray(transform, dtype=np.float64) @ homo.T).T[:, :3]


def depth_from_level_points(
    points_level: np.ndarray,
    T_level_da360: np.ndarray,
    R_body_optical: np.ndarray,
    width: int,
    height: int,
    hfov_deg: float,
    z_min: float = 0.25,
    z_max: float = 20.0,
) -> np.ndarray:
    """Nearest-z metric depth, metres, optical z. Pixels with no return stay 0."""
    depth = np.zeros((height, width), dtype=np.float32)
    xyz = np.asarray(points_level, dtype=np.float64).reshape(-1, 3)
    if xyz.size == 0:
        return depth
    optical = (T_optical_level(T_level_da360, R_body_optical) @ np.concatenate(
        [xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1
    ).T).T[:, :3]
    z = optical[:, 2]
    fx, fy, cx, cy = pinhole_intrinsics(width, height, hfov_deg)
    keep = (z > z_min) & (z < z_max)
    if not np.any(keep):
        return depth
    optical = optical[keep]
    z = z[keep]
    u = np.rint(fx * optical[:, 0] / z + cx).astype(np.int32)
    v = np.rint(fy * optical[:, 1] / z + cy).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(inside):
        return depth
    u = u[inside]
    v = v[inside]
    z = z[inside]
    # Farther points first, so a nearer return overwrites the same pixel.
    order = np.argsort(-z)
    depth[v[order], u[order]] = z[order].astype(np.float32)
    return depth


def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    if area <= 0.0:
        return 0.0
    return inter / area


def attach_track_masks(tracks: list[dict], detections: list[dict], min_iou: float = 0.3) -> list[dict]:
    """Copy the YOLOE mask onto the track with the same box."""
    for track in tracks:
        best = None
        best_iou = min_iou
        for det in detections:
            if not det.get("mask"):
                continue
            score = _box_iou(track["xyxy"], det["xyxy"])
            if score > best_iou:
                best_iou = score
                best = det
        if best is not None:
            track["mask"] = best["mask"]
    return tracks


def _track_pixel_mask(track: dict, width: int, height: int) -> np.ndarray:
    """Boolean HxW mask from a polygon, else the box."""
    mask = np.zeros((height, width), dtype=bool)
    poly = track.get("mask")
    if poly:
        polygon = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
        if polygon.shape[0] >= 3:
            canvas = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(canvas, [polygon], 1)
            return canvas.astype(bool)
    xyxy = track.get("xyxy")
    if not xyxy:
        return mask
    x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
    x1 = min(max(x1, 0), width - 1)
    x2 = min(max(x2, 0), width)
    y1 = min(max(y1, 0), height - 1)
    y2 = min(max(y2, 0), height)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def lift_track_obb_from_depth(
    track: dict,
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    T_odom_optical: np.ndarray,
    min_pixels: int = 16,
) -> dict | None:
    """Median-depth lift to an axis-aligned OBB in ``odom`` (yaw from XY PCA).

    This is the lightweight path for semantic map objects. Hydra itself still
    consumes the dense depth + label images, not these boxes.
    """
    height, width = depth_m.shape[:2]
    pixels = _track_pixel_mask(track, width, height)
    vs, us = np.where(pixels & (depth_m > 0.0))
    if us.size < min_pixels:
        return None
    z = depth_m[vs, us].astype(np.float64)
    x = (us.astype(np.float64) - cx) * z / fx
    y = (vs.astype(np.float64) - cy) * z / fy
    optical = np.stack([x, y, z], axis=1)
    homo = np.concatenate([optical, np.ones((optical.shape[0], 1), dtype=np.float64)], axis=1)
    odom = (np.asarray(T_odom_optical, dtype=np.float64) @ homo.T).T[:, :3]
    center = np.median(odom, axis=0)
    # Yaw from the dominant horizontal axis; size from percentile extents in that frame.
    xy = odom[:, :2] - center[:2]
    if xy.shape[0] >= 3:
        _, _, vt = np.linalg.svd(xy, full_matrices=False)
        axis = vt[0]
        yaw = float(np.arctan2(axis[1], axis[0]))
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, s], [-s, c]], dtype=np.float64)
        local = xy @ R.T
        qx = np.percentile(local[:, 0], [5, 95])
        qy = np.percentile(local[:, 1], [5, 95])
        qz = np.percentile(odom[:, 2], [5, 95])
        size = np.array([max(0.05, qx[1] - qx[0]), max(0.05, qy[1] - qy[0]), max(0.05, qz[1] - qz[0])])
        # Re-centre along the yawed axes.
        mid = np.array([(qx[0] + qx[1]) * 0.5, (qy[0] + qy[1]) * 0.5])
        center = center.copy()
        center[0] += c * mid[0] - s * mid[1]
        center[1] += s * mid[0] + c * mid[1]
        center[2] = 0.5 * (qz[0] + qz[1])
    else:
        yaw = 0.0
        span = np.percentile(odom, [5, 95], axis=0)
        size = np.maximum(span[1] - span[0], 0.05)
        center = 0.5 * (span[0] + span[1])
    return {
        "track_id": track.get("track_id"),
        "view": track.get("view"),
        "name": track.get("name"),
        "cls": int(track.get("cls", -1)),
        "score": float(track.get("score", 0.0)),
        "center_odom": [float(v) for v in center],
        "size": [float(v) for v in size],
        "yaw_rad": float(yaw),
        "depth_m": float(np.median(z)),
        "n_points": int(us.size),
    }


def paint_label_image(width: int, height: int, tracks: list[dict]) -> np.ndarray:
    """Background is 0. A track paints ``cls + 1``. Higher scores are drawn last."""
    labels = np.zeros((height, width), dtype=np.int32)
    ordered = sorted(tracks, key=lambda det: float(det.get("score", 0.0)))
    for det in ordered:
        value = int(det.get("label", int(det.get("cls", 0)) + 1))
        if value <= 0:
            continue
        mask = det.get("mask")
        if mask:
            polygon = np.asarray(mask, dtype=np.int32).reshape(-1, 2)
            if polygon.shape[0] >= 3:
                cv2.fillPoly(labels, [polygon], int(value))
                continue
        xyxy = det.get("xyxy")
        if not xyxy:
            continue
        x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
        x1 = min(max(x1, 0), width - 1)
        x2 = min(max(x2, 0), width)
        y1 = min(max(y1, 0), height - 1)
        y2 = min(max(y2, 0), height)
        if x2 > x1 and y2 > y1:
            labels[y1:y2, x1:x2] = value
    return labels


def mask_to_json(mask) -> list[list[float]] | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] < 3:
        return None
    return [[round(float(x), 1), round(float(y), 1)] for x, y in arr]


def load_level_transforms(calib_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(calib_dir) / "lidar_extrinsic.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for key, frame in (("T_level_front", "front_lidar"), ("T_level_rear", "rear_lidar")):
        if key in data:
            out[frame] = np.asarray(data[key], dtype=np.float64)
    return out


class HydraStepper:
    """Thin wrapper around ``DaaamHydraBridge`` for older call sites / tests."""

    def __init__(self, config_path: str = "") -> None:
        from scene_memory.mapping.daaam_mapping import DaaamHydraBridge

        self._bridge = DaaamHydraBridge(config_path=config_path)
        self.config_path = config_path
        self.reason = self._bridge.reason
        # Empty config is OK now (packaged habitat configs); tests that pass ""
        # still expect unavailable only when import fails.
        if self.reason:
            return

    @property
    def available(self) -> bool:
        return self._bridge.available

    @property
    def frames(self) -> int:
        return self._bridge.frames

    @property
    def pipeline(self):
        return self._bridge.pipeline

    def step(
        self,
        stamp_sec: float,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        labels: np.ndarray,
        pose7: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> bool:
        return self._bridge.process_frame(
            stamp_sec, rgb, depth_m, labels, pose7, fx, fy, cx, cy
        )
