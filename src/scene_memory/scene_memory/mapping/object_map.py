"""Persistent open-vocabulary object map in the SuperMap live schema.

SuperMap's code is not public yet. The README documents the live ROS interface:

* ``/obj_boxes`` / ``/obj_points`` — disabled in the live Hydra path
  (use Boxer or another 3D estimator for accurate OBBs)
* annotated image
* JSON with ``bbox3d``, ``label``, ``id``, ``center``, ``spatial_relations``,
  ``status``, ``latest_stamp``

This module keeps a living map in ``odom`` from depth-lifted YOLOE+ByteTrack
observations so Foxglove / RViz look close to the SuperMap demos.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from scene_memory.common.obb import yaw_to_quaternion_xyzw


def _color_for_id(obj_id: int) -> tuple[float, float, float]:
    """Stable RGB in [0,1] from the object id (SuperMap-style distinct colours)."""
    golden = 0.61803398875
    hue = (obj_id * golden) % 1.0
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return float(r), float(g), float(b)


def _bbox_corners(center, size, yaw: float) -> list[list[float]]:
    cx, cy, cz = (float(v) for v in center)
    sx, sy, sz = (float(v) for v in size)
    c, s = math.cos(yaw), math.sin(yaw)
    corners = []
    for dx in (-0.5 * sx, 0.5 * sx):
        for dy in (-0.5 * sy, 0.5 * sy):
            for dz in (-0.5 * sz, 0.5 * sz):
                x = cx + c * dx - s * dy
                y = cy + s * dx + c * dy
                z = cz + dz
                corners.append([x, y, z])
    return corners


def _spatial_relation(a_center, a_size, b_center, b_size) -> str | None:
    """Coarse SuperMap-style predicates in the odom frame (Z up)."""
    ax, ay, az = a_center
    bx, by, bz = b_center
    dx, dy, dz = bx - ax, by - ay, bz - az
    horiz = math.hypot(dx, dy)
    a_h = max(float(a_size[2]), 0.05)
    b_h = max(float(b_size[2]), 0.05)
    a_r = 0.5 * max(float(a_size[0]), float(a_size[1]), 0.05)
    b_r = 0.5 * max(float(b_size[0]), float(b_size[1]), 0.05)
    if horiz < max(a_r + b_r, 0.4) and bz > az + 0.4 * a_h:
        return "on"
    if horiz < max(a_r + b_r, 0.4) and bz < az - 0.4 * b_h:
        return "under"
    if horiz < max(a_r + b_r + 1.5, 1.0) and abs(dz) < 0.75 * max(a_h, b_h):
        return "beside"
    return None


@dataclass
class MappedObject:
    id: int
    label: str
    cls: int
    center: np.ndarray
    size: np.ndarray
    yaw: float
    score: float
    status: str = "active"
    latest_stamp: float = 0.0
    voxels: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    hits: int = 1
    view_keys: set = field(default_factory=set)

    def to_dict(self, relations: list[dict]) -> dict:
        return {
            "id": int(self.id),
            "label": str(self.label),
            "cls": int(self.cls),
            "center": [float(v) for v in self.center],
            "bbox3d": {
                "center": [float(v) for v in self.center],
                "size": [float(v) for v in self.size],
                "yaw_rad": float(self.yaw),
                "corners": _bbox_corners(self.center, self.size, self.yaw),
            },
            "spatial_relations": relations,
            "status": self.status,
            "latest_stamp": float(self.latest_stamp),
            "score": float(self.score),
            "hits": int(self.hits),
        }


class SuperMapMemory:
    """Accumulate depth-lifted tracks into a persistent odom-frame object map."""

    def __init__(
        self,
        match_dist: float = 1.0,
        stale_sec: float = 8.0,
        gone_sec: float = 30.0,
        voxel_m: float = 0.08,
        max_voxels: int = 2500,
        store_voxels: bool = False,
    ) -> None:
        self.match_dist = float(match_dist)
        self.stale_sec = float(stale_sec)
        self.gone_sec = float(gone_sec)
        self.voxel_m = float(voxel_m)
        self.max_voxels = int(max_voxels)
        self.store_voxels = bool(store_voxels)
        self.objects: dict[int, MappedObject] = {}
        self._next_id = 1
        self._view_track_to_id: dict[tuple[str, int], int] = {}

    def update(self, observations: list[dict], stamp_sec: float) -> list[dict]:
        for obs in observations:
            self._ingest(obs, stamp_sec)
        self._age(stamp_sec)
        return self.serialize(stamp_sec)

    def serialize(self, stamp_sec: float | None = None) -> list[dict]:
        alive = [obj for obj in self.objects.values() if obj.status != "gone"]
        rows = []
        for obj in alive:
            relations = []
            for other in alive:
                if other.id == obj.id:
                    continue
                rel = _spatial_relation(obj.center, obj.size, other.center, other.size)
                if rel:
                    relations.append({"relation": rel, "id": int(other.id), "label": other.label})
            rows.append(obj.to_dict(relations))
        if stamp_sec is not None:
            for row in rows:
                row.setdefault("frame_stamp", float(stamp_sec))
        return rows

    def _ingest(self, obs: dict, stamp_sec: float) -> None:
        label = str(obs.get("name") or obs.get("label") or "object")
        cls = int(obs.get("cls", -1))
        center = np.asarray(obs.get("center_odom"), dtype=np.float64).reshape(3)
        size = np.asarray(obs.get("size"), dtype=np.float64).reshape(3)
        yaw = float(obs.get("yaw_rad", 0.0))
        score = float(obs.get("score", 0.0))
        view = str(obs.get("view") or "")
        track_id = obs.get("track_id")
        points_arr = np.zeros((0, 3), dtype=np.float32)
        if self.store_voxels:
            points = obs.get("points_odom")
            if points is not None:
                points_arr = np.asarray(points, dtype=np.float32).reshape(-1, 3)

        obj_id = None
        key = None
        if view and track_id is not None:
            key = (view, int(track_id))
            obj_id = self._view_track_to_id.get(key)
            if obj_id is not None and obj_id not in self.objects:
                obj_id = None

        if obj_id is None:
            obj_id = self._match_by_geometry(label, center)

        if obj_id is None:
            obj_id = self._next_id
            self._next_id += 1
            self.objects[obj_id] = MappedObject(
                id=obj_id,
                label=label,
                cls=cls,
                center=center.copy(),
                size=np.maximum(size, 0.05),
                yaw=yaw,
                score=score,
                latest_stamp=stamp_sec,
                voxels=self._voxelize(points_arr) if self.store_voxels else np.zeros((0, 3), np.float32),
                view_keys={key} if key else set(),
            )
        else:
            obj = self.objects[obj_id]
            alpha = 0.35
            obj.center = (1.0 - alpha) * obj.center + alpha * center
            obj.size = np.maximum((1.0 - alpha) * obj.size + alpha * size, 0.05)
            # Wrap yaw blend into the nearer half-turn.
            dyaw = (yaw - obj.yaw + math.pi) % (2.0 * math.pi) - math.pi
            obj.yaw = float(obj.yaw + alpha * dyaw)
            obj.score = max(obj.score, score)
            obj.label = label if score >= obj.score else obj.label
            obj.cls = cls
            obj.status = "active"
            obj.latest_stamp = stamp_sec
            obj.hits += 1
            if key:
                obj.view_keys.add(key)
            if self.store_voxels and points_arr.size:
                merged = np.concatenate([obj.voxels, points_arr], axis=0) if obj.voxels.size else points_arr
                obj.voxels = self._voxelize(merged)

        if key is not None:
            self._view_track_to_id[key] = obj_id

    def _match_by_geometry(self, label: str, center: np.ndarray) -> int | None:
        best_id = None
        best_dist = self.match_dist
        for obj in self.objects.values():
            if obj.status == "gone":
                continue
            if obj.label != label:
                continue
            dist = float(np.linalg.norm(obj.center - center))
            if dist < best_dist:
                best_dist = dist
                best_id = obj.id
        return best_id

    def _age(self, stamp_sec: float) -> None:
        for obj in self.objects.values():
            age = stamp_sec - obj.latest_stamp
            if age > self.gone_sec:
                obj.status = "gone"
            elif age > self.stale_sec:
                obj.status = "stale"
            else:
                obj.status = "active"

    def _voxelize(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if pts.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        keep = np.isfinite(pts).all(axis=1)
        pts = pts[keep]
        if pts.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        keys = np.floor(pts / self.voxel_m).astype(np.int32)
        # Unique voxels via structured view.
        flat = np.ascontiguousarray(keys).view([("x", np.int32), ("y", np.int32), ("z", np.int32)])
        _, index = np.unique(flat, return_index=True)
        voxels = pts[index]
        if voxels.shape[0] > self.max_voxels:
            pick = np.linspace(0, voxels.shape[0] - 1, self.max_voxels).astype(int)
            voxels = voxels[pick]
        return voxels.astype(np.float32)


def objects_to_pointcloud2(objects: list[MappedObject], stamp, frame_id: str = "odom"):
    """One coloured cloud for ``/obj_points``."""
    from sensor_msgs.msg import PointField
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header

    points = []
    for obj in objects:
        if obj.status == "gone" or obj.voxels.size == 0:
            continue
        r, g, b = _color_for_id(obj.id)
        rgb_u32 = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
        rgb = np.frombuffer(np.uint32([rgb_u32]).tobytes(), dtype=np.float32)[0]
        for p in obj.voxels:
            points.append((float(p[0]), float(p[1]), float(p[2]), float(rgb)))
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return point_cloud2.create_cloud(
        header,
        [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ],
        points,
    )


def objects_to_box_markers(objects: list[MappedObject], stamp, frame_id: str = "odom"):
    """``/obj_boxes`` as cubes + floating labels."""
    from visualization_msgs.msg import Marker, MarkerArray

    array = MarkerArray()
    clear = Marker()
    clear.header.stamp = stamp
    clear.header.frame_id = frame_id
    clear.ns = "obj_boxes"
    clear.action = Marker.DELETEALL
    array.markers.append(clear)
    for obj in objects:
        if obj.status == "gone":
            continue
        r, g, b = _color_for_id(obj.id)
        alpha = 0.55 if obj.status == "active" else 0.25
        cube = Marker()
        cube.header.stamp = stamp
        cube.header.frame_id = frame_id
        cube.ns = "obj_boxes"
        cube.id = int(obj.id)
        cube.type = Marker.CUBE
        cube.action = Marker.ADD
        cube.pose.position.x = float(obj.center[0])
        cube.pose.position.y = float(obj.center[1])
        cube.pose.position.z = float(obj.center[2])
        q = yaw_to_quaternion_xyzw(obj.yaw)
        cube.pose.orientation.x = float(q[0])
        cube.pose.orientation.y = float(q[1])
        cube.pose.orientation.z = float(q[2])
        cube.pose.orientation.w = float(q[3])
        cube.scale.x = float(obj.size[0])
        cube.scale.y = float(obj.size[1])
        cube.scale.z = float(obj.size[2])
        cube.color.r, cube.color.g, cube.color.b, cube.color.a = r, g, b, alpha
        cube.lifetime.sec = 0
        cube.lifetime.nanosec = 0
        array.markers.append(cube)

        text = Marker()
        text.header.stamp = stamp
        text.header.frame_id = frame_id
        text.ns = "obj_labels"
        text.id = int(obj.id)
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(obj.center[0])
        text.pose.position.y = float(obj.center[1])
        text.pose.position.z = float(obj.center[2] + 0.5 * obj.size[2] + 0.15)
        text.pose.orientation.w = 1.0
        text.scale.z = 0.18
        text.color.r = text.color.g = text.color.b = 1.0
        text.color.a = 0.95 if obj.status == "active" else 0.5
        text.text = f"{obj.label}#{obj.id}"
        array.markers.append(text)
    return array


def map_payload(objects_json: list[dict], stamp_sec: float) -> str:
    return json.dumps(
        {
            "frame_id": "odom",
            "latest_stamp": float(stamp_sec),
            "objects": objects_json,
        }
    )
