"""Depth-free layered semantic topo map: detector tracks + VIO → objects / places / regions / floors / buildings.

Floors come from settled VIO height changes (stairs / elevator / ramp), indoor vs outdoor from a SigLIP
classifier on the equirect (``--environment`` forces one).

Inputs: ``/detector/detections_json`` (tracked 2D detections) and the odometry, either equirect VIO
(``--pose_source vio``) or Super-LIO on the MID-360S (``--pose_source lio``, camera pose through the rig
calibration in data/calib). With ``--cloud_topic`` (the LIO deskewed scans in odom) an object is placed
at the nearest lidar cluster inside its mask in the view it was detected in (``center_source`` lidar);
otherwise, and for objects without returns, centroids come from ray triangulation across the frames of a
track (class-size prior until there is enough parallax). The object memory re-identifies and
label-fuses them, and the place layer follows the robot.

Outputs: ``<scene_dir>/objects_4d.json`` (same format as the Hydra node, so
``query_4d`` keeps working), ``<scene_dir>/topo_graph.json``, ``/topo/graph``
(the same graph as JSON), ``/topo/markers`` and ``/topo/track_centers`` (the
3D-to-2D tracking prior for the detector).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from scene_memory.perception.boxer_lift import load_pinhole_views
from scene_memory.mapping.boxer_objects import BoxerObjectMap
from scene_memory.common.calib_io import load_rig_calibration
from scene_memory.mapping.centroid_lift import CentroidLifter, class_max_size, class_size, pixel_center
from scene_memory.mapping.instance_points import InstancePoints, MapCloud
from scene_memory.common.labelspace import load_labelspace
from scene_memory.mapping.object_views import KeyViewStore, descriptions_by_key, load_description_cache
from scene_memory.mapping.scene_graph import (T_da360_optical, T_odom_optical, pinhole_intrinsics, pose_frame_to_da360,
                                              pose_to_matrix)
from scene_memory.common.pose_buffer import pose_at
from scene_memory.perception.lidar_depth import CloudWindow, mask_depth, project_points
from scene_memory.mapping.dsg_export import write_dsg
from scene_memory.query.region_summary import load_summaries
from scene_memory import paths
from scene_memory.mapping.environment import EnvironmentClassifier
from scene_memory.mapping.topo_graph import PlaceLayer, VerticalTracker, build_graph, save_graph

DEFAULT_VIO_CONFIG = paths.VIO_CONFIG


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib_dir", default=str(paths.CALIB))
    p.add_argument("--pose_source", choices=["vio", "lio"], default="vio",
                   help="vio: equirect_vio imu pose; lio: Super-LIO livox_frame pose + rig calibration")
    p.add_argument("--vio_config", default=str(DEFAULT_VIO_CONFIG))
    p.add_argument("--lio_config", default=str(paths.LIO_CONFIG), help="Super-LIO YAML (lio.extrinsic.lidar_imu)")
    p.add_argument("--prompts", default=str(paths.CONFIG / "prompts.txt"))
    p.add_argument("--detections_topic", default="/detector/detections_json")
    p.add_argument("--odom_topic", default="/equirect_vio/odometry")
    p.add_argument("--world_frame", default="odom",
                   help="frame of --odom_topic's poses, written into the map files (map with QiNav localization)")
    p.add_argument("--reset_topic", default="/lio/reset",
                   help="odometry restarted (lio_watchdog_node): save the map and start a new segment <scene_dir>_segN")
    p.add_argument("--cloud_topic", default="", help="LIO scans in odom (/lio/cloud_world); empty = no lidar depth")
    p.add_argument("--cloud_window_s", type=float, default=1.0, help="s of scans projected into a view")
    p.add_argument("--cloud_voxel", type=float, default=0.03, help="m; per-scan voxel filter (0 = off)")
    p.add_argument("--cloud_keep_s", type=float, default=10.0,
                   help="s of scans kept; must exceed the detection latency (camera + detector, ~3 s on Orin NX)")
    p.add_argument("--lidar_min_points", type=int, default=12, help="returns in a mask needed for a lidar centre")
    p.add_argument("--lidar_max_range", type=float, default=20.0)
    p.add_argument("--image_prefix", default="/insta360", help="<prefix>/<view>/image/compressed for object key views")
    p.add_argument("--centers_topic", default="/topo/track_centers")
    p.add_argument("--graph_topic", default="/topo/graph")
    p.add_argument("--markers_topic", default="/topo/markers")
    p.add_argument("--scene_dir", default="")
    p.add_argument("--save_period_s", type=float, default=5.0)
    p.add_argument("--object_ttl", type=float, default=5.0, help="s unseen before an object is marked hidden")
    p.add_argument("--label_accuracy", type=float, default=0.7, help="detector confusion-matrix diagonal")
    p.add_argument("--reid_radius", type=float, default=0.8, help="m; re-id and duplicate merge radius")
    p.add_argument("--min_parallax_deg", type=float, default=3.0)
    p.add_argument("--min_rays", type=int, default=3)
    p.add_argument("--min_prior_obs", type=int, default=5, help="frames before a non-triangulated track uses the size prior")
    p.add_argument("--place_spacing", type=float, default=1.0, help="m between place nodes")
    p.add_argument("--region_radius", type=float, default=4.0, help="m; max distance from a region's seed place")
    p.add_argument("--door_radius", type=float, default=0.8, help="m; traversable edges passing a door split regions")
    p.add_argument("--min_region_places", type=int, default=3, help="smaller regions merge into a neighbour")
    p.add_argument("--dynamic_classes", default="person", help="comma list; tracked but kept out of the static map")
    p.add_argument("--miss_range", type=float, default=5.0,
                   help="m; a triangulated object in view closer than this but not detected counts as a miss "
                        "(0 disables existence tracking)")
    p.add_argument("--miss_margin", type=float, default=0.1, help="ignore projections within this image fraction of the border")
    p.add_argument("--miss_min_px", type=float, default=24.0, help="ignore objects whose projection is smaller (px)")
    p.add_argument("--environment", choices=["auto", "indoor", "outdoor"], default="auto",
                   help="auto = SigLIP indoor/outdoor classifier on the equirect")
    p.add_argument("--equirect_topic", default="/insta360/equirect/image/compressed")
    p.add_argument("--env_period_s", type=float, default=2.0, help="s between indoor/outdoor classifications")
    p.add_argument("--floor_height", type=float, default=3.0, help="m per storey (stairs / elevator)")
    p.add_argument("--floor_min_dz", type=float, default=2.0, help="m of settled height change that counts as a floor")
    p.add_argument("--instance_voxel", type=float, default=0.05, help="m; per-object lidar point voxels (3D boxes)")
    p.add_argument("--map_voxel", type=float, default=0.1, help="m; accumulated map cloud for display (0 = off)")
    p.add_argument("--map_cloud_topic", default="/topo/map_cloud")
    p.add_argument("--room_dz", type=float, default=3.0, help="m; regions drawn this far above their places")
    p.add_argument("--pano_overlay", action="store_true",
                   help="publish 2D detections + projected 3D boxes on the equirect (SuperMap style)")
    p.add_argument("--overlay_topic", default="/topo/pano_overlay/compressed")
    p.add_argument("--no_dsg", action="store_true", help="skip the spark_dsg (DAAAM-format) export")
    p.add_argument("--no_views", action="store_true", help="do not save object key views (DAM input)")
    return p.parse_args(argv)


class TopoMapper:
    """ROS-free core so it can be replayed from a bag or unit-tested."""

    def __init__(self, args: argparse.Namespace, env: EnvironmentClassifier | None = None) -> None:
        self.args = args
        calib = load_rig_calibration(args.calib_dir)
        self.views = {v["name"]: v for v in load_pinhole_views(calib.source_dir)}
        self.T_imu_da360 = pose_frame_to_da360(args.pose_source, args.vio_config, args.lio_config, calib)
        self.cloud = (CloudWindow(args.cloud_window_s, keep_s=args.cloud_keep_s, voxel_m=args.cloud_voxel)
                      if args.cloud_topic else None)
        names = load_labelspace(args.prompts).names[1:]
        self.objects = BoxerObjectMap(
            ttl_s=args.object_ttl,
            reid_radius_m=args.reid_radius,
            class_names=names,
            label_accuracy=args.label_accuracy,
            dynamic_classes=tuple(args.dynamic_classes.split(",")),
        )
        self.instances = InstancePoints(voxel_m=args.instance_voxel)
        self.objects.on_absorb = self.instances.merge
        self.map_cloud = MapCloud(args.map_voxel) if args.map_voxel > 0 else None
        self.lifter = CentroidLifter(
            min_rays=args.min_rays, min_parallax_deg=args.min_parallax_deg, min_prior_obs=args.min_prior_obs
        )
        self.places = PlaceLayer(spacing_m=args.place_spacing,
                                 vertical=VerticalTracker(floor_dz=args.floor_min_dz, floor_height=args.floor_height))
        if args.environment != "auto":
            self.places.set_environment(args.environment)
        self.env = None if args.environment != "auto" else (env or EnvironmentClassifier())
        self._odom_t: list[float] = []
        self._odom_T: list[np.ndarray] = []
        self.stats = {"frames": 0, "lidar": 0, "tri": 0, "prior": 0, "none": 0, "merged": 0, "odom": 0, "odom_old": 0,
                      "no_pose": 0, "miss": 0, "absent": 0, "clouds": 0}

    def on_odom(self, t: float, T_odom_imu: np.ndarray, quat_xyzw=None) -> None:
        self.stats["odom"] += 1
        if self._odom_t and t <= self._odom_t[-1]:
            self.stats["odom_old"] += 1
            return
        self._odom_t.append(t)
        self._odom_T.append(T_odom_imu)
        if len(self._odom_t) > 4000:
            del self._odom_t[:1000], self._odom_T[:1000]
        self.places.update(T_odom_imu[:3, 3], t, quat_xyzw)

    def on_equirect(self, t: float, bgr: np.ndarray) -> str:
        """Indoor / outdoor update from one 360° frame (call at ``env_period_s``)."""
        if self.env is None:
            return self.places.environment
        self.places.set_environment(self.env.update(bgr), t)
        return self.places.environment

    def last_odom_t(self) -> float:
        return self._odom_t[-1] if self._odom_t else 0.0

    def on_cloud(self, t: float, xyz_odom: np.ndarray) -> None:
        if self.cloud is not None:
            self.cloud.add(t, xyz_odom)
            self.stats["clouds"] += 1
        if self.map_cloud is not None:
            self.map_cloud.add(xyz_odom)

    def pose_at(self, t: float, max_dt: float = 0.1) -> np.ndarray | None:
        """Odometry pose at ``t``: interpolated between the bracketing poses (LIO runs at 10 Hz), else the
        nearest one within ``max_dt``."""
        return pose_at(self._odom_t, self._odom_T, t, max_dt)

    def on_detections(self, body: dict) -> list[dict]:
        """One detector frame → object memory update. Returns the lifted detections."""
        t = body["stamp"]["sec"] + 1e-9 * body["stamp"]["nanosec"]
        T_odom_imu = self.pose_at(t)
        if T_odom_imu is None:
            self.stats["no_pose"] += 1
            return []
        size = body.get("view_image") or body["image"]
        w, h = int(size["width"]), int(size["height"])
        cloud = self.cloud.points(t) if self.cloud is not None else None
        projected: dict[str, tuple] = {}
        lifted = []
        for det in body.get("detections", []):
            tid, view = det.get("track_id"), self.views.get(det.get("view") or "")
            if tid is None or view is None:
                continue
            K = pinhole_intrinsics(w, h, view["hfov_deg"])
            T = T_odom_optical(T_odom_imu, self.T_imu_da360, view["R_body_optical"])
            center, source = self.lifter.observe(tid, det["name"], det["xyxy"], pixel_center(det), K, T, w, h, t)
            det = dict(det)
            if cloud is not None and len(cloud):
                if view["name"] not in projected:
                    projected[view["name"]] = project_points(cloud, T, K, w, h, z_max=self.args.lidar_max_range)
                hit = mask_depth(projected[view["name"]], w, h, det, min_points=self.args.lidar_min_points)
                if hit is not None:
                    center, source = hit["center"], "lidar"
                    det["lidar_depth"] = hit["depth"]
                    det["lidar_points"] = hit["n_points"]
                    det["_points"] = hit["points"]
            self.stats[source or "none"] += 1
            if center is None:
                continue
            det["cls_scores"] = {int(c): float(s) for c, s in (det.get("cls_scores") or {}).items()}
            det["obb"] = {"center": np.asarray(center).tolist(), "size": list(class_size(det["name"])),
                          "source": source}
            lifted.append(det)
        self.objects.update(lifted, t, w, h)
        for det in lifted:
            pts, obj = det.pop("_points", None), self.objects.objects.get(det.get("memory_key", ""))
            if pts is not None and obj is not None:
                self.instances.add(obj["key"], pts, replace=self.objects.is_dynamic(obj))
                self._refresh_box(obj)
        if self.args.miss_range > 0 and body.get("views"):
            self._negative_evidence(body, T_odom_imu, w, h, t)
        self.objects.expire(t)
        self.lifter.forget(t)
        self.stats["frames"] += 1
        return lifted

    def _negative_evidence(self, body: dict, T_odom_imu: np.ndarray, w: int, h: int, t: float) -> None:
        """Visibility test without depth: a triangulated static object that projects well inside a processed
        view, within ``miss_range``, large enough to be detected and not behind a nearer detection, but was not
        detected in this frame, gets one miss."""
        a = self.args
        detected = {}
        for det in body.get("detections", []):
            tid = det.get("track_id")
            key = self.objects._alias.get(int(tid)) if tid is not None else None
            # Unknown objects (key None) still occlude whatever projects inside their box.
            detected.setdefault(det.get("view"), []).append((det["xyxy"], key))
        seen_keys = {k for dets in detected.values() for _, k in dets if k}
        cands = [o for o in self.objects.objects.values()
                 if o.get("center_rank", 0) >= 2 and o.get("status", "present") == "present"
                 and t - o.get("last_miss", -np.inf) >= self.objects.miss_interval_s
                 and o["key"] not in seen_keys and not self.objects.is_dynamic(o) and t - o["last_seen"] > 0.3]
        if not cands:
            return
        centers = np.stack([o["center"] for o in cands])
        radius = np.array([0.5 * float(np.max(o["size"])) for o in cands])
        mx, my = a.miss_margin * w, a.miss_margin * h
        missed: set[str] = set()
        for name in body["views"]:
            view = self.views.get(name)
            if view is None:
                continue
            fx, fy, cx, cy = pinhole_intrinsics(w, h, view["hfov_deg"])
            T = T_odom_optical(T_odom_imu, self.T_imu_da360, view["R_body_optical"])
            p = (centers - T[:3, 3]) @ T[:3, :3]
            z = p[:, 2]
            dist = np.linalg.norm(p, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                u = fx * p[:, 0] / z + cx
                v = fy * p[:, 1] / z + cy
                px = 2.0 * fx * radius / z
            ok = (z > 0.5) & (dist < a.miss_range) & (u > mx) & (u < w - mx) & (v > my) & (v < h - my) & (px >= a.miss_min_px)
            boxes = []
            for xyxy, key in detected.get(name, []):
                obj = self.objects.objects.get(key or "")
                d = float(np.linalg.norm(obj["center"] - T[:3, 3])) if obj is not None else 0.0
                boxes.append((xyxy, d))
            for i in np.flatnonzero(ok):
                if any(x1 <= u[i] <= x2 and y1 <= v[i] <= y2 and d < dist[i] - 0.3 for (x1, y1, x2, y2), d in boxes):
                    continue
                missed.add(cands[i]["key"])
        for key in missed:
            self.stats["miss"] += 1
            self.stats["absent"] += int(self.objects.miss(key, t))

    def _refresh_box(self, obj: dict) -> None:
        box = self.instances.box(obj["key"], anchor=obj["center"], max_size=class_max_size(obj["name"]))
        if box is not None:
            obj["bbox3d"] = {**box, "source": "lidar"}

    def consolidate(self) -> None:
        merged = self.objects.consolidate()
        self.stats["merged"] += merged
        if merged:
            for obj in self.objects.objects.values():
                self._refresh_box(obj)

    def graph(self, summaries: dict | None = None) -> dict:
        objs = self.objects.to_json()["objects"]
        return build_graph(objs, self.places, region_radius_m=self.args.region_radius,
                           door_radius_m=self.args.door_radius, min_region_places=self.args.min_region_places,
                           summaries=summaries)


def _xyz_cloud(xyz: np.ndarray, frame_id: str, stamp):
    from sensor_msgs.msg import PointCloud2, PointField

    msg = PointCloud2()
    msg.header.frame_id, msg.header.stamp = frame_id, stamp
    msg.height, msg.width = 1, len(xyz)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
    msg.point_step, msg.row_step, msg.is_dense = 12, 12 * len(xyz), True
    msg.data = np.ascontiguousarray(xyz, dtype=np.float32).tobytes()
    return msg


def _markers(graph: dict, stamp, instances: InstancePoints | None = None, room_dz: float = 3.0):
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray

    from scene_memory.mapping.pano_overlay import box_edge_points, class_rgb

    out = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    out.markers.append(clear)

    def base(ns, mid, mtype, scale, rgba):
        m = Marker()
        m.header.frame_id = graph["frame_id"]
        m.header.stamp = stamp
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
        return m

    def pt(xyz):
        return Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))

    layers = graph["layers"]
    places = {p["key"]: p for p in layers["places"]}
    sph = base("places", 0, Marker.SPHERE_LIST, 0.15, (0.6, 0.6, 0.6, 0.9))
    for p in layers["places"]:
        sph.points.append(pt(p["position"]))
        hue = (p.get("region", 0) * 0.618) % 1.0
        sph.colors.append(ColorRGBA(r=0.3 + 0.7 * hue, g=0.8 - 0.6 * hue, b=0.5, a=0.9))
    out.markers.append(sph)
    lines = base("traversable", 1, Marker.LINE_LIST, 0.03, (0.4, 0.4, 0.4, 0.8))
    links = base("object_place", 2, Marker.LINE_LIST, 0.01, (0.2, 0.6, 1.0, 0.4))
    rel = base("object_relations", 3, Marker.LINE_LIST, 0.02, (1.0, 0.6, 0.0, 0.8))
    objs = {o["key"]: o for o in layers["objects"]}
    for e in graph["edges"]:
        if e["relation"] == "traversable":
            lines.points += [pt(places[e["source"]]["position"]), pt(places[e["target"]]["position"])]
        elif e["relation"] == "reachable_from":
            links.points += [pt(objs[e["source"]]["center"]), pt(places[e["target"]]["position"])]
        elif e["source"] in objs and e["target"] in objs:
            rel.points += [pt(objs[e["source"]]["center"]), pt(objs[e["target"]]["center"])]
    out.markers += [lines, links, rel]
    # Rooms (regions) float room_dz above their places, Hydra style.
    rooms = base("rooms", 4, Marker.CUBE_LIST, 0.35, (1.0, 1.0, 1.0, 0.9))
    room_links = base("room_places", 5, Marker.LINE_LIST, 0.01, (0.8, 0.8, 0.8, 0.35))
    for r in layers.get("regions", []):
        hue = (r["id"] * 0.618) % 1.0
        top = [r["center"][0], r["center"][1], r["center"][2] + room_dz]
        rooms.points.append(pt(top))
        rooms.colors.append(ColorRGBA(r=0.3 + 0.7 * hue, g=0.8 - 0.6 * hue, b=0.5, a=0.9))
        for pid in r["places"]:
            if f"place{pid}" in places:
                room_links.points += [pt(top), pt(places[f"place{pid}"]["position"])]
    out.markers += [rooms, room_links]
    for i, o in enumerate(layers["objects"]):
        absent = o.get("status") == "absent"
        rgb = (0.5, 0.5, 0.5) if absent else class_rgb(o["name"])
        box = o.get("bbox3d") or {"center": o["center"], "size": o["size"], "source": "prior"}
        measured = box.get("source") != "prior"
        alpha = 0.35 if absent or not measured else (1.0 if o.get("visible") else 0.7)
        edges = base("boxes", 10 + i, Marker.LINE_LIST, 0.025 if measured else 0.01, (*rgb, alpha))
        for e in box_edge_points(box["center"], box["size"], samples=2):
            edges.points += [pt(e[0]), pt(e[1])]
        out.markers.append(edges)
        if instances is not None and o["key"] in instances:
            cloud = base("instance_points", 10 + i, Marker.POINTS, 0.04, (*rgb, 0.9))
            cloud.points = [pt(p) for p in instances.points(o["key"])]
            out.markers.append(cloud)
        txt = base("labels", 10 + i, Marker.TEXT_VIEW_FACING, 0.22, (*rgb, max(alpha, 0.8)))
        c, s = box["center"], box["size"]
        txt.pose.position = pt([c[0], c[1], c[2] + 0.5 * s[2] + 0.15])
        txt.text = f"{o['name']} {o['key']}" + (" (absent)" if absent else "")
        out.markers.append(txt)
    return out


def main(argv=None, mapper_factory=None, node_name: str = "topo_map", mapper_name: str = "topo") -> None:
    """``mapper_factory(args, env=None)`` builds the core (default ``TopoMapper``); a core with
    ``handles_reset`` keeps its map across ``/lio/reset`` (``core.on_reset()``) instead of starting a segment."""
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, PointCloud2
    from std_msgs.msg import Header, String
    from visualization_msgs.msg import MarkerArray

    from scene_memory.perception.lidar_depth import cloud_xyz_fast

    args = argv if isinstance(argv, argparse.Namespace) else _parse_args(argv)
    factory = mapper_factory or TopoMapper
    rclpy.init()
    node = rclpy.create_node(node_name)
    log = node.get_logger()
    core = factory(args)
    scene_dir = Path(args.scene_dir) if args.scene_dir else paths.SCENES / time.strftime("%Y%m%d_%H%M%S")
    views_store = None if args.no_views else KeyViewStore(scene_dir)
    frames: dict[tuple[str, int], np.ndarray] = {}

    pub_centers = node.create_publisher(String, args.centers_topic, 10)
    pub_graph = node.create_publisher(String, args.graph_topic, 1)
    pub_markers = node.create_publisher(MarkerArray, args.markers_topic, 1)
    pub_map = node.create_publisher(PointCloud2, args.map_cloud_topic, 1)
    pub_overlay = node.create_publisher(CompressedImage, args.overlay_topic, 2) if args.pano_overlay else None

    first_dir = scene_dir
    segment = {"n": 1, "reset_at": -1e9}

    def restarting() -> bool:
        # Poses of the diverged LIO may still arrive until it exits; the respawned one starts later.
        return time.monotonic() - segment["reset_at"] < 1.5

    def on_reset(msg) -> None:
        nonlocal core, scene_dir, views_store
        export()
        segment["reset_at"] = time.monotonic()
        if getattr(core, "handles_reset", False):
            core.on_reset()
            log.warn(f"odometry restarted; {mapper_name} keeps the map and re-localises the new segment")
            return
        log.warn(f"odometry restarted; map segment {segment['n']} saved to {scene_dir}")
        segment["n"] += 1
        environment = core.places.environment
        core = factory(args, env=core.env)
        core.places.set_environment(environment)
        scene_dir = first_dir.with_name(f"{first_dir.name}_seg{segment['n']}")
        views_store = None if args.no_views else KeyViewStore(scene_dir)

    def on_odom(msg) -> None:
        if restarting():
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        core.on_odom(t, pose_to_matrix((p.x, p.y, p.z), (q.x, q.y, q.z, q.w)), (q.x, q.y, q.z, q.w))

    def on_cloud(msg) -> None:
        if restarting():
            return
        try:
            xyz = cloud_xyz_fast(msg)
        except ValueError as exc:
            log.warn(f"cloud decode failed: {exc}", throttle_duration_sec=10.0)
            return
        core.on_cloud(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec, xyz)

    def on_view(name, msg) -> None:
        import cv2

        ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        frames[(name, ns)] = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        while len(frames) > 60:
            frames.pop(next(iter(frames)))

    def on_detections(msg) -> None:
        try:
            body = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        lifted = core.on_detections(body)
        if views_store is not None and lifted:
            ns = body["stamp"]["sec"] * 1_000_000_000 + body["stamp"]["nanosec"]
            for det in lifted:
                bgr = frames.get((det.get("view"), ns))
                obj = core.objects.objects.get(det.get("memory_key", ""))
                if bgr is not None and obj is not None:
                    views_store.offer(obj, det, bgr, ns * 1e-9)
        pub_centers.publish(String(data=json.dumps({"frame_id": args.world_frame, "stamp": body["stamp"],
                                                    "objects": core.objects.track_centers()})))
        if pub_overlay is not None:
            try:
                publish_overlay(body)
            except Exception as exc:  # noqa: BLE001 — a display must not stop the mapping
                log.warn(f"pano overlay failed: {exc}", throttle_duration_sec=30.0)

    equirects: deque = deque(maxlen=40)

    def on_equirect_frame(msg) -> None:
        equirects.append((msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec, bytes(msg.data)))

    def publish_overlay(body: dict) -> None:
        import cv2

        from scene_memory.mapping.pano_overlay import render

        t = body["stamp"]["sec"] + 1e-9 * body["stamp"]["nanosec"]
        if not equirects:
            return
        stamp_eq, data = min(equirects, key=lambda f: abs(f[0] - t))
        T_odom_imu = core.pose_at(t)
        if abs(stamp_eq - t) > 0.15 or T_odom_imu is None:
            return
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
        size = body.get("view_image") or body["image"]
        vw, vh = int(size["width"]), int(size["height"])
        views = core.views
        img = render(bgr, body.get("detections", []), list(core.objects.objects.values()),
                     T_odom_imu @ core.T_imu_da360, views, (vw, vh),
                     {n: T_da360_optical(v["R_body_optical"])[:3, :3] for n, v in views.items()},
                     lambda n: pinhole_intrinsics(vw, vh, views[n]["hfov_deg"]))
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            out = CompressedImage(format="jpeg", data=jpg.tobytes())
            out.header.stamp.sec, out.header.stamp.nanosec = body["stamp"]["sec"], body["stamp"]["nanosec"]
            out.header.frame_id = "insta360"
            pub_overlay.publish(out)

    dsg_state = {"enabled": not args.no_dsg}
    cloud_state = {"n": -1}  # map_cloud.npy (plan_check obstacle test) rewritten only when the map grew

    def export() -> None:
        core.consolidate()
        graph = core.graph(load_summaries(scene_dir))
        graph["frame_id"] = args.world_frame
        try:
            scene_dir.mkdir(parents=True, exist_ok=True)
            latest = scene_dir.parent / "latest"
            if latest.is_symlink() and latest.resolve() != scene_dir.resolve():
                latest.unlink()
            if not latest.exists():
                latest.symlink_to(scene_dir.name)
            by_key = descriptions_by_key(load_description_cache(scene_dir))
            for key, obj in core.objects.objects.items():
                if key in by_key:
                    obj["descriptions"] = by_key[key]
            core.objects.save(scene_dir / "objects_4d.json", extra={"saved_at": time.time(), "mapper": mapper_name,
                                                                    "frame_id": args.world_frame,
                                                                    "stats": dict(core.stats)})
            save_graph(graph, scene_dir / "topo_graph.json")
            if core.map_cloud is not None:
                pts = core.map_cloud.points()
                if len(pts) != cloud_state["n"]:
                    tmp = scene_dir / "map_cloud.tmp.npy"
                    np.save(tmp, np.asarray(pts, dtype=np.float32)[:, :3])
                    tmp.replace(scene_dir / "map_cloud.npy")
                    cloud_state["n"] = len(pts)
        except Exception as exc:  # noqa: BLE001 — export must not kill the node
            log.warn(f"export failed: {exc}")
        if dsg_state["enabled"]:
            try:
                write_dsg(scene_dir, graph, core.objects.to_json()["objects"])
            except ImportError as exc:
                dsg_state["enabled"] = False
                log.warn(f"spark_dsg unavailable, DSG export disabled: {exc}")
            except Exception as exc:  # noqa: BLE001
                log.warn(f"DSG export failed: {exc}")
        try:
            stamp = node.get_clock().now().to_msg()
            pub_graph.publish(String(data=json.dumps(graph)))
            pub_markers.publish(_markers(graph, stamp, core.instances, args.room_dz))
            if core.map_cloud is not None:
                pub_map.publish(_xyz_cloud(core.map_cloud.points(), args.world_frame, stamp))
        except Exception:  # noqa: BLE001 — context already shut down on exit
            pass
        layers = graph["layers"]
        s = core.stats
        pl = core.places
        log.info(
            f"[{pl.environment} floor={pl.floor()}{' climbing' if pl.vertical.transit else ''}] "
            f"buildings={sum(b['kind'] == 'building' for b in layers['buildings'])} floors={len(layers['floors'])} "
            f"objects={len(layers['objects'])} places={len(layers['places'])} regions={len(layers['regions'])} "
            f"agents={len(layers['agents'])} "
            f"edges={len(graph['edges'])} | obs lidar={s['lidar']} tri={s['tri']} prior={s['prior']} none={s['none']} "
            f"merged={s['merged']} clouds={s['clouds']} "
            f"miss={s['miss']} absent={s['absent']} "
            f"| odom={s['odom']} old={s['odom_old']} last={core.last_odom_t():.2f} no_pose={s['no_pose']}"
        )

    env_state = {"last": -1e9}

    def on_equirect(msg) -> None:
        import cv2

        now = time.monotonic()
        if now - env_state["last"] < args.env_period_s:
            return
        env_state["last"] = now
        bgr = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
        if bgr is None:
            return
        before = core.places.environment
        try:
            after = core.on_equirect(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec, bgr)
        except Exception as exc:  # noqa: BLE001 — keep mapping with the last state
            log.warn(f"indoor/outdoor classifier failed, keeping '{before}': {exc}")
            core.env = None
            return
        if after != before:
            log.info(f"environment: {before} -> {after} (p_outdoor={core.env.p:.2f})")

    if core.env is not None:
        node.create_subscription(CompressedImage, args.equirect_topic, on_equirect, qos_profile_sensor_data)
    if args.pano_overlay:
        node.create_subscription(CompressedImage, args.equirect_topic, on_equirect_frame, qos_profile_sensor_data)
    node.create_subscription(Odometry, args.odom_topic, on_odom, 200)
    if args.reset_topic:
        node.create_subscription(Header, args.reset_topic, on_reset, 10)
    node.create_subscription(String, args.detections_topic, on_detections, 50)
    if args.cloud_topic:
        node.create_subscription(PointCloud2, args.cloud_topic, on_cloud, qos_profile_sensor_data)
    if views_store is not None:
        for name, view in core.views.items():
            node.create_subscription(CompressedImage, view["image_topic"].replace("/insta360", args.image_prefix, 1),
                                     lambda m, n=name: on_view(n, m), qos_profile_sensor_data)
    node.create_timer(args.save_period_s, export)
    log.info(f"{mapper_name} map: {args.detections_topic} + {args.odom_topic} ({args.pose_source})"
             + (f" + lidar {args.cloud_topic}" if args.cloud_topic else "") + f" → {scene_dir}")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        export()
        log.info(f"saved {scene_dir / 'topo_graph.json'}")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
