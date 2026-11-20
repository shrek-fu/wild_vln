"""2D detection + tracking on the Insta360 views; publishes /detector/* and rays/markers.

The detector is chosen with ``--detector`` (yoloe | sam3 | gdino_sam2); topics,
node name and marker namespaces do not depend on it.

Rays are computed in the calibrated ``lidar_level`` math frame, then expressed
in ``--frame_id`` for ROS (default ``imu``, which matches equirect_vio TF).
"""

from __future__ import annotations

import argparse
import json
import os
from array import array
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from scene_memory import paths
from scene_memory.viz.annotate import color_bgr, draw_detections
from scene_memory.perception.byte_track import ViewByteTracker
from scene_memory.perception.pose_track import ViewPoseTracker, project_centers
from scene_memory.perception.gdino_sam2_detector import (
    DEFAULT_GDINO,
    DEFAULT_GROUND_CLASSES,
    DEFAULT_SAM2,
    GdinoSam2ViewDetector,
    parse_ground_classes,
)
from scene_memory.perception.sam3_detector import DEFAULT_SAM3, Sam3ViewDetector
from scene_memory.mapping.scene_graph import (
    T_odom_optical,
    attach_track_masks,
    mask_to_json,
    pose_frame_to_da360,
    pinhole_intrinsics,
    pose_to_matrix,
)
from scene_memory.perception.boxer_lift import DEFAULT_CKPT, BoxerLifter, load_pinhole_views, view_center_ray_level
from scene_memory.common.calib_io import load_rig_calibration, rotation_to_quaternion_xyzw
from scene_memory.perception.yoloe_detector import YoloeEquirectDetector
from scene_memory.common.obb import yaw_to_quaternion_xyzw
from scene_memory.common.geometry import (
    azimuth_elevation,
    camera_origin_level,
    equirect_uv_to_da360_rays,
    rays_to_level,
)
from scene_memory.common.tiles import box_center_uv

DEFAULT_CALIB = paths.CALIB
DEFAULT_VIO_CONFIG = paths.VIO_CONFIG
DEFAULT_NAMES = (
    "person door chair table sofa bed stairs backpack bottle cup "
    "potted plant tv refrigerator sink toilet laptop book clock"
)


def default_names_file() -> Path:
    return paths.CONFIG / "prompts.txt"


def load_names(names: str, names_file: str) -> list[str]:
    if names.strip():
        return [part for part in names.replace(",", " ").split() if part]
    path = Path(names_file).expanduser()
    if path.is_file():
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
        if lines:
            return lines
    return [part for part in DEFAULT_NAMES.split() if part]


def attach_level_rays(detections: list[dict], width: int, height: int, T_level_da360: np.ndarray) -> np.ndarray:
    """Add ``uv``, ``ray_level``, azimuth and elevation. Returns the camera origin in ``lidar_level``."""
    origin = camera_origin_level(T_level_da360)
    if not detections:
        return origin
    uv = [box_center_uv(det["xyxy"], width) for det in detections]
    u = np.array([p[0] for p in uv], dtype=np.float64)
    v = np.array([p[1] for p in uv], dtype=np.float64)
    rays = rays_to_level(equirect_uv_to_da360_rays(u, v, width, height), T_level_da360)
    azimuth, elevation = azimuth_elevation(rays)
    for det, (uu, vv), ray, az, el in zip(detections, uv, rays, azimuth, elevation):
        det["uv"] = [float(uu), float(vv)]
        det["ray_level"] = [float(c) for c in ray]
        det["azimuth_rad"] = float(az)
        det["elevation_rad"] = float(el)
    return origin


def cloud_xyz(msg) -> np.ndarray:
    """Read XYZ from a PointCloud2 already expressed in ``lidar_level``."""
    from sensor_msgs_py import point_cloud2

    if hasattr(point_cloud2, "read_points_numpy"):
        pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        if getattr(pts, "dtype", None) is not None and pts.dtype.names:
            return np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float64)
        return np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    rows = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    return np.array([[p[0], p[1], p[2]] for p in rows], dtype=np.float64)


def detections_payload(detections: list[dict], width: int, height: int, origin: np.ndarray, stamp, frame_id: str,
                       extra: dict | None = None) -> str:
    """``extra`` (views mode): ``views`` = views processed this frame (lets the mapper infer misses) and
    ``view_image`` = the size of one view (``image`` is the size of the mosaic)."""
    body = {
        **(extra or {}),
        "frame_id": frame_id,
        "stamp": {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)},
        "image": {"width": int(width), "height": int(height)},
        "camera_origin_level": [float(c) for c in origin],
        "detections": [
            {
                "name": det["name"],
                "score": float(det["score"]),
                "cls": int(det["cls"]),
                "cls_scores": {str(c): float(s) for c, s in (det.get("cls_scores") or {}).items()},
                "xyxy": [float(v) for v in det["xyxy"]],
                "parts": [[float(v) for v in part] for part in det.get("parts", [])],
                "uv": det.get("uv", []),
                "view": det.get("view"),
                "track_id": det.get("track_id"),
                "mask": mask_to_json(det.get("mask")),
                "ray_level": det.get("ray_level", []),
                "azimuth_rad": det.get("azimuth_rad"),
                "elevation_rad": det.get("elevation_rad"),
                "obb": det.get("obb"),
            }
            for det in detections
        ],
    }
    return json.dumps(body)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Text-prompted detection (YOLOE / SAM3 / GroundingDINO+SAM2) on the Insta360 views. "
            "Calibration is data/calib. Default output is 2D boxes only."
        )
    )
    p.add_argument("--detector", choices=("yoloe", "sam3", "gdino_sam2"), default="yoloe",
                   help="yoloe: YOLOE-26 seg; sam3: SAM3 text-prompted masks; "
                        "gdino_sam2: GroundingDINO boxes + SAM2.1 masks (SuperMap)")
    p.add_argument("--gdino_model", default=str(DEFAULT_GDINO))
    p.add_argument("--sam2_model", default=str(DEFAULT_SAM2))
    p.add_argument("--distractors_file", default=str(paths.CONFIG / "distractors.txt"),
                   help="gdino_sam2: extra prompts whose detections are dropped; empty disables")
    p.add_argument("--ground_classes", default="",
                   help="gdino_sam2: ground classes file (config/ground_classes.txt) or inline spec, e.g. '"
                        + DEFAULT_GROUND_CLASSES + "'; their SAM2 masks "
                        "are published as /detector/<view>/ground_labels (mono8 PNG, pixel = class index + 1, "
                        "image stamp) for ground_segmentation; empty disables")
    p.add_argument("--ground_conf", type=float, default=0.3)
    p.add_argument("--gdino_size", type=int, default=800,
                   help="gdino_sam2: GroundingDINO input shortest side (640 is faster but drops ~1/3 of the boxes)")
    p.add_argument("--sam_imgsz", type=int, default=640,
                   help="gdino_sam2: SAM2 input side (1024 = model default; 640 keeps the boxes, mask IoU ~0.88)")
    p.add_argument("--no_trt", action="store_true",
                   help="gdino_sam2: stay on PyTorch even if TensorRT engines exist "
                        "(build with scripts/gdino_trt.py and scripts/sam2_trt.py)")
    p.add_argument("--gdino_precision", default="fp16", choices=("fp16", "mixed", "int8"),
                   help="gdino_sam2: which GroundingDINO engine to load (scripts/gdino_trt.py --precision)")
    p.add_argument("--sam3_model", default=str(DEFAULT_SAM3))
    p.add_argument("--sam3_imgsz", type=int, default=0, help="SAM3 input side; 0 = ultralytics default")
    p.add_argument("--tracker", choices=("bytetrack", "botsort"), default="bytetrack",
                   help="botsort: ultralytics BoT-SORT with the 3D-to-2D pose prior (SuperMap)")
    p.add_argument("--no_track_prior", action="store_true",
                   help="botsort without projecting the 4D-memory object centres with the VIO pose")
    p.add_argument("--gmc", default="none", help="BoT-SORT image motion compensation: none|sparseOptFlow|orb|ecc")
    p.add_argument("--odom_topic", default="/equirect_vio/odometry")
    p.add_argument("--centers_topic", default="/hydra/track_centers")
    p.add_argument("--yoloe_model", "--checkpoint", dest="yoloe_model",
                   default=str(paths.YOLOE))
    p.add_argument("--yoloe_root", default=str(paths.REFER / "yoloe"))
    p.add_argument("--mobileclip", default="", help="directory or file for mobileclip_blt.pt")
    p.add_argument("--calib_dir", default=str(DEFAULT_CALIB))
    p.add_argument("--input", choices=("views", "equirect"), default="views",
                   help="views: the pinhole view topics, one batch; equirect: tiled panorama (yoloe only)")
    p.add_argument(
        "--views",
        default="front",
        help="Comma-separated pinhole views to detect and track on (baseline: front). Empty = all.",
    )
    p.add_argument("--self_mask", default=str(paths.CONFIG / "self_mask.yaml"),
                   help="per-view robot self-occlusion polygons (painted out, overlapping detections dropped); "
                        "empty disables")
    p.add_argument("--self_mask_overlap", type=float, default=0.5,
                   help="drop detections with at least this fraction of their box on the robot")
    p.add_argument("--image_topic", default="/insta360/equirect/image/compressed")
    p.add_argument("--lidar_topic", default="",
                   help="optional lidar_level cloud, used only with --boxer")
    p.add_argument("--boxer", action="store_true",
                   help="also run BoxerNet and publish 3D OBBs; default is 2D boxes only")
    p.add_argument("--boxer_ckpt", default="", help="BoxerNet checkpoint; default is refer/boxer/ckpts")
    p.add_argument("--boxer_thresh", type=float, default=0.3, help="minimum BoxerNet aleatoric probability")
    p.add_argument("--boxer_size", type=int, default=256,
                   help="BoxerNet pinhole side; 256 is the fast setting, 640 matches the camera view, 960 matches the checkpoint")
    p.add_argument("--no_boxer", action="store_true",
                   help="keep 2D boxes only (default); overrides --boxer")
    p.add_argument("--box_depth", type=float, default=1.5, help="metres kept behind the near lidar hits")
    p.add_argument("--min_box_points", type=int, default=20)
    p.add_argument("--max_box_m", type=float, default=4.0)
    p.add_argument("--max_range", type=float, default=20.0)
    p.add_argument(
        "--frame_id",
        default="imu",
        help="ROS frame for markers/pose (default imu = equirect_vio base). "
        "Use lidar_level only if that TF exists.",
    )
    p.add_argument("--pose_source", choices=("vio", "lio"), default="vio",
                   help="odometry child frame: vio = equirect_vio imu; lio = Super-LIO livox_frame (with --frame_id livox_frame)")
    p.add_argument("--vio_config", default=str(DEFAULT_VIO_CONFIG), help="OpenVIO YAML for T_imu_da360")
    p.add_argument("--lio_config", default=str(paths.LIO_CONFIG), help="Super-LIO YAML (lio.extrinsic.lidar_imu)")
    p.add_argument("--names", default="", help="space or comma separated prompts; empty uses --names_file")
    p.add_argument("--names_file", default=str(default_names_file()))
    p.add_argument("--device", default="")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=0,
                   help="YOLOE resize side; 0 keeps the split view at its own resolution")
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--mode", choices=("tiles", "full"), default="tiles")
    p.add_argument("--max_hz", type=float, default=30.0)
    p.add_argument("--no_track", action="store_true", help="publish detections without track ids")
    p.add_argument("--track_thresh", type=float, default=0.25, help="ByteTrack high-score association threshold")
    p.add_argument("--track_buffer", type=int, default=30, help="frames a lost track id is kept")
    p.add_argument("--match_thresh", type=float, default=0.8, help="ByteTrack IoU distance match threshold")
    p.add_argument("--track_fps", type=float, default=10.0, help="frame rate used to scale the lost-track buffer")
    p.add_argument("--jpeg_quality", type=int, default=80)
    return p.parse_known_args(argv)[0]


def _mosaic(images: list[np.ndarray]) -> np.ndarray:
    if len(images) <= 1:
        return images[0] if images else np.zeros((64, 64, 3), np.uint8)
    height = min(im.shape[0] for im in images)
    width = min(im.shape[1] for im in images)
    tiles = [
        im if im.shape[0] == height and im.shape[1] == width else cv2.resize(im, (width, height))
        for im in images
    ]
    if len(tiles) == 2:
        return np.hstack(tiles)
    mid = (len(tiles) + 1) // 2
    top = np.hstack(tiles[:mid])
    bottom_tiles = tiles[mid:]
    while len(bottom_tiles) < mid:
        bottom_tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([top, np.hstack(bottom_tiles)])


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    import rclpy
    from geometry_msgs.msg import PoseStamped, Point
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2
    from std_msgs.msg import String
    from visualization_msgs.msg import Marker, MarkerArray

    def _wire_box(size) -> list:
        """12 edges of a box centered at the origin, in the object frame."""
        hx, hy, hz = (max(float(v), 0.05) * 0.5 for v in size)
        corners = (
            (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
            (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
        )
        edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        points = []
        for a, b in edges:
            points.append(Point(x=corners[a][0], y=corners[a][1], z=corners[a][2]))
            points.append(Point(x=corners[b][0], y=corners[b][1], z=corners[b][2]))
        return points

    class LiveDetectorNode(Node):
        def __init__(self) -> None:
            super().__init__("detector_live")
            self.args = args
            self.calib = load_rig_calibration(args.calib_dir)
            self.get_logger().info(self.calib.summary())
            # Markers are authored in lidar_level math, then mapped to --frame_id.
            # Default imu: no lidar_level TF needed (VIO already publishes odom→imu).
            self.T_imu_da360 = pose_frame_to_da360(args.pose_source, args.vio_config, args.lio_config, self.calib)
            self.T_imu_level = self.T_imu_da360 @ np.linalg.inv(self.calib.T_level_da360)
            self.T_pub_level = (
                np.eye(4, dtype=np.float64)
                if str(args.frame_id) == "lidar_level"
                else self.T_imu_level
            )
            self.get_logger().info(f"marker frame_id={args.frame_id}")
            names = load_names(args.names, args.names_file)
            device = args.device
            if not device:
                device = _default_device()
            if args.detector != "yoloe" and args.input != "views":
                raise SystemExit(f"--detector {args.detector} needs --input views")
            if args.detector == "gdino_sam2":
                self.get_logger().info(f"loading GroundingDINO {args.gdino_model} + {args.sam2_model} on {device}")
                self.detector = GdinoSam2ViewDetector(
                    names,
                    args.gdino_model,
                    args.sam2_model,
                    device=device,
                    conf=args.conf,
                    iou=args.iou,
                    distractors=load_names("", args.distractors_file) if args.distractors_file else [],
                    ground=parse_ground_classes(args.ground_classes),
                    ground_conf=args.ground_conf,
                    gdino_size=args.gdino_size,
                    sam_imgsz=args.sam_imgsz,
                    trt=not args.no_trt,
                    gdino_precision=args.gdino_precision,
                )
                self.get_logger().info(f"GroundingDINO+SAM2 ready prompt='{self.detector.text}'")
                if self.detector.ground_classes:
                    self.get_logger().info(f"ground classes {self.detector.ground_classes}")
            elif args.ground_classes:
                self.get_logger().warn(f"--ground_classes needs --detector gdino_sam2; ignored for {args.detector}")
            elif args.detector == "sam3":
                self.get_logger().info(f"loading SAM3 {args.sam3_model} on {device} prompts={names}")
                self.detector = Sam3ViewDetector(
                    args.sam3_model, names, device=device, conf=args.conf, iou=args.iou, imgsz=args.sam3_imgsz
                )
                self.get_logger().info("SAM3 ready")
            else:
                self.get_logger().info(f"loading YOLOE {args.yoloe_model} on {device} prompts={names}")
                self.detector = YoloeEquirectDetector(
                    checkpoint=args.yoloe_model,
                    yoloe_root=args.yoloe_root,
                    names=names,
                    device=device,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    overlap=args.overlap,
                    mode=args.mode,
                    mobileclip=args.mobileclip or None,
                )
                head = "e2e nms=off" if self.detector.nms_free else "one-to-many nms"
                self.get_logger().info(f"YOLOE ready head={head}")
            self.tracks = None
            self.pose_tracks = None
            self._odom: list[tuple[float, np.ndarray]] = []
            self._centers: list[dict] = []
            if not args.no_track and args.tracker == "botsort":
                self.pose_tracks = ViewPoseTracker(
                    [v.strip() for v in str(args.views).split(",") if v.strip()] or ["front", "left", "rear", "right"],
                    track_thresh=args.track_thresh,
                    track_buffer=args.track_buffer,
                    match_thresh=args.match_thresh,
                    gmc=args.gmc,
                )
                self.get_logger().info(
                    f"BoT-SORT ready buffer={args.track_buffer} gmc={args.gmc} "
                    f"3d_prior={'off' if args.no_track_prior else args.centers_topic}"
                )
            elif not args.no_track:
                self.tracks = ViewByteTracker(
                    paths.REFER / "ByteTrack",
                    names,
                    track_thresh=args.track_thresh,
                    track_buffer=args.track_buffer,
                    match_thresh=args.match_thresh,
                    frame_rate=args.track_fps,
                )
                self.get_logger().info(
                    f"ByteTrack ready thresh={args.track_thresh} buffer={args.track_buffer} "
                    f"match={args.match_thresh} fps={args.track_fps}"
                )
            self.boxer = None
            self._detect_device = device
            if args.boxer and not args.no_boxer:
                ckpt = args.boxer_ckpt or str(DEFAULT_CKPT)
                self.get_logger().info(f"loading BoxerNet {ckpt}")
                self.boxer = BoxerLifter(
                    args.calib_dir, ckpt, device, thresh=args.boxer_thresh, image_size=args.boxer_size
                )
                self.get_logger().info(f"BoxerNet ready view={self.boxer.size} thresh={args.boxer_thresh}")
            self._queue: queue.Queue = queue.Queue(maxsize=1)
            self._stop = threading.Event()
            self._lidar_lock = threading.Lock()
            self._lidar_xyz: np.ndarray | None = None
            self._view_lock = threading.Lock()
            self._views_bgr: dict[str, np.ndarray] = {}
            self._view_stamps: dict[str, object] = {}
            self._view_stamp = None
            self._last_log = 0.0
            self._frames = 0
            self.view_calib = load_pinhole_views(self.calib.source_dir) if args.input == "views" else []
            wanted = [v.strip() for v in str(args.views).split(",") if v.strip()]
            if args.input == "views" and wanted:
                missing = [n for n in wanted if n not in {view["name"] for view in self.view_calib}]
                if missing:
                    raise SystemExit(f"unknown --views {missing}; have {[v['name'] for v in self.view_calib]}")
                self.view_calib = [view for view in self.view_calib if view["name"] in set(wanted)]
            self._view_by_name = {view["name"]: view for view in self.view_calib}
            self.self_mask = None
            if args.input == "views" and args.self_mask and Path(args.self_mask).is_file():
                from scene_memory.perception.self_mask import SelfMask

                self.self_mask = SelfMask(args.self_mask)
                masked = [v for v in self._view_by_name if self.self_mask.polygons.get(v)]
                self.get_logger().info(f"robot self mask {args.self_mask} on views {masked or 'none'}")

            qos = qos_profile_sensor_data
            if args.input == "views":
                for view in self.view_calib:
                    self.create_subscription(
                        CompressedImage,
                        view["image_topic"],
                        lambda msg, name=view["name"]: self._on_view(name, msg),
                        qos,
                    )
            else:
                self.create_subscription(CompressedImage, args.image_topic, self._on_image, qos)
            if args.lidar_topic:
                self.create_subscription(PointCloud2, args.lidar_topic, self._on_lidar, qos)
            if self.pose_tracks is not None and not args.no_track_prior:
                from nav_msgs.msg import Odometry

                self.create_subscription(Odometry, args.odom_topic, self._on_odom, 50)
                self.create_subscription(String, args.centers_topic, self._on_centers, 10)
            self.pub_image = self.create_publisher(CompressedImage, "/detector/image/compressed", qos)
            self.pub_info = self.create_publisher(CameraInfo, "/detector/camera_info", qos)
            self.pub_pose = self.create_publisher(PoseStamped, "/detector/camera_pose", qos)
            self.pub_json = self.create_publisher(String, "/detector/detections_json", 10)
            self.pub_markers = self.create_publisher(MarkerArray, "/detector/markers", 10)
            self.pub_ground = {
                view["name"]: self.create_publisher(CompressedImage, f"/detector/{view['name']}/ground_labels", 2)
                for view in self.view_calib
            } if getattr(self.detector, "ground_classes", None) else {}
            self._worker = threading.Thread(target=self._loop, daemon=True)
            self._worker.start()
            topics = ",".join(view["image_topic"] for view in self.view_calib) if args.input == "views" else args.image_topic
            lifter = "boxer" if self.boxer is not None else "2d"
            self.get_logger().info(
                f"waiting for {topics}; input={args.input} mode={args.mode} "
                f"max_hz={args.max_hz} lift={lifter} calib={self.calib.source_dir}"
                + (f"; lidar {args.lidar_topic}" if args.lidar_topic else "")
            )

        def _on_lidar(self, msg: PointCloud2) -> None:
            try:
                xyz = cloud_xyz(msg)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"lidar decode failed: {exc}")
                return
            if xyz.size == 0:
                return
            with self._lidar_lock:
                self._lidar_xyz = xyz

        def _on_odom(self, msg) -> None:
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._odom.append((t, pose_to_matrix((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))))
            if len(self._odom) > 300:
                del self._odom[:100]

        def _on_centers(self, msg) -> None:
            try:
                self._centers = json.loads(msg.data).get("objects", [])
            except (json.JSONDecodeError, AttributeError):
                pass

        def _pose_at(self, stamp, max_dt: float = 0.2):
            """VIO pose closest to the image stamp (odom stamps are image stamps)."""
            if not self._odom:
                return None
            t = stamp.sec + stamp.nanosec * 1e-9
            dt, T = min(((abs(t - ts), T) for ts, T in self._odom[-200:]), key=lambda x: x[0])
            return T if dt <= max_dt else None

        def _priors(self, view_name: str, width: int, height: int, stamp) -> dict:
            if self.args.no_track_prior or not self._centers:
                return {}
            T_odom_imu = self._pose_at(stamp)
            if T_odom_imu is None:
                return {}
            view = self._view_by_name[view_name]
            fx, fy, cx, cy = pinhole_intrinsics(width, height, float(view["hfov_deg"]))
            pose = T_odom_optical(T_odom_imu, self.T_imu_da360, view["R_body_optical"])
            return project_centers(self._centers, pose, fx, fy, cx, cy, width, height)

        def _on_image(self, msg: CompressedImage) -> None:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return
            stamped = (bgr, msg.header.stamp)
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put(stamped)

        def _on_view(self, name: str, msg) -> None:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return
            with self._view_lock:
                self._views_bgr[name] = bgr
                self._view_stamps[name] = msg.header.stamp
                self._view_stamp = msg.header.stamp
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put(None)

        def _loop(self) -> None:
            period = 1.0 / max(args.max_hz, 0.1)
            while not self._stop.is_set():
                try:
                    item = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                t0 = time.perf_counter()
                try:
                    if self.args.input == "views":
                        with self._view_lock:
                            frames = dict(self._views_bgr)
                            stamps = dict(self._view_stamps)
                            stamp = self._view_stamp
                        if stamp is None or stamp == getattr(self, "_seen_stamp", None):
                            continue
                        self._seen_stamp = stamp
                        self._infer_views(frames, stamp, stamps)
                    else:
                        bgr, stamp = item
                        self._infer(bgr, stamp)
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().error(f"inference failed: {exc}")
                elapsed = time.perf_counter() - t0
                remain = period - elapsed
                if remain > 0:
                    time.sleep(remain)

        def _infer_views(self, frames: dict[str, np.ndarray], stamp, stamps: dict | None = None) -> None:
            order = [view["name"] for view in self.view_calib if view["name"] in frames]
            if not order or stamp is None:
                return
            started = time.perf_counter()
            images = [frames[name] for name in order]
            if not getattr(self, "_logged_yolo_size", False):
                self._logged_yolo_size = True
                height0, width0 = images[0].shape[:2]
                self.get_logger().info(f"detector input {width0}x{height0} from the split views")
                if hasattr(self.detector, "trt_engine"):
                    from scene_memory.perception.gdino_trt import engine_path

                    h, w = self.detector.input_hw(height0, width0)
                    precision = self.detector.gdino_precision
                    path = engine_path(self.detector._model_dir, self.detector.text, h, w, precision)
                    if self.detector.trt_engine(height0, width0) is not None:
                        self.get_logger().info(f"GroundingDINO TensorRT engine {path.name}")
                    else:
                        self.get_logger().warn(f"GroundingDINO on PyTorch (no TensorRT engine {path.name}; "
                                               f"build with scripts/gdino_trt.py --precision {precision})")
                    from scene_memory.perception import sam2_trt

                    sam_engine = sam2_trt.engine_path(self.detector._sam_path, self.detector.sam_imgsz)
                    if self.detector.trt and sam_engine.is_file():
                        self.get_logger().info(f"SAM2 encoder TensorRT engine {sam_engine.name}")
                    elif self.detector.trt:
                        self.get_logger().warn(f"SAM2 encoder on PyTorch (no TensorRT engine {sam_engine.name}; "
                                               "build with scripts/sam2_trt.py)")
            if self.self_mask is not None:
                images = [self.self_mask.paint(name, image) for name, image in zip(order, images)]
            per_view = self.detector.predict_views(images)
            yolo_ms = (time.perf_counter() - started) * 1000.0
            if self.self_mask is not None:
                per_view = [
                    [det for det in dets
                     if self.self_mask.overlap(name, det["xyxy"], *image.shape[:2]) < self.args.self_mask_overlap]
                    for name, image, dets in zip(order, images, per_view)
                ]
            if self.pub_ground:
                self._publish_ground(order, (stamps or {}), stamp)
            raw_dets = sum(len(dets) for dets in per_view)
            track_started = time.perf_counter()
            n_prior = 0
            if self.pose_tracks is not None:
                tracked = []
                for name, image, dets in zip(order, images, per_view):
                    height, width = image.shape[:2]
                    priors = self._priors(name, width, height, stamp)
                    tracked.append(self.pose_tracks.update(name, dets, width, height, image, priors))
                    n_prior += self.pose_tracks.n_prior
                per_view = tracked
            elif self.tracks is not None:
                per_view = [
                    attach_track_masks(
                        self.tracks.update(name, dets, image.shape[1], image.shape[0]),
                        dets,
                    )
                    for name, image, dets in zip(order, images, per_view)
                ]
            track_ms = (time.perf_counter() - track_started) * 1000.0
            grouped: dict[str, list[dict]] = {}
            flat: list[dict] = []
            for name, image, dets in zip(order, images, per_view):
                view = self._view_by_name[name]
                height, width = image.shape[:2]
                dets.sort(key=lambda det: -float(det["score"]))
                dets = dets[:16]
                for det in dets:
                    x1, y1, x2, y2 = det["xyxy"]
                    det["view"] = name
                    ray = view_center_ray_level(
                        0.5 * (x1 + x2),
                        0.5 * (y1 + y2),
                        width,
                        height,
                        view["hfov_deg"],
                        view["R_body_optical"],
                        self.calib.T_level_da360,
                    )
                    det["ray_level"] = [float(v) for v in ray]
                grouped[name] = dets
                flat.extend(dets)
            with self._lidar_lock:
                cloud = None if self._lidar_xyz is None else self._lidar_xyz
            boxer_started = time.perf_counter()
            if self.boxer is not None:
                n_obb = self.boxer.lift_views(frames, grouped, cloud, self.calib.T_level_da360)
            else:
                n_obb = 0
            boxer_ms = (time.perf_counter() - boxer_started) * 1000.0
            if self.boxer is not None and self.boxer._batch_failed and not getattr(self, "_batch_warned", False):
                self._batch_warned = True
                self.get_logger().error(f"boxer batch failed, falling back per view: {self.boxer._batch_error}")
            canvas = _mosaic([draw_detections(frames[name], grouped[name]) for name in order])
            origin = camera_origin_level(self.calib.T_level_da360)
            height, width = canvas.shape[:2]
            vh, vw = images[0].shape[:2]
            self._publish(canvas, flat, origin, stamp, width, height,
                          extra={"views": order, "view_image": {"width": int(vw), "height": int(vh)}})
            self._frames += 1
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            fps = 1000.0 / elapsed_ms if elapsed_ms > 0.0 else 0.0
            now = time.time()
            if now - self._last_log > 2.0:
                self._last_log = now
                self.get_logger().info(
                    f"views={len(order)} dets={raw_dets} tracks={len(flat)} prior={n_prior} "
                    f"obb={n_obb} {self.args.detector}={yolo_ms:.0f}ms track={track_ms:.0f}ms boxer={boxer_ms:.0f}ms "
                    f"rate={fps:.1f}fps frames={self._frames}"
                )

        def _publish_ground(self, order: list[str], stamps: dict, stamp) -> None:
            classes = ",".join(self.detector.ground_classes)
            for name, labels in zip(order, self.detector.ground_labels):
                if labels is None:
                    continue
                robot = self.self_mask.mask(name, *labels.shape[:2]) if self.self_mask is not None else None
                if robot is not None:
                    labels = labels.copy()
                    labels[robot] = 0
                ok, png = cv2.imencode(".png", labels)
                if not ok:
                    continue
                msg = CompressedImage()
                msg.header.stamp = stamps.get(name, stamp)
                msg.header.frame_id = f"insta360_{name}_optical"
                msg.format = f"mono8; png compressed; classes={classes}"
                msg.data = array("B", png.tobytes())
                self.pub_ground[name].publish(msg)

        def _infer(self, bgr: np.ndarray, stamp) -> None:
            height, width = bgr.shape[:2]
            raw = self.detector.predict(bgr)
            if self.tracks is not None:
                detections = attach_track_masks(
                    self.tracks.update("equirect", raw, width, height),
                    raw,
                )
            else:
                detections = raw
            origin = attach_level_rays(detections, width, height, self.calib.T_level_da360)
            with self._lidar_lock:
                cloud = None if self._lidar_xyz is None else self._lidar_xyz
            if self.boxer is not None:
                n_obb = self.boxer.lift(bgr, detections, cloud, self.calib.T_level_da360)
            else:
                n_obb = 0
            annotated = draw_detections(bgr, detections)
            self._publish(annotated, detections, origin, stamp, width, height)
            self._frames += 1
            now = time.time()
            if now - self._last_log > 2.0:
                self._last_log = now
                self.get_logger().info(
                    f"frame {width}x{height} dets={len(detections)} obb={n_obb} frames={self._frames}"
                )

        def _publish(self, bgr, detections, origin, stamp, width, height, extra=None) -> None:
            header_frame = "insta360_equirect"
            image = CompressedImage()
            image.header.stamp = stamp
            image.header.frame_id = header_frame
            image.format = "jpeg"
            ok, buf = cv2.imencode(
                ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.jpeg_quality)]
            )
            if ok:
                image.data = buf.tobytes()
                self.pub_image.publish(image)

            info = CameraInfo()
            info.header.stamp = stamp
            info.header.frame_id = header_frame
            info.width = width
            info.height = height
            info.distortion_model = "equirectangular"
            info.d = []
            info.k = [
                float(width), 0.0, width / 2.0 - 0.5,
                0.0, float(height), height / 2.0 - 0.5,
                0.0, 0.0, 1.0,
            ]
            info.p = [
                float(width), 0.0, width / 2.0 - 0.5, 0.0,
                0.0, float(height), height / 2.0 - 0.5, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ]
            self.pub_info.publish(info)

            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self.args.frame_id
            # Camera pose in publish frame (level→imu when frame_id=imu).
            T_pub_cam = self.T_pub_level @ self.calib.T_level_insta360
            t = T_pub_cam[:3, 3]
            q = rotation_to_quaternion_xyzw(T_pub_cam[:3, :3])
            pose.pose.position.x = float(t[0])
            pose.pose.position.y = float(t[1])
            pose.pose.position.z = float(t[2])
            pose.pose.orientation.x = float(q[0])
            pose.pose.orientation.y = float(q[1])
            pose.pose.orientation.z = float(q[2])
            pose.pose.orientation.w = float(q[3])
            self.pub_pose.publish(pose)

            origin_pub = self._level_to_pub_point(origin)
            text = String()
            text.data = detections_payload(detections, width, height, origin_pub, stamp, self.args.frame_id, extra)
            self.pub_json.publish(text)
            self.pub_markers.publish(self._markers(detections, origin_pub, stamp))

        def _level_to_pub_point(self, p) -> np.ndarray:
            xyz = np.asarray(p, dtype=np.float64).reshape(3)
            return self.T_pub_level[:3, :3] @ xyz + self.T_pub_level[:3, 3]

        def _level_to_pub_dir(self, d) -> np.ndarray:
            return self.T_pub_level[:3, :3] @ np.asarray(d, dtype=np.float64).reshape(3)

        def _markers(self, detections, origin, stamp) -> MarkerArray:
            array = MarkerArray()
            for namespace in ("det_ray", "det_label", "det_obb"):
                clear = Marker()
                clear.header.stamp = stamp
                clear.header.frame_id = self.args.frame_id
                clear.ns = namespace
                clear.id = 0
                clear.action = Marker.DELETEALL
                array.markers.append(clear)
            for i, det in enumerate(detections):
                ray = det.get("ray_level")
                if not ray:
                    continue
                ray_pub = self._level_to_pub_dir(ray)
                arrow = Marker()
                arrow.header.stamp = stamp
                arrow.header.frame_id = self.args.frame_id
                arrow.ns = "det_ray"
                arrow.id = i
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.scale.x = 0.02
                arrow.scale.y = 0.04
                arrow.scale.z = 0.06
                arrow.color.a = 0.9
                arrow.color.r = 0.1
                arrow.color.g = 0.8
                arrow.color.b = 0.2
                start = Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2]))
                end = Point(
                    x=float(origin[0] + ray_pub[0]),
                    y=float(origin[1] + ray_pub[1]),
                    z=float(origin[2] + ray_pub[2]),
                )
                arrow.points = [start, end]
                array.markers.append(arrow)

                label = Marker()
                label.header = arrow.header
                label.ns = "det_label"
                label.id = i
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position = end
                obb = det.get("obb")
                if obb:
                    center = self._level_to_pub_point(obb["center"])
                    label.pose.position = Point(
                        x=float(center[0]),
                        y=float(center[1]),
                        z=float(center[2]) + 0.5 * float(obb["size"][2]) + 0.2,
                    )
                color = color_bgr(str(det["name"]))
                label.pose.orientation.w = 1.0
                label.scale.z = 0.4
                label.color.a = 1.0
                label.color.r = color[2] / 255.0
                label.color.g = color[1] / 255.0
                label.color.b = color[0] / 255.0
                track_id = det.get("track_id")
                prefix = f"#{int(track_id)} " if track_id is not None else ""
                label.text = f"{prefix}{det['name']} {float(det['score']):.2f}"
                if obb and obb.get("score3d") is not None:
                    label.text += f" / {float(obb['score3d']):.2f}"
                array.markers.append(label)
                if not obb:
                    continue
                box = Marker()
                box.header.stamp = stamp
                box.header.frame_id = self.args.frame_id
                box.ns = "det_obb"
                box.id = i
                box.type = Marker.LINE_LIST
                box.action = Marker.ADD
                center = self._level_to_pub_point(obb["center"])
                size = obb["size"]
                yaw = float(obb["yaw"])
                c, s = np.cos(yaw), np.sin(yaw)
                R_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                quat = rotation_to_quaternion_xyzw(self.T_pub_level[:3, :3] @ R_yaw)
                box.pose.position.x = float(center[0])
                box.pose.position.y = float(center[1])
                box.pose.position.z = float(center[2])
                box.pose.orientation.x = float(quat[0])
                box.pose.orientation.y = float(quat[1])
                box.pose.orientation.z = float(quat[2])
                box.pose.orientation.w = float(quat[3])
                box.scale.x = 0.03
                box.points = _wire_box(size)
                box.color.a = 1.0
                box.color.r = color[2] / 255.0
                box.color.g = color[1] / 255.0
                box.color.b = color[0] / 255.0
                array.markers.append(box)
            return array

        def destroy_node(self):
            self._stop.set()
            return super().destroy_node()

    def _default_device() -> str:
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _stop_now(signum, frame) -> None:
        del signum, frame
        # Inference is inside CUDA on another thread. Letting rclpy/PyTorch
        # unwind from Ctrl+C calls std::terminate. Exit the process instead.
        os.write(2, b"\nstopping detector\n")
        os._exit(0)

    signal.signal(signal.SIGINT, _stop_now)
    signal.signal(signal.SIGTERM, _stop_now)
    rclpy.init()
    node = LiveDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node._stop.set()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
