"""DAAAM-aligned Hydra input publisher for Insta360 views.

Baseline: front RGB-D only (``--views front``):

1. Depth from DA360 equirect (preferred), lidar, or mono fallback
2. RGB + optional label/color images from detector tracks (zeros if no dets)
3. ``T_odom_optical`` as 7D pose; ``pipeline.step`` when ``hydra_python`` exists

Depth source (``--depth_source``): ``da360`` | ``mono`` | ``lidar`` | ``auto``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scene_memory import paths
from scene_memory.perception.boxer_lift import DEFAULT_CKPT, equirect_depth_to_pinhole, load_pinhole_views
from scene_memory.mapping.boxer_objects import BoxerObjectMap, backproject_depth
from scene_memory.common.calib_io import load_rig_calibration
from scene_memory.mapping.daaam_mapping import (
    DaaamHydraBridge,
    clip_depth_meters,
    create_label_and_color,
    depth_edge_mask,
    erode_label_edges,
)
from scene_memory.common.labelspace import load_labelspace, write_labelspace_files
from scene_memory.mapping.object_views import KeyViewStore, descriptions_by_key, load_description_cache
from scene_memory.perception.mono_depth import DEFAULT_WEIGHTS, MetricMonoDepth
from scene_memory.mapping.scene_graph import (
    T_odom_optical,
    depth_from_level_points,
    load_imu_from_camera,
    load_level_transforms,
    pinhole_intrinsics,
    points_to_level,
    pose7_xyzw,
    pose_to_matrix,
)

DEFAULT_VIO_CONFIG = paths.VIO_CONFIG
DEFAULT_PROMPTS = paths.CONFIG / "prompts.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib_dir", default=str(paths.CALIB))
    parser.add_argument("--vio_config", default=str(DEFAULT_VIO_CONFIG))
    parser.add_argument("--lidar_topic", default="", help="PointCloud2 in lidar_level, front_lidar or rear_lidar")
    parser.add_argument(
        "--depth_source",
        default="da360",
        choices=("da360", "depthart", "mono", "lidar", "auto"),
        help="da360 = warp /da360/depth_metric to pinhole; depthart = DepthART metric TensorRT per view; "
             "mono = DA2 Metric; lidar = project cloud",
    )
    parser.add_argument("--depthart_domain", default="outdoor", choices=("indoor", "outdoor"),
                        help="DepthART weights: indoor (max 10 m) | outdoor (max 80 m)")
    parser.add_argument(
        "--da360_depth_topic",
        default="/da360/depth_metric",
        help="Equirect depth Image (32FC1); depth_metric preferred, depth is relative",
    )
    parser.add_argument("--depth_weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--depth_device", default="", help="cuda / cpu; empty picks cuda if available")
    parser.add_argument("--odom_topic", default="/equirect_vio/odometry")
    parser.add_argument("--detections_topic", default="/detector/detections_json")
    parser.add_argument("--image_prefix", default="/insta360")
    parser.add_argument(
        "--hydra_config",
        default="",
        help="Optional YAML file or Hydra config dir; empty uses packaged dataset configs",
    )
    parser.add_argument("--hydra_dataset", default="habitat", help="Packaged Hydra dataset name")
    parser.add_argument("--hydra_labelspace", default="ade20k_mp3d", help="Packaged label space")
    parser.add_argument("--hydra_output", default="", help="Optional Hydra log/output directory")
    parser.add_argument("--hydra_view", default="front", help="The one view passed to pipeline.step")
    parser.add_argument(
        "--views",
        default="front",
        help="Comma-separated views to process (baseline: front). Empty = all calib views.",
    )
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--depth_lb", type=float, default=0.25, help="DAAAM depth_lb metres")
    parser.add_argument("--depth_ub", type=float, default=20.0, help="DAAAM depth_ub metres")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument(
        "--depth_edge_ratio",
        type=float,
        default=0.08,
        help="Drop depth where local 3x3 max-min > ratio*depth (flying pixels); 0 disables",
    )
    parser.add_argument("--label_erode_px", type=int, default=3, help="Shrink detector masks before Hydra; 0 disables")
    parser.add_argument("--no_boxer", action="store_true", help="Skip BoxerNet; /hydra/objects falls back to Hydra DSG")
    parser.add_argument("--boxer_ckpt", default="")
    parser.add_argument("--boxer_device", default="", help="cuda / cpu; empty picks cuda if available")
    parser.add_argument("--boxer_thresh", type=float, default=0.3)
    parser.add_argument("--boxer_size", type=int, default=0, help="BoxerNet input size; 0 = trained size")
    parser.add_argument("--boxer_max_hz", type=float, default=2.0)
    parser.add_argument("--boxer_stride", type=int, default=8, help="Depth pixel stride for Boxer sdp_w points")
    parser.add_argument("--object_ttl", type=float, default=5.0, help="Seconds a Boxer track box survives unseen")
    parser.add_argument("--label_accuracy", type=float, default=0.7,
                        help="detector confusion-matrix diagonal for the Bayesian label fusion")
    parser.add_argument("--track_centers_topic", default="/hydra/track_centers",
                        help="object centres + track ids in odom, the 3D-to-2D prior for the 2D tracker")
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=0.0,
        help="Multiply depth by this; 0 = auto from image-only BoxerNet (DA360 without lidar is not metric)",
    )
    parser.add_argument(
        "--scale_warmup",
        type=int,
        default=5,
        help="Auto scale: Boxer frames before Hydra steps and before depth points feed Boxer",
    )
    parser.add_argument(
        "--scale_lock_after",
        type=int,
        default=30,
        help="Auto scale: freeze after this many Boxer frames so the TSDF keeps one scale; 0 never locks",
    )
    parser.add_argument(
        "--scene_dir",
        default="",
        help="Where dsg.json + objects_4d.json are written; empty = data/scene4d/<start time>",
    )
    parser.add_argument("--save_period", type=float, default=30.0, help="Seconds between exports; 0 = only on exit")
    parser.add_argument(
        "--no_views",
        action="store_true",
        help="Don't save per-object key views (image + mask) for offline DAM",
    )
    args = parser.parse_args()

    import rclpy

    rclpy.init()
    node = HydraScene(args)
    try:
        rclpy.spin(node.node)
    except KeyboardInterrupt:
        pass
    finally:
        node.export()
        node.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class HydraScene:
    def __init__(self, args) -> None:
        from geometry_msgs.msg import Point, PoseStamped
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2
        from std_msgs.msg import String
        from visualization_msgs.msg import Marker, MarkerArray

        self._Image = Image
        self._CameraInfo = CameraInfo
        self._PoseStamped = PoseStamped
        self._Point = Point
        self._Marker = Marker
        self._MarkerArray = MarkerArray
        self.args = args
        self.node = Node("hydra_scene")
        self._n_objects = 0
        rig = load_rig_calibration(args.calib_dir)
        self.T_level_da360 = rig.T_level_da360
        all_views = {view["name"]: view for view in load_pinhole_views(args.calib_dir)}
        wanted = [v.strip() for v in str(args.views).split(",") if v.strip()]
        if wanted:
            missing = [n for n in wanted if n not in all_views]
            if missing:
                raise SystemExit(f"unknown --views {missing}; have {sorted(all_views)}")
            self.views = {n: all_views[n] for n in wanted}
        else:
            self.views = all_views
        self._last_status_ns = 0
        self.T_imu_da360 = load_imu_from_camera(args.vio_config)
        self.level_tf = load_level_transforms(args.calib_dir)
        self.T_odom_imu = None
        self.cloud_xyz = np.zeros((0, 3), dtype=np.float64)
        self.cloud_frame = ""
        self.detections: list[dict] = []
        self.labelspace = load_labelspace(args.prompts)
        write_labelspace_files(self.labelspace, paths.CONFIG)
        self.stepper = DaaamHydraBridge(
            config_path=args.hydra_config,
            depth_lb=args.depth_lb,
            depth_ub=args.depth_ub,
            dataset=args.hydra_dataset,
            labelspace=args.hydra_labelspace,
            output_path=args.hydra_output,
            hydra_label_config=self.labelspace.to_hydra_label_config(),
        )
        self.mono = None
        self._mono_cache: dict[str, tuple[float, np.ndarray]] = {}
        self._da360_depth: np.ndarray | None = None
        self._da360_stamp_ns = 0
        # Single-view baseline can run a bit faster than the old 4-view budget.
        self._mono_min_dt = 0.08 if len(self.views) == 1 else 0.15
        if args.depth_source in ("mono", "auto"):
            try:
                self.mono = MetricMonoDepth(weights=args.depth_weights, device=args.depth_device)
                self.node.get_logger().info(
                    f"mono depth ready ({args.depth_weights}) on {self.mono.device}"
                )
            except Exception as exc:  # noqa: BLE001 — surface weight/import failures once
                self.node.get_logger().error(f"mono depth unavailable: {exc}")
                if args.depth_source == "mono":
                    raise
        self.depthart = {}
        if args.depth_source == "depthart":
            from scene_memory.perception.depthart_trt import DepthArtMetric

            for spec in self.views.values():
                size = (int(spec["width"] or args.width), int(spec["height"] or args.height), float(spec["hfov_deg"]))
                if size not in self.depthart:
                    self.depthart[size] = DepthArtMetric(*size, domain=args.depthart_domain)
            engines = ", ".join(sorted({m.engine_file.name for m in self.depthart.values()}))
            self.node.get_logger().info(f"DepthART depth ready ({engines})")
        self.pubs = {}
        for name in self.views:
            self.pubs[name] = {
                "rgb": self.node.create_publisher(Image, f"/hydra/{name}/rgb", 10),
                "depth": self.node.create_publisher(Image, f"/hydra/{name}/depth", 10),
                # Same style as /da360/depth_viz (inverse-depth colormap) for Foxglove compare.
                "depth_viz": self.node.create_publisher(Image, f"/hydra/{name}/depth_viz", 10),
                "labels": self.node.create_publisher(Image, f"/hydra/{name}/labels", 10),
                "semantic": self.node.create_publisher(Image, f"/hydra/{name}/semantic_color", 10),
                "pose": self.node.create_publisher(PoseStamped, f"/hydra/{name}/pose", 10),
                "info": self.node.create_publisher(CameraInfo, f"/hydra/{name}/camera_info", 10),
            }
            topic = f"{args.image_prefix}/{name}/image/compressed"
            self.node.create_subscription(
                CompressedImage, topic, lambda msg, view=name: self._on_image(view, msg), qos_profile_sensor_data
            )
        self.node.create_subscription(Odometry, args.odom_topic, self._on_odom, 50)
        self.node.create_subscription(String, args.detections_topic, self._on_dets, 10)
        if args.depth_source in ("da360", "auto"):
            self.node.create_subscription(
                Image, args.da360_depth_topic, self._on_da360_depth, qos_profile_sensor_data
            )
        if args.lidar_topic:
            self.node.create_subscription(PointCloud2, args.lidar_topic, self._on_cloud, qos_profile_sensor_data)
        self.status = self.node.create_publisher(String, "/hydra/status", 10)
        # Object boxes in odom: BoxerNet OBBs on /hydra/objects, Hydra DSG clusters on /hydra/objects_dsg.
        self.pub_objects = self.node.create_publisher(MarkerArray, "/hydra/objects", 10)
        self.pub_objects_dsg = self.node.create_publisher(MarkerArray, "/hydra/objects_dsg", 10)
        self.boxer = None
        self.boxer_map = BoxerObjectMap(
            ttl_s=args.object_ttl,
            class_names=self.labelspace.names[1:],
            label_accuracy=args.label_accuracy,
        )
        self.pub_centers = self.node.create_publisher(String, args.track_centers_topic, 10)
        self._boxer_last_s = 0.0
        self._boxer_ms = 0.0
        self.depth_scale = float(args.depth_scale) if args.depth_scale > 0 else 1.0
        self._scale_samples = 0
        self._scale_locked = args.depth_scale > 0
        import time as _time

        self.scene_dir = Path(args.scene_dir) if args.scene_dir else (
            paths.SCENES / _time.strftime("%Y%m%d_%H%M%S")
        )
        self._last_save_s = _time.monotonic()
        self._layer_counts: dict[str, int] = {}
        self.views_store = None if args.no_views else KeyViewStore(self.scene_dir)
        if not args.no_boxer:
            try:
                from scene_memory.perception.boxer_lift import BoxerLifter

                device = args.boxer_device
                if not device:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                self.boxer = BoxerLifter(
                    args.calib_dir,
                    args.boxer_ckpt or str(DEFAULT_CKPT),
                    device,
                    thresh=args.boxer_thresh,
                    image_size=args.boxer_size or None,
                )
                self.node.get_logger().info(f"BoxerNet ready size={self.boxer.size} on {device}")
            except Exception as exc:  # noqa: BLE001 — keep Hydra running without Boxer
                self.node.get_logger().error(f"BoxerNet unavailable, using Hydra DSG boxes: {exc}")
        self.node.get_logger().info(
            f"front RGBD baseline views={list(self.views)} step={self.stepper.reason or 'ready'} "
            f"depth_source={args.depth_source} da360={args.da360_depth_topic if args.depth_source in ('da360', 'auto') else '(off)'} "
            f"lidar={args.lidar_topic or '(none)'} boxer={'on' if self.boxer else 'off'} objects=/hydra/objects "
            f"scene_dir={self.scene_dir}"
        )

    def export(self) -> None:
        """Write Hydra DSG (``dsg.json``) and the 4D object memory (``objects_4d.json``)."""
        import time

        self._last_save_s = time.monotonic()
        try:
            self.scene_dir.mkdir(parents=True, exist_ok=True)
            latest = self.scene_dir.parent / "latest"
            if latest.is_symlink() and latest.resolve() != self.scene_dir.resolve():
                latest.unlink()
            if not latest.exists():
                latest.symlink_to(self.scene_dir.name)
            # DAM worker (describe_objects --watch) writes descriptions.json next to us.
            by_key = descriptions_by_key(load_description_cache(self.scene_dir))
            for key, obj in self.boxer_map.objects.items():
                if key in by_key:
                    obj["descriptions"] = by_key[key]
            saved = self.stepper.save_graph(self.scene_dir / "dsg.json")
            self._layer_counts = self.stepper.layer_counts()
            self.boxer_map.save(
                self.scene_dir / "objects_4d.json",
                extra={
                    "saved_at": time.time(),
                    "depth_scale": self.depth_scale,
                    "depth_scale_locked": self._scale_locked,
                    "hydra_frames": self.stepper.frames,
                    "dsg_saved": saved,
                    "dsg_layers": self._layer_counts,
                },
            )
        except Exception as exc:  # noqa: BLE001 — export must not kill the node
            self.node.get_logger().warn(f"export failed: {exc}")
    def _on_odom(self, msg) -> None:
        pose = msg.pose.pose
        q = pose.orientation
        self.T_odom_imu = pose_to_matrix(
            (pose.position.x, pose.position.y, pose.position.z),
            (q.x, q.y, q.z, q.w),
        )

    def _on_da360_depth(self, msg) -> None:
        """Cache equirect radial depth from DA360 (32FC1 metres or relative)."""
        if msg.encoding not in ("32FC1", "32FC"):
            self.node.get_logger().warn(f"unsupported DA360 depth encoding {msg.encoding}")
            return
        arr = np.frombuffer(bytes(msg.data), dtype=np.float32)
        if arr.size != int(msg.height) * int(msg.width):
            return
        self._da360_depth = arr.reshape(int(msg.height), int(msg.width)).copy()
        self._da360_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _on_dets(self, msg) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        rows = payload.get("detections", []) if isinstance(payload, dict) else payload
        self.detections = rows if isinstance(rows, list) else []

    def _on_cloud(self, msg) -> None:
        from sensor_msgs_py import point_cloud2

        raw = np.array(list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        if raw.size == 0:
            self.cloud_xyz = np.zeros((0, 3), dtype=np.float64)
        elif raw.dtype.names:
            self.cloud_xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float64)
        else:
            self.cloud_xyz = raw.reshape(-1, 3).astype(np.float64)
        self.cloud_frame = msg.header.frame_id

    def _on_image(self, view: str, msg) -> None:
        import cv2

        if self.T_odom_imu is None or view not in self.views:
            return
        spec = self.views[view]
        width = int(spec["width"] or self.args.width)
        height = int(spec["height"] or self.args.height)
        fx, fy, cx, cy = pinhole_intrinsics(width, height, float(spec["hfov_deg"]))
        tracks = [self._fused(det) for det in self.detections if det.get("view") == view]
        # DAAAM: label_image + color_image from track masks / semantic lookup.
        labels_u16, color_bgr = create_label_and_color(width, height, tracks, self.labelspace)
        labels = labels_u16.astype(np.int32)
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        if bgr.shape[0] != height or bgr.shape[1] != width:
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        raw_depth, depth_kind = self._make_depth(bgr, width, height, spec)
        raw_depth = np.array(raw_depth, dtype=np.float32)
        raw_depth[depth_edge_mask(raw_depth, self.args.depth_edge_ratio)] = 0.0
        auto_scale = depth_kind == "da360" and self.args.depth_scale <= 0
        scale = self.depth_scale if (auto_scale or self.args.depth_scale > 0) else 1.0
        depth = clip_depth_meters(raw_depth * np.float32(scale), self.args.depth_lb, self.args.depth_ub)
        labels = erode_label_edges(labels, self.args.label_erode_px)
        pose = T_odom_optical(self.T_odom_imu, self.T_imu_da360, spec["R_body_optical"])
        stamp = msg.header.stamp
        frame = f"insta360_{view}_optical"
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.pubs[view]["rgb"].publish(self._image(stamp, frame, rgb, "rgb8"))
        self.pubs[view]["depth"].publish(self._image(stamp, frame, depth, "32FC1"))
        self.pubs[view]["depth_viz"].publish(
            self._image(stamp, frame, self._depth_viz_bgr(depth), "bgr8")
        )
        self.pubs[view]["labels"].publish(self._image(stamp, frame, labels, "32SC1"))
        self.pubs[view]["semantic"].publish(self._image(stamp, frame, color_bgr, "bgr8"))
        self.pubs[view]["pose"].publish(self._pose(stamp, pose))
        self.pubs[view]["info"].publish(self._info(stamp, frame, spec))

        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        stepped = False
        scale_ready = not (auto_scale and self.boxer is not None) or self._scale_samples >= self.args.scale_warmup
        if view == self.args.hydra_view and scale_ready:
            stepped = self.stepper.process_frame(
                stamp_sec, rgb, depth, labels, pose7_xyzw(pose), fx, fy, cx, cy
            )
            if stepped:
                dsg = self.stepper.iter_object_boxes()
                self.boxer_map.link_dsg(dsg)
                self.pub_objects_dsg.publish(self._object_markers(stamp, dsg, "hydra_dsg"))
                if self.boxer is None:
                    self._n_objects = len(dsg)
                    self.pub_objects.publish(self._object_markers(stamp, dsg, "hydra_objects"))
        if view == self.args.hydra_view and self.boxer is not None:
            self._run_boxer(
                view,
                bgr,
                raw_depth if (auto_scale and not self._scale_locked) else None,
                depth,
                tracks,
                pose,
                fx,
                fy,
                cx,
                cy,
                stamp,
                stamp_sec,
            )
        import time

        if self.args.save_period > 0 and time.monotonic() - self._last_save_s >= self.args.save_period:
            self.export()
        now_ns = self.node.get_clock().now().nanoseconds
        if now_ns - self._last_status_ns > 1_000_000_000:
            self._last_status_ns = now_ns
            from std_msgs.msg import String

            status = String()
            err = getattr(self.stepper, "_last_error", "") or ""
            status.data = (
                f"baseline view={view} depth={depth_kind} depth_px={int(np.count_nonzero(depth))} "
                f"rgb={width}x{height} labels={int(np.count_nonzero(labels))} tracks={len(tracks)} "
                f"hydra_step={int(stepped)} hydra_frames={self.stepper.frames} "
                f"hydra_attempts={getattr(self.stepper, 'attempts', 0)} "
                f"objects={self._n_objects} boxer_ms={self._boxer_ms:.0f} "
                f"depth_scale={self.depth_scale:.3f}({self._scale_samples}{',locked' if self._scale_locked else ''}) "
                f"layers={','.join(f'{k}:{v}' for k, v in self._layer_counts.items()) or '-'} "
                f"binding={self.stepper.reason or 'ready'}"
                + (f" err={err}" if err else "")
            )
            self.status.publish(status)

    def _make_depth(self, bgr, width: int, height: int, spec) -> tuple[np.ndarray, str]:
        want = self.args.depth_source
        if want == "depthart":
            if bgr is None or not bgr.size:
                return np.zeros((height, width), dtype=np.float32), "empty"
            return self.depthart[(width, height, float(spec["hfov_deg"]))].infer_bgr(bgr), "depthart"
        if want in ("da360", "auto") and self._da360_depth is not None:
            depth = equirect_depth_to_pinhole(
                self._da360_depth,
                spec["R_body_optical"],
                width,
                height,
                float(spec["hfov_deg"]),
            )
            if int(np.count_nonzero(depth)) > 0 or want == "da360":
                return depth, "da360"
        use_lidar = want == "lidar" or (want == "auto" and self.cloud_xyz.shape[0] > 0)
        if use_lidar:
            level = points_to_level(self.cloud_xyz, self.cloud_frame, self.level_tf)
            depth = depth_from_level_points(
                level,
                self.T_level_da360,
                spec["R_body_optical"],
                width,
                height,
                float(spec["hfov_deg"]),
            )
            if int(np.count_nonzero(depth)) > 0 or want == "lidar":
                return depth, "lidar"
        if self.mono is not None and bgr is not None and bgr.size:
            name = str(spec.get("name", "view"))
            now = self.node.get_clock().now().nanoseconds * 1e-9
            cached = self._mono_cache.get(name)
            if cached is not None and (now - cached[0]) < self._mono_min_dt:
                return cached[1], "mono"
            if bgr.shape[0] != height or bgr.shape[1] != width:
                import cv2

                bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
            depth = self.mono.infer_bgr(bgr)
            self._mono_cache[name] = (now, depth)
            return depth, "mono"
        return np.zeros((height, width), dtype=np.float32), "empty"

    def _update_depth_scale(self, dets: list[dict], raw_depth: np.ndarray, T_odom_opt: np.ndarray) -> None:
        """Fit depth scale = image-only Boxer object depth / raw depth at the box core (log-space EMA)."""
        T_opt_odom = np.linalg.inv(T_odom_opt)
        height, width = raw_depth.shape
        ratios = []
        for det in dets:
            obb = det.get("obb")
            if not obb:
                continue
            z_box = float((T_opt_odom[:3, :3] @ np.asarray(obb["center"]) + T_opt_odom[:3, 3])[2])
            x1, y1, x2, y2 = det["xyxy"]
            qx, qy = 0.25 * (x2 - x1), 0.25 * (y2 - y1)
            core = raw_depth[
                int(np.clip(y1 + qy, 0, height - 1)) : int(np.clip(y2 - qy, 1, height)),
                int(np.clip(x1 + qx, 0, width - 1)) : int(np.clip(x2 - qx, 1, width)),
            ]
            core = core[core > 0]
            if z_box > 0.2 and core.size >= 20:
                ratios.append(z_box / float(np.median(core)))
        if not ratios:
            return
        frame_scale = float(np.clip(np.median(ratios), 0.02, 50.0))
        if self._scale_samples == 0:
            self.depth_scale = frame_scale
        else:
            a = 0.2
            self.depth_scale = float(np.exp((1.0 - a) * np.log(self.depth_scale) + a * np.log(frame_scale)))
        self._scale_samples += 1
        if self.args.scale_lock_after > 0 and self._scale_samples >= self.args.scale_lock_after:
            self._scale_locked = True
            self.node.get_logger().info(
                f"depth scale locked at {self.depth_scale:.4f} after {self._scale_samples} Boxer frames "
                f"(reuse with --depth_scale {self.depth_scale:.4f})"
            )

    def _fused(self, det: dict) -> dict:
        """Detection with the object's fused label, so Hydra integrates the stable class."""
        fused = self.boxer_map.fused_label(det.get("track_id"))
        if fused is None or fused[0] == int(det.get("cls", -1)):
            return det
        return {**det, "cls": fused[0], "name": fused[1]}

    def _offer_views(self, dets: list[dict], bgr, stamp_sec: float) -> None:
        if self.views_store is None:
            return
        for det in dets:
            obj = self.boxer_map.objects.get(det.get("memory_key", ""))
            if obj is not None:
                self.views_store.offer(obj, det, bgr, stamp_sec)

    def _run_boxer(self, view, bgr, raw_depth, depth, tracks, pose, fx, fy, cx, cy, stamp, stamp_sec) -> None:
        """BoxerNet OBBs for the detector tracks of this view, in odom.

        With ``raw_depth`` (auto scale), an image-only pass first fits the depth
        scale; the scaled depth points then refine the boxes once warmed up.
        """
        import time

        if self.args.boxer_max_hz > 0 and stamp_sec - self._boxer_last_s < 1.0 / self.args.boxer_max_hz:
            return
        self._boxer_last_s = stamp_sec
        dets = [dict(det) for det in tracks if det.get("xyxy")]
        img_h, img_w = bgr.shape[:2]
        if dets:
            started = time.perf_counter()
            T_odom_da360 = np.asarray(self.T_odom_imu, dtype=np.float64) @ np.asarray(self.T_imu_da360, dtype=np.float64)
            try:
                if raw_depth is not None:
                    self.boxer.lift_views({view: bgr}, {view: dets}, None, T_odom_da360)
                    self._update_depth_scale(dets, raw_depth, pose)
                    if self._scale_samples < self.args.scale_warmup:
                        self.boxer_map.update(dets, stamp_sec, img_w, img_h)
                        self._offer_views(dets, bgr, stamp_sec)
                        dets = []
                if dets:
                    for det in dets:
                        det.pop("obb", None)
                    points = backproject_depth(depth, fx, fy, cx, cy, pose, stride=max(1, self.args.boxer_stride))
                    self.boxer.lift_views({view: bgr}, {view: dets}, points, T_odom_da360)
                    self.boxer_map.update(dets, stamp_sec, img_w, img_h)
                    self._offer_views(dets, bgr, stamp_sec)
            except Exception as exc:  # noqa: BLE001 — one bad frame must not stop Hydra
                self.node.get_logger().warn(f"BoxerNet failed: {exc}", throttle_duration_sec=5.0)
                return
            self._boxer_ms = (time.perf_counter() - started) * 1000.0
        self.boxer_map.expire(stamp_sec)
        from std_msgs.msg import String

        self.pub_centers.publish(
            String(data=json.dumps({"frame_id": "odom", "stamp": stamp_sec, "objects": self.boxer_map.track_centers()}))
        )
        objs = []
        for obj in self.boxer_map.boxes():
            sid = obj["cls"] + 1 if obj["cls"] >= 0 else 0
            label = f"{obj['name'] or self.labelspace.name(sid)} {obj['key']}"
            descs = obj.get("descriptions") or []
            if descs:
                words = descs[-1]["text"].split()
                label += "\n" + " ".join(words[:8]) + ("…" if len(words) > 8 else "")
            if not obj["visible"]:
                label += f" (seen {stamp_sec - obj['last_seen']:.0f}s ago)"
            objs.append(
                {
                    "semantic_id": sid,
                    "name": label,
                    "center": obj["center"],
                    "dims": obj["size"],
                    "corners": obj["corners"],
                    "alpha": 0.9 if obj["visible"] else 0.3,
                }
            )
        self._n_objects = len(objs)
        self.pub_objects.publish(self._object_markers(stamp, objs, "hydra_objects"))

    def _object_markers(self, stamp, objs: list[dict], ns: str):
        """Box edges + text labels in ``odom`` for ``iter_object_boxes``-style dicts."""
        Marker = self._Marker
        array = self._MarkerArray()
        clear = Marker()
        clear.header.stamp = stamp
        clear.header.frame_id = "odom"
        clear.ns = ns
        clear.id = 0
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        clear_txt = Marker()
        clear_txt.header = clear.header
        clear_txt.ns = f"{ns}_labels"
        clear_txt.id = 0
        clear_txt.action = Marker.DELETEALL
        array.markers.append(clear_txt)

        # AABB/OBB edge pairs for spark_dsg BoundingBox.corners() ordering.
        edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        for i, obj in enumerate(objs):
            sid = int(obj["semantic_id"])
            bgr = self.labelspace.color_bgr(sid)
            name = obj["name"] or self.labelspace.name(sid)
            corners = obj.get("corners")
            box = Marker()
            box.header.stamp = stamp
            box.header.frame_id = "odom"
            box.ns = ns
            box.id = i + 1
            box.action = Marker.ADD
            box.pose.orientation.w = 1.0
            box.color.a = float(obj.get("alpha", 0.9))
            box.color.r = bgr[2] / 255.0
            box.color.g = bgr[1] / 255.0
            box.color.b = bgr[0] / 255.0
            if corners is not None and corners.shape[0] >= 8:
                box.type = Marker.LINE_LIST
                box.scale.x = 0.03
                pts = []
                for a, b in edges:
                    if a < corners.shape[0] and b < corners.shape[0]:
                        pts.append(self._Point(x=float(corners[a, 0]), y=float(corners[a, 1]), z=float(corners[a, 2])))
                        pts.append(self._Point(x=float(corners[b, 0]), y=float(corners[b, 1]), z=float(corners[b, 2])))
                box.points = pts
            else:
                center = obj["center"]
                dims = np.maximum(obj["dims"], 0.05)
                box.type = Marker.CUBE
                box.pose.position.x = float(center[0])
                box.pose.position.y = float(center[1])
                box.pose.position.z = float(center[2])
                box.scale.x = float(dims[0])
                box.scale.y = float(dims[1])
                box.scale.z = float(dims[2])
                box.color.a = 0.35
            array.markers.append(box)

            text = Marker()
            text.header = box.header
            text.ns = f"{ns}_labels"
            text.id = i + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            c = obj["center"]
            top = float(corners[:, 2].max()) if corners is not None else float(c[2] + 0.5 * obj["dims"][2])
            text.pose.position.x = float(c[0])
            text.pose.position.y = float(c[1])
            text.pose.position.z = top + 0.15
            text.pose.orientation.w = 1.0
            text.scale.z = 0.25
            text.color.a = 1.0
            text.color.r = box.color.r
            text.color.g = box.color.g
            text.color.b = box.color.b
            text.text = f"{name}#{obj['id']}" if "id" in obj else name
            array.markers.append(text)
        return array

    @staticmethod
    def _depth_viz_bgr(depth_m: np.ndarray) -> np.ndarray:
        """Inverse-depth color preview (same idea as DA360 ``depth_viz``). Not metric."""
        import cv2

        d = np.asarray(depth_m, dtype=np.float32)
        inv = np.zeros_like(d, dtype=np.float32)
        valid = np.isfinite(d) & (d > 1e-3)
        if np.any(valid):
            inv[valid] = 1.0 / d[valid]
            inv[valid] /= max(float(inv[valid].max()), 1e-6)
        u8 = np.clip(inv * 255.0, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)

    def _image(self, stamp, frame: str, array: np.ndarray, encoding: str):
        msg = self._Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame
        msg.height = int(array.shape[0])
        msg.width = int(array.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = 0
        contiguous = np.ascontiguousarray(array)
        channels = 1 if contiguous.ndim == 2 else int(contiguous.shape[2])
        msg.step = int(msg.width * channels * contiguous.dtype.itemsize)
        msg.data = contiguous.tobytes()
        return msg

    def _pose(self, stamp, transform: np.ndarray):
        msg = self._PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "odom"
        pose7 = pose7_xyzw(transform)
        msg.pose.position.x = float(pose7[0])
        msg.pose.position.y = float(pose7[1])
        msg.pose.position.z = float(pose7[2])
        msg.pose.orientation.x = float(pose7[3])
        msg.pose.orientation.y = float(pose7[4])
        msg.pose.orientation.z = float(pose7[5])
        msg.pose.orientation.w = float(pose7[6])
        return msg

    def _info(self, stamp, frame: str, spec):
        msg = self._CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = frame
        width = int(spec["width"] or self.args.width)
        height = int(spec["height"] or self.args.height)
        fx, fy, cx, cy = pinhole_intrinsics(width, height, float(spec["hfov_deg"]))
        msg.width = width
        msg.height = height
        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        return msg


if __name__ == "__main__":
    main()
