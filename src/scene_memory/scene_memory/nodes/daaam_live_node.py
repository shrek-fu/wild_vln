"""Live DAAAM stack (FastSAM + BotSort + optional DAM) on one Insta360 view.

YOLOE+ByteTrack is parked for later VLN target labels. This node:

* consumes ``/insta360/{view}/image/compressed`` + ``/equirect_vio/odometry``
* estimates metric depth (Depth-Anything) when no lidar is given
* runs ``PipelineOrchestrator.process_frame`` (SAM → BotSort → assignment/DAM)
* publishes DAAAM label/color/depth/pose (no depth-lift ``/obj_boxes``)

Requires the editable ``daaam`` package and FastSAM weights under
``refer/DAAAM/checkpoints/``. On 8 GB GPUs keep ``--enable_grounding`` off
until VRAM allows DAM workers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from scene_memory import paths
from scene_memory.mapping.daaam_mapping import DaaamHydraBridge, clip_depth_meters
from scene_memory.perception.mono_depth import DEFAULT_WEIGHTS, MetricMonoDepth
from scene_memory.mapping.scene_graph import (
    T_odom_optical,
    load_imu_from_camera,
    pinhole_intrinsics,
    pose7_xyzw,
    pose_to_matrix,
)
from scene_memory.perception.boxer_lift import load_pinhole_views

DAAAM_ROOT = paths.REFER / "DAAAM"
DEFAULT_VIO_CONFIG = paths.VIO_CONFIG
DEFAULT_DAAAM_CFG = paths.CONFIG / "daaam_live.yaml"


def _ensure_daaam_on_path() -> None:
    src = str(DAAAM_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _ensure_label_map() -> tuple[Path, Path]:
    """DAAAM needs 30k pseudo ``label_names`` slots for incremental track ids."""
    yaml_path = DAAAM_ROOT / "config" / "labels_pseudo.yaml"
    csv_path = DAAAM_ROOT / "config" / "labels_pseudo.csv"
    if not yaml_path.is_file() or not csv_path.is_file():
        from daaam.utils.vision import create_label_map

        create_label_map(str(yaml_path), 30000)
    return yaml_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib_dir", default=str(paths.CALIB))
    parser.add_argument("--vio_config", default=str(DEFAULT_VIO_CONFIG))
    parser.add_argument("--daaam_config", default=str(DEFAULT_DAAAM_CFG))
    parser.add_argument("--view", default="front", help="Single pinhole view for DAAAM (GPU budget)")
    parser.add_argument("--image_prefix", default="/insta360")
    parser.add_argument("--odom_topic", default="/equirect_vio/odometry")
    parser.add_argument("--depth_source", default="mono", choices=("mono", "none"))
    parser.add_argument("--depth_weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--depth_lb", type=float, default=0.25)
    parser.add_argument("--depth_ub", type=float, default=20.0)
    parser.add_argument("--hydra_config", default="")
    parser.add_argument(
        "--enable_grounding",
        action="store_true",
        help="Start DAM assignment/grounding workers (heavy VRAM)",
    )
    args = parser.parse_args()

    _ensure_daaam_on_path()
    try:
        from daaam.config import PipelineConfig
        from daaam.pipeline import PipelineOrchestrator
        from daaam.pipeline.models import Frame
    except ImportError as exc:
        raise SystemExit(
            f"DAAAM import failed ({exc}). Install with:\n"
            f"  pip install --user -e {DAAAM_ROOT} --no-deps\n"
            f"  pip install --user boxmot ultralytics pydantic python-dotenv"
        ) from exc

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, Image
    from std_msgs.msg import String

    yaml_path, csv_path = _ensure_label_map()
    cfg = PipelineConfig.from_yaml(args.daaam_config, validate_files=False)
    # SegmentationService joins ROOT_DIR/checkpoints|config with these relative paths.
    # TrackingService joins ROOT_DIR with reid_weights.
    cfg.semantic_config_path = str(yaml_path)
    cfg.labelspace_colors_path = str(csv_path)
    ckpt = DAAAM_ROOT / "checkpoints" / cfg.segmentation.model_name
    if not ckpt.is_file():
        raise SystemExit(f"FastSAM checkpoint missing: {ckpt}")
    cfg.depth.depth_lb = args.depth_lb
    cfg.depth.depth_ub = args.depth_ub
    if not args.enable_grounding:
        cfg.workers.num_grounding_workers = 0
        cfg.workers.num_assignment_workers = 0
        cfg.grounding.enable_perframe_clip_features = False

    rclpy.init()
    node = Node("daaam_live")
    mono = None
    if args.depth_source == "mono":
        mono = MetricMonoDepth(weights=args.depth_weights)

    node.get_logger().info(
        "Loading DAAAM orchestrator (SAM + BotSort"
        + (", DAM)" if args.enable_grounding else "; grounding deferred)")
    )
    orchestrator = PipelineOrchestrator(cfg)
    try:
        orchestrator.segmentation_service.warmup()
    except Exception as exc:  # noqa: BLE001
        node.get_logger().warn(f"segmentation warmup skipped: {exc}")
    try:
        orchestrator.tracking_service.warmup()
    except Exception as exc:  # noqa: BLE001
        node.get_logger().warn(f"tracking warmup skipped: {exc}")

    views = {v["name"]: v for v in load_pinhole_views(args.calib_dir)}
    if args.view not in views:
        raise SystemExit(f"view {args.view} not in calib")
    spec = views[args.view]
    T_imu_da360 = load_imu_from_camera(args.vio_config)
    hydra = DaaamHydraBridge(args.hydra_config, depth_lb=args.depth_lb, depth_ub=args.depth_ub)
    T_odom_imu: dict = {"T": None, "pose7": None, "T_opt": None}
    frame_id = {"n": 0}
    last_pose7 = {"v": None}
    last_vel = {"lin": np.zeros(3), "t": None}

    pubs = {
        "depth": node.create_publisher(Image, f"/hydra/{args.view}/depth", 10),
        "labels": node.create_publisher(Image, f"/hydra/{args.view}/labels", 10),
        "semantic": node.create_publisher(Image, f"/hydra/{args.view}/semantic_color", 10),
        "pose": node.create_publisher(PoseStamped, f"/hydra/{args.view}/pose", 10),
        "status": node.create_publisher(String, "/daaam/status", 10),
        "json": node.create_publisher(String, "/daaam/tracks_json", 10),
    }

    def _image(stamp, frame, array, encoding):
        msg = Image()
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

    def _on_odom(msg):
        pose = msg.pose.pose
        q = pose.orientation
        T = pose_to_matrix(
            (pose.position.x, pose.position.y, pose.position.z),
            (q.x, q.y, q.z, q.w),
        )
        T_opt = T_odom_optical(T, T_imu_da360, spec["R_body_optical"])
        pose7 = pose7_xyzw(T_opt)
        now = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if last_pose7["v"] is not None and last_vel["t"] is not None:
            dt = max(1e-3, now - last_vel["t"])
            last_vel["lin"] = (pose7[:3] - last_pose7["v"][:3]) / dt
        last_pose7["v"] = pose7
        last_vel["t"] = now
        T_odom_imu["T"] = T
        T_odom_imu["pose7"] = pose7
        T_odom_imu["T_opt"] = T_opt

    def _on_image(msg):
        if T_odom_imu.get("pose7") is None:
            return
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        fx, fy, cx, cy = pinhole_intrinsics(w, h, float(spec["hfov_deg"]))
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if mono is not None:
            depth = clip_depth_meters(mono.infer_bgr(bgr), args.depth_lb, args.depth_ub)
        else:
            depth = np.zeros((h, w), dtype=np.float32)

        frame_id["n"] += 1
        frame = Frame(
            frame_id=frame_id["n"],
            timestamp=stamp_sec,
            rgb_image=rgb,
            depth_image=depth,
            transform=np.asarray(T_odom_imu["pose7"], dtype=np.float64),
            lin_vel=np.asarray(last_vel["lin"], dtype=np.float64),
            ang_vel=np.zeros(3, dtype=np.float64),
            camera_intrinsics={"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        )
        t0 = time.perf_counter()
        try:
            label_image, color_image = orchestrator.process_frame(frame)
        except Exception as exc:  # noqa: BLE001
            node.get_logger().error(f"DAAAM process_frame failed: {exc}")
            return
        dt_ms = (time.perf_counter() - t0) * 1000.0

        optical_frame = f"insta360_{args.view}_optical"
        pubs["depth"].publish(_image(msg.header.stamp, optical_frame, depth, "32FC1"))
        labels_i32 = np.asarray(label_image, dtype=np.int32)
        pubs["labels"].publish(_image(msg.header.stamp, optical_frame, labels_i32, "32SC1"))
        pubs["semantic"].publish(_image(msg.header.stamp, optical_frame, color_image, "bgr8"))
        pose_msg = PoseStamped()
        pose_msg.header.stamp = msg.header.stamp
        pose_msg.header.frame_id = "odom"
        p7 = T_odom_imu["pose7"]
        pose_msg.pose.position.x = float(p7[0])
        pose_msg.pose.position.y = float(p7[1])
        pose_msg.pose.position.z = float(p7[2])
        pose_msg.pose.orientation.x = float(p7[3])
        pose_msg.pose.orientation.y = float(p7[4])
        pose_msg.pose.orientation.z = float(p7[5])
        pose_msg.pose.orientation.w = float(p7[6])
        pubs["pose"].publish(pose_msg)

        stepped = hydra.process_frame(
            stamp_sec, rgb, depth, labels_i32, p7, fx, fy, cx, cy
        )
        status = String()
        status.data = (
            f"view={args.view} daaam_ms={dt_ms:.0f} depth_px={int(np.count_nonzero(depth))} "
            f"label_px={int(np.count_nonzero(labels_i32))} "
            f"tracks={len(getattr(frame, 'tracks', []) or [])} "
            f"hydra_step={int(stepped)} grounding={'on' if args.enable_grounding else 'off'} "
            f"binding={hydra.reason or 'ready'}"
        )
        pubs["status"].publish(status)
        pubs["json"].publish(
            String(
                data=json.dumps(
                    {
                        "frame_id": frame_id["n"],
                        "tracks": [
                            {
                                "id": int(getattr(t, "id", -1)),
                                "semantic_id": int(
                                    orchestrator.object_labels.get(int(getattr(t, "id", -1)), 0)
                                ),
                            }
                            for t in (getattr(frame, "tracks", []) or [])
                        ],
                    }
                )
            )
        )

    topic = f"{args.image_prefix}/{args.view}/image/compressed"
    node.create_subscription(CompressedImage, topic, _on_image, qos_profile_sensor_data)
    node.create_subscription(Odometry, args.odom_topic, _on_odom, 50)
    node.get_logger().info(
        f"DAAAM live on {topic}; YOLOE parked; "
        f"grounding={'ON' if args.enable_grounding else 'OFF (pass --enable_grounding)'}"
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            orchestrator.stop()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
