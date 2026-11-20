"""CLI for Insta360 X4 Air equirectangular VIO around OpenVIO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .prepare import default_openvio_bin, import_video_session, run_openvio


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="x4_equirect_vio", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_imp = sub.add_parser("import-video", help="Extract frames + IMU CSV into OpenVIO session dir")
    p_imp.add_argument("--video", type=Path, required=True)
    p_imp.add_argument("--camera-data", type=Path, required=True, help="Gyroflow/Insta cameradata CSV")
    p_imp.add_argument("--session-dir", type=Path, required=True)
    p_imp.add_argument("--width", type=int, default=1440)
    p_imp.add_argument("--height", type=int, default=720)

    p_bag = sub.add_parser("import-bag", help="Import ROS2 bag (needs sourced ROS 2)")
    p_bag.add_argument("--bag", type=Path, required=True)
    p_bag.add_argument("--session-dir", type=Path, required=True)
    p_bag.add_argument("--image-topic", default="/camera/image_raw")
    p_bag.add_argument("--imu-topic", default="/imu/data_raw")
    p_bag.add_argument("--width", type=int, default=1440)
    p_bag.add_argument("--height", type=int, default=720)

    p_run = sub.add_parser("run", help="Run OpenVIO offline on a prepared session")
    p_run.add_argument("--session-dir", type=Path, required=True)
    p_run.add_argument("--config", type=Path, default=None)
    p_run.add_argument("--output", type=Path, required=True)
    p_run.add_argument("--openvio-bin", type=Path, default=None)
    p_run.add_argument("--gui", action="store_true")

    p_all = sub.add_parser("all", help="import-video then run")
    p_all.add_argument("--video", type=Path, required=True)
    p_all.add_argument("--camera-data", type=Path, required=True)
    p_all.add_argument("--session-dir", type=Path, required=True)
    p_all.add_argument("--output", type=Path, required=True)
    p_all.add_argument("--config", type=Path, default=None)
    p_all.add_argument("--width", type=int, default=1440)
    p_all.add_argument("--height", type=int, default=720)
    p_all.add_argument("--openvio-bin", type=Path, default=None)

    args = parser.parse_args(argv)
    root = _repo_root()

    if args.cmd == "import-video":
        summary = import_video_session(
            video=args.video,
            camera_data=args.camera_data,
            session_dir=args.session_dir,
            width=args.width,
            height=args.height,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.cmd == "import-bag":
        from .bag_import import import_ros2_bag

        summary = import_ros2_bag(
            bag_path=args.bag,
            session_dir=args.session_dir,
            image_topic=args.image_topic,
            imu_topic=args.imu_topic,
            width=args.width,
            height=args.height,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.cmd in ("run", "all"):
        openvio_bin = args.openvio_bin or default_openvio_bin(root)
        config = args.config
        if args.cmd == "all":
            width = args.width
            height = args.height
            print(
                json.dumps(
                    import_video_session(
                        video=args.video,
                        camera_data=args.camera_data,
                        session_dir=args.session_dir,
                        width=width,
                        height=height,
                    ),
                    indent=2,
                )
            )
            if config is None:
                config = (
                    root / "config" / "x4_air_equirect_1280.yaml"
                    if width == 1280
                    else root / "config" / "x4_air_equirect_1440.yaml"
                )
        if config is None:
            config = root / "config" / "x4_air_equirect_1440.yaml"
        summary = run_openvio(
            session_dir=args.session_dir,
            config=config,
            output=args.output,
            openvio_bin=openvio_bin,
            gui=getattr(args, "gui", False),
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary.get("ok") else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())