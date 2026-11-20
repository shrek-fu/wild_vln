"""Run YOLOE once on a saved equirectangular image and optionally lift rays with the rig calib."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from scene_memory.viz.annotate import draw_detections
from scene_memory.common.calib_io import load_rig_calibration
from scene_memory.perception.yoloe_detector import YoloeEquirectDetector
from scene_memory import paths
from scene_memory.nodes.detector_node import attach_level_rays, default_names_file, load_names


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="equirectangular image path")
    p.add_argument("--output", default="", help="annotated image path")
    p.add_argument("--checkpoint", default=str(paths.YOLOE))
    p.add_argument("--yoloe_root", default=str(paths.REFER / "yoloe"))
    p.add_argument("--mobileclip", default="")
    p.add_argument("--calib_dir", default="", help="if set, attach lidar_level rays")
    p.add_argument("--names", default="")
    p.add_argument("--names_file", default=str(default_names_file()))
    p.add_argument("--device", default="")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--mode", choices=("tiles", "full"), default="tiles")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    bgr = cv2.imread(args.source, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"failed to read {args.source}")
    device = args.device
    if not device:
        try:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    detector = YoloeEquirectDetector(
        checkpoint=args.checkpoint,
        yoloe_root=args.yoloe_root,
        names=load_names(args.names, args.names_file),
        device=device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        overlap=args.overlap,
        mode=args.mode,
        mobileclip=args.mobileclip or None,
    )
    detections = detector.predict(bgr)
    height, width = bgr.shape[:2]
    origin = None
    if args.calib_dir:
        calib = load_rig_calibration(args.calib_dir)
        origin = attach_level_rays(detections, width, height, calib.T_level_da360)
        print(calib.summary())
    annotated = draw_detections(bgr, detections)
    output = args.output
    if not output:
        src = Path(args.source)
        output = str(src.with_name(src.stem + "_yoloe" + src.suffix))
    cv2.imwrite(output, annotated)
    payload = {
        "image": {"width": width, "height": height, "path": str(args.source)},
        "camera_origin_level": None if origin is None else [float(c) for c in origin],
        "detections": detections,
    }
    print(json.dumps(payload, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main(sys.argv[1:])
