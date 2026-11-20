"""Prepare OpenVIO dataset folders for X4 Air sessions."""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def write_imu_csv(path: Path, rows: Iterable[tuple[float, float, float, float, float, float, float]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "ax", "ay", "az", "gx", "gy", "gz"])
        for row in rows:
            w.writerow(
                [
                    f"{row[0]:.9f}",
                    f"{row[1]:.9f}",
                    f"{row[2]:.9f}",
                    f"{row[3]:.9f}",
                    f"{row[4]:.9f}",
                    f"{row[5]:.9f}",
                    f"{row[6]:.9f}",
                ]
            )
            n += 1
    return n


def write_cam_timestamps(path: Path, stamps_s: list[float]) -> None:
    path.write_text("".join(f"{t:.9f}\n" for t in stamps_s), encoding="utf-8")


def parse_cameradata_csv(path: Path) -> list[tuple[float, float, float, float, float, float, float]]:
    """Parse Gyroflow / Insta360 cameradata CSV (timestamp_ms + org_acc_* + org_gyro_* deg/s)."""
    rows: list[tuple[float, float, float, float, float, float, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = {k.lower(): k for k in (reader.fieldnames or [])}

        def col(*names: str) -> str:
            for n in names:
                if n.lower() in fields:
                    return fields[n.lower()]
            raise KeyError(f"Missing columns {names} in {path}; have {list(fields)}")

        t_key = col("timestamp_ms", "timestamp", "t")
        ax = col("org_acc_x", "acc_x", "ax")
        ay = col("org_acc_y", "acc_y", "ay")
        az = col("org_acc_z", "acc_z", "az")
        gx = col("org_gyro_x", "gyro_x", "gx")
        gy = col("org_gyro_y", "gyro_y", "gy")
        gz = col("org_gyro_z", "gyro_z", "gz")
        gyro_is_deg = "org_gyro_x" in {k.lower() for k in fields}
        for r in reader:
            t_raw = float(r[t_key])
            key_l = t_key.lower()
            if "ns" in key_l or t_raw > 1e15:
                t = t_raw * 1e-9
            elif "ms" in key_l or t_raw > 1e11:
                t = t_raw * 1e-3
            else:
                t = t_raw
            gscale = math.pi / 180.0 if gyro_is_deg else 1.0
            rows.append(
                (
                    t,
                    float(r[ax]),
                    float(r[ay]),
                    float(r[az]),
                    float(r[gx]) * gscale,
                    float(r[gy]) * gscale,
                    float(r[gz]) * gscale,
                )
            )
    if not rows:
        raise RuntimeError(f"No IMU rows in {path}")
    t0 = rows[0][0]
    return [(t - t0, ax, ay, az, gx, gy, gz) for t, ax, ay, az, gx, gy, gz in rows]


def extract_frames(
    video: Path,
    out_dir: Path,
    *,
    width: int,
    height: int,
    jpeg_quality: int = 92,
) -> tuple[list[Path], float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*"):
        p.unlink()
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    num, den = probe.stdout.strip().split("/")
    fps = float(num) / float(den)
    pattern = str(out_dir / "%06d.jpg")
    vf = f"scale={width}:{height}"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            vf,
            "-q:v",
            str(max(2, min(31, int(31 - jpeg_quality * 0.3)))),
            pattern,
        ],
        check=True,
        capture_output=True,
    )
    frames = sorted(out_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames in {out_dir}")
    return frames, fps


def import_video_session(
    *,
    video: Path,
    camera_data: Path,
    session_dir: Path,
    width: int = 1440,
    height: int = 720,
) -> dict:
    session_dir = session_dir.resolve()
    panocam = session_dir / "panocam"
    frames, fps = extract_frames(video.resolve(), panocam, width=width, height=height)
    stamps = [i / fps for i in range(len(frames))]
    write_cam_timestamps(session_dir / "cam_timestamps.txt", stamps)
    imu_rows = parse_cameradata_csv(camera_data.resolve())
    n_imu = write_imu_csv(session_dir / "imu_data.csv", imu_rows)
    summary = {
        "session_dir": str(session_dir),
        "num_images": len(frames),
        "num_imu": n_imu,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_s": stamps[-1] if stamps else 0.0,
        "image_dir": "panocam",
    }
    (session_dir / "session_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def resize_existing_images(src_dir: Path, dst_dir: Path, width: int, height: int) -> list[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    outs: list[Path] = []
    paths = sorted(
        list(src_dir.glob("*.png"))
        + list(src_dir.glob("*.jpg"))
        + list(src_dir.glob("*.jpeg"))
        + list(src_dir.glob("*.JPG"))
    )
    for i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        out = dst_dir / f"{i:06d}.png"
        cv2.imwrite(str(out), img)
        outs.append(out)
    return outs


def run_openvio(
    *,
    session_dir: Path,
    config: Path,
    output: Path,
    openvio_bin: Path,
    gui: bool = False,
) -> dict:
    session_dir = session_dir.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not openvio_bin.is_file():
        raise FileNotFoundError(
            f"OpenVIO binary not found: {openvio_bin}. Build ../openvio with ./build.sh first."
        )
    mode = "gui" if gui else "odometry"
    cmd = [
        str(openvio_bin),
        mode,
        "--dataset_path",
        str(session_dir),
        "--config_path",
        str(config.resolve()),
        "--output_path",
        str(output),
    ]
    if gui:
        cmd.append("--exit_at_end")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    summary = {
        "ok": proc.returncode == 0 and output.is_file() and output.stat().st_size > 0,
        "returncode": proc.returncode,
        "cmd": cmd,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "output": str(output),
        "num_poses": sum(1 for _ in output.open()) if output.is_file() else 0,
    }
    (output.parent / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"OpenVIO failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
    return summary


def default_openvio_bin(repo_root: Path) -> Path:
    """Prefer the vendored OpenVIO build inside this repo."""
    candidates = [
        repo_root / "build" / "openvio" / "bin" / "vio",
        repo_root / "third_party" / "openvio" / "build" / "bin" / "vio",
        # Legacy sibling clone (optional fallback)
        repo_root.parent / "openvio" / "build" / "bin" / "vio",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]