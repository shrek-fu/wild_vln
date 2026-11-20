"""Post-process a collection_data session.

    ros2 run collection_data export_session <session_dir> [--frames 10] [--no_mp4]

Writes into the session:
    trajectory_tum.txt          VIO odometry, TUM format (t x y z qx qy qz qw)
    equirect_stamps.csv         stitched frame stamps from the bag
    equirect/<stamp>.jpg        every Nth stitched frame (--frames N)
    raw/seg_NNN/frame_poses.csv raw fisheye frame device_ts -> ROS time -> interpolated VIO pose
    raw/seg_NNN/stream_K.mp4    bitstream remuxed (no re-encode) for players / ffmpeg / SfM tools
    summary.json                counts, rates, durations, gaps
"""

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml

ODOM_TOPIC = "/equirect_vio/odometry"
EQUIRECT_TOPIC = "/insta360/equirect/image/compressed"
IMU_TOPIC = "/imu/data_raw"


def stamp_s(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def read_bag(bag_dir: Path, topics):
    """Yield (topic, msg) for the requested topics."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    meta = yaml.safe_load((bag_dir / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=meta["storage_identifier"]),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = [t for t in topics if t in types]
    if wanted:
        reader.set_filter(rosbag2_py.StorageFilter(topics=wanted))
    classes = {t: get_message(types[t]) for t in wanted}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in classes:
            yield topic, deserialize_message(data, classes[topic])


def slerp(q0: np.ndarray, q1: np.ndarray, a: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where(dot[:, None] < 0, -q1, q1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    small = theta < 1e-6
    sin = np.where(small, 1.0, np.sin(theta))
    w0 = np.where(small, 1 - a, np.sin((1 - a) * theta) / sin)
    w1 = np.where(small, a, np.sin(a * theta) / sin)
    q = w0[:, None] * q0 + w1[:, None] * q1
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def interpolate_poses(traj: np.ndarray, t: np.ndarray, max_gap_s: float = 0.2):
    """traj: Nx8 TUM rows sorted by time. Returns Mx7 poses and a validity mask."""
    out = np.full((len(t), 7), np.nan)
    valid = np.zeros(len(t), dtype=bool)
    if len(traj) < 2 or len(t) == 0:
        return out, valid
    idx = np.searchsorted(traj[:, 0], t)
    inside = (idx > 0) & (idx < len(traj))
    i1 = np.clip(idx, 1, len(traj) - 1)
    i0 = i1 - 1
    t0, t1 = traj[i0, 0], traj[i1, 0]
    inside &= (t1 - t0) <= max_gap_s
    a = np.clip((t - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0)
    out[:, :3] = traj[i0, 1:4] + a[:, None] * (traj[i1, 1:4] - traj[i0, 1:4])
    out[:, 3:] = slerp(traj[i0, 4:8], traj[i1, 4:8], a)
    out[~inside] = np.nan
    valid[inside] = True
    return out, valid


def rate_stats(t: np.ndarray) -> dict:
    t = np.sort(t[np.isfinite(t)])
    if len(t) < 2:
        return {"count": int(len(t))}
    dt = np.diff(t)
    return {
        "count": int(len(t)),
        "duration_s": round(float(t[-1] - t[0]), 3),
        "rate_hz": round(float((len(t) - 1) / max(t[-1] - t[0], 1e-9)), 2),
        "max_gap_s": round(float(dt.max()), 4),
    }


def load_csv(path: Path) -> list:
    if not path.is_file():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def remux(stream: Path, fps: float) -> dict:
    mp4 = stream.with_suffix(".mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-fflags", "+genpts", "-r", f"{fps:g}",
           "-i", str(stream), "-c", "copy", str(mp4)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return {"mp4_error": res.stderr.strip()[-400:]}
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=codec_name,width,height,nb_read_packets", "-of", "json", str(mp4)],
        capture_output=True, text=True,
    )
    info = json.loads(probe.stdout or "{}").get("streams", [{}])
    return {"mp4": mp4.name, **(info[0] if info else {})}


def export_raw_segment(seg: Path, traj: np.ndarray, make_mp4: bool) -> dict:
    report = {"segment": seg.name}
    camera = seg / "camera.json"
    if camera.is_file():
        cam = json.loads(camera.read_text())
        report["camera"] = {k: cam.get(k) for k in ("camera_name", "serial", "encode", "stream_resolution",
                                                    "stream_bitrate_mbps")}
        report["calibration_strings"] = len(cam.get("calibration", []))

    packets = load_csv(seg / "video_index.csv")
    by_stream = {}
    for p in packets:
        s = by_stream.setdefault(p["stream_index"], {"packets": 0, "bytes": 0, "file": p["file"], "arrival": []})
        s["packets"] += 1
        s["bytes"] += int(p["size"])
        s["arrival"].append(float(p["arrival_ros_s"]))
    report["streams"] = {}
    for idx, s in sorted(by_stream.items()):
        entry = {"file": s["file"], "packets": s["packets"], "MB": round(s["bytes"] / 1e6, 2),
                 **rate_stats(np.asarray(s["arrival"]))}
        if make_mp4 and shutil.which("ffmpeg") and (seg / s["file"]).is_file():
            entry.update(remux(seg / s["file"], entry.get("rate_hz") or 30.0))
        report["streams"][idx] = entry

    gyro = load_csv(seg / "gyro.csv")
    if gyro:
        mapped = np.array([float(g["ros_s"]) for g in gyro])
        report["gyro"] = rate_stats(mapped[mapped > 0])
        report["gyro"]["unmapped"] = int((mapped <= 0).sum())
    exposure = load_csv(seg / "exposure.csv")
    if exposure:
        report["exposure"] = rate_stats(np.array([float(e["arrival_ros_s"]) for e in exposure]))

    frames = load_csv(seg / "frames.csv")
    if frames:
        dev = np.array([int(f["device_ts"]) for f in frames], dtype=np.int64)
        ros = np.array([float(f["ros_s"]) for f in frames])
        report["stitched_frames"] = rate_stats(ros)
        poses, valid = interpolate_poses(traj, ros)
        with (seg / "frame_poses.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "device_ts", "ros_s", "valid", "x", "y", "z", "qx", "qy", "qz", "qw"])
            for i, (d, t, ok, p) in enumerate(zip(dev, ros, valid, poses)):
                w.writerow([i, int(d), f"{t:.9f}", int(ok), *(f"{v:.6f}" for v in p)])
        report["frames_with_pose"] = int(valid.sum())
        packet_ts = {int(p["device_ts"]) for p in packets}
        report["frames_matching_packet_ts"] = int(sum(int(d) in packet_ts for d in dev))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--frames", type=int, default=0, help="save every Nth stitched JPEG to equirect/ (0 = none)")
    ap.add_argument("--no_mp4", action="store_true", help="skip remuxing the raw bitstream")
    args = ap.parse_args()
    session = args.session.expanduser().resolve()
    summary = {"session": str(session)}

    odom, equirect_t, imu_t = [], [], []
    bag = session / "bag"
    if (bag / "metadata.yaml").is_file():
        out_dir = session / "equirect"
        if args.frames > 0:
            out_dir.mkdir(exist_ok=True)
        n_eq = 0
        for topic, msg in read_bag(bag, [ODOM_TOPIC, EQUIRECT_TOPIC, IMU_TOPIC]):
            if topic == ODOM_TOPIC:
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                odom.append([stamp_s(msg.header), p.x, p.y, p.z, q.x, q.y, q.z, q.w])
            elif topic == IMU_TOPIC:
                imu_t.append(stamp_s(msg.header))
            else:
                t = stamp_s(msg.header)
                name = ""
                if args.frames > 0 and n_eq % args.frames == 0:
                    name = f"{t:.6f}.jpg"
                    (out_dir / name).write_bytes(bytes(msg.data))
                equirect_t.append((t, name))
                n_eq += 1
    else:
        summary["bag"] = "missing"

    traj = np.asarray(sorted(odom), dtype=float).reshape(-1, 8)
    np.savetxt(session / "trajectory_tum.txt", traj, fmt="%.9f %.6f %.6f %.6f %.7f %.7f %.7f %.7f")
    with (session / "equirect_stamps.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ros_s", "file"])
        w.writerows([f"{t:.9f}", n] for t, n in equirect_t)

    summary["odometry"] = rate_stats(traj[:, 0]) if len(traj) else {"count": 0}
    if len(traj) > 1:
        steps = np.linalg.norm(np.diff(traj[:, 1:4], axis=0), axis=1)
        summary["odometry"]["path_length_m"] = round(float(steps.sum()), 3)
        summary["odometry"]["extent_m"] = np.round(np.ptp(traj[:, 1:4], axis=0), 3).tolist()
    summary["equirect_bag"] = rate_stats(np.array([t for t, _ in equirect_t]))
    summary["imu_bag"] = rate_stats(np.array(imu_t))
    summary["raw"] = [export_raw_segment(seg, traj, not args.no_mp4)
                      for seg in sorted((session / "raw").glob("seg_*"))]

    (session / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
