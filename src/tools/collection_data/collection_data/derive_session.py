"""Turn a raw-only collection_data session (record.launch.py raw_only:=true) into a replayable one.

    python3 -m collection_data.derive_session <session_dir> [--out <session>_proc] [--every 2] [--no_lio]

The raw session holds the fisheye bitstream (raw/seg_NNN) and a bag with the sensor topics only. This
writes ``<out>/bag`` with what ``replay_session`` / ``pipeline.launch.py session:=`` expect:

    /insta360/equirect/image/compressed   raw/seg_*/stream_0 stitched by the CUDA stitcher (stitch_segment
                                          --pipe-every), JPEG, stamped with frames.csv (the live stamps)
    /lio/odom, /lio/cloud_world, /tf      Super-LIO re-run on the bagged MID-360S CustomMsg + IMU (real time,
                                          own ROS_DOMAIN_ID so a running robot stack is not disturbed)
    /imu/data_raw, /tf_static             copied from the raw bag

plus ``config/`` (copied), ``raw`` (symlink) and ``session.json``. Then:

    src/scripts/stvln.sh replay <session>_proc hydra_fake
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import heapq
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

EQUIRECT_TOPIC = "/insta360/equirect/image/compressed"
LIDAR_TOPICS = ["/livox/front_lidar/custom", "/livox/front_lidar/imu"]
# recorded while QiNav's driver owned the lidar (lio.launch.py then does not start ours)
QINAV_LIDAR_TOPICS = {"/livox/lidar": LIDAR_TOPICS[0], "/livox/imu": LIDAR_TOPICS[1]}
LIO_TOPICS = {"/lio/odom": "nav_msgs/msg/Odometry", "/lio/cloud_world": "sensor_msgs/msg/PointCloud2",
              "/tf": "tf2_msgs/msg/TFMessage"}
COPY_TOPICS = ["/imu/data_raw", "/tf_static"]


def _src_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "sensors" / "insta360_cuda_stitch").is_dir():
            return parent
        if (parent / "src" / "sensors" / "insta360_cuda_stitch").is_dir():
            return parent / "src"
    raise FileNotFoundError("src/sensors/insta360_cuda_stitch not found above this file")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("session", type=Path)
    p.add_argument("--out", type=Path, default=None, help="default <session>_proc next to the session")
    p.add_argument("--every", type=int, default=2, help="keep every Nth stitched frame (live JPEG topic ~15 Hz)")
    p.add_argument("--width", type=int, default=0, help="equirect width; 0 = the session's equirect_width")
    p.add_argument("--jpeg_quality", type=int, default=80)
    p.add_argument("--encoders", type=int, default=4, help="JPEG encoder threads")
    p.add_argument("--no_lio", action="store_true", help="skip the Super-LIO re-run")
    p.add_argument("--lio_rate", type=float, default=1.0, help="bag play rate for Super-LIO (>1 risks drops)")
    p.add_argument("--domain_id", type=int, default=77, help="ROS_DOMAIN_ID of the Super-LIO re-run")
    p.add_argument("--keep_lidar", action="store_true", help="also copy the raw MID-360S topics")
    p.add_argument("--stitch_bin", type=Path, default=None)
    return p.parse_args(argv)


def _writer(path: Path, topics: dict[str, str]):
    import rosbag2_py

    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
           rosbag2_py.ConverterOptions("cdr", "cdr"))
    for name, typ in topics.items():
        w.create_topic(rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format="cdr"))
    return w


def _reader(path: Path, topics: list[str] | None = None):
    import rosbag2_py

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=""), rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    if topics is not None:
        r.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    return r, types


# ------------------------------------------------------------------ stitching
def stitch(session: Path, out_bag: Path, args) -> int:
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import CompressedImage

    stitch_bin = args.stitch_bin or _src_root() / "sensors" / "insta360_cuda_stitch" / "build" / "stitch_segment"
    if not stitch_bin.exists():
        raise SystemExit(f"{stitch_bin} missing: build src/sensors/insta360_cuda_stitch (cmake) first")
    width = args.width or _equirect_width(session)
    segs = sorted((session / "raw").glob("seg_*"))
    if not segs:
        raise SystemExit(f"no raw/seg_* in {session}")
    w = _writer(out_bag, {EQUIRECT_TOPIC: "sensor_msgs/msg/CompressedImage"})
    params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    total = 0
    with cf.ThreadPoolExecutor(args.encoders) as pool:
        for seg in segs:
            with open(seg / "frames.csv") as f:
                stamp = {int(r["device_ts"]): float(r["ros_s"]) for r in csv.DictReader(f)}
            t0 = time.monotonic()
            proc = subprocess.Popen([str(stitch_bin), str(seg), str(seg), "--width", str(width),
                                     "--pipe-every", str(args.every)], stdout=subprocess.PIPE)
            pending: list[tuple[float, cf.Future]] = []
            n = skipped = 0

            def flush(limit: int) -> None:
                nonlocal n
                while len(pending) > limit:
                    t, fut = pending.pop(0)
                    ok, jpg = fut.result()
                    if not ok:
                        continue
                    msg = CompressedImage()
                    msg.header.stamp.sec, msg.header.stamp.nanosec = int(t), int(round((t % 1) * 1e9)) % 1_000_000_000
                    msg.header.frame_id = "insta360"
                    msg.format = "jpeg"
                    msg.data = jpg.tobytes()
                    w.write(EQUIRECT_TOPIC, serialize_message(msg), int(round(t * 1e9)))
                    n += 1

            while True:
                head = proc.stdout.read(16)
                if len(head) < 16:
                    break
                ts, fw, fh = struct.unpack("<qii", head)
                buf = proc.stdout.read(fw * fh * 3)
                if len(buf) < fw * fh * 3:
                    break
                t = stamp.get(ts)
                if t is None:  # before the live clock mapping settled: no live stamp either
                    skipped += 1
                    continue
                rgb = np.frombuffer(buf, np.uint8).reshape(fh, fw, 3)
                pending.append((t, pool.submit(lambda a: cv2.imencode(".jpg", a[:, :, ::-1], params), rgb)))
                flush(2 * args.encoders)
            flush(0)
            proc.wait()
            dt = time.monotonic() - t0
            print(f"[derive] {seg.name}: {n} frames {width}x{width // 2} in {dt:.0f} s ({n / max(dt, 1e-3):.1f} fps), "
                  f"{skipped} without a live stamp", flush=True)
            total += n
    del w
    return total


def _equirect_width(session: Path) -> int:
    import yaml

    for f in (session / "config" / "insta360_views.yaml",):
        if f.is_file():
            p = yaml.safe_load(f.read_text()).get("insta360_views", {}).get("ros__parameters", {})
            return int(p.get("equirect_width", 3840))
    return 3840


# ------------------------------------------------------------------ Super-LIO
def lidar_topics(bag: Path) -> tuple[list[str], list[str]]:
    """Lidar/IMU topics to play from `bag` and the `ros2 bag play --remap` args onto LIDAR_TOPICS."""
    _, types = _reader(bag, None)
    if all(t in types for t in LIDAR_TOPICS) or not all(t in types for t in QINAV_LIDAR_TOPICS):
        return list(LIDAR_TOPICS), []
    print("[derive] lidar recorded on QiNav's /livox/lidar, /livox/imu; remapped for the re-run", flush=True)
    return list(QINAV_LIDAR_TOPICS), ["--remap", *(f"{a}:={b}" for a, b in QINAV_LIDAR_TOPICS.items())]


class LioRerun:
    """Super-LIO (lio.launch.py, driver off) + recorder on their own domain while the raw bag plays."""

    def __init__(self, session: Path, out_bag: Path, args):
        self.session, self.out_bag, self.args = session, out_bag, args
        self.env = dict(os.environ, ROS_DOMAIN_ID=str(args.domain_id))
        self.procs: list[subprocess.Popen] = []
        self.log = open(out_bag.parent / "lio_rerun.log", "w")

    def _spawn(self, cmd: list[str]) -> subprocess.Popen:
        p = subprocess.Popen(cmd, env=self.env, stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        self.procs.append(p)
        return p

    def start(self) -> None:
        lio_launch = _src_root() / "scene_memory" / "launch" / "lio.launch.py"
        # force: a QiNav super_lio_node on this machine runs on another domain and must not be "reused"
        self._spawn(["ros2", "launch", str(lio_launch), "start_driver:=false", "start_lio:=force",
                     "start_watchdog:=false"])
        self._spawn(["ros2", "bag", "record", "-s", "mcap", "-o", str(self.out_bag), *LIO_TOPICS])
        time.sleep(6.0)  # Super-LIO up, recorder subscribed
        topics, remap = lidar_topics(self.session / "bag")
        self.play = self._spawn(["ros2", "bag", "play", str(self.session / "bag"), "--topics", *topics, *remap,
                                 "--rate", str(self.args.lio_rate), "--disable-keyboard-controls"])

    def wait(self) -> None:
        self.play.wait()
        time.sleep(3.0)
        self.stop()

    def stop(self) -> None:
        for p in reversed(self.procs):
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for p in self.procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        if not self.log.closed:
            self.log.close()


# ------------------------------------------------------------------ merge
def _stamp_ns(msg) -> int:
    h = msg.transforms[0].header if hasattr(msg, "transforms") else msg.header
    return h.stamp.sec * 1_000_000_000 + h.stamp.nanosec


def _messages(path: Path, topics: list[str] | None, restamp: bool, window: int = 200):
    """(t_ns, topic, data) in time order. ``restamp``: bag time := header stamp (recorder bags hold arrival time,
    a reorder window absorbs the small disorder)."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    r, types = _reader(path, topics)
    heap: list[tuple[int, int, str, bytes]] = []
    seq = 0
    classes = {}
    while r.has_next():
        topic, data, t = r.read_next()
        if restamp:
            cls = classes.setdefault(topic, get_message(types[topic]))
            msg = deserialize_message(data, cls)
            if hasattr(msg, "transforms") and not msg.transforms:
                continue
            t = _stamp_ns(msg)
        heapq.heappush(heap, (t, seq, topic, data))
        seq += 1
        if len(heap) > window:
            t, _, topic, data = heapq.heappop(heap)
            yield t, topic, data
    while heap:
        t, _, topic, data = heapq.heappop(heap)
        yield t, topic, data


def merge(out_bag: Path, sources: list[tuple[Path, list[str] | None, bool]]) -> dict[str, int]:
    types: dict[str, str] = {}
    for path, topics, _ in sources:
        _, t = _reader(path, topics)
        types.update({k: v for k, v in t.items() if topics is None or k in topics})
    w = _writer(out_bag, types)
    counts: dict[str, int] = {}
    for t, topic, data in heapq.merge(*(_messages(p, tp, rs) for p, tp, rs in sources), key=lambda m: m[0]):
        w.write(topic, data, t)
        counts[topic] = counts.get(topic, 0) + 1
    del w
    return counts


def main(argv=None) -> None:
    args = _parse_args(argv)
    session = args.session.expanduser().resolve()
    if not (session / "bag" / "metadata.yaml").is_file():
        raise SystemExit(f"{session}/bag/metadata.yaml missing (still recording, or not a session)")
    out = (args.out or session.with_name(session.name + "_proc")).expanduser().resolve()
    if (out / "bag").exists():
        raise SystemExit(f"{out}/bag already exists")
    work = out / "_work"
    if work.exists():  # left by an interrupted run of this tool
        shutil.rmtree(work)
    work.mkdir(parents=True)
    if (session / "config").is_dir():
        shutil.copytree(session / "config", out / "config", dirs_exist_ok=True)
    if not (out / "raw").exists():
        (out / "raw").symlink_to(session / "raw")

    t0 = time.monotonic()
    lio = None
    if not args.no_lio:
        lio = LioRerun(session, work / "lio", args)
        lio.start()
        print(f"[derive] Super-LIO re-run on ROS_DOMAIN_ID={args.domain_id} (log {work / 'lio_rerun.log'})",
              flush=True)
    try:
        frames = stitch(session, work / "equirect", args)
        if lio is not None:
            print("[derive] stitching done; waiting for the Super-LIO re-run (real time) ...", flush=True)
            lio.wait()
    finally:
        if lio is not None:
            lio.stop()

    copy = COPY_TOPICS + (LIDAR_TOPICS + list(QINAV_LIDAR_TOPICS) if args.keep_lidar else [])
    sources = [(session / "bag", copy, False), (work / "equirect", None, False)]
    if lio is not None and (work / "lio" / "metadata.yaml").is_file():
        sources.append((work / "lio", list(LIO_TOPICS), True))
    counts = merge(out / "bag", sources)
    info = {"derived_from": str(session), "created": time.strftime("%Y-%m-%d %H:%M:%S"), "equirect_frames": frames,
            "every": args.every, "lio_rerun": lio is not None, "topics": counts,
            "seconds": round(time.monotonic() - t0, 1)}
    src_info = session / "session.json"
    if src_info.is_file():
        info["source_session"] = json.loads(src_info.read_text())
    (out / "session.json").write_text(json.dumps(info, indent=2))
    if "/lio/odom" not in counts and lio is not None:
        print(f"[derive] WARNING: no /lio/odom from the re-run; see {work / 'lio_rerun.log'} (kept _work)")
    else:
        shutil.rmtree(work)
    print(f"[derive] {out}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"[derive] replay: src/scripts/stvln.sh replay {out} hydra_fake")


if __name__ == "__main__":
    sys.exit(main())
