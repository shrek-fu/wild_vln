"""Follow QiNav's mapping: start semantic mapping with it, file the semantic map next to the saved map.

    src/scripts/stvln.sh qinav_bridge [topo|hydra_fake] [--bg] [--maps_dir DIR] [k:=v ...]

* /ql/nav/task_status (every 2 s): mapping_ready false -> true (api 1031, 2D or 3D) starts
  ``pipeline.launch.py odom:=qinav_mapping`` once QiNav's super_lio_node publishes /lio/odom. Both QiNav
  mapping stacks build the saved map in that odom frame (2D: octomap on /cloud_registered -> <map>.pgm/.yaml;
  3D: dlio_multi_map_node -> <map>_map.pcd), i.e. QiNav's map frame later.
  mapping_ready true -> false (kill_slam, also after 1032 save + stop) stops the pipeline (SIGINT; the
  mapper writes its final graph on exit).
* /ql/nav/map_saved (success): the scene's topo_graph.json, objects_4d.json (+ dsg / descriptions /
  map_cloud.npy) are written next to QiNav's map files as <map>_topo_graph.json, <map>_objects_4d.json, ...
  (frame_id "map") plus <map>_semantic.json; maps dir = that of the saved zip (2D ~/.ros/nav2d/maps,
  3D ~/.ros/nav3d/maps). Written again with the final graph when the pipeline stops.

The pipeline runs as log/stvln_live.pid, so ``stvln.sh status`` / ``stop live`` see it, and a manual
``stvln.sh live`` that is already running is left alone (the camera takes one client).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from scene_memory import paths
from scene_memory.common import livox_probe

PIPELINE = paths.PKG / "launch" / "pipeline.launch.py"
LOG = paths.WS_ROOT / "log"
PID_FILE = LOG / "stvln_live.pid"
# scene file -> suffix after "<map_name>_". QiNav's map_manager lists every sub-directory of a maps dir and
# every .yaml/.pgm/.pcd/.zip/.bt file as a map, so these stay flat files with other extensions.
SCENE_FILES = {"topo_graph.json": "topo_graph.json", "objects_4d.json": "objects_4d.json",
               "clustered_dsg_with_summaries.json": "dsg.json", "descriptions.json": "descriptions.json",
               "map_cloud.npy": "map_cloud.npy"}
REFRAMED = ("topo_graph.json", "objects_4d.json")
MAPS_DIRS = {1: Path("~/.ros/nav2d/maps").expanduser(), 2: Path("~/.ros/nav3d/maps").expanduser()}  # NavMode


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mapper", default="topo")
    p.add_argument("--maps_dir", default="", help="empty = QiNav's maps dir of the saved map (dir of its zip)")
    p.add_argument("--lio_wait_s", type=float, default=30.0, help="wait for QiNav's /lio/odom after 1031")
    p.add_argument("--save_period_s", type=float, default=5.0, help="topo_map_node export period")
    p.add_argument("launch_args", nargs="*", help="extra key:=value for pipeline.launch.py")
    return p.parse_args(argv)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _manual_live() -> int | None:
    try:
        pgid = int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None
    return pgid if _group_alive(pgid) else None


def _restore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def maps_dir_for(zip_path: str, map_type: int) -> Path:
    return Path(zip_path).parent if zip_path else MAPS_DIRS.get(int(map_type), MAPS_DIRS[2])


QINAV_SOURCE_FRAME = ("odom of QiNav's Super-LIO during mapping, the frame its octomap (2D) / "
                      "dlio_multi_map_node (3D) saved the map in")


def file_semantic_map(scene_dir: Path, maps_dir: Path, map_name: str, map_type: int, zip_path: str,
                      final: bool, source_frame: str = QINAV_SOURCE_FRAME) -> list[str]:
    """Copy the scene's map files to <maps_dir>/<map_name>_<file> in QiNav's map frame; returns the names written."""
    maps_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, suffix in SCENE_FILES.items():
        src = scene_dir / name
        if not src.is_file():
            continue
        dst = maps_dir / f"{map_name}_{suffix}"
        tmp = maps_dir / f".{dst.name}.tmp"
        if name in REFRAMED:
            data = json.loads(src.read_text(encoding="utf-8"))
            data["frame_id"] = "map"
            tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        else:
            shutil.copyfile(src, tmp)
        tmp.replace(dst)
        written.append(dst.name)
    is_2d = int(map_type) == 1
    meta = {
        "map_name": map_name,
        "map_type": "2d" if is_2d else "3d",
        "map_zip": zip_path,
        "map_files": [str(maps_dir / f"{map_name}{s}") for s in ((".yaml", ".pgm", ".pcd") if is_2d else ("_map.pcd",))],
        "frame_id": "map",
        "source_frame": source_frame,
        "scene_dir": str(scene_dir),
        "final": final,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": written,
    }
    (maps_dir / f"{map_name}_semantic.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                                        encoding="utf-8")
    return written


class Bridge:
    def __init__(self, node, args):
        self.node, self.args, self.log = node, args, node.get_logger()
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.scene_dir: Path | None = None
        self.saved: dict[str, tuple[int, str]] = {}  # map_name -> (map_type, zip_path) saved in this run
        self.mapping = None
        self.starting = False
        self.last_odom = 0.0

    # ---------------------------------------------------------------- QiNav events
    def on_status(self, msg) -> None:
        mapping = bool(msg.mapping_ready)
        if mapping == self.mapping:
            return
        first, self.mapping = self.mapping is None, mapping
        if mapping:
            self.log.info(f"QiNav mapping {'running' if first else 'started'} (status {msg.status}/{msg.phase})")
            if not self.starting:
                self.starting = True
                threading.Thread(target=self._start, daemon=True).start()
        elif not first:
            self.log.info(f"QiNav mapping ended (status {msg.status})")
            threading.Thread(target=self._stop, daemon=True).start()

    def on_map_saved(self, msg) -> None:
        if not msg.success:
            self.log.warn(f"QiNav map save failed ({msg.map_name}): {msg.error_message}")
            return
        with self.lock:
            scene = self.scene_dir
            if scene is None:
                self.log.warn(f"map '{msg.map_name}' saved but no semantic mapping ran with it")
                return
            self.saved[msg.map_name] = (int(msg.map_type), msg.zip_path)
        # one more export period so the graph covers everything up to the save
        threading.Timer(self.args.save_period_s + 1.0, self._file,
                        args=(msg.map_name, int(msg.map_type), msg.zip_path, False)).start()

    def on_odom(self, _msg) -> None:
        self.last_odom = time.monotonic()

    # ---------------------------------------------------------------- pipeline
    def _start(self) -> None:
        try:
            if (pgid := _manual_live()) is not None:
                self.log.warn(f"stvln live already running (process group {pgid}); not starting another")
                return
            deadline = time.monotonic() + self.args.lio_wait_s
            while time.monotonic() < deadline and self.mapping:
                lio = livox_probe.processes(livox_probe.LIO_PATTERN)
                if lio and time.monotonic() - self.last_odom < 1.0:
                    break
                time.sleep(0.5)
            else:
                if self.mapping:
                    self.log.error(f"no QiNav super_lio_node / /lio/odom within {self.args.lio_wait_s:.0f} s; "
                                   "semantic mapping not started")
                return
            extra = [a for a in self.args.launch_args if not a.startswith("scene_dir:=")]
            given = [a.split(":=", 1)[1] for a in self.args.launch_args if a.startswith("scene_dir:=")]
            scene = Path(given[-1]).expanduser() if given else paths.SCENES / f"qinav_{time.strftime('%Y%m%d_%H%M%S')}"
            cmd = ["ros2", "launch", str(PIPELINE), "odom:=qinav_mapping", f"mapper:={self.args.mapper}",
                   f"scene_dir:={scene}", *extra]
            LOG.mkdir(parents=True, exist_ok=True)
            logf = open(LOG / "stvln_live.log", "w")
            with self.lock:
                self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                             start_new_session=True, preexec_fn=_restore_sigint)
                self.scene_dir, self.saved = scene, {}
            logf.close()
            PID_FILE.write_text(str(self.proc.pid))
            self.log.info(f"semantic mapping started (pid {self.proc.pid}) -> {scene}; log {LOG / 'stvln_live.log'}")
        finally:
            self.starting = False

    def _stop(self) -> None:
        with self.lock:
            proc, scene, saved = self.proc, self.scene_dir, dict(self.saved)
            self.proc = None
        if proc is not None and proc.poll() is None:
            self.log.info("stopping semantic mapping (final save) ...")
            # launch process only: a group-wide SIGINT lets respawn=True children come back
            proc.send_signal(signal.SIGINT)
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    proc.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    self.log.warn(f"pipeline still running; {sig.name} to its process group")
                    os.killpg(proc.pid, sig)
            proc.wait()
        if proc is not None and PID_FILE.is_file() and PID_FILE.read_text().strip() == str(proc.pid):
            PID_FILE.unlink()
        for map_name, (map_type, zip_path) in saved.items():
            self._file(map_name, map_type, zip_path, True, scene)
        with self.lock:
            if self.proc is None:
                self.scene_dir = None

    def _file(self, map_name: str, map_type: int, zip_path: str, final: bool, scene: Path | None = None) -> None:
        scene = scene or self.scene_dir
        if scene is None:
            return
        maps_dir = Path(self.args.maps_dir).expanduser() if self.args.maps_dir else maps_dir_for(zip_path, map_type)
        with self.lock:
            written = file_semantic_map(scene, maps_dir, map_name, map_type, zip_path, final)
        if f"{map_name}_topo_graph.json" in written:
            self.log.info(f"semantic map '{map_name}' ({'final' if final else 'at save'}) -> {maps_dir}: "
                          f"{', '.join(written)}")
        else:
            self.log.warn(f"semantic map '{map_name}': {scene} has no topo_graph.json yet")

    def shutdown(self) -> None:
        self._stop()


def main(argv=None) -> None:
    args = _parse_args(argv)
    import rclpy
    from nav_msgs.msg import Odometry
    from qi_nav_msgs.msg import MapSaved, NavigationStatus
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data

    rclpy.init(args=[])  # key:=value in argv are launch arguments, not remap rules
    node = rclpy.create_node("qinav_mapping_bridge")
    bridge = Bridge(node, args)
    node.create_subscription(NavigationStatus, "/ql/nav/task_status", bridge.on_status, 10)
    node.create_subscription(MapSaved, "/ql/nav/map_saved", bridge.on_map_saved, 10)
    node.create_subscription(Odometry, "/lio/odom", bridge.on_odom, qos_profile_sensor_data)
    node.get_logger().info(f"following QiNav mapping (mapper {args.mapper}); semantic maps -> "
                           f"{args.maps_dir or 'QiNav maps dir (~/.ros/nav2d/maps | ~/.ros/nav3d/maps)'}")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        bridge.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
