"""Super-LIO divergence watchdog.

Super-LIO has no failure recovery: once the filter diverges the pose runs off (km/s) and never comes back,
and every object placed afterwards is wrong. This node watches ``/lio/odom``; ``--trips`` consecutive
poses implying more than ``--max_speed`` (or non-finite values) count as a divergence. It then

* publishes ``/lio/reset`` (``std_msgs/Header``, stamp = last good pose) so the mapper can close the
  current map segment, the restarted LIO starting again at the origin;
* stops ``super_lio_node`` (same user and ROS_DOMAIN_ID only); ``lio.launch.py`` respawns it.

``/lio/health`` (JSON string, 1 Hz) reports the state for Foxglove. No restart within ``--cooldown_s``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

LIO_PROCESS = "super_lio_node"


class DivergenceDetector:
    """ROS-free: feed ``(t, xyz)`` poses, ``update`` returns True when the LIO is judged diverged."""

    def __init__(self, max_speed: float = 5.0, trips: int = 3):
        self.max_speed = float(max_speed)
        self.trips = int(trips)
        self.bad = 0
        self.last: tuple[float, tuple[float, float, float]] | None = None
        self.last_good_t: float | None = None
        self.speed = 0.0

    def reset(self) -> None:
        self.bad = 0
        self.last = None
        self.speed = 0.0

    def update(self, t: float, xyz: tuple[float, float, float]) -> bool:
        finite = all(math.isfinite(v) for v in xyz)
        if finite and self.last is not None and t > self.last[0]:
            self.speed = math.dist(xyz, self.last[1]) / (t - self.last[0])
        bad = not finite or self.speed > self.max_speed
        if finite:
            self.last = (t, xyz)
        if bad:
            self.bad += 1
        else:
            self.bad = 0
            self.last_good_t = t
        return self.bad >= self.trips


def lio_pids(name: str = LIO_PROCESS) -> list[int]:
    """``name`` processes of this user in this ROS domain."""
    domain = os.environ.get("ROS_DOMAIN_ID", "")
    pids = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if (proc / "comm").read_text().strip() != name[:15] or proc.stat().st_uid != os.getuid():
                continue
            env = dict(item.split("=", 1) for item in (proc / "environ").read_bytes().decode(errors="replace")
                       .split("\0") if "=" in item)
        except OSError:
            continue
        if env.get("ROS_DOMAIN_ID", "") == domain:
            pids.append(int(proc.name))
    return pids


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--odom_topic", default="/lio/odom")
    p.add_argument("--reset_topic", default="/lio/reset")
    p.add_argument("--health_topic", default="/lio/health")
    p.add_argument("--max_speed", type=float, default=5.0, help="m/s between consecutive poses")
    p.add_argument("--trips", type=int, default=3, help="consecutive bad poses before acting")
    p.add_argument("--cooldown_s", type=float, default=15.0)
    p.add_argument("--stall_s", type=float, default=5.0, help="warn when no odometry arrives for this long")
    p.add_argument("--no_restart", action="store_true", help="only report, do not stop the LIO")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.executors import ExternalShutdownException
    from std_msgs.msg import Header, String

    args = _parse_args(argv)
    rclpy.init()
    node = rclpy.create_node("lio_watchdog")
    log = node.get_logger()
    detector = DivergenceDetector(args.max_speed, args.trips)
    pub_reset = node.create_publisher(Header, args.reset_topic, 10)
    pub_health = node.create_publisher(String, args.health_topic, 10)
    state = {"restarts": 0, "last_restart": -1e9, "last_odom": None, "status": "waiting", "pending_kill": []}

    def stop_lio() -> None:
        pids = lio_pids()
        for pid in pids:
            try:
                os.kill(pid, signal.SIGINT)
            except OSError:
                pass
        state["pending_kill"] = pids
        log.error(f"LIO diverged (speed {detector.speed:.1f} m/s); stopping {LIO_PROCESS} {pids}")

    def on_odom(msg) -> None:
        now = time.monotonic()
        state["last_odom"] = now
        if now - state["last_restart"] < args.cooldown_s:
            state["status"] = "restarting"
            detector.reset()
            return
        p = msg.pose.pose.position
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if not detector.update(t, (p.x, p.y, p.z)):
            state["status"] = "ok"
            return
        state["status"] = "diverged"
        reset = Header(frame_id="odom")
        good = detector.last_good_t or t
        reset.stamp.sec, reset.stamp.nanosec = int(good), int((good % 1.0) * 1e9)
        pub_reset.publish(reset)
        if not args.no_restart:
            stop_lio()
        state["restarts"] += 1
        state["last_restart"] = now
        detector.reset()

    def tick() -> None:
        # SIGINT first; whatever is left after a second is killed so the launch can respawn it.
        for pid in state["pending_kill"]:
            if Path(f"/proc/{pid}").exists() and time.monotonic() - state["last_restart"] > 1.0:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        if time.monotonic() - state["last_restart"] > 1.0:
            state["pending_kill"] = []
        last = state["last_odom"]
        if last is not None and time.monotonic() - last > args.stall_s and state["status"] == "ok":
            state["status"] = "stalled"
            log.warn(f"no {args.odom_topic} for {args.stall_s:.0f} s")
        pub_health.publish(String(data=json.dumps({"status": state["status"], "speed": round(detector.speed, 2),
                                                   "restarts": state["restarts"]})))

    node.create_subscription(Odometry, args.odom_topic, on_odom, 200)
    node.create_timer(1.0, tick)
    log.info(f"watching {args.odom_topic}: > {args.max_speed} m/s for {args.trips} poses = diverged"
             + (" (report only)" if args.no_restart else f"; restarts {LIO_PROCESS}"))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
