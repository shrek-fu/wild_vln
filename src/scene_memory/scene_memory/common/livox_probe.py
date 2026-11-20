"""Is a MID-360S driver / Super-LIO already running (e.g. QiNav's mapping or localization stack)?

QiNav starts livox_ros_driver2 (/livox/lidar CustomMsg, /livox/imu) and its own super_lio_node (same
/lio/odom, /lio/cloud_world, TF odom -> livox_frame and lidar-IMU extrinsic as ours) when it gets a
mapping / localization command. A second driver fights over the lidar UDP ports and a second Super-LIO
publishes the same topics, so lio.launch.py reuses whatever is already up.

    python -m scene_memory.common.livox_probe          # prints the status, exit 0
"""

from __future__ import annotations

import os
import subprocess
import time

DRIVER_PATTERN = "livox_ros_driver2_node"
LIO_PATTERN = "super_lio_node"
CUSTOM_MSG = "livox_ros_driver2/msg/CustomMsg"
IMU_MSG = "sensor_msgs/msg/Imu"


def processes(pattern: str) -> list[tuple[int, str]]:
    """(pid, cmdline) of running processes whose cmdline contains `pattern`, wrapper shells excluded."""
    try:
        out = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    found = []
    for line in out.splitlines():
        pid, _, cmd = line.partition(" ")
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        exe = cmd.split(" ", 1)[0]
        if exe.endswith(pattern):
            found.append((int(pid), cmd))
    return found


def _topics(timeout: float) -> dict[str, str]:
    try:
        out = subprocess.run(["ros2", "topic", "list", "-t"], capture_output=True, text=True,
                             timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    topics = {}
    for line in out.splitlines():
        name, _, kind = line.partition(" ")
        topics[name] = kind.strip(" []")
    return topics


def livox_topics(wait_s: float = 8.0) -> tuple[str | None, str | None]:
    """(CustomMsg topic, IMU topic) published under /livox, waiting up to `wait_s` for discovery."""
    deadline = time.monotonic() + wait_s
    while True:
        topics = _topics(timeout=5.0)
        lidar = sorted(t for t, k in topics.items() if k == CUSTOM_MSG and t.startswith("/livox"))
        imu = sorted(t for t, k in topics.items() if k == IMU_MSG and t.startswith("/livox"))
        if lidar and imu or time.monotonic() > deadline:
            return (lidar[0] if lidar else None), (imu[0] if imu else None)
        time.sleep(1.0)


def owner(cmd: str) -> str:
    return "QiNav" if "qinav" in cmd.lower() or "QiNav_ws" in cmd else "other"


def main() -> None:
    drivers, lios = processes(DRIVER_PATTERN), processes(LIO_PATTERN)
    for label, procs in (("livox driver", drivers), ("super_lio", lios)):
        if not procs:
            print(f"{label}: not running")
        for pid, cmd in procs:
            print(f"{label}: pid {pid} ({owner(cmd)}) {cmd[:160]}")
    if drivers:
        lidar, imu = livox_topics(wait_s=3.0)
        print(f"livox topics: lidar={lidar} imu={imu}")


if __name__ == "__main__":
    main()
