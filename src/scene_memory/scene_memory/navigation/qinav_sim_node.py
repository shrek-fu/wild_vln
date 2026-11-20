"""QiNav nav_server stand-in for offline VLN runs: QiNav's ROS interface on an exported scene, kinematic robot.

    python -m scene_memory.navigation.qinav_sim_node --scene_dir data/sessions/<s>/scene4d/latest

* ``/slam/navigation/execute`` (``qi_nav_msgs/srv/NavigationExecute``): api 1001 with waypoints is accepted when
  every waypoint is reachable; the robot plans A* on the scene's ``grid_nav.NavGrid`` (like QiNav's global
  planner on its map) from waypoint to waypoint and drives the path at ``--speed`` with ``--yaw_rate`` turns,
  finally turning to the last waypoint's yaw. Other api ids are refused.
* ``/ql/nav/task_status`` (``NavigationStatus``): idle / running / completed / error, waypoint progress,
  ``localization_ready`` true.
* ``/ql/nav/localization`` (``nav_msgs/Odometry``) in the scene frame, the goal server's robot pose; TF
  <frame> -> base_link on the floor (sensor height below it) for the Go2-W model (go2w_description).
* ``/qinav_sim/reset`` (``geometry_msgs/PoseStamped``): teleport (x, y, yaw), cancel the task.
* ``/qinav_sim/result`` (``std_msgs/String``): JSON per finished task: driven path, length, minimum clearance.

``--time_scale`` runs the robot faster than real time (the status / odometry rates stay in wall time).
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from scene_memory.navigation.grid_nav import NavGrid, path_length


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene_dir", required=True)
    p.add_argument("--frame_id", default="", help="default: the scene graph's frame_id")
    p.add_argument("--start", default="robot", help="robot (end of the recording) | place<N> | x,y[,yaw]")
    p.add_argument("--speed", type=float, default=0.6, help="m/s")
    p.add_argument("--yaw_rate", type=float, default=1.0, help="rad/s")
    p.add_argument("--time_scale", type=float, default=4.0)
    p.add_argument("--rate", type=float, default=20.0, help="Hz of simulation and odometry")
    p.add_argument("--robot_radius", type=float, default=0.3)
    p.add_argument("--reach_m", type=float, default=3.0, help="mapped area: this far from the driven trajectory")
    p.add_argument("--snap_m", type=float, default=1.5, help="waypoints this far into obstacles are moved out")
    p.add_argument("--service", default="/slam/navigation/execute")
    p.add_argument("--status_topic", default="/ql/nav/task_status")
    p.add_argument("--odom_topic", default="/ql/nav/localization")
    return p.parse_args(argv)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def start_pose(spec: str, graph: dict) -> tuple[float, float, float]:
    agents = graph["layers"]["agents"]
    if spec == "robot":
        a = agents[-1]
        x, y, z, w = a["orientation"]
        return a["position"][0], a["position"][1], math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    if spec.startswith("place"):
        places = graph["layers"]["places"]
        p = places[int(spec.removeprefix("place"))] if isinstance(places, list) else places[spec]
        return p["position"][0], p["position"][1], 0.0
    v = [float(s) for s in spec.split(",")]
    return v[0], v[1], v[2] if len(v) > 2 else 0.0


class Robot:
    """Pose + the current task: a polyline with waypoint boundaries, driven turn-then-move."""

    def __init__(self, x: float, y: float, yaw: float):
        self.x, self.y, self.yaw = x, y, yaw
        self.task = None

    def start(self, task_id: str, path: np.ndarray, wp_ends: list[int], final_yaw: float, ids: list[str]) -> None:
        self.task = {"id": task_id, "path": path, "ends": wp_ends, "i": 1, "final_yaw": final_yaw, "ids": ids,
                     "reached": -1, "driven": [[self.x, self.y]], "t": 0.0, "status": "running"}

    def step(self, dt: float, speed: float, yaw_rate: float) -> None:
        t = self.task
        if t is None or t["status"] != "running":
            return
        t["t"] += dt
        path, i = t["path"], t["i"]
        if i >= len(path):
            err = _wrap(t["final_yaw"] - self.yaw)
            turn = max(-yaw_rate * dt, min(yaw_rate * dt, err))
            self.yaw = _wrap(self.yaw + turn)
            if abs(err) < 0.05:
                t["status"], t["reached"] = "completed", len(t["ids"]) - 1
            return
        dx, dy = path[i][0] - self.x, path[i][1] - self.y
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            err = _wrap(math.atan2(dy, dx) - self.yaw)
            if abs(err) > 0.1:  # turn in place first
                self.yaw = _wrap(self.yaw + max(-yaw_rate * dt, min(yaw_rate * dt, err)))
                return
            self.yaw = _wrap(self.yaw + err)
        move = speed * dt
        if move >= dist:
            self.x, self.y = float(path[i][0]), float(path[i][1])
            while t["reached"] + 1 < len(t["ends"]) and t["ends"][t["reached"] + 1] <= i:
                t["reached"] += 1
            t["i"] += 1
        else:
            self.x += dx / dist * move
            self.y += dy / dist * move
        t["driven"].append([self.x, self.y])


def main(argv=None) -> None:
    import rclpy
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import Odometry
    from qi_nav_msgs.msg import NavigationStatus
    from qi_nav_msgs.srv import NavigationExecute
    from rclpy.executors import ExternalShutdownException
    from std_msgs.msg import String
    from tf2_ros import TransformBroadcaster

    args = _parse_args(argv)
    from pathlib import Path

    scene_dir = Path(args.scene_dir).resolve()
    graph = json.loads((scene_dir / "topo_graph.json").read_text(encoding="utf-8"))
    frame = args.frame_id or graph.get("frame_id", "map")
    t0 = time.monotonic()
    grid = NavGrid.from_scene(scene_dir, robot_radius=args.robot_radius, reach_m=args.reach_m)
    robot = Robot(*start_pose(args.start, graph))

    rclpy.init()
    node = rclpy.create_node("qinav_sim")
    log = node.get_logger()
    log.info(f"qinav_sim: scene {scene_dir} frame {frame}, grid {grid.shape} at {grid.res} m, sensor height "
             f"{grid.sensor_height:.2f} m, {int(grid.free.sum())} free cells ({time.monotonic() - t0:.1f} s); "
             f"start ({robot.x:.2f}, {robot.y:.2f}, {math.degrees(robot.yaw):.0f} deg)")
    pub_status = node.create_publisher(NavigationStatus, args.status_topic, 10)
    pub_odom = node.create_publisher(Odometry, args.odom_topic, 10)
    pub_result = node.create_publisher(String, "/qinav_sim/result", 10)
    tf_pub = TransformBroadcaster(node)
    session = f"qinav_sim_{int(time.time())}"
    state = {"error": ("", ""), "published": None}

    def status_msg() -> NavigationStatus:
        m = NavigationStatus()
        t = robot.task
        m.status = "idle" if t is None else t["status"]
        m.phase = "navigating" if t is not None and t["status"] == "running" else ""
        m.error_code, m.error_message = state["error"]
        m.service_session = session
        m.device_connect = m.localization_ready = m.navigation_ready = m.robot_in_control = True
        if t is not None:
            m.current_waypoint_index = t["reached"]
            m.current_waypoint_id = t["ids"][t["reached"]] if t["reached"] >= 0 else ""
            m.total_waypoints = len(t["ids"])
        else:
            m.current_waypoint_index = -1
        return m

    def finish(t: dict) -> None:
        driven = np.array(t["driven"])
        clear = [grid.clearance_at(p) for p in driven[:: max(len(driven) // 400, 1)]]
        body = {"task_id": t["id"], "status": t["status"], "waypoint_ids": t["ids"], "error": state["error"][1],
                "planned_length_m": round(path_length(t["path"]), 2), "driven_length_m": round(path_length(driven), 2),
                "min_clearance_m": round(float(min(clear)), 2) if clear else None, "sim_s": round(t["t"], 1),
                "final": [round(robot.x, 3), round(robot.y, 3), round(robot.yaw, 4)],
                "path": np.round(driven[:: max(len(driven) // 300, 1)], 3).tolist() + [[robot.x, robot.y]]}
        pub_result.publish(String(data=json.dumps(body)))
        log.info(f"task {t['id']} {t['status']}: driven {body['driven_length_m']} m in {body['sim_s']} s (sim), "
                 f"min clearance {body['min_clearance_m']} m, final ({robot.x:.2f}, {robot.y:.2f})")

    def on_execute(req, resp):
        r = req.request
        if r.api_id != 1001 or not r.waypoints:
            resp.response.success, resp.response.message = False, f"qinav_sim: api {r.api_id} not supported"
            return resp
        legs, ends, pos = [np.array([[robot.x, robot.y]])], [], np.array([robot.x, robot.y])
        for w in r.waypoints:
            path = grid.plan(pos, (w.x, w.y), args.snap_m)
            if path is None:
                state["error"] = ("NO_PATH", f"waypoint {w.waypoint_id} ({w.x:.2f}, {w.y:.2f}) unreachable on the map")
                if robot.task is not None:
                    robot.task["status"] = "error"
                resp.response.success, resp.response.message = False, state["error"][1]
                log.warn(state["error"][1])
                return resp
            legs.append(path[1:] if len(path) > 1 else path)
            ends.append(sum(len(leg) for leg in legs) - 1)
            pos = path[-1]
        if robot.task is not None and robot.task["status"] == "running":
            robot.task["status"] = "canceled"
            finish(robot.task)
        state["error"] = ("", "")
        robot.start(r.task_id or str(r.request_id), np.concatenate(legs), ends, r.waypoints[-1].yaw,
                    [w.waypoint_id for w in r.waypoints])
        resp.response.success = True
        resp.response.message = f"accepted {len(r.waypoints)} waypoint(s), planned {path_length(robot.task['path']):.1f} m"
        log.info(f"execute {robot.task['id']}: {resp.response.message}")
        return resp

    def on_reset(msg: PoseStamped) -> None:
        q = msg.pose.orientation
        if robot.task is not None and robot.task["status"] == "running":
            robot.task["status"] = "canceled"
            finish(robot.task)
        robot.x, robot.y = msg.pose.position.x, msg.pose.position.y
        robot.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        robot.task, state["error"] = None, ("", "")
        log.info(f"reset to ({robot.x:.2f}, {robot.y:.2f}, {math.degrees(robot.yaw):.0f} deg)")

    def tick() -> None:
        before = robot.task["status"] if robot.task else None
        robot.step(args.time_scale / args.rate, args.speed, args.yaw_rate)
        if robot.task is not None and before == "running" and robot.task["status"] != "running":
            finish(robot.task)
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = node.get_clock().now().to_msg(), frame, "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y = robot.x, robot.y
        odom.pose.pose.position.z = grid.pose_z((robot.x, robot.y))
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(robot.yaw / 2), math.cos(robot.yaw / 2)
        pub_odom.publish(odom)
        tf = TransformStamped(header=odom.header, child_frame_id="base_link")
        tf.transform.translation.x, tf.transform.translation.y = robot.x, robot.y
        tf.transform.translation.z = odom.pose.pose.position.z - grid.sensor_height
        tf.transform.rotation = odom.pose.pose.orientation
        tf_pub.sendTransform(tf)

    node.create_service(NavigationExecute, args.service, on_execute)
    node.create_subscription(PoseStamped, "/qinav_sim/reset", on_reset, 10)
    node.create_timer(1.0 / args.rate, tick)
    node.create_timer(0.2, lambda: pub_status.publish(status_msg()))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
