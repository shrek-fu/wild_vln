#!/usr/bin/env python3

import os, time, math, subprocess
from typing import List, Dict, Optional
import yaml
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import py_trees

try:
    from nav_btree.bt_node_initial_loc import InitialLocalization
    from nav_btree.bt_node_navigation import NavigateToPoseBt
except ImportError:
    from bt_node_initial_loc import InitialLocalization
    from bt_node_navigation import NavigateToPoseBt


def _pulse_player_env() -> dict:
    env = os.environ.copy()
    uid = os.getuid()
    runtime = env.get("XDG_RUNTIME_DIR") or f"/run/user/{uid}"
    if os.path.isdir(runtime):
        env["XDG_RUNTIME_DIR"] = runtime
    pulse_sock = os.path.join(env.get("XDG_RUNTIME_DIR") or runtime, "pulse", "native")
    if os.path.exists(pulse_sock):
        env.setdefault("PULSE_SERVER", f"unix:{pulse_sock}")
    return env


def _play_wav(path: str, logger) -> bool:
    if not path or not os.path.isfile(path):
        logger.error(f"voice file missing: {path!r}")
        return False
    env = _pulse_player_env()
    last_error = "no audio player (paplay/play)"
    for command in (["paplay", path], ["play", "-q", path]):
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
                env=env,
                text=True,
            )
        except FileNotFoundError:
            continue
        except Exception as exc:
            last_error = str(exc)
            continue
        if completed.returncode == 0:
            logger.info(f"voice played with {command[0]}: {path}")
            return True
        err = (completed.stderr or f"exit {completed.returncode}").strip()
        last_error = f"{command[0]}: {err}"
        logger.warn(f"voice player failed ({command[0]}): {err}")
    logger.error(f"voice playback failed: {last_error}")
    return False


class PlaySpeech(py_trees.behaviour.Behaviour):
    """播放一段语音"""
    def __init__(self, name: str, node: Node, speech_file: Dict, status_pub: rclpy.publisher.Publisher):
        super().__init__(name)
        self.node = node
        self.file = speech_file
        self.pub = status_pub

    def update(self):
        voice_id = self.file['voice_id']
        play_delay = self.file['play_delay']
        self.node.get_logger().info(f"voice_id: {voice_id}, play_delay: {play_delay}s")
        time.sleep(play_delay)
        self.publish_status(f"voice_playing:{self.name}")
        self.node.get_logger().info(f"voice playing: {self.file['voice_content']}")
        ok = _play_wav(self.file.get('file_path', ''), self.node.get_logger())
        if ok:
            self.node.get_logger().info(f"Finished playing voice {self.name}")
        else:
            self.node.get_logger().error(f"Finished without audio {self.name}")
        return py_trees.common.Status.SUCCESS

    def publish_status(self, status: str):
        msg = String(data=status)
        self.pub.publish(msg)

class TourBehaviorTree:
    def __init__(self, node: Node, waypoints: List[Dict]):
        self.node = node
        self.waypoints = waypoints
        self.tree: Optional[py_trees.behaviour.Behaviour] = None
        self.status_pub = node.create_publisher(String, "/bt_status", 10)

    def _publish_status(self, status: str):
        msg = String(data=status)
        self.status_pub.publish(msg)

    # ---------- 拼树 ----------
    def _create_waypoint_sequence(self) -> py_trees.composites.Sequence:
        """根据 waypoints 动态拼装导航+语音子树"""
        root = py_trees.composites.Sequence("TourSequence", memory=True)
        for idx, wp in enumerate(self.waypoints):
            if wp.get('status') == 'reached':
                self.node.get_logger().info(f"Skipping waypoint {wp['id']} as completed.")
                continue
            seq = py_trees.composites.Sequence(f"Spot{idx}", memory=True)
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = wp['x']
            pose.pose.position.y = wp['y']
            pose.pose.position.z = wp['z']
            pose.pose.orientation.z = math.sin(wp['yaw'] / 2.0)
            pose.pose.orientation.w = math.cos(wp['yaw'] / 2.0)

            nav = NavigateToPoseBt(wp['id'], pose, self.node, self.status_pub)
            speech = PlaySpeech(f"{wp['id']}-voice", self.node, wp['voice'], self.status_pub)
            seq.add_children([nav, speech])
            root.add_child(seq)
        return root

    def _create_full_tree(self) -> py_trees.behaviour.Behaviour:
        """初始定位 -> waypoint 序列"""
        root = py_trees.composites.Sequence("FullTask", memory=True)

        # 1. 初始定位
        init_loc = InitialLocalization("InitialLocalization",
                                       self.node,
                                       self.status_pub,
                                       tol=0.3,
                                       max_retry=3)
        retry_init = py_trees.decorators.Retry("RetryInit", num_failures=2, child=init_loc)

        # 2. waypoint 序列
        tour_seq = self._create_waypoint_sequence()

        root.add_children([retry_init, tour_seq])
        return root

    # ---------- 外部接口 ----------
    def setup(self):
        self.tree = self._create_full_tree()
        self.tree.setup_with_descendants()

    def tick_until_done(self):
        """一直 tick 直到树返回 SUCCESS / FAILURE"""
        self._publish_status("running")
        try:
            while rclpy.ok():
                self.tree.tick_once()
                if self.tree.status == py_trees.common.Status.SUCCESS:
                    self.node.get_logger().info('全部点位完成！')
                    self._publish_status("finished")
                    break
                if self.tree.status == py_trees.common.Status.FAILURE:
                    self.node.get_logger().error('行为树执行失败！')
                    self._publish_status("failed")
                    break
                rclpy.spin_once(self.node, timeout_sec=0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.tree.shutdown()


def load_waypoints(waypoints_file: str, node: Node) -> Optional[List[Dict]]:
    if not waypoints_file:
        node.get_logger().error('waypoints_file 参数为空！')
        return None
    if not os.path.exists(waypoints_file):
        node.get_logger().error(f'waypoints_file 文件不存在：{waypoints_file}')
        return None
    node.get_logger().info(f'load waypoints_file: {waypoints_file}')
    try:
        with open(waypoints_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)['waypoints']
    except Exception as e:
        node.get_logger().error(f'加载 waypoints_file 出错：{type(e).__name__} - {str(e)}')
        return None


def main():
    rclpy.init()
    node = rclpy.create_node('tour_bt_node')
    node.declare_parameter('waypoints_file', '')
    waypoints_file = node.get_parameter('waypoints_file').value
    waypoints = load_waypoints(waypoints_file, node)
    if not waypoints:
        node.get_logger().error('waypoints 为空，请检查 waypoints_file')
        return

    bt = TourBehaviorTree(node, waypoints)
    bt.setup()
    bt.tick_until_done()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
