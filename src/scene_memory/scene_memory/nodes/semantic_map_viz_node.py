"""ROS 2 node: publish a saved semantic map (topo graph: places, rooms, object boxes + labels) for Foxglove.

    python -m scene_memory.nodes.semantic_map_viz_node --map_prefix data/maps/demo

Reads the newer of ``<prefix>_topo_graph.json`` and ``<prefix>_scene/topo_graph.json`` (the scene copy says
``odom``: the mapping LIO frame, which is the frame of ``<prefix>_map.pcd``, i.e. ``map``) and publishes ``/topo/markers`` (transient local, re-sent every
``--period_s``), the same topic and markers ``topo_map_node`` publishes while mapping.
"""

import argparse
import json
from pathlib import Path


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--map_prefix", required=True, help="data/maps/<name> (without _map.pcd)")
    p.add_argument("--markers_topic", default="/topo/markers")
    p.add_argument("--period_s", type=float, default=5.0)
    p.add_argument("--room_dz", type=float, default=3.0)
    return p.parse_args(argv)


def graph_file(prefix: str) -> Path | None:
    files = [f for f in (Path(f"{prefix}_topo_graph.json"), Path(f"{prefix}_scene") / "topo_graph.json")
             if f.is_file()]
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def main(argv=None) -> None:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from visualization_msgs.msg import MarkerArray

    from scene_memory.nodes.topo_map_node import _markers

    args = _parse_args(argv)
    rclpy.init()
    node = rclpy.create_node("semantic_map_viz")
    path = graph_file(args.map_prefix)
    if path is None:
        node.get_logger().warning(f"no semantic map for {args.map_prefix} (_topo_graph.json); nothing to show")
        node.destroy_node()
        rclpy.try_shutdown()
        return
    graph = json.loads(path.read_text())
    graph["frame_id"] = "map"
    layers = graph["layers"]
    node.get_logger().info(
        f"{path}: {len(layers['objects'])} objects, {len(layers['places'])} places, "
        f"{len(layers.get('regions', []))} rooms -> {args.markers_topic} (frame {graph['frame_id']})")
    pub = node.create_publisher(MarkerArray, args.markers_topic,
                                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def publish():
        pub.publish(_markers(graph, node.get_clock().now().to_msg(), room_dz=args.room_dz))

    publish()
    node.create_timer(args.period_s, publish)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
