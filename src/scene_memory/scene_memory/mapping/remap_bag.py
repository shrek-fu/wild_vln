"""Rebuild the topo map offline from recorded detections + odometry (CPU only, seconds instead of a replay).

Record the inputs during any run::

    ros2 bag record -o /tmp/det_bag /detector/detections_json /equirect_vio/odometry
    # LIO odometry + lidar depth:
    ros2 bag record -o /tmp/det_bag /detector/detections_json /lio/odom /lio/cloud_world

then map with any ``topo_map_node`` options (``--odom_topic`` / ``--cloud_topic`` select the bag topics)::

    python3 -m scene_memory.mapping.remap_bag /tmp/det_bag --scene_dir /tmp/remap --miss_range 4
    python3 -m scene_memory.mapping.remap_bag /tmp/det_bag --pose_source lio --odom_topic /lio/odom \
        --cloud_topic /lio/cloud_world

Key-view crops need the images, so the result has no ``views/`` (SigLIP falls back to labels).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from scene_memory.mapping.scene_graph import pose_to_matrix
from scene_memory.mapping.topo_graph import save_graph
from scene_memory.nodes.topo_map_node import TopoMapper, _parse_args
from scene_memory.perception.lidar_depth import cloud_xyz_fast


def read_bag(bag_dir: Path, topics: list[str], raw: frozenset[str] = frozenset()):
    """``(topic, message)``; topics in ``raw`` yield the serialized bytes instead."""
    import rosbag2_py
    import yaml
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    meta = yaml.safe_load((bag_dir / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=meta["storage_identifier"]),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    classes = {t: None if t in raw else get_message(types[t]) for t in topics if t in types}
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(classes)))
    while reader.has_next():
        topic, data, _ = reader.read_next()
        yield topic, data if classes[topic] is None else deserialize_message(data, classes[topic])


def remap(bag_dir: Path, args: argparse.Namespace, consolidate_s: float = 5.0, core=None) -> TopoMapper:
    """``core``: any mapper with the ``TopoMapper`` replay interface (default: a new ``TopoMapper(args)``)."""
    core = core or TopoMapper(args)
    # Odometry and detections reach the live node through separate queues; detections lag their image by
    # the detector latency, so feed each detection only after odometry has passed its stamp.
    pending: list[tuple[float, dict]] = []
    last_consolidate = None
    topics = [args.detections_topic, args.odom_topic] + ([args.cloud_topic] if args.cloud_topic else [])
    # A core that keeps its map across LIO restarts (hydra_fake) is told about them; TopoMapper ignores them.
    resets = bool(getattr(core, "handles_reset", False) and args.reset_topic)
    if resets:
        topics.append(args.reset_topic)
    for topic, msg in read_bag(bag_dir, topics):
        if resets and topic == args.reset_topic:
            core.on_reset()
        elif topic == args.odom_topic:
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
            core.on_odom(t, pose_to_matrix((p.x, p.y, p.z), (q.x, q.y, q.z, q.w)), (q.x, q.y, q.z, q.w))
        elif topic == args.cloud_topic:
            core.on_cloud(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec, cloud_xyz_fast(msg))
        else:
            body = json.loads(msg.data)
            pending.append((body["stamp"]["sec"] + 1e-9 * body["stamp"]["nanosec"], body))
        while pending and pending[0][0] <= core.last_odom_t():
            stamp, body = pending.pop(0)
            core.on_detections(body)
            if last_consolidate is None or stamp - last_consolidate >= consolidate_s:
                core.consolidate()
                last_consolidate = stamp
    for _, body in pending:
        core.on_detections(body)
    core.consolidate()
    return core


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     add_help=False)
    parser.add_argument("bag", type=Path)
    known, rest = parser.parse_known_args(argv)
    args = _parse_args(rest)
    scene_dir = Path(args.scene_dir or f"/tmp/remap_{time.strftime('%Y%m%d_%H%M%S')}")
    t0 = time.perf_counter()
    core = remap(known.bag, args)
    graph = core.graph()
    core.objects.save(scene_dir / "objects_4d.json", extra={"saved_at": time.time(), "mapper": "topo-remap",
                                                             "stats": dict(core.stats)})
    save_graph(graph, scene_dir / "topo_graph.json")
    objs = core.objects.to_json()["objects"]
    layers = graph["layers"]
    print(f"{scene_dir}: {time.perf_counter() - t0:.1f} s | objects={len(objs)} "
          f"status={dict(Counter('dynamic' if o['dynamic'] else o['status'] for o in objs))} "
          f"graph objects={len(layers['objects'])} places={len(layers['places'])} regions={len(layers['regions'])}")
    print("stats", core.stats)


if __name__ == "__main__":
    main()
