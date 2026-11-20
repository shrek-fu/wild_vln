"""Live traversable ground by class: Insta360 pinhole views segmented, lifted with the LIO cloud, voted into an
odom grid.

Subscribes the view images (``--segmenter gdino_sam2``: segmented here) or the detector's
``/detector/<view>/ground_labels`` (``--segmenter detector``: the ground classes ride along in the semantic
detector's GroundingDINO + SAM2 pass, no extra model), plus ``/lio/odom`` and ``/lio/cloud_world``.
``--classes`` (same spec as ``detector_node --ground_classes``) names the labels; ``--terrain_classes`` are taken
from the lidar alone (stairs, steps are not on the ground plane). Publishes, all in odom:

* ``/ground/grid`` (``nav_msgs/OccupancyGrid``): 0 traversable, 100 obstacle or ``--blocked_classes``, -1 unknown;
* ``/ground/class_grid`` (``OccupancyGrid``): traversable cells hold their label (class index + 1), 100 obstacle;
* ``/ground/traversable_cloud`` (``PointCloud2`` x y z label): traversable cell centres at their ground height;
* ``/ground/markers`` (``MarkerArray``): per-class region outlines (``outline/<class>``), enclosed holes
  (orange), translucent surface coloured by class and red obstacle cells;
* ``/ground/<view>/overlay/compressed`` (``--overlay``): labels, lidar ground / obstacle returns, ray fill.

See ``ground_segmentation.mapper`` for the per-image steps.
"""

from __future__ import annotations

import argparse
import time
from array import array
from collections import deque

import cv2
import numpy as np

from scene_memory import paths
from scene_memory.common.calib_io import load_rig_calibration
from scene_memory.mapping.scene_graph import pose_frame_to_da360, pose_to_matrix
from scene_memory.perception.boxer_lift import load_pinhole_views
from scene_memory.perception.lidar_depth import cloud_xyz_fast

from ground_segmentation.mapper import TERRAIN_CLASSES, GroundMapper, class_rgb, draw_overlay
from ground_segmentation.segmenters import blocked_classes, class_names, make_segmenter, terrain_classes


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib_dir", default=str(paths.CALIB))
    p.add_argument("--pose_source", choices=["vio", "lio"], default="lio")
    p.add_argument("--vio_config", default=str(paths.VIO_CONFIG))
    p.add_argument("--lio_config", default=str(paths.LIO_CONFIG))
    p.add_argument("--odom_topic", default="/lio/odom")
    p.add_argument("--cloud_topic", default="/lio/cloud_world")
    p.add_argument("--views", default="front,left,rear,right", help="comma list; empty = all calibrated views")
    p.add_argument("--max_hz", type=float, default=1.0, help="per view")
    p.add_argument("--segmenter", choices=["detector", "gdino_sam2"], default="detector",
                   help="detector: use /detector/<view>/ground_labels (detector_node --ground_classes); "
                        "gdino_sam2: own GroundingDINO + SAM2 instance")
    p.add_argument("--classes", default=str(paths.CONFIG / "ground_classes.txt"),
                   help="ground classes file (config/ground_classes.txt) or inline '<class>:<prompt>|<prompt>;...'; "
                        "must match detector_node --ground_classes in detector mode")
    p.add_argument("--terrain_classes", default="",
                   help="comma list of classes taken from the lidar alone (not on the ground plane); default: the "
                        "@terrain classes of --classes, or " + ",".join(TERRAIN_CLASSES) + " if it has no flags")
    p.add_argument("--blocked_classes", default="",
                   help="comma list of classes marked 100 in /ground/grid, added to the @blocked classes of --classes")
    p.add_argument("--max_step", type=float, default=0.18,
                   help="m; highest step the robot climbs (terrain classes; taller jumps are obstacles)")
    p.add_argument("--max_slope_deg", type=float, default=35.0,
                   help="steepest ramp / stair pitch the robot climbs over ~0.9 m; <= 0 disables")
    p.add_argument("--device", default="")
    p.add_argument("--resolution", type=float, default=0.1, help="m per grid cell")
    p.add_argument("--fill_range", type=float, default=6.0, help="m; ray-plane fill of mask pixels without lidar")
    p.add_argument("--max_wait_s", type=float, default=1.0, help="hold an image this long for its odometry")
    p.add_argument("--publish_hz", type=float, default=1.0, help="grid + cloud rate")
    p.add_argument("--window_m", type=float, default=0.0,
                   help="m; keep only the square of this side around the robot (navigation: 30); 0 = whole run")
    p.add_argument("--grid_topic", default="/ground/grid")
    p.add_argument("--class_grid_topic", default="/ground/class_grid")
    p.add_argument("--cloud_out_topic", default="/ground/traversable_cloud")
    p.add_argument("--marker_topic", default="/ground/markers", help="outlines, surface and obstacles for 3D views")
    p.add_argument("--overlay", action="store_true", help="publish /ground/<view>/overlay/compressed")
    return p.parse_args(argv)


def labelled_cloud_msg(xyz: np.ndarray, labels: np.ndarray, header):
    from sensor_msgs.msg import PointCloud2, PointField

    msg = PointCloud2()
    msg.header = header
    msg.height, msg.width = 1, len(xyz)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(("x", "y", "z", "label"))]
    msg.is_bigendian, msg.point_step, msg.is_dense = False, 16, True
    msg.row_step = 16 * len(xyz)
    data = np.column_stack([np.asarray(xyz, np.float32).reshape(-1, 3), np.asarray(labels, np.float32)])
    msg.data = array("B", data.astype(np.float32).tobytes())
    return msg


def ground_markers(grid, classes: list[str], header, ground_z: float | None):
    """MarkerArray for the 3D panel: ``outline/<class>`` region boundaries in the class colour and ``holes``
    (orange, areas enclosed by ground) 3 cm above the ground, ``surface`` translucent cells coloured by class,
    ``obstacles`` red cells at ground height. Every marker is always sent (possibly empty) so stale drawings clear."""
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray

    def marker(ns: str, kind: int, rgba, scale) -> Marker:
        m = Marker()
        m.header = header
        m.ns, m.id, m.type, m.action = ns, 0, kind, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
        m.frame_locked = True
        return m

    def add_ring(target: Marker, xyz: np.ndarray) -> None:
        ring = np.vstack([xyz, xyz[:1]])
        for a, b in zip(ring[:-1], ring[1:]):
            target.points += [Point(x=float(a[0]), y=float(a[1]), z=float(a[2]) + 0.03),
                              Point(x=float(b[0]), y=float(b[1]), z=float(b[2]) + 0.03)]

    res = float(grid.res)
    out = MarkerArray()
    for label, name in enumerate(classes, start=1):
        outline = marker(f"outline/{name}", Marker.LINE_LIST, (*class_rgb(name), 1.0), (0.05, 0.0, 0.0))
        for xyz, is_hole in grid.outlines(label):
            if not is_hole:
                add_ring(outline, xyz)
        out.markers.append(outline)
    holes = marker("holes", Marker.LINE_LIST, (1.0, 0.45, 0.0, 1.0), (0.04, 0.0, 0.0))
    for xyz, is_hole in grid.outlines():
        if is_hole:
            add_ring(holes, xyz)
    surface = marker("surface", Marker.CUBE_LIST, (1.0, 1.0, 1.0, 0.35), (res, res, 0.02))
    xyz, labels = grid.traversable(with_class=True)
    palette = [ColorRGBA(r=0.2, g=0.85, b=0.3, a=0.35)] + [ColorRGBA(r=r, g=g, b=b, a=0.35)
                                                           for r, g, b in (class_rgb(c) for c in classes)]
    surface.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in xyz]
    surface.colors = [palette[int(lab)] for lab in labels]
    obstacles = marker("obstacles", Marker.CUBE_LIST, (0.9, 0.15, 0.15, 0.6), (res, res, 0.1))
    z0 = 0.05 + (ground_z if ground_z is not None else 0.0)
    obstacles.points = [Point(x=float(x), y=float(y), z=z0) for x, y in grid.obstacles()]
    out.markers += [holes, surface, obstacles]
    return out


def main(argv=None) -> None:
    import rclpy
    from nav_msgs.msg import OccupancyGrid, Odometry
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, PointCloud2
    from std_msgs.msg import Header
    from visualization_msgs.msg import MarkerArray

    args = _parse_args(argv)
    calib = load_rig_calibration(args.calib_dir)
    T_child_da360 = pose_frame_to_da360(args.pose_source, args.vio_config, args.lio_config, calib)
    wanted = {v.strip() for v in args.views.split(",") if v.strip()}
    views = [v for v in load_pinhole_views(calib.source_dir) if not wanted or v["name"] in wanted]
    from_detector = args.segmenter == "detector"
    segmenter = None if from_detector else make_segmenter(args.segmenter, args.classes, device=args.device)
    classes = class_names(args.classes)
    terrain = terrain_classes(args.classes, args.terrain_classes)
    blocked = tuple(classes.index(c) + 1 for c in blocked_classes(args.classes, args.blocked_classes) if c in classes)
    mapper = GroundMapper(segmenter, T_child_da360, classes=classes, terrain_classes=terrain,
                          max_step=args.max_step, max_slope_deg=args.max_slope_deg,
                          fill_range=args.fill_range, resolution=args.resolution)

    rclpy.init()
    node = rclpy.create_node("ground_segmentation")
    log = node.get_logger()
    grid_pub = node.create_publisher(OccupancyGrid, args.grid_topic, 1)
    class_grid_pub = node.create_publisher(OccupancyGrid, args.class_grid_topic, 1)
    cloud_pub = node.create_publisher(PointCloud2, args.cloud_out_topic, 1)
    marker_pub = node.create_publisher(MarkerArray, args.marker_topic, 1)
    overlay_pubs = {v["name"]: node.create_publisher(CompressedImage, f"/ground/{v['name']}/overlay/compressed", 1)
                    for v in views} if args.overlay else {}
    last = {v["name"]: 0.0 for v in views}
    pending: dict[str, tuple[dict, object, float]] = {}
    # detector mode + overlay: the labels arrive after inference, so keep recent images to draw on.
    recent: dict[str, deque] = {v["name"]: deque(maxlen=30) for v in views}
    frame = {"odom": "odom"}
    timing = {"seg_ms": 0.0, "lift_ms": 0.0}
    warned = {"classes": False}

    def on_odom(msg) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        frame["odom"] = msg.header.frame_id or frame["odom"]
        mapper.on_odom(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec,
                       pose_to_matrix((p.x, p.y, p.z), (q.x, q.y, q.z, q.w)))
        for name, (view, img, t) in list(pending.items()):
            if mapper.poses.last_t() >= t:
                del pending[name]
                handle(view, img, t)

    def on_cloud(msg) -> None:
        try:
            mapper.on_cloud(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec, cloud_xyz_fast(msg))
        except ValueError as exc:
            log.warn(f"cloud decode failed: {exc}", throttle_duration_sec=10.0)

    def on_view(view: dict, msg) -> None:
        name = view["name"]
        if args.max_hz > 0 and time.monotonic() - last[name] < 1.0 / args.max_hz:
            return
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if mapper.poses.last_t() < t < mapper.poses.last_t() + args.max_wait_s:
            pending[name] = (view, msg, t)
            return
        handle(view, msg, t)

    def handle(view: dict, msg, t: float) -> None:
        name = view["name"]
        if from_detector:
            sent = msg.format.partition("classes=")[2]
            if sent and sent.split(",") != classes and not warned["classes"]:
                warned["classes"] = True
                log.error(f"detector ground classes {sent.split(',')} != --classes {classes}; labels will be wrong")
            labels = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if labels is None:
                return
            image = next((m for s, m in reversed(recent[name]) if s == t), None)
            bgr = None if image is None else cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None and bgr.shape[:2] != labels.shape:
                bgr = None
        else:
            labels = None
            bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                return
        last[name] = time.monotonic()
        res = mapper.process(view, bgr, t, labels=labels)
        if res is None:
            return
        timing["seg_ms"] = 1e3 * res["seg_s"]
        timing["lift_ms"] = 1e3 * res.get("lift_s", 0.0)
        pub = overlay_pubs.get(name)
        if pub is not None and bgr is not None:
            ok, jpg = cv2.imencode(".jpg", draw_overlay(bgr, res), [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                out = CompressedImage()
                out.header = msg.header
                out.format = "jpeg"
                out.data = array("B", jpg.tobytes())
                pub.publish(out)

    def grid_msg(raster, header, ground_z):
        (ox, oy), w, h, data = raster
        grid = OccupancyGrid()
        grid.header = header
        grid.info.map_load_time = header.stamp
        grid.info.resolution = float(mapper.grid.res)
        grid.info.width, grid.info.height = w, h
        grid.info.origin.position.x, grid.info.origin.position.y = ox, oy
        grid.info.origin.position.z = ground_z if ground_z is not None else 0.0
        grid.info.origin.orientation.w = 1.0
        grid.data = array("b", data.tobytes())
        return grid

    def publish() -> None:
        if args.window_m > 0 and (pose := mapper.poses.at(mapper.poses.last_t())) is not None:
            mapper.grid.crop(pose[:2, 3], 0.5 * args.window_m)
        header = Header()
        header.stamp = node.get_clock().now().to_msg()
        header.frame_id = frame["odom"]
        ground_z = None if mapper.plane is None else float(-mapper.plane[1] / mapper.plane[0][2])
        if len(mapper.grid):
            grid_pub.publish(grid_msg(mapper.grid.occupancy(blocked), header, ground_z))
            class_grid_pub.publish(grid_msg(mapper.grid.class_occupancy(), header, ground_z))
        cloud_pub.publish(labelled_cloud_msg(*mapper.grid.traversable(with_class=True), header))
        marker_pub.publish(ground_markers(mapper.grid, classes, header, ground_z))

    node.create_subscription(Odometry, args.odom_topic, on_odom, 200)
    node.create_subscription(PointCloud2, args.cloud_topic, on_cloud, qos_profile_sensor_data)
    for view in views:
        if from_detector:
            node.create_subscription(CompressedImage, f"/detector/{view['name']}/ground_labels",
                                     lambda m, v=view: on_view(v, m), 5)
            if args.overlay:
                node.create_subscription(
                    CompressedImage, view["image_topic"],
                    lambda m, n=view["name"]: recent[n].append((m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, m)),
                    qos_profile_sensor_data)
        else:
            node.create_subscription(CompressedImage, view["image_topic"], lambda m, v=view: on_view(v, m),
                                     qos_profile_sensor_data)
    node.create_timer(1.0 / max(args.publish_hz, 0.1), publish)

    def report() -> None:
        _, cls = mapper.grid.traversable(with_class=True)
        counts = {name: int((cls == i).sum()) for i, name in enumerate(classes, start=1)}
        log.info(f"ground {mapper.stats} cells={len(mapper.grid)} traversable={counts} "
                 f"seg={timing['seg_ms']:.0f}ms lift={timing['lift_ms']:.0f}ms")

    node.create_timer(10.0, report)
    log.info(f"ground segmentation ({args.segmenter}) classes {classes} (terrain {terrain}, blocked "
             f"{[classes[i - 1] for i in blocked]}, climbs step <= {args.max_step} m, slope <= "
             f"{args.max_slope_deg} deg) for {[v['name'] for v in views]} at {args.max_hz} Hz/view: "
             f"{args.odom_topic} + {args.cloud_topic} -> {args.grid_topic}, {args.class_grid_topic}, {args.marker_topic}"
             + (f", {args.window_m:g} m window around the robot" if args.window_m > 0 else ""))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
