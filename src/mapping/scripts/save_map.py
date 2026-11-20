#!/usr/bin/env python3
"""Save the running STVLN mapping (src/mapping/launch/map.launch.py) like QiNav's save_map3d.sh, into data/maps/<name>/.

    src/scripts/stvln.sh map_save <name> [--semantic_only]

* /save_pcd (dlio_multi_map_node) -> <name>_map.pcd, _ground.pcd, _nonground.pcd, _nonground_raw.pcd, _poses.pcd
* planning ground (--ground auto): ground_segmentation's cells when its node runs (map ... ground:=semantic) ->
  <name>_ground_sem.pcd + _nonground_sem.pcd (semantic_ground.py), else linefit's _ground / _nonground;
  perception_3d export_traversable_pcd on them -> <name>_traversable.pcd (global planner ground)
* semantic files from data/maps/<name>/<name>_scene (pipeline export, every few seconds and on stop) ->
  <name>_topo_graph.json, _objects_4d.json, _dsg.json, ... with frame_id "map"
  --semantic_only: just these, e.g. again after `stvln.sh stop map` for the final export
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from scene_memory import paths
from scene_memory.nodes.qinav_mapping_bridge_node import file_semantic_map

WS_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WS_ROOT / "src" / "planning" / "launch"))
from stack_common import MAPS_DIR  # noqa: E402

PCD_FILES = {"map.pcd": "_map.pcd", "ground_map.pcd": "_ground.pcd", "non_ground_map_raw.pcd": "_nonground_raw.pcd",
             "non_ground_map.pcd": "_nonground.pcd", "odom_poses.pcd": "_poses.pcd"}
SOURCE_FRAME = "odom of STVLN's Super-LIO during stvln.sh map, the frame dlio_multi_map_node saved the map in"


def save_pcd(prefix: Path, leaf: float, timeout: float) -> None:
    import rclpy
    from direct_lidar_inertial_odometry.srv import SavePCD

    rclpy.init(args=[])
    node = rclpy.create_node("stvln_map_save")
    try:
        client = node.create_client(SavePCD, "/save_pcd")
        if not client.wait_for_service(timeout_sec=10.0):
            raise SystemExit("[map_save] /save_pcd not available: is `stvln.sh map` running?")
        with tempfile.TemporaryDirectory(dir=prefix.parent, prefix=".save_") as tmp:
            t0 = time.monotonic()
            future = client.call_async(SavePCD.Request(leaf_size=leaf, save_path=tmp))
            rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
            if not future.done() or not future.result().success:
                raise SystemExit(f"[map_save] /save_pcd failed ({'timeout' if not future.done() else 'success=false'})")
            for src, suffix in PCD_FILES.items():
                if (Path(tmp) / src).is_file():
                    shutil.move(str(Path(tmp) / src), f"{prefix}{suffix}")
                else:
                    print(f"[map_save] warning: /save_pcd wrote no {src}")
            print(f"[map_save] PCD maps saved in {time.monotonic() - t0:.1f} s")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def semantic_planning_ground(prefix: Path, classes: str, blocked: str, tol: float) -> tuple[str, str] | None:
    """<prefix>_ground_sem.pcd (ground_segmentation cells, intensity = class label) and
    <prefix>_nonground_sem.pcd (map minus points on that ground); None when the ground node is not running."""
    from semantic_ground import blocked_labels, grab_ground_cloud, near_ground, read_pcd_xyz, write_pcd_xyzi

    got = grab_ground_cloud()
    if got is None:
        return None
    xyz, labels, _ = got
    keep = ~np.isin(labels, blocked_labels(classes, blocked))
    xyz, labels = xyz[keep], labels[keep]
    if len(xyz) < 50:
        print(f"[map_save] ground_segmentation has only {len(xyz)} traversable cells: using linefit's ground")
        return None
    cloud = read_pcd_xyz(f"{prefix}_map.pcd")
    obstacle = cloud[~near_ground(cloud, xyz, tol=tol)]
    ground, nonground = f"{prefix}_ground_sem.pcd", f"{prefix}_nonground_sem.pcd"
    write_pcd_xyzi(ground, xyz, labels)
    write_pcd_xyzi(nonground, obstacle)
    print(f"[map_save] planning ground from ground_segmentation: {len(xyz)} cells "
          f"({int((~keep).sum())} blocked dropped), non-ground {len(obstacle)} of {len(cloud)} map points")
    return ground, nonground


def export_traversable(prefix: Path, ground: str, nonground: str, leaf: float, obstacle_half: float) -> None:
    # QiNav exports with ground leaf 0.3 and a +-0.5 m obstacle box: in offices most of the floor then turns
    # lethal and the free graph falls apart (demo map: 64 % lethal, start / goal unreachable).
    subprocess.run(["ros2", "run", "perception_3d", "export_traversable_pcd", ground, nonground,
                    f"{prefix}_traversable.pcd", str(leaf), "0.2", "1.5", str(obstacle_half)], check=True)


def planning_sources(prefix: Path, ground: str) -> tuple[str, str]:
    sem = (f"{prefix}_ground_sem.pcd", f"{prefix}_nonground_sem.pcd")
    if ground != "linefit" and all(Path(p).is_file() for p in sem):
        return sem
    if ground == "semantic":
        raise SystemExit(f"[map_save] --ground semantic: no {sem[0]} / {sem[1]}")
    return f"{prefix}_ground.pcd", f"{prefix}_nonground.pcd"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("name")
    p.add_argument("--maps_dir", type=Path, default=MAPS_DIR, help="the map goes to <maps_dir>/<name>/")
    p.add_argument("--leaf", type=float, default=0.1, help="PCD voxel size (QiNav: 0.1)")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--semantic_only", action="store_true")
    p.add_argument("--ground", choices=["auto", "semantic", "linefit"], default="auto",
                   help="planning ground: auto = ground_segmentation's /ground/traversable_cloud when it runs "
                        "(map.launch.py ground:=semantic), else linefit's (DLIO ground map)")
    p.add_argument("--ground_classes", default=str(paths.CONFIG / "ground_classes.txt"),
                   help="same file as the ground node; its @blocked classes are not planning ground")
    p.add_argument("--blocked_classes", default="", help="comma list, on top of the file's @blocked")
    p.add_argument("--ground_tol", type=float, default=0.12,
                   help="m; map points this close to a ground cell's height are ground, not obstacles")
    p.add_argument("--trav_leaf", type=float, default=0.2, help="m; planner graph vertex spacing (QiNav: 0.3)")
    p.add_argument("--obstacle_half", type=float, default=0.3,
                   help="m; a vertex is lethal with obstacles 0.1-1 m above it within +-this (QiNav: 0.5)")
    p.add_argument("--traversable_only", action="store_true",
                   help="only re-export <name>_traversable.pcd from the saved ground / non-ground files")
    args = p.parse_args()

    folder = args.maps_dir / args.name
    folder.mkdir(parents=True, exist_ok=True)
    prefix = folder / args.name
    if args.traversable_only:
        src = planning_sources(prefix, args.ground)
        print(f"[map_save] re-exporting {prefix}_traversable.pcd from {Path(src[0]).name} + {Path(src[1]).name}")
        export_traversable(prefix, *src, args.trav_leaf, args.obstacle_half)
        return
    if not args.semantic_only:
        save_pcd(prefix, args.leaf, args.timeout)
        sem = None if args.ground == "linefit" else \
            semantic_planning_ground(prefix, args.ground_classes, args.blocked_classes, args.ground_tol)
        if sem is None and args.ground == "semantic":
            raise SystemExit("[map_save] --ground semantic: no /ground/traversable_cloud (map ... ground:=semantic?)")
        if sem is None:
            print("[map_save] planning ground from linefit (DLIO ground map)")
        export_traversable(prefix, *(sem or (f"{prefix}_ground.pcd", f"{prefix}_nonground.pcd")),
                           args.trav_leaf, args.obstacle_half)
    scene_dir = folder / f"{args.name}_scene"
    if scene_dir.is_dir():
        written = file_semantic_map(scene_dir, folder, args.name, 2, "", final=args.semantic_only,
                                    source_frame=SOURCE_FRAME)
        print(f"[map_save] semantic: {', '.join(written) or 'nothing exported yet'}")
    else:
        print(f"[map_save] no {scene_dir} (semantic:=false?): PCD maps only")
    for f in sorted(folder.glob(f"{args.name}_*")):
        if f.is_file():
            print(f"  {f}  {f.stat().st_size / 1e6:.1f} MB")
    print(f"[map_save] navigate on it: src/scripts/stvln.sh nav {args.name} --bg")


if __name__ == "__main__":
    main()
