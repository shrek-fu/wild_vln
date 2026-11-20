"""Per-class limits on object boxes, centred on the object's anchor (its fused centre).

Live, ``InstancePoints.box`` only counts voxels inside the class's largest plausible box around the anchor.
``limit_object`` is the guard on export (any mapper) and the fix for scenes saved before that: ``size`` is
capped, ``bbox3d`` is intersected with the anchor-centred limit box.

    python3 -m scene_memory.mapping.box_limits <scene dir> [--save]

rewrites ``objects_4d.json`` and the object nodes of ``topo_graph.json``; then ``stvln.sh map_save <name>
--semantic_only`` (and ``map_push``) for a map.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scene_memory.mapping.centroid_lift import class_max_size


def clamp_box(box: dict, anchor, max_size) -> dict:
    """``box`` intersected with the ``max_size`` box around ``anchor``; along an axis where the two miss each
    other, an anchor-centred extent of the smaller size."""
    c, s = np.asarray(box["center"], dtype=np.float64), np.asarray(box["size"], dtype=np.float64)
    a, m = np.asarray(anchor, dtype=np.float64), np.asarray(max_size, dtype=np.float64)
    lo, hi = np.maximum(c - s / 2, a - m / 2), np.minimum(c + s / 2, a + m / 2)
    bad = hi <= lo
    lo[bad], hi[bad] = a[bad] - np.minimum(s, m)[bad] / 2, a[bad] + np.minimum(s, m)[bad] / 2
    return {**box, "center": ((lo + hi) / 2).round(3).tolist(), "size": (hi - lo).round(3).tolist()}


def limit_object(obj: dict) -> dict:
    """``obj`` (an ``objects_4d`` row or graph node) with ``size`` / ``bbox3d`` within its class's limits."""
    m = np.asarray(class_max_size(obj.get("name", "")))
    if obj.get("size") is not None:
        obj["size"] = np.minimum(np.asarray(obj["size"], dtype=np.float64), m).round(3).tolist()
    if obj.get("bbox3d") and obj.get("center") is not None:
        obj["bbox3d"] = clamp_box(obj["bbox3d"], obj["center"], m)
    return obj


def _max_dims(objects) -> dict[str, float]:
    by = defaultdict(list)
    for o in objects:
        if o.get("bbox3d"):
            by[o["name"]].append(max(o["bbox3d"]["size"]))
    return {k: max(v) for k, v in by.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene")
    p.add_argument("--save", action="store_true", help="write the files (default: only print the change)")
    args = p.parse_args()
    scene = Path(args.scene)
    files = {"objects_4d.json": lambda b: b["objects"], "topo_graph.json": lambda b: b["layers"]["objects"]}
    for name, rows_of in files.items():
        path = scene / name
        if not path.exists():
            continue
        body = json.loads(path.read_text(encoding="utf-8"))
        rows = rows_of(body)
        before = _max_dims(rows)
        for o in rows:
            limit_object(o)
        after = _max_dims(rows)
        print(f"{name}: largest bbox3d side per class, before -> after")
        for k in sorted(before):
            print(f"  {k:14s} {before[k]:6.2f} -> {after[k]:5.2f}  (limit {max(class_max_size(k)):.2f})")
        if args.save:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(body, indent=1, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
            print(f"  saved {path}")


if __name__ == "__main__":
    main()
