"""Where/when queries over an exported 4D scene (``objects_4d.json`` + ``dsg.json``).

Examples::

    python3 -m scene_memory.query.query_4d data/scene4d/latest summary
    python3 -m scene_memory.query.query_4d data/scene4d/latest where chair
    python3 -m scene_memory.query.query_4d data/scene4d/latest when laptop
    python3 -m scene_memory.query.query_4d data/scene4d/latest at 18:30:05
    python3 -m scene_memory.query.query_4d data/scene4d/latest near 1.0 2.0 0.0 --radius 1.5
    python3 -m scene_memory.query.query_4d data/scene4d/latest moved
    python3 -m scene_memory.query.query_4d data/scene4d/latest describe laptop
    python3 -m scene_memory.query.query_4d data/scene4d/latest search black mug
    python3 -m scene_memory.query.query_4d data/scene4d/latest find "red chair"
    python3 -m scene_memory.query.query_4d data/scene4d/latest find "trash can near door"
    python3 -m scene_memory.query.query_4d data/scene4d/latest find 门旁边的垃圾桶 --json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from scene_memory.mapping.object_views import descriptions_by_key, load_description_cache


def load_scene(path: str | Path, include_rejected: bool = False) -> dict:
    """``objects_4d.json``; objects the mapper rejected as false positives are dropped unless asked for."""
    path = Path(path)
    if path.is_dir():
        path = path / "objects_4d.json"
    scene = json.loads(path.read_text(encoding="utf-8"))
    if not include_rejected:
        scene["objects"] = [o for o in scene.get("objects", []) if o.get("status") != "rejected"]
    # descriptions.json (DAM worker) is newer than the last objects_4d.json export.
    by_key = descriptions_by_key(load_description_cache(path.parent))
    for obj in scene.get("objects", []):
        if obj["key"] in by_key:
            obj["descriptions"] = by_key[obj["key"]]
    return scene


def latest_description(obj: dict) -> str:
    descs = obj.get("descriptions") or []
    return descs[-1]["text"] if descs else ""


def fmt_time(stamp: float) -> str:
    return datetime.fromtimestamp(stamp).strftime("%H:%M:%S")


def parse_time(text: str, reference: float) -> float:
    """Epoch seconds, ``HH:MM[:SS]`` on the reference day, or ``-N`` seconds before ``reference``."""
    if text.startswith("-"):
        return reference + float(text)
    if ":" in text:
        parts = [int(p) for p in text.split(":")]
        day = datetime.fromtimestamp(reference)
        return day.replace(hour=parts[0], minute=parts[1], second=parts[2] if len(parts) > 2 else 0).timestamp()
    return float(text)


def seen_intervals(obj: dict, gap_s: float = 5.0) -> list[tuple[float, float]]:
    """Contiguous observation spans from the object's history."""
    stamps = [row[0] for row in obj.get("history", [])] or [obj["first_seen"], obj["last_seen"]]
    stamps = sorted(stamps + [obj["last_seen"]])
    spans = [[stamps[0], stamps[0]]]
    for stamp in stamps[1:]:
        if stamp - spans[-1][1] > gap_s:
            spans.append([stamp, stamp])
        else:
            spans[-1][1] = stamp
    return [(a, b) for a, b in spans]


def match(objects: list[dict], name: str) -> list[dict]:
    name = name.lower()
    return [o for o in objects if name in o["name"].lower() or name == o["key"].lower()]


def describe(obj: dict) -> str:
    c = np.round(obj["center"], 2).tolist()
    s = np.round(obj["size"], 2).tolist()
    state = "visible" if obj["visible"] else f"hidden since {fmt_time(obj['last_seen'])}"
    link = " dsg=yes" if obj.get("dsg") else ""
    views = f" views={len(obj.get('views') or [])}" if obj.get("views") else ""
    return f"{obj['key']} {obj['name']:<12} at {c} size {s} {state} obs={obj['n_obs']}{link}{views}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="scene4d directory or objects_4d.json")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary")
    p_list = sub.add_parser("list")
    p_list.add_argument("--visible", action="store_true")
    sub.add_parser("where").add_argument("name")
    sub.add_parser("when").add_argument("name")
    sub.add_parser("describe").add_argument("name")
    sub.add_parser("search").add_argument("words", nargs="+", help="keywords matched against names + DAM text")
    p_at = sub.add_parser("at")
    p_at.add_argument("time", help="epoch s, HH:MM[:SS], or -N s before save")
    p_at.add_argument("--window", type=float, default=2.0)
    p_near = sub.add_parser("near")
    p_near.add_argument("xyz", nargs=3, type=float)
    p_near.add_argument("--radius", type=float, default=1.0)
    p_moved = sub.add_parser("moved")
    p_moved.add_argument("--min_m", type=float, default=0.5)
    p_index = sub.add_parser("index", help="(re)build the SigLIP object embeddings")
    p_index.add_argument("--rebuild", action="store_true")
    p_find = sub.add_parser("find", help="open-vocabulary search, optionally '<target> near|on <anchor>'")
    p_find.add_argument("query", nargs="+")
    p_find.add_argument("--near", default=None, help="anchor query for a 'near' relation")
    p_find.add_argument("--on", default=None, help="anchor query for an 'on' relation")
    p_find.add_argument("--near_m", type=float, default=1.5)
    p_find.add_argument("--top", type=int, default=5)
    p_find.add_argument("--min_score", type=float, default=0.01, help="below this the target is reported as not found")
    p_find.add_argument("--json", action="store_true")
    for p in (p_index, p_find):
        p.add_argument("--model", default=None, help="SigLIP model dir")
        p.add_argument("--device", default=None)
    args = parser.parse_args()

    scene = load_scene(args.scene)
    objects = scene.get("objects", [])
    saved_at = float(scene.get("saved_at", time.time()))

    if args.cmd == "summary":
        names: dict[str, int] = {}
        for obj in objects:
            names[obj["name"]] = names.get(obj["name"], 0) + 1
        print(f"saved {fmt_time(saved_at)} frame={scene.get('frame_id')} hydra_frames={scene.get('hydra_frames')}")
        print(f"depth_scale={scene.get('depth_scale')} locked={scene.get('depth_scale_locked')}")
        print(f"dsg layers={scene.get('dsg_layers')}")
        visible = sum(1 for o in objects if o["visible"])
        print(f"objects={len(objects)} visible={visible} classes={names}")
    elif args.cmd == "list":
        for obj in objects:
            if args.visible and not obj["visible"]:
                continue
            print(describe(obj))
    elif args.cmd == "where":
        hits = match(objects, args.name)
        if not hits:
            print(f"no '{args.name}' in memory")
        for obj in sorted(hits, key=lambda o: -o["last_seen"]):
            print(describe(obj))
            if latest_description(obj):
                print(f"    \"{latest_description(obj)}\"")
    elif args.cmd == "describe":
        hits = match(objects, args.name)
        if not hits:
            print(f"no '{args.name}' in memory")
        for obj in hits:
            print(describe(obj))
            descs = obj.get("descriptions") or []
            if not descs:
                print("    (no DAM description yet; run describe_objects)")
            for d in descs:
                print(f"    [{fmt_time(d['stamp'])} visit {d['visit']}] {d['text']}")
    elif args.cmd == "search":
        words = [w.lower() for w in args.words]
        scored = []
        for obj in objects:
            text = " ".join([obj["name"]] + [d["text"] for d in obj.get("descriptions") or []]).lower()
            score = sum(1 for w in words if w in text)
            if score:
                scored.append((score, obj))
        if not scored:
            print(f"nothing matches {args.words}")
        for score, obj in sorted(scored, key=lambda r: (-r[0], -r[1]["last_seen"])):
            print(f"[{score}/{len(words)}] {describe(obj)}")
            if latest_description(obj):
                print(f"    \"{latest_description(obj)}\"")
    elif args.cmd == "when":
        hits = match(objects, args.name)
        if not hits:
            print(f"no '{args.name}' in memory")
        for obj in hits:
            spans = ", ".join(f"{fmt_time(a)}-{fmt_time(b)}" for a, b in seen_intervals(obj))
            print(f"{obj['key']} {obj['name']}: first {fmt_time(obj['first_seen'])}, last {fmt_time(obj['last_seen'])}; seen {spans}")
    elif args.cmd == "at":
        t = parse_time(args.time, saved_at)
        print(f"objects observed around {fmt_time(t)} (±{args.window:.0f}s):")
        for obj in objects:
            near = [row for row in obj.get("history", []) if abs(row[0] - t) <= args.window]
            inside = any(a - args.window <= t <= b + args.window for a, b in seen_intervals(obj))
            if near or inside:
                pos = np.round(near[-1][1:], 2).tolist() if near else np.round(obj["center"], 2).tolist()
                print(f"  {obj['key']} {obj['name']} at {pos}")
    elif args.cmd == "near":
        q = np.asarray(args.xyz)
        rows = sorted(((float(np.linalg.norm(np.asarray(o["center"]) - q)), o) for o in objects), key=lambda r: r[0])
        for dist, obj in rows:
            if dist <= args.radius:
                print(f"{dist:5.2f} m  {describe(obj)}")
    elif args.cmd == "moved":
        for obj in objects:
            hist = np.asarray([row[1:] for row in obj.get("history", [])])
            if len(hist) < 2:
                continue
            spread = float(np.linalg.norm(hist.max(0) - hist.min(0)))
            if spread >= args.min_m:
                print(f"{obj['key']} {obj['name']} moved {spread:.2f} m ({fmt_time(obj['first_seen'])}-{fmt_time(obj['last_seen'])})")
    elif args.cmd in ("index", "find"):
        semantic_query(args, objects)


def semantic_query(args, objects: list[dict]) -> None:
    from scene_memory.query import semantic_index as si

    scene_dir = Path(args.scene)
    scene_dir = scene_dir if scene_dir.is_dir() else scene_dir.parent
    encoder = si.SiglipEncoder(args.model or si.DEFAULT_SIGLIP, device=args.device)
    index = si.SemanticIndex(scene_dir, objects, encoder, rebuild=getattr(args, "rebuild", False))
    if args.cmd == "index":
        print(f"indexed {len(objects)} objects ({int(index.has_view.sum())} with views, {index.encoded} re-encoded)")
        return
    query = " ".join(args.query)
    relation, anchor = ("near", args.near) if args.near else ("on", args.on) if args.on else (None, None)
    if relation is None:
        target, relation, anchor = si.parse_query(query)
    else:
        target = query
    hits = index.search(target, top_k=args.top, relation=relation or "", anchor=anchor or "", near_m=args.near_m)
    graph_objects, places = si.load_graph_nodes(scene_dir)
    rows = [si.hit_to_json(h, graph_objects, places) for h in hits]
    found = bool(rows) and rows[0]["semantic"] >= args.min_score
    if args.json:
        print(json.dumps({"query": query, "target": target, "relation": relation, "anchor": anchor, "found": found,
                          "hits": rows}, ensure_ascii=False, indent=1))
        return
    rel = f" [{relation} '{anchor}']" if relation else ""
    print(f"target '{target}'{rel}" + ("" if found else f"  (low confidence < {args.min_score}: probably not in the map)"))
    for r in rows:
        line = f"{r['score']:.3f} {r['key']} {r['name']:<12} at {r['center']} region={r['region']} waypoint={r['place']}"
        if "anchor" in r:
            a = r["anchor"]
            line += f"  ← {relation} {a['key']} {a['name']} (sem {a['semantic']:.3f}, geo {a['relation_w']:.2f})"
        print(line + (f"  view={r['view']}" if r["view"] else ""))


if __name__ == "__main__":
    main()
