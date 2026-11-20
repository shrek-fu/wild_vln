"""Export the topo scene as a DAAAM/Hydra spark_dsg scene graph.

Writes ``clustered_dsg_with_summaries.json`` — the file DAAAM's QA tools load —
with Hydra's layer layout:

=====================  =========  ======  ==================================
layer                  key        symbol  attributes
=====================  =========  ======  ==================================
OBJECTS                2          O<id>   ObjectNodeAttributes (triangulated)
BACKGROUND_OBJECTS     2[2]       o<id>   ObjectNodeAttributes (size-prior only)
AGENTS                 2[97]      a<i>    AgentNodeAttributes (trajectory)
TRAVERSABILITY         3[2]       t<id>   TraversabilityNodeAttributes
ROOMS                  4          R<id>   RoomNodeAttributes (+ region summary)
BUILDINGS              5          B0      SemanticNodeAttributes
=====================  =========  ======  ==================================

Edges: t–t (traversable), R–R (adjacent), R→t, B→R, t→O/o (parent = reachable
place). Object metadata follows DAAAM: ``description`` and ``temporal_history``
{timestamps, first_observed, last_observed, observation_count}. SigLIP image
embeddings go to graph metadata ``features[<id>]["siglip_feature"]`` (not
``clip_feature``: DAAAM's CLIP text encoder is a different model).

Offline use (re-segments regions, optional GLM region summaries)::

    python3 -m scene_memory.mapping.dsg_export data/scene4d/latest --summarize
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

import numpy as np

DSG_FILE = "clustered_dsg_with_summaries.json"


def _obj_id(key: str) -> int:
    return int("".join(ch for ch in key if ch.isdigit()) or 0)


def _description(obj: dict) -> str:
    descs = obj.get("descriptions") or []
    return descs[-1]["text"] if descs else obj["name"]


def to_dsg(graph: dict, objects: list[dict], features: dict[str, np.ndarray] | None = None):
    import spark_dsg as sd

    names = {"OBJECTS": sd.LayerKey(2), "BACKGROUND_OBJECTS": sd.LayerKey(2, 2), "AGENTS": sd.LayerKey(2, 97),
             "TRAVERSABILITY": sd.LayerKey(3, 2), "ROOMS": sd.LayerKey(4), "BUILDINGS": sd.LayerKey(5)}
    G = sd.DynamicSceneGraph(list(names.values()), names)
    layers = graph["layers"]
    nodes = {o["key"]: o for o in layers["objects"]}

    for p in layers["places"]:
        a = sd.TraversabilityNodeAttributes()
        a.position = np.asarray(p["position"], dtype=np.float64)
        a.name = p["key"]
        a.first_observed_ns = int(p["first_seen"] * 1e9)
        a.last_observed_ns = int(p["last_seen"] * 1e9)
        a.metadata.set({"visits": int(p.get("visits", 1)), "first_seen": p["first_seen"], "last_seen": p["last_seen"]})
        G.add_node("TRAVERSABILITY", sd.NodeSymbol("t", p["id"]), a)

    for r in layers["regions"]:
        a = sd.RoomNodeAttributes()
        a.position = np.asarray(r["center"], dtype=np.float64)
        a.semantic_label = int(r["id"])
        a.name = r.get("label") or r["key"]
        md = {"key": r["key"], "num_objects": len(r["objects"]), "num_places": len(r["places"]),
              "adjacent": [int(x) for x in r.get("adjacent", [])], "kind": r.get("kind", "room"),
              "environment": r.get("environment", "indoor"), "building": r.get("building"),
              "floor": r.get("floor"), "floors": r.get("floors")}
        if r.get("label"):
            md.update(region_label=r["label"], region_description=r.get("description", ""),
                      reasoning=r.get("reasoning", ""),
                      description=f"{r['label']}: {r.get('description', '')}")
        G.add_node(sd.DsgLayers.ROOMS, sd.NodeSymbol("R", r["id"]), a)
        G.get_node(sd.NodeSymbol("R", r["id"])).attributes.metadata.set(md)

    building_id = {}
    for b in layers.get("buildings", []):
        a = sd.SemanticNodeAttributes()
        a.position = np.asarray(b["center"], dtype=np.float64)
        a.name = b.get("key", "building")
        G.add_node(sd.DsgLayers.BUILDINGS, sd.NodeSymbol("B", b["id"]), a)
        G.get_node(sd.NodeSymbol("B", b["id"])).attributes.metadata.set(
            {"key": b.get("key"), "kind": b.get("kind", "building"), "floors": b.get("floors", []),
             "entrances": b.get("entrances", [])})
        building_id[b.get("key", f"building{b['id']}")] = b["id"]

    for i, kf in enumerate(layers.get("agents", [])):
        a = sd.AgentNodeAttributes()
        a.position = np.asarray(kf["position"], dtype=np.float64)
        qx, qy, qz, qw = kf.get("orientation", [0.0, 0.0, 0.0, 1.0])
        a.world_R_body = sd.Quaternion(float(qw), float(qx), float(qy), float(qz))
        a.timestamp = datetime.timedelta(seconds=float(kf["stamp"]))
        a.metadata.set({"timestamp": float(kf["stamp"]), "place": kf.get("place")})
        G.add_node(2, sd.NodeSymbol("a", i), a, 97)

    feats = {}
    objects = [o for o in objects if not o.get("dynamic") and o.get("status") != "rejected"]
    for obj in objects:
        node = nodes.get(obj["key"], {})
        oid = _obj_id(obj["key"])
        background = obj.get("center_source", "tri") == "prior"
        a = sd.ObjectNodeAttributes()
        a.position = np.asarray(obj["center"], dtype=np.float64)
        a.name = obj["name"]
        a.semantic_label = oid
        a.bounding_box = sd.BoundingBox(np.asarray(obj["size"], np.float32), np.asarray(obj["center"], np.float32))
        stamps = [float(row[0]) for row in obj.get("history", [])] or [obj["first_seen"], obj["last_seen"]]
        a.metadata.set({
            "description": _description(obj),
            "label": obj["name"],
            "key": obj["key"],
            "center_source": obj.get("center_source", "tri"),
            "label_belief": obj.get("label_belief", {}),
            "views": [v["image"] for v in obj.get("views") or []],
            "temporal_history": {"timestamps": stamps, "first_observed": float(obj["first_seen"]),
                                 "last_observed": float(obj["last_seen"]), "observation_count": int(obj.get("n_obs", 0)),
                                 "frame_ids": []},
        })
        symbol = sd.NodeSymbol("o" if background else "O", oid)
        G.add_node("BACKGROUND_OBJECTS" if background else sd.DsgLayers.OBJECTS, symbol, a)
        if node.get("place") is not None:
            G.insert_edge(sd.NodeSymbol("t", node["place"]), symbol)
        if features and obj["key"] in features:
            feats[str(oid)] = {"siglip_feature": np.asarray(features[obj["key"]], dtype=float).round(5).tolist()}

    room_of = {}
    for e in graph["edges"]:
        s, t, rel = e["source"], e["target"], e["relation"]
        if rel == "traversable":
            G.insert_edge(sd.NodeSymbol("t", int(s[5:])), sd.NodeSymbol("t", int(t[5:])))
        elif rel == "adjacent":
            G.insert_edge(sd.NodeSymbol("R", int(s[6:])), sd.NodeSymbol("R", int(t[6:])))
        elif rel == "in" and s.startswith("place"):
            room_of[int(s[5:])] = int(t[6:])
        elif rel == "in" and s.startswith("region") and t in building_id:
            G.insert_edge(sd.NodeSymbol("B", building_id[t]), sd.NodeSymbol("R", int(s[6:])), True)
    for place, room in room_of.items():
        G.insert_edge(sd.NodeSymbol("R", room), sd.NodeSymbol("t", place), True)

    stamps = [p["first_seen"] for p in layers["places"]] + [kf["stamp"] for kf in layers.get("agents", [])]
    stamps += [o["first_seen"] for o in objects]
    # DAAAM's preprocess_scene_graph subtracts start_time from unsigned ns fields: it must precede everything.
    G.metadata.set({"features": feats, "embedding_model": "siglip-base-patch16-256-multilingual" if feats else None,
                    "start_time": float(min(stamps)) if stamps else 0.0,
                    "source": "scene_memory.topo_map", "frame_id": graph.get("frame_id", "odom")})
    return G


def load_features(scene_dir: Path) -> dict[str, np.ndarray]:
    path = scene_dir / "semantic_index.npz"
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    return {str(k): e for k, e, v in zip(data["keys"], data["image_emb"], data["has_view"]) if v}


def write_dsg(scene_dir: str | Path, graph: dict, objects: list[dict]) -> Path:
    scene_dir = Path(scene_dir)
    path = scene_dir / DSG_FILE
    tmp = scene_dir / ("tmp_" + DSG_FILE)
    to_dsg(graph, objects, load_features(scene_dir)).save(str(tmp))
    tmp.replace(path)
    return path


def main() -> None:
    from scene_memory.query.query_4d import load_scene
    from scene_memory.query.region_summary import load_summaries, summarize_regions
    from scene_memory.mapping.topo_graph import regraph, save_graph

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="scene4d directory with objects_4d.json + topo_graph.json")
    parser.add_argument("--region_radius", type=float, default=4.0)
    parser.add_argument("--door_radius", type=float, default=0.8)
    parser.add_argument("--min_region_places", type=int, default=3)
    parser.add_argument("--summarize", action="store_true", help="LLM region summaries (GLM, needs ZHIPUAI_API_KEY)")
    parser.add_argument("--model", default="glm-4-plus")
    parser.add_argument("--language", default="English")
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()

    scene_dir = Path(args.scene)
    objects = load_scene(scene_dir)["objects"]
    old = json.loads((scene_dir / "topo_graph.json").read_text(encoding="utf-8"))
    kwargs = dict(region_radius_m=args.region_radius, door_radius_m=args.door_radius,
                  min_region_places=args.min_region_places)
    graph = regraph(old, objects, **kwargs)
    summaries = load_summaries(scene_dir)
    if args.summarize:
        chat = None
        key = os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("GLM_API_KEY")
        if key:
            from scene_memory.query.glm_retriever import GlmClient

            client = GlmClient(key)
            chat = lambda prompt: client.chat(args.model, [{"role": "user", "content": prompt}]).get("content") or ""
        else:
            print("no ZHIPUAI_API_KEY: region labels fall back to class counts")
        summaries = summarize_regions(graph, objects, scene_dir, chat, args.n_samples, args.language, log=print)
    for region in graph["layers"]["regions"]:
        region.update(summaries.get(tuple(sorted(region["places"])), {}))
    save_graph(graph, scene_dir / "topo_graph.json")
    path = write_dsg(scene_dir, graph, objects)
    L = graph["layers"]
    print(f"{path}: objects={len(L['objects'])} places={len(L['places'])} regions={len(L['regions'])} "
          f"agents={len(L['agents'])} edges={len(graph['edges'])}")


if __name__ == "__main__":
    main()
