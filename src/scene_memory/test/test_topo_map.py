import numpy as np
import pytest

from scene_memory.mapping.boxer_objects import BoxerObjectMap
from scene_memory.mapping.centroid_lift import CentroidLifter, prior_depth, triangulate_rays
from scene_memory.mapping.topo_graph import PlaceLayer, build_graph, cluster_regions, object_relations, regraph, segment_regions

K = (320.0, 320.0, 319.5, 319.5)


def _camera(x, y):
    """Optical frame at (x, y, 1) looking along world +X (optical z = world x, x = -y, y = -z)."""
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    T[:3, 3] = (x, y, 1.0)
    return T


def _project(X, T):
    p = np.linalg.inv(T) @ np.append(X, 1.0)
    return np.array([K[0] * p[0] / p[2] + K[2], K[1] * p[1] / p[2] + K[3]])


def test_triangulation_needs_parallax_and_rejects_outlier():
    X = np.array([5.0, 1.0, 1.2])
    cams = [_camera(0.0, y) for y in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    O = np.array([T[:3, 3] for T in cams])
    D = np.array([(X - o) / np.linalg.norm(X - o) for o in O])
    D[2] = (D[2] + np.array([0, 0.3, 0])) / np.linalg.norm(D[2] + np.array([0, 0.3, 0]))
    est = triangulate_rays(O, D)
    assert est is not None and np.linalg.norm(est - X) < 0.05
    same_spot = np.repeat(O[:1], 4, axis=0)
    assert triangulate_rays(same_spot, np.repeat(D[:1], 4, axis=0)) is None


def test_prior_depth_from_height_and_truncation():
    # person 1.7 m tall, 136 px high with f=320 → 4 m
    assert abs(prior_depth("person", (300, 200, 340, 336), K, 640, 640) - 4.0) < 1e-6
    assert prior_depth("person", (0, 0, 639, 639), K, 640, 640) is None


def test_lifter_waits_then_triangulates():
    X = np.array([4.0, 0.5, 0.8])
    lifter = CentroidLifter(min_prior_obs=5)
    out = []
    for i, y in enumerate((0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 1.2)):
        T = _camera(0.0, y)
        uv = _project(X, T)
        box = (uv[0] - 20, uv[1] - 40, uv[0] + 20, uv[1] + 40)
        out.append(lifter.observe(7, "chair", box, uv, K, T, 640, 640, float(i)))
    assert [s for _, s in out[:4]] == [None] * 4
    assert out[4][1] == "prior"
    assert out[-1][1] == "tri" and np.linalg.norm(out[-1][0] - X) < 0.05


def test_object_map_prefers_triangulated_and_consolidates():
    m = BoxerObjectMap(reid_radius_m=0.8, class_names=["chair"])
    size = [0.5, 0.5, 0.9]
    m.update([{"obb": {"center": [3, 0, 0], "size": size, "source": "prior"}, "track_id": 1, "cls": 0, "name": "chair"}], 0.0)
    m.update([{"obb": {"center": [1, 0, 0], "size": size, "source": "tri"}, "track_id": 1, "cls": 0, "name": "chair"}], 1.0)
    m.update([{"obb": {"center": [3, 0, 0], "size": size, "source": "prior"}, "track_id": 1, "cls": 0, "name": "chair"}], 2.0)
    obj = next(iter(m.objects.values()))
    assert np.allclose(obj["center"], [1, 0, 0]) and obj["center_source"] == "tri"
    m.expire(100.0)
    m.update([{"obb": {"center": [1.3, 0, 0], "size": size, "source": "tri"}, "track_id": 2, "cls": 0, "name": "chair"}], 101.0)
    assert len(m.objects) == 1
    m.update([{"obb": {"center": [4, 0, 0], "size": size, "source": "tri"}, "track_id": 3, "cls": 0, "name": "chair"}], 102.0)
    m.objects[next(k for k, o in m.objects.items() if 3 in o["track_ids"])]["centers"] = [np.array([1.5, 0, 0])]
    m.objects[next(k for k, o in m.objects.items() if 3 in o["track_ids"])]["center"] = np.array([1.5, 0, 0])
    m.expire(200.0)
    assert m.consolidate() == 1 and len(m.objects) == 1
    assert sorted(next(iter(m.objects.values()))["track_ids"]) == [1, 2, 3]


def test_place_layer_edges_follow_trajectory():
    layer = PlaceLayer(spacing_m=1.0)
    for i, x in enumerate(np.arange(0, 3.01, 0.25)):
        layer.update([x, 0, 0], float(i))
    assert len(layer.places) == 4
    assert layer.edges == {(0, 1), (1, 2), (2, 3)}
    layer.update([0.1, 0.0, 0.0], 100.0)
    assert (0, 3) in layer.edges


def test_relations_on_and_near():
    objs = [
        {"key": "table", "center": [0, 0, 0.375], "size": [1.2, 0.7, 0.75]},
        {"key": "cup", "center": [0.1, 0, 0.8], "size": [0.08, 0.08, 0.1]},
        {"key": "chair", "center": [0.9, 0, 0.45], "size": [0.5, 0.5, 0.9]},
        {"key": "far", "center": [9, 9, 0.45], "size": [0.5, 0.5, 0.9]},
    ]
    edges = {(e["source"], e["target"], e["relation"]) for e in object_relations(objs)}
    assert ("cup", "table", "on") in edges
    assert ("table", "chair", "near") in edges
    assert not any("far" in e for e in edges)


def test_build_graph_layers():
    layer = PlaceLayer(spacing_m=1.0)
    for i, x in enumerate(np.arange(0, 10.01, 0.5)):
        layer.update([x, 0, 0], float(i))
    regions = cluster_regions([dict(p) for p in layer.places], 4.0)
    assert len(regions) == 3
    graph = build_graph([{"key": "obj1", "name": "door", "cls": 1, "center": [9.5, 1.0, 1.0], "size": [0.9, 0.1, 2.0]}], layer)
    obj = graph["layers"]["objects"][0]
    assert obj["place"] in (9, 10) and obj["region"] == 2
    assert {"source": "obj1", "target": f"place{obj['place']}", "relation": "reachable_from"} in graph["edges"]
    rels = {(e["source"], e["target"], e["relation"]) for e in graph["edges"]}
    assert ("region0", "region1", "adjacent") in rels and ("region1", "region2", "adjacent") in rels
    assert ("region0", "region2", "adjacent") not in rels
    assert ("region2", "building0", "in") in rels
    assert len(graph["layers"]["agents"]) == 21 and graph["layers"]["agents"][-1]["place"] == 10


def _corridor(n, y=0.0):
    return [{"id": i, "position": [float(i), y, 0.0]} for i in range(n)]


def test_segment_regions_cuts_at_door_and_merges_small():
    places = _corridor(6)
    edges = {(i, i + 1) for i in range(5)}
    regions, adjacent = segment_regions(places, edges, doors=[[2.5, 0.2, 1.0]], radius_m=10.0, min_places=2)
    assert [r["places"] for r in regions] == [[0, 1, 2], [3, 4, 5]] and adjacent == {(0, 1)}
    regions, _ = segment_regions(places, edges, doors=[[4.5, 0.0, 1.0]], radius_m=10.0, min_places=2)
    assert [r["places"] for r in regions] == [[0, 1, 2, 3, 4, 5]]


def test_regraph_keeps_summary_of_unchanged_region():
    layer = PlaceLayer(spacing_m=1.0)
    for i, x in enumerate(np.arange(0, 5.01, 0.5)):
        layer.update([x, 0, 0], float(i))
    graph = build_graph([], layer, region_radius_m=10.0)
    graph["layers"]["regions"][0].update(label="corridor", description="long corridor")
    again = regraph(graph, [], region_radius_m=10.0)
    assert again["layers"]["regions"][0]["label"] == "corridor"
    assert len(again["layers"]["agents"]) == len(graph["layers"]["agents"])
    split = regraph(graph, [], region_radius_m=2.0, min_region_places=1)
    assert all("label" not in r for r in split["layers"]["regions"])


def test_dsg_export_roundtrip(tmp_path):
    sd = pytest.importorskip("spark_dsg")
    from scene_memory.mapping.dsg_export import write_dsg

    layer = PlaceLayer(spacing_m=1.0)
    for i, x in enumerate(np.arange(0, 10.01, 0.5)):
        layer.update([x, 0, 0], 100.0 + i, (0, 0, 0, 1))
    objs = [{"key": "obj0007", "name": "door", "center": [5.5, 0.5, 1.0], "size": [0.9, 0.1, 2.0], "first_seen": 101.0,
             "last_seen": 105.0, "n_obs": 4, "history": [[101.0, 5.5, 0.5, 1.0]], "center_source": "tri"},
            {"key": "obj0008", "name": "chair", "center": [1.0, 0.8, 0.4], "size": [0.5, 0.5, 0.9], "first_seen": 101.0,
             "last_seen": 102.0, "n_obs": 6, "center_source": "prior"}]
    graph = build_graph(objs, layer)
    assert [r["places"] for r in graph["layers"]["regions"]] == [list(range(6)), list(range(6, 11))]
    graph["layers"]["regions"][0].update(label="corridor", description="start of the corridor")
    G = sd.DynamicSceneGraph.load(str(write_dsg(tmp_path, graph, objs)))
    assert G.get_layer(sd.DsgLayers.OBJECTS).num_nodes() == 1 and G.get_layer("BACKGROUND_OBJECTS").num_nodes() == 1
    assert G.get_layer("TRAVERSABILITY").num_nodes() == 11 and G.get_layer(2, 97).num_nodes() == 21
    rooms = G.get_layer(sd.DsgLayers.ROOMS)
    assert rooms.num_nodes() == 2 and rooms.num_edges() == 1
    r0 = G.get_node(sd.NodeSymbol("R", 0))
    assert r0.attributes.metadata.get()["region_label"] == "corridor"
    assert all(G.get_node(c).layer.layer == 3 for c in r0.children())
    door = G.get_node(sd.NodeSymbol("O", 7))
    md = door.attributes.metadata.get()
    assert md["description"] == "door" and md["temporal_history"]["first_observed"] == 101.0
    assert G.get_node(door.get_parent()).layer.layer == 3
    assert G.get_node(sd.NodeSymbol("B", 0)).has_children()
