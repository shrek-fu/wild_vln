import numpy as np

from test_semantic_index import _index, _obj

from scene_memory.mapping.topo_graph import PlaceLayer, VerticalTracker, build_graph
from scene_memory.navigation.planner import PlaceGraph
from scene_memory.nodes.goal_server_node import route_legs
from scene_memory.query.glm_retriever import SceneTools

DT = 0.2


def _drive(layer, t, start, end, duration, environment=None):
    """Move linearly from start to end in ``duration`` s (standing still if start == end)."""
    if environment:
        layer.set_environment(environment, t)
    start, end = np.asarray(start, float), np.asarray(end, float)
    n = max(int(round(duration / DT)), 1)
    for k in range(1, n + 1):
        layer.update(start + (end - start) * k / n, t + k * DT)
    return t + n * DT


def _two_floors_and_elevator():
    """Floor 0 corridor, stairs up to floor 1 (walked back above the corridor), elevator up to floor 2."""
    layer = PlaceLayer()
    t = _drive(layer, 0.0, [0, 0, 0], [0, 0, 0], 1.0)
    t = _drive(layer, t, [0, 0, 0], [8, 0, 0], 16.0)
    t = _drive(layer, t, [8, 0, 0], [14, 0, 3], 12.0)        # stairs, slope 0.5
    t = _drive(layer, t, [14, 0, 3], [14, 1, 3], 2.0)
    t = _drive(layer, t, [14, 1, 3], [4, 1, 3], 20.0)
    t = _drive(layer, t, [4, 1, 3], [4, 1, 6], 10.0)         # elevator: no horizontal motion
    t = _drive(layer, t, [4, 1, 6], [4, 1, 6], 4.0)
    t = _drive(layer, t, [4, 1, 6], [9, 1, 6], 10.0)
    return layer


def _two_buildings():
    """Building A (x 0..6), outdoors (x 6..20), building B (x 20..26)."""
    layer = PlaceLayer()
    t = _drive(layer, 0.0, [0, 0, 0], [6, 0, 0], 12.0)
    t = _drive(layer, t, [6, 0, 0], [20, 0, 0.5], 28.0, environment="outdoor")
    _drive(layer, t, [20, 0, 0.5], [26, 0, 0.5], 12.0, environment="indoor")
    return layer


def test_vertical_tracker_kinds_and_outdoor_terrain():
    vt = VerticalTracker()
    events = [vt.update(0.2 * k, [0.0, 0.0, min(0.3 * k, 3.0)]) for k in range(40)]
    floor = [e for e in events if e and e["event"] == "floor"]
    assert len(floor) == 1 and floor[0]["kind"] == "elevator" and floor[0]["level"] == 1

    vt = VerticalTracker()
    events = [vt.update(0.2 * k, [0.1 * k, 0.0, 0.0]) for k in range(20)]  # flat
    events += [vt.update(4.0 + 0.2 * k, [2.0 + 0.1 * k, 0.0, min(0.01 * k, 3.0)]) for k in range(340)]
    floor = [e for e in events if e and e["event"] == "floor"]
    assert len(floor) == 1 and floor[0]["kind"] == "ramp"  # 1:10 gradient

    vt = VerticalTracker()
    events = [vt.update(0.2 * k, [0.1 * k, 0.0, min(0.1 * k, 3.0)], indoor=False) for k in range(80)]
    assert not [e for e in events if e and e["event"] == "floor"] and vt.level == 0

    vt = VerticalTracker()  # bobbing within half a metre is never a transit
    events = [vt.update(0.2 * k, [0.1 * k, 0.0, 0.5 * np.sin(0.3 * k)]) for k in range(200)]
    assert not any(events)


def test_floors_stay_apart_and_connectors_are_marked():
    layer = _two_floors_and_elevator()
    assert [(tr["kind"], tr["level"]) for tr in layer.transitions] == [("stairs", 1), ("elevator", 2)]
    floors = {pl["floor"] for pl in layer.places if not pl.get("connector")}
    assert floors == {0, 1, 2}
    # floor-1 places above the floor-0 corridor were not merged into it
    above = [pl for pl in layer.places if pl.get("floor") == 1 and 4 <= pl["position"][0] <= 8]
    assert above and all(pl["position"][2] > 2.5 for pl in above)
    kinds = {pl.get("connector") for pl in layer.places} - {None}
    assert kinds == {"stairs", "elevator"}

    objects = [_obj("o0", "chair", (6.0, -0.8, 0.4)), _obj("o1", "chair", (6.0, 1.8, 3.4)),
               _obj("o2", "tv", (8.0, 1.8, 6.6))]
    graph = build_graph(objects, layer)
    by_key = {o["key"]: o for o in graph["layers"]["objects"]}
    assert [by_key[k]["floor"] for k in ("o0", "o1", "o2")] == [0, 1, 2]
    regions = graph["layers"]["regions"]
    assert {r["kind"] for r in regions} == {"room", "stairs", "elevator"}
    assert {r["building"] for r in regions} == {"building0"}
    assert [f["key"] for f in graph["layers"]["floors"]] == [f"building0_floor{i}" for i in range(3)]
    link_kinds = {link["kind"] for r in regions for link in r["links"]}
    assert {"stairs", "elevator"} <= link_kinds


def test_planner_segments_floor_changes_and_bans():
    objects = [_obj("o2", "tv", (8.0, 1.8, 6.6))]
    graph = build_graph(objects, _two_floors_and_elevator())
    plan = PlaceGraph(graph).plan([0.0, 0.0, 0.0], [8.0, 1.8, 6.6])
    assert not plan["partial"] and plan["changes_floor"] and plan["goal_floor"] == 2
    assert [s["mode"] for s in plan["segments"]] == ["walk", "stairs", "walk", "elevator", "walk"]
    stairs, elevator = plan["segments"][1], plan["segments"][3]
    assert (stairs["from_floor"], stairs["to_floor"]) == (0, 1)
    assert (elevator["from_floor"], elevator["to_floor"]) == (1, 2)
    assert "elevator up from floor 1 to floor 2" in plan["instructions"][3]
    places = [w["place"] for w in plan["waypoints"]]
    assert len(places) == len(set(places))

    # a wheeled robot without stairs cannot reach floor 2: partial, ends on floor 0
    plan = PlaceGraph(graph, allow_stairs=False).plan([0.0, 0.0, 0.0], [8.0, 1.8, 6.6])
    assert plan["partial"] and not plan["changes_floor"]

    legs = route_legs(PlaceGraph(graph).plan([0.0, 0.0, 0.0], [8.0, 1.8, 6.6]))
    assert [leg["mode"] for leg in legs] == ["walk", "stairs", "walk", "elevator", "walk"]
    assert legs[-1]["index"] == 4 and legs[0]["count"] == 5


def test_buildings_outdoor_entrances_and_one_walking_leg():
    layer = _two_buildings()
    assert [tr["kind"] for tr in layer.transitions] == ["exit", "entrance"]
    objects = [_obj("a", "chair", (2.0, 0.8, 0.4)), _obj("b", "tv", (24.0, 0.8, 1.0)),
               _obj("c", "car", (13.0, 2.0, 0.8))]
    for o in objects:
        o.update(first_seen=1.0, last_seen=50.0, status="present")
    graph = build_graph(objects, layer)
    buildings = {b["key"]: b for b in graph["layers"]["buildings"]}
    assert set(buildings) == {"building0", "building1", "outdoor"}
    assert buildings["building0"]["entrances"] and buildings["building1"]["entrances"]
    by_key = {o["key"]: o for o in graph["layers"]["objects"]}
    assert by_key["a"]["building"] == 0 and by_key["b"]["building"] == 1 and by_key["c"]["building"] is None
    regions = graph["layers"]["regions"]
    assert "entrance" in {link["kind"] for r in regions for link in r["links"]}

    plan = PlaceGraph(graph).plan([0.0, 0.0, 0.0], [24.0, 0.8, 1.0])
    assert not plan["partial"] and not plan["changes_floor"] and plan["goal_building"] == 1
    assert [(s["environment"], s["building"]) for s in plan["segments"]] == [
        ("indoor", 0), ("outdoor", None), ("indoor", 1)]
    assert "outdoors" in plan["instructions"][1]
    legs = route_legs(plan)
    assert len(legs) == 1 and legs[0]["mode"] == "walk" and len(legs[0]["instructions"]) == 3

    tools = SceneTools("/nonexistent", _index(objects), objects, graph)
    assert tools.multi_level
    assert tools._obj_row(tools.by_key["b"])["building"] == "building1"
    assert tools._obj_row(tools.by_key["c"])["building"] == "outdoor"
    info = tools.region_information()
    assert {b["key"] for b in info["buildings"]} == {"building0", "building1", "outdoor"}
    on_b = tools.objects_on_floor("building1", 0)
    assert [o["key"] for o in on_b["objects"]] == ["b"]
    assert tools.objects_on_floor("outdoor")["counts"] == {"car": 1}
    assert "Buildings:" in tools.summary()
