import argparse
import json
import math

import numpy as np

from test_semantic_index import _index, _obj

from scene_memory.mapping.boxer_objects import BoxerObjectMap
from scene_memory.mapping.scene_graph import T_odom_optical
from scene_memory.mapping.topo_graph import PlaceLayer, build_graph
from scene_memory.navigation.planner import PlaceGraph
from scene_memory.nodes.topo_map_node import TopoMapper
from scene_memory.query.glm_retriever import SceneTools, resolve

SIZE = [0.5, 0.5, 0.9]


def _det(center, track, name="chair", cls=0, source="tri", view="front", xyxy=(300, 300, 340, 340)):
    return {"obb": {"center": list(center), "size": SIZE, "source": source}, "track_id": track, "cls": cls,
            "name": name, "view": view, "xyxy": list(xyxy)}


def test_existence_absent_after_misses_and_reappears():
    m = BoxerObjectMap(class_names=["chair"])
    for t in range(3):
        m.update([_det([2, 0, 0], 1)], float(t))
    key = next(iter(m.objects))
    assert m.objects[key]["status"] == "present"
    became = [m.miss(key, 10.0 + i) for i in range(12)]
    assert any(became) and became.count(True) == 1
    assert m.objects[key]["status"] == "absent" and not m.objects[key]["visible"]
    m.update([_det([2, 0, 0], 5)], 30.0)
    obj = m.objects[key]
    assert len(m.objects) == 1 and obj["status"] == "present"
    assert [e for _, e in obj["events"]] == ["appeared", "disappeared", "reappeared"]
    row = m.to_json()["objects"][0]
    assert row["status"] == "present" and 0.5 < row["existence"] < 1.0 and not row["dynamic"]
    # consecutive frames within miss_interval_s are one piece of evidence
    assert m.miss(key, 40.0) is False and m.miss(key, 40.5) is False and m.objects[key]["misses"] == 13


def test_dynamic_objects_stay_out_of_the_static_graph():
    m = BoxerObjectMap(class_names=["chair", "person"])
    m.update([_det([2, 0, 0], 1), _det([2.5, 0, 0], 2, name="person", cls=1)], 0.0)
    rows = m.to_json()["objects"]
    assert {r["name"]: r["dynamic"] for r in rows} == {"chair": False, "person": True}
    layer = PlaceLayer()
    layer.update([0, 0, 0], 0.0)
    graph = build_graph(rows, layer)
    assert [o["name"] for o in graph["layers"]["objects"]] == ["chair"]


def _mapper():
    mapper = TopoMapper.__new__(TopoMapper)
    mapper.args = argparse.Namespace(miss_range=5.0, miss_margin=0.1, miss_min_px=24.0)
    R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    mapper.views = {"front": {"hfov_deg": 90.0, "R_body_optical": R}}
    mapper.T_imu_da360 = np.eye(4)
    mapper.objects = BoxerObjectMap(class_names=["chair", "table"])
    mapper.stats = {"miss": 0, "absent": 0}
    T = T_odom_optical(np.eye(4), np.eye(4), R)
    ahead = lambda d, lateral=0.0: (T @ np.array([lateral, 0.0, d, 1.0]))[:3]  # noqa: E731
    return mapper, ahead


def test_visibility_test_counts_misses_but_respects_occlusion_and_range():
    mapper, ahead = _mapper()
    objs = mapper.objects
    for t in (0.0, 0.1, 0.2):
        objs.update([_det(ahead(3.0), 1), _det(ahead(9.0), 2), _det(ahead(4.0, lateral=-4.0), 3)], t)
    objs.update([_det(ahead(2.5, lateral=0.5), 4, name="table", cls=1)], 0.3)  # a single false positive
    keys = {o["track_ids"][0]: k for k, o in objs.objects.items()}
    body = {"views": ["front"], "detections": []}
    for i in range(40):
        mapper._negative_evidence(body, np.eye(4), 640, 640, 1.0 + 0.25 * i)  # 4 Hz, 10 s
    assert objs.objects[keys[1]]["status"] == "absent"          # 3 m ahead, image centre
    assert objs.objects[keys[2]]["status"] == "present"         # beyond miss_range
    assert objs.objects[keys[3]]["status"] == "present"         # outside the view
    assert objs.objects[keys[4]]["status"] == "rejected"
    assert mapper.stats["absent"] == 1
    # one miss per second at most: absent after 8 s in view (3 hits → 8 misses), not after 8 frames
    assert objs.objects[keys[1]]["misses"] == 8

    mapper, ahead = _mapper()
    objs = mapper.objects
    objs.update([_det(ahead(4.0), 1), _det(ahead(2.0), 2, name="table", cls=1)], 0.0)
    keys = {o["track_ids"][0]: k for k, o in objs.objects.items()}
    # the nearer table is detected over the chair's projection → chair is occluded, not missing
    body = {"views": ["front"], "detections": [{"track_id": 2, "view": "front", "xyxy": [250, 250, 390, 390]}]}
    for i in range(12):
        mapper._negative_evidence(body, np.eye(4), 640, 640, 1.0 + i)
    assert objs.objects[keys[1]]["status"] == "present" and mapper.stats["miss"] == 0


def _corridor_graph():
    """Places 0..5 along x (1 m apart), a dead-end branch 6 at (2, 3); traversable 0-1-2-3-4-5 and 2-6."""
    pos = [[float(i), 0.0, 0.0] for i in range(6)] + [[2.0, 3.0, 0.0]]
    places = [{"id": i, "key": f"place{i}", "position": p, "region": 0 if i in (0, 1, 2, 6) else 1}
              for i, p in enumerate(pos)]
    edges = [(i, i + 1) for i in range(5)] + [(2, 6)]
    return {"frame_id": "odom", "layers": {"places": places},
            "edges": [{"source": f"place{a}", "target": f"place{b}", "relation": "traversable"} for a, b in edges]}


def test_planner_routes_over_places_and_faces_target():
    pg = PlaceGraph(_corridor_graph())
    plan = pg.plan([0.1, 0.0, 0.0], [5.0, 1.0, 0.5], seen_places={4})
    assert plan["start_place"] == "place0" and plan["goal_place"] == "place4"
    assert plan["place_path"] == [f"place{i}" for i in range(5)] and not plan["partial"]
    assert plan["path_length_m"] == 4.0 and plan["target_seen_from_goal"]
    assert math.isclose(plan["goal_pose"]["yaw"], math.atan2(1.0, 1.0), abs_tol=1e-3)
    assert plan["waypoints"][0]["position"] == [0.0, 0.0, 0.0]
    assert plan["waypoints"][-1]["position"] == [4.0, 0.0, 0.0]
    # the branch place is closer to (2, 4) as the crow flies and is reachable only through place2
    plan = pg.plan([5.0, 0.0, 0.0], [2.0, 4.0, 0.5])
    assert plan["goal_place"] == "place6" and plan["place_path"][-2] == "place2"


def test_planner_partial_when_target_area_disconnected():
    graph = _corridor_graph()
    graph["layers"]["places"].append({"id": 7, "key": "place7", "position": [20.0, 0.0, 0.0], "region": 2})
    plan = PlaceGraph(graph).plan([0.0, 0.0, 0.0], [20.5, 0.0, 0.0])
    assert plan["partial"] and plan["goal_place"] == "place5"


def _tools(robot_pose=None):
    objects = [_obj("o1", "chair", (4.5, 0.8, 0.4)), _obj("o2", "chair", (2.0, 3.6, 0.4))]
    # o1 seen while the robot was at places 3-4, o2 while at places 2/6
    for o, (start, status) in zip(objects, ((106.0, "present"), (112.0, "absent"))):
        o.update(first_seen=start, last_seen=start + 4, status=status, history=[[start + s, *o["center"]] for s in range(5)])
    graph = _corridor_graph()
    agents = [{"key": f"agent{i}", "stamp": 100.0 + 2 * i, "position": graph["layers"]["places"][p]["position"],
               "orientation": [0, 0, 0, 1], "place": p} for i, p in enumerate([0, 1, 2, 3, 4, 3, 2, 6, 2, 1, 0])]
    graph["layers"].update(agents=agents, objects=[{"key": "o1", "place": 4, "region": 1},
                                                   {"key": "o2", "place": 6, "region": 0}],
                           regions=[{"id": 0, "label": "hall", "objects": ["o2"], "adjacent": [1]},
                                    {"id": 1, "label": "office", "objects": ["o1"], "adjacent": [0]}])
    return SceneTools("/nonexistent", _index(objects), objects, graph, robot_pose=robot_pose)


def test_answer_carries_plan_and_status():
    tools = _tools()
    script = iter([
        {"tool_calls": [{"id": "r", "type": "function", "function": {"name": "route", "arguments": json.dumps({"key": "o1"})}}]},
        {"tool_calls": [{"id": "a", "type": "function", "function": {
            "name": "answer", "arguments": json.dumps({"target": "o1", "reply": "去椅子", "alternatives": ["o2"]})}}]},
    ])
    out = resolve("去椅子", tools, lambda m: next(script))
    plan = out["plan"]
    assert out["target"] == "o1" and out["status"] == "present"
    assert plan["start_place"] == "place0" and plan["goal_place"] == "place4" and plan["target_seen_from_goal"]
    route = tools.route("o2")
    assert route["status"] == "absent" and route["goal_place"] == "place6"
    assert [r["label"] for r in route["regions_on_route"]] == ["hall"]
    assert tools.find_objects("chair")["candidates"]


def test_live_robot_pose_overrides_trajectory_end():
    tools = _tools(robot_pose={"position": [5.0, 0.0, 0.0], "orientation": [0, 0, 0, 1]})
    assert tools.plan_to("o1")["start_place"] == "place5"
    assert tools._obj_row(tools.by_key["o1"])["from_robot_now_m"] == round(math.sqrt(0.25 + 0.64 + 0.16), 1)
    assert tools._obj_row(tools.by_key["o2"])["status"] == "absent"


def test_find_objects_by_distance():
    objects = [_obj(f"c{i}", "chair", (float(x), 0.0, 0.4)) for i, x in enumerate((1, 9, 4, 6))]
    for o in objects:
        o.update(first_seen=0.0, last_seen=1.0, status="present")
    graph = {"layers": {"places": [{"id": 0, "key": "place0", "position": [0, 0, 0], "region": 0}],
                        "agents": [{"key": "agent0", "stamp": 0.0, "position": [0, 0, 0],
                                    "orientation": [0, 0, 0, 1], "place": 0}],
                        "objects": [{"key": o["key"], "place": 0, "region": 0} for o in objects],
                        "regions": [{"id": 0, "label": "hall", "objects": [o["key"] for o in objects]}]}}
    tools = SceneTools("/nonexistent", _index(objects), objects, graph)
    far = tools.call("find_objects", {"query": "chair", "order": "farthest", "top_k": 2})
    assert far["num_matching"] == 4 and [c["key"] for c in far["candidates"]] == ["c1", "c3"]
    near = tools.find_objects("chair", order="nearest", top_k=1)
    assert [c["key"] for c in near["candidates"]] == ["c0"]
