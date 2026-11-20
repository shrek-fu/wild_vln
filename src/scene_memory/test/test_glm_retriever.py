import json

from test_semantic_index import _index, _obj

from scene_memory.query.glm_retriever import SceneTools, resolve


def _scene():
    """Two rooms: R0 (x<5, door + near trash can, t=100..110) and R1 (x>=5, far trash can, t=120..130)."""
    objects = [
        _obj("o1", "trash can", (10, 0, 0.3)),
        _obj("o2", "trash can", (0.8, 0, 0.3)),
        _obj("o3", "door", (0, 0, 1.0), (1.0, 0.1, 2.0)),
    ]
    for o, start in zip(objects, (120.0, 100.0, 100.0)):
        o.update(first_seen=start, last_seen=start + 10, visible=False,
                 history=[[start + s, *o["center"]] for s in range(0, 11, 2)])
    places = [{"id": 0, "key": "place0", "position": [1.0, 0.5, 0.0], "region": 0},
              {"id": 1, "key": "place1", "position": [9.0, 0.5, 0.0], "region": 1}]
    agents = [{"key": f"agent{i}", "stamp": 100.0 + 5 * i, "position": [1.0 + 2 * i, 0.5, 0.0],
               "orientation": [0, 0, 0, 1], "place": 0 if i < 3 else 1} for i in range(7)]
    graph = {"layers": {"objects": [{"key": "o2", "place": 0, "region": 0}, {"key": "o3", "place": 0, "region": 0},
                                    {"key": "o1", "place": 1, "region": 1}],
                        "places": places, "agents": agents,
                        "regions": [{"id": 0, "key": "region0", "label": "entrance", "objects": ["o2", "o3"],
                                     "adjacent": [1]},
                                    {"id": 1, "key": "region1", "label": "office", "objects": ["o1"],
                                     "adjacent": [0]}]},
             "edges": [{"source": "o2", "target": "o3", "relation": "near"}]}
    return SceneTools("/nonexistent", _index(objects), objects, graph)


def _call(name, **args):
    return {"tool_calls": [{"id": name, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}


def test_tool_loop_rejects_invented_key_and_returns_waypoint():
    tools = _scene()
    script = iter([
        _call("find_objects", query="trash can", relation="near", anchor="door"),
        _call("answer", target="o99", reply="x"),
        _call("answer", target="o2", reply=""),
        _call("answer", target="o2", alternatives=["o1", "o42"], reply="门旁边的垃圾桶"),
    ])
    seen = []

    def chat(messages):
        seen.append([m["role"] for m in messages])
        return next(script)

    out = resolve("门旁边的垃圾桶", tools, chat)
    assert out["target"] == "o2" and out["alternatives"] == ["o1"]
    assert out["place"] == "place0" and out["waypoint"] == [1.0, 0.5, 0.0]
    assert out["steps"] == 4
    assert all(roles[-1] == "tool" for roles in seen[1:])


def test_tool_results_and_not_found_answer():
    tools = _scene()
    found = tools.find_objects("trash can", relation="near", anchor="door", top_k=2)
    assert found["found"] and found["candidates"][0]["key"] == "o2"
    info = tools.object_info("o2")
    assert info["relations"][0]["other_name"] == "door" and info["moved_m"] == 0.0
    assert info["region"] == {"region": 0, "label": "entrance"}
    assert tools.objects_in_radius("o3", radius=1.5)["objects"][0]["key"] == "o2"
    assert "error" in tools.look_at(["o2"], "is it black?")
    out = resolve("灭火器", tools, lambda m: _call("answer", target=None, reply="地图里没有灭火器"))
    assert out["target"] is None and out["waypoint"] is None


def test_spatio_temporal_tools():
    tools = _scene()
    assert tools.t0 == 100.0 and tools.t_end == 130.0

    loc = tools.robot_location(0)
    assert loc["t"] == 0.0 and loc["region"]["label"] == "entrance"
    assert tools.robot_location("now")["region"]["label"] == "office"
    assert tools.robot_location(-30)["t"] == 0.0

    traj = tools.trajectory(0, 30)
    assert traj["distance_m"] == 12.0
    assert [r["label"] for r in traj["regions_passed"]] == ["entrance", "office"]

    info = tools.region_information()
    assert info["current_region"] == 1
    by_id = {r["region"]: r for r in info["regions"]}
    assert by_id[0]["is_neighbor"] and by_id[0]["visits"] == [{"entered": 0.0, "left": 10.0}]

    region = tools.objects_in_region("R0", query="trash can")
    assert region["counts"] == {"trash can": 1, "door": 1} and region["objects"][0]["key"] == "o2"

    assert {o["key"] for o in tools.objects_in_view(25, window=2)["objects"]} == {"o1"}
    assert {o["key"] for o in tools.objects_in_view(5)["objects"]} == {"o2", "o3"}
    near_robot = tools.objects_in_radius("robot", time=0, radius=2.0)
    assert near_robot["counts"] == {"trash can": 1, "door": 1}
    assert "error" in tools.call("trajectory", {"start": 0})


def test_label_matches_and_distance_from_robot_now():
    tools = _scene()
    found = tools.find_objects("door", top_k=1)
    assert [c["key"] for c in found["candidates"]] == ["o3"] and "label_matches" not in found
    found = tools.find_objects("trash can", top_k=1)
    assert len(found["candidates"]) == 1 and len(found["label_matches"]) == 1
    assert tools.object_info("o1")["from_robot_now_m"] == 3.1  # robot ends at (13, 0.5, 0)


def test_follow_up_history_reaches_model():
    tools = _scene()
    seen = []

    def chat(messages):
        seen.append(list(messages))
        return _call("answer", reply="两个")

    history = [{"role": "user", "content": "门在哪"}, {"role": "assistant", "content": "在入口"}]
    resolve("它旁边有几个垃圾桶", tools, chat, history=history)
    assert [m["role"] for m in seen[0]] == ["system", "user", "assistant", "user"]
