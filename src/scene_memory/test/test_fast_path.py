import json

from test_existence_nav import _corridor_graph
from test_semantic_index import _index, _obj

from scene_memory.query import fast_path as fp
from scene_memory.query.glm_retriever import SceneTools, answer_instruction, resolve


def _tools():
    objects = [_obj("t1", "trash can", (4.5, 0.8, 0.4)), _obj("t2", "trash can", (2.0, 3.6, 0.4)),
               _obj("d1", "door", (2.0, 4.3, 1.0), (1.0, 0.1, 2.0)), _obj("c1", "cup", (1.0, -0.8, 0.4))]
    for o in objects:
        o.update(first_seen=100.0, last_seen=110.0, status="present", history=[[100.0, *o["center"]]])
    graph = _corridor_graph()
    agents = [{"key": f"agent{i}", "stamp": 100.0 + i, "position": graph["layers"]["places"][p]["position"],
               "orientation": [0, 0, 0, 1], "place": p} for i, p in enumerate([5, 4, 3, 2, 1, 0])]
    graph["layers"].update(agents=agents, objects=[{"key": "t1", "place": 4, "region": 1},
                                                   {"key": "t2", "place": 6, "region": 0},
                                                   {"key": "d1", "place": 6, "region": 0},
                                                   {"key": "c1", "place": 1, "region": 0}],
                           regions=[{"id": 0, "label": "hall", "objects": ["t2", "d1", "c1"], "adjacent": [1]},
                                    {"id": 1, "label": "office", "objects": ["t1"], "adjacent": [0]}])
    return SceneTools("/nonexistent", _index(objects), objects, graph)


def _script(*answers):
    msgs = iter([{"tool_calls": [{"id": f"a{i}", "type": "function", "function": {
        "name": "answer", "arguments": json.dumps(a)}}]} for i, a in enumerate(answers)])
    return lambda m: next(msgs)


def test_parse_rules():
    assert fp.parse("有几把椅子？", {})["slow"] == "question"
    assert fp.parse("刚才经过的椅子", {})["slow"] == "time"
    assert fp.parse("去找到第一个机器人", {})["slow"] == "ordinal / direction"
    assert fp.parse("去另一个椅子", {})["slow"] == "follow-up"
    assert fp.parse("去远处那把椅子", {})["slow"] == "distance"
    assert fp.parse("go to the farthest chair", {})["slow"] == "distance"
    assert fp.parse("去最近的椅子", {})["slow"] is None
    assert fp.parse("找到 办公室的第一排", {})["slow"] == "ordinal / direction"
    assert fp.parse("去办公室的椅子", {})["slow"].startswith("unknown qualifier")   # no such region label
    p = fp.parse("请带我去办公室里的那把黑色的椅子", {3: "办公室"})
    assert (p["region"], p["phrase"], p["slow"]) == (3, "黑色的椅子", None)
    assert fp.parse("垃圾桶在哪里？", {})["phrase"] == "垃圾桶"
    assert fp.parse("去茶几", {})["slow"] is None
    p = fp.parse("去2楼的椅子", {})
    assert p["level"] == ("floor", 2) and p["phrase"] == "椅子"
    assert fp.parse("去楼下的门", {})["level"] == ("relative", -1)
    assert fp.parse("去外面的垃圾桶", {})["level"] == ("outdoor", None)
    p = fp.parse("门旁边的垃圾桶", {})
    assert (p["phrase"], p["relation"], p["anchor"]) == ("垃圾桶", "near", "门")
    assert fp.parse("go to the trash can in the office", {1: "office"})["region"] == 1


def test_fast_path_picks_shortest_route_and_uses_relations():
    tools = _tools()
    out = answer_instruction("go to the trash can", tools, chat=None)
    assert out["path"] == "fast" and out["steps"] == 0 and out["verified"] is None
    assert out["target"] == "t1" and out["plan"]["goal_place"] == "place4"    # 4 m vs 5 m via the branch
    assert out["alternatives"] == ["t2"]
    out = answer_instruction("trash can near the door", tools, chat=None)
    assert out["target"] == "t2" and out["path"] == "fast"
    out = answer_instruction("find the trash can in the hall", tools, chat=None)
    assert out["target"] == "t2" and "hall" in out["reply"]


def test_vision_check_reorders_and_rejections_go_to_the_agent():
    tools = _tools()
    tools.vision = lambda image, text: ""
    tools.verify = lambda keys, request: [k for k in keys if k == "t2"]
    out = answer_instruction("go to the trash can", tools, chat=None)
    assert out["target"] == "t2" and out["verified"] is True and out["path"] == "fast"

    tools.verify = lambda keys, request: []
    seen = []

    def chat(messages):
        seen.append(messages)
        return _script({"target": None, "reply": "没有找到"})(messages)

    out = answer_instruction("go to the trash can", tools, chat)
    assert out["path"] == "slow" and out["target"] is None and "visual check rejected" in out["fast_reason"]
    assert any("quick visual check" in m["content"] for m in seen[0] if m["role"] == "system")


def test_agent_answer_is_verified_and_retried():
    tools = _tools()
    tools.vision = lambda image, text: ""
    calls = []

    def verify(keys, request):
        calls.append(list(keys))
        return ["t2"]

    tools.verify = verify
    out = resolve("去那个垃圾桶", tools, _script({"target": "t1", "reply": "去t1"}, {"target": "t2", "reply": "去t2"}))
    assert out["target"] == "t2" and out["verified"] is True and out["steps"] == 2
    assert calls == [["t1"], ["t2"]]

    out = answer_instruction("有几个垃圾桶？", tools, _script({"target": None, "reply": "两个"}))
    assert out["path"] == "slow" and out["fast_reason"] == "question" and out["reply"] == "两个"
