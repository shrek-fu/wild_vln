import numpy as np

from scene_memory.query.semantic_index import SemanticIndex, parse_query, relation_weight

VOCAB = ["trash can", "door", "cup", "table"]


class FakeEncoder:
    def texts(self, texts):
        out = np.zeros((len(texts), len(VOCAB)))
        for r, t in enumerate(texts):
            for c, word in enumerate(VOCAB):
                out[r, c] = float(word in t)
        return out

    def prob(self, cos):
        return np.asarray(cos, float)


def _obj(key, name, center, size=(0.4, 0.4, 0.4)):
    return {"key": key, "name": name, "center": list(center), "size": list(size)}


def _index(objects):
    index = SemanticIndex.__new__(SemanticIndex)
    index.objects = objects
    index.encoder = FakeEncoder()
    index.image_emb = FakeEncoder().texts([o["name"] for o in objects])
    index.fallback_emb = index.image_emb
    index.has_view = np.ones(len(objects), bool)
    return index


def test_parse_query_en_zh():
    assert parse_query("trash can near the door") == ("trash can", "near", "door")
    assert parse_query("cup on table") == ("cup", "on", "table")
    assert parse_query("门旁边的垃圾桶") == ("垃圾桶", "near", "门")
    assert parse_query("桌子上的杯子") == ("杯子", "on", "桌子")
    assert parse_query("杯子在桌子上") == ("杯子", "on", "桌子")
    assert parse_query("red chair") == ("red chair", None, None)


def test_relation_weight_near_and_on():
    table = _obj("t", "table", (0, 0, 0.375), (1.2, 0.7, 0.75))
    cup_on = _obj("c1", "cup", (0.1, 0.1, 0.80), (0.08, 0.08, 0.1))
    cup_floor = _obj("c2", "cup", (2.0, 0, 0.05), (0.08, 0.08, 0.1))
    assert relation_weight(cup_on, table, "on") > 0.9
    assert relation_weight(cup_floor, table, "on") < 0.01
    assert relation_weight(cup_on, table, "near") > relation_weight(cup_floor, table, "near")


def test_search_relation_picks_target_by_anchor():
    objects = [
        _obj("o1", "trash can", (10, 0, 0.3)),
        _obj("o2", "trash can", (0.8, 0, 0.3)),
        _obj("o3", "door", (0, 0, 1.0), (1.0, 0.1, 2.0)),
    ]
    index = _index(objects)
    plain = index.search("trash can", top_k=3, relation="", anchor="")
    assert {h.obj["key"] for h in plain[:2]} == {"o1", "o2"}
    hits = index.search("trash can near door", top_k=2)
    assert hits[0].obj["key"] == "o2" and hits[0].anchor["key"] == "o3"
    assert hits[0].score > 5 * hits[1].score
