import json

import numpy as np

from scene_memory.mapping.pcd_rooms import segment
from scene_memory.mapping.topo_graph import regraph
from scene_memory.mapping.view_layer import INDEX_FILE, KEYFRAMES_FILE, crop_wrapped, sector_boxes
from scene_memory.query import view_search as vs


def _plane(xs, ys, z):
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)


def _wall(a, b, z0=0.0, z1=3.0, step=0.05):
    n = int(np.linalg.norm(np.subtract(b, a)) / step) + 1
    t = np.linspace(0, 1, n)
    xy = np.asarray(a) + t[:, None] * (np.asarray(b) - np.asarray(a))
    zs = np.arange(z0, z1, step)
    return np.array([[x, y, z] for x, y in xy for z in zs])


def _two_rooms():
    """Rooms x 0-5 and 5-10 (y 0-5), joined by a 1 m doorway (lintel above 2 m) in the wall at x = 5."""
    pts = [_plane(np.arange(0, 10, 0.1), np.arange(0, 5, 0.1), 0.0),
           _plane(np.arange(0, 10, 0.1), np.arange(0, 5, 0.1), 3.0),
           _wall((0, 0), (10, 0)), _wall((10, 0), (10, 5)), _wall((10, 5), (0, 5)), _wall((0, 5), (0, 0)),
           _wall((5, 0), (5, 2)), _wall((5, 3), (5, 5)), _wall((5, 2), (5, 3), z0=2.0)]
    return np.concatenate(pts)


def _graph(xs):
    places = [{"id": i, "position": [float(x), 2.5, 0.4], "first_seen": float(i), "last_seen": float(i)}
              for i, x in enumerate(xs)]
    edges = [{"source": f"place{i}", "target": f"place{i + 1}", "relation": "traversable"} for i in range(len(xs) - 1)]
    agents = [{"stamp": float(i), "position": p["position"], "orientation": [0, 0, 0, 1]} for i, p in enumerate(places)]
    return {"layers": {"places": places, "agents": agents, "objects": []}, "edges": edges}


def test_pcd_rooms_split_at_the_wall():
    xs = np.arange(1.0, 9.5, 0.5)
    graph = _graph(xs)
    objects = [{"key": "o1", "name": "chair", "center": [2.0, 4.0, 0.5], "size": [0.5, 0.5, 0.9],
                "first_seen": 0.0, "last_seen": 1.0},
               {"key": "o2", "name": "chair", "center": [5.4, 4.0, 0.5], "size": [0.5, 0.5, 0.9],
                "first_seen": 0.0, "last_seen": 1.0}]
    place_room, object_room, grids = segment(graph, objects, _two_rooms())
    assert grids[0].n_rooms == 2 and abs(grids[0].floor_z) < 0.06 and abs(grids[0].ceiling_z - 3.0) < 0.06
    left = {place_room[i] for i, x in enumerate(xs) if x < 4.5}
    right = {place_room[i] for i, x in enumerate(xs) if x > 5.5}
    assert len(left) == 1 and len(right) == 1 and left != right
    assert object_room["o1"] in left and object_room["o2"] in right

    for p in graph["layers"]["places"]:
        p["room_pcd"] = place_room[p["id"]]
    out = regraph(graph, objects, object_rooms=object_room)
    assert out["segmentation"]["method"] == "pcd" and len(out["layers"]["regions"]) == 2
    region = {o["key"]: o["region"] for o in out["layers"]["objects"]}
    place_region = {p["id"]: p["region"] for p in out["layers"]["places"]}
    # o2 sits 0.4 m behind the wall from the left room's last place, but belongs to the right room
    assert region["o2"] == place_region[len(xs) - 1] and region["o1"] == place_region[0]
    # room_pcd survives a later regraph (e.g. dsg_export)
    again = regraph(out, objects)
    assert again["segmentation"]["method"] == "pcd" and len(again["layers"]["regions"]) == 2


def test_parse_box_and_lift_ray():
    assert vs.parse_box('{"found": true, "box": [100, 200, 300, 400]}') == [100, 200, 300, 400]
    assert vs.parse_box("目标在 [[120,80,560,900]] 处") == [120, 80, 560, 900]
    assert vs.parse_box('{"found": false}') is None and vs.parse_box("[5, 5, 1, 1]") is None
    wall = np.array([[3.0, y, z] for y in np.linspace(-0.5, 0.5, 21) for z in np.linspace(-0.5, 0.5, 21)])
    behind = wall + [4.0, 0, 0]
    p = vs.lift_ray(np.concatenate([wall, behind]), np.zeros(3), np.array([1.0, 0.0, 0.0]))
    assert p is not None and abs(p[0] - 3.0) < 0.3 and abs(p[1]) < 0.2


def test_search_views_with_a_vision_model(tmp_path):
    from PIL import Image

    width, height = 400, 200
    (tmp_path / "keyframes").mkdir()
    boxes = sector_boxes(width, height, 4, 90.0)
    frames = []
    for k in range(2):
        Image.new("RGB", (width, height), (40 * k, 90, 160)).save(tmp_path / "keyframes" / f"kf{k:04d}.jpg")
        T = np.eye(4)
        # DA360 optical (+Z forward, +Y down) -> odom (X forward, Z up)
        T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
        T[:3, 3] = [float(k), 0.0, 0.5]
        frames.append({"id": k, "file": f"keyframes/kf{k:04d}.jpg", "stamp": 10.0 + k, "position": T[:3, 3].tolist(),
                       "T_odom_da360": T.tolist(), "width": width, "height": height, "sectors": [list(b) for b in boxes]})
    (tmp_path / KEYFRAMES_FILE).write_text(json.dumps({"keyframes": frames, "sectors": 4}), encoding="utf-8")
    emb = np.zeros((8, 3), np.float32)
    emb[:, 0] = 0.1
    emb[6] = [0.0, 1.0, 0.0]      # keyframe 1, sector 2 matches the query best
    np.savez(tmp_path / INDEX_FILE, emb=emb, keyframe=np.repeat([0, 1], 4), sector=np.tile(np.arange(4), 2))
    # a wall 3 m in front of the cameras (sector 2 = u 150-250 = straight ahead)
    wall = np.array([[5.0, y, z] for y in np.linspace(-2, 2, 81) for z in np.linspace(-1, 2, 61)])
    np.save(tmp_path / "map_cloud.npy", wall.astype(np.float32))

    views = vs.ViewIndex(tmp_path)
    assert crop_wrapped(Image.open(tmp_path / "keyframes" / "kf0000.jpg"), (-50, 50, 50, 150)).size == (100, 100)
    asked = []

    def vision(image: bytes, text: str) -> str:
        asked.append(text)
        if text.startswith("Numbered"):
            return '{"view": 1, "reason": "fridge on the left"}'
        return '{"found": true, "box": [400, 300, 600, 700]}'

    res = vs.search(views, np.array([0.0, 1.0, 0.0]), "去冰箱", "fridge", vision, k=2)
    assert res["keyframe"] == 1 and res["sector"] == 2 and res["verified"]
    assert abs(res["point"][0] - 5.0) < 0.2 and abs(res["point"][1]) < 0.3 and len(asked) == 2

    none = vs.search(views, np.array([0.0, 1.0, 0.0]), "去冰箱", "fridge", lambda i, t: '{"view": 0}', k=2)
    assert "error" in none


def test_fast_path_falls_back_to_the_views():
    from scene_memory.query import fast_path as fp

    class Tools:
        vision = staticmethod(lambda image, text: "")
        asked = []

        def views_available(self):
            return True

        def search_views(self, query, request=""):
            self.asked.append(query)
            return {"key": "view0", "verified": query == "蓝色的周转箱"}

    tools = Tools()
    hit = fp._from_views(fp.FastResult(phrase="蓝色的周转箱旁边"), tools, "去蓝色的周转箱旁边", "no class")
    assert hit.target == "view0" and hit.verified and tools.asked == ["蓝色的周转箱"]
    miss = fp._from_views(fp.FastResult(phrase="盆栽"), tools, "去盆栽那里", "no class")
    assert miss.target is None and miss.reason.startswith("no class; view search")
    tools.vision = None
    assert fp._from_views(fp.FastResult(phrase="窗户"), tools, "去窗户", "no class").reason == "no class"
