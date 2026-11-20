import numpy as np

from scene_memory.mapping.revisit import apply_revisits, find_revisits
from scene_memory.mapping.topo_graph import assemble_graph, regraph


def _two_passes(drift_x: float = 4.0):
    """Out along x=0 (y 0..10), a 3-place narrow link, back along x=drift (VIO drift) through the same room."""
    places, traj = [], []
    out = [(0.0, float(y)) for y in range(11)]
    link = [(drift_x * k / 4, 11.0) for k in range(1, 4)]
    back = [(drift_x, float(y)) for y in range(10, -1, -1)]
    for i, (x, y) in enumerate(out + link + back):
        c = 0.5 if len(out) <= i < len(out) + len(link) else 2.0
        places.append({"id": i, "position": [x, y, 0.0], "first_seen": 0.0, "last_seen": 0.0, "clearance": c})
    edges = [(i, i + 1) for i in range(len(places) - 1)]
    t = 0.0
    for i, p in enumerate(places):
        t = 4.0 * i
        traj.append({"stamp": t, "position": p["position"], "orientation": [0, 0, 0, 1]})
    rng = np.random.default_rng(0)
    location = {y: rng.normal(size=(2, 16)) for y in range(11)}
    stamps, desc = [], []
    for i, (x, y) in enumerate(out + link + back):
        if (x, y) in link:
            d = rng.normal(size=(2, 16))
        else:
            d = location[int(y)] if i < len(out) else np.roll(location[int(y)], 1, axis=0)  # opposite heading
        stamps.append(4.0 * i)
        desc.append(d / np.linalg.norm(d, axis=1, keepdims=True))
    graph = assemble_graph([], places, edges, traj)
    return graph, np.array(stamps), np.array(desc), len(out), len(link)


def test_revisit_links_and_corrects_drift():
    graph, stamps, desc, n_out, n_link = _two_passes()
    assert len(graph["layers"]["regions"]) == 2

    loops = find_revisits(graph, stamps, desc, min_dt=20.0, sim_min=0.9)
    assert len(loops) >= 3
    for lp in loops:
        a, b = sorted((lp["a"], lp["b"]))
        assert a < n_out <= n_out + n_link <= b
        pa, pb = graph["layers"]["places"][a]["position"], graph["layers"]["places"][b]["position"]
        assert abs(pa[1] - pb[1]) < 0.5

    objects = [{"key": "obj1", "name": "chair", "center": [4.0, 2.0, 0.5], "first_seen": 90.0, "last_seen": 92.0,
                "history": [[90.0, 4.0, 2.0, 0.5], [92.0, 4.0, 2.0, 0.5]]}]
    stats = apply_revisits(graph, objects, loops)
    assert stats["edges_added"] == len(loops)
    after = regraph(graph, objects)
    assert len(after["layers"]["regions"]) == 1
    assert after["loop_corrected"] and after["revisits"] == loops
    gap_before = 4.0
    pos = {p["id"]: p["position"] for p in after["layers"]["places"]}
    gaps = [abs(pos[lp["a"]][0] - pos[lp["b"]][0]) for lp in loops]
    assert np.median(gaps) < 0.5 * gap_before
    assert objects[0]["center_odom"] == [4.0, 2.0, 0.5] and objects[0]["center"][0] < 3.0
    assert np.allclose(pos[0], [0.0, 0.0, 0.0], atol=1e-6)


def test_no_revisit_without_appearance_match():
    graph, stamps, desc, _, _ = _two_passes()
    rng = np.random.default_rng(1)
    desc = rng.normal(size=desc.shape)
    desc /= np.linalg.norm(desc, axis=2, keepdims=True)
    assert find_revisits(graph, stamps, desc, min_dt=20.0, sim_min=0.9) == []
