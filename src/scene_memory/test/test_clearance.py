import numpy as np

from scene_memory.mapping.clearance import frame_clearance, place_clearances
from scene_memory.mapping.topo_graph import assemble_graph, segment_regions


def _rooms_and_corridor():
    """Two 4x4 grids of places (1 m apart, clearance 2) joined by a 4-place corridor (clearance 0.5), no doors."""
    places, edges, index = [], [], {}

    def add(xy, c):
        index[xy] = len(places)
        places.append({"id": len(places), "position": [float(xy[0]), float(xy[1]), 0.0], "clearance": c})

    for ox in (0, 8):
        for x in range(4):
            for y in range(4):
                add((ox + x, y), 2.0)
    for x in range(4, 8):
        add((x, 1), 0.5)
    for (x, y), i in index.items():
        for dx, dy in ((1, 0), (0, 1)):
            if (x + dx, y + dy) in index:
                edges.append((i, index[(x + dx, y + dy)]))
    return places, edges, index


def test_filtration_splits_rooms_at_narrow_passage():
    places, edges, index = _rooms_and_corridor()
    info = {}
    regions, adjacent = segment_regions(places, edges, radius_m=100.0, info=info)
    assert info["method"] == "clearance" and info["seeds"] == 2
    assert len(regions) == 2 and adjacent == {(0, 1)}
    assert places[index[(0, 0)]]["region"] != places[index[(11, 3)]]["region"]
    # corridor places split between the two rooms, each half to the nearer one
    assert places[index[(4, 1)]]["region"] == places[index[(0, 0)]]["region"]
    assert places[index[(7, 1)]]["region"] == places[index[(11, 3)]]["region"]

    for p in places:
        p.pop("clearance")
    info = {}
    regions, _ = segment_regions(places, edges, radius_m=100.0, info=info)
    assert info["method"] == "radius" and len(regions) == 1


def test_assemble_graph_reports_segmentation_and_fills_gaps():
    places, edges, index = _rooms_and_corridor()
    places[index[(1, 1)]]["clearance"] = None
    for p in places:
        p.update(first_seen=0.0, last_seen=1.0)
    graph = assemble_graph([], places, edges, [])
    assert graph["segmentation"]["method"] == "clearance"
    assert len(graph["layers"]["regions"]) == 2


def test_frame_clearance_and_place_mapping():
    prof = [3.0] * 360
    prof[150:210] = [0.7] * 60            # operator behind the camera: ignored
    prof[70:110] = [1.0] * 40
    assert frame_clearance(prof, ignore_sector=(120, 240), percentile=10) == 1.0
    assert frame_clearance(prof, ignore_sector=None, percentile=10) == 0.7
    assert frame_clearance([None] * 360) is None

    graph = {"layers": {
        "places": [{"id": 0, "position": [0.0, 0.0, 0.0]}, {"id": 1, "position": [5.0, 0.0, 0.0]}],
        "agents": [{"stamp": 0.0, "position": [0.0, 0.0, 0.0]}, {"stamp": 10.0, "position": [5.0, 0.0, 0.0]}]}}
    frames = [{"stamp": 0.1, "profile": [2.0] * 360}, {"stamp": 9.9, "profile": [4.0] * 360},
              {"stamp": 5.0, "profile": [9.0] * 360}]   # 5 s from any keyframe: skipped
    clear = place_clearances(graph, frames, scale=0.5, ignore_sector=None)
    assert clear == {0: 1.0, 1: 2.0}
