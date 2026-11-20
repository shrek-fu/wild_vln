"""3D viewer for the layered topo scene graph (Hydra/DAAAM style: layers stacked in z).

Layers: objects (3D boxes at their true pose) -> places + trajectory -> regions (lifted by
``--region_dz``) -> building (lifted by ``--building_dz``); interlayer and object relation
edges (near/on) are drawn as lines, everything coloured by region.

Backends::

    # rerun: 3D + timeline (objects/places appear at first_seen, robot moves along the trajectory)
    python3 -m scene_memory.viz.dsg_viz data/scene4d/latest
    python3 -m scene_memory.viz.dsg_viz data/scene4d/latest --save /tmp/scene.rrd   # rerun /tmp/scene.rrd

    # plotly: single HTML file, Chinese labels, hover shows region summaries / object details
    python3 -m scene_memory.viz.dsg_viz data/scene4d/latest --html /tmp/scene.html

    # text: building -> floor -> region -> objects
    python3 -m scene_memory.viz.dsg_viz data/scene4d/latest --tree
"""

from __future__ import annotations

import argparse
import colorsys
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scene_memory.query.query_4d import load_scene

RELATION_COLORS = {
    "traversable": (60, 60, 60), "reachable_from": (150, 150, 150), "in": (90, 140, 255),
    "adjacent": (255, 120, 0), "near": (0, 200, 120), "on": (220, 0, 220),
}


def palette(n: int) -> list[tuple[int, int, int]]:
    return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.75, 0.95)) for i in range(max(n, 1))]


class Layout:
    """Positions and colours of every node key after lifting the upper layers."""

    def __init__(self, graph: dict, region_dz: float, building_dz: float):
        layers = graph["layers"]
        self.objects = layers.get("objects", [])
        self.places = layers.get("places", [])
        self.regions = layers.get("regions", [])
        self.buildings = layers.get("buildings", [])
        self.agents = sorted(layers.get("agents", []), key=lambda a: a["stamp"])
        self.edges = graph.get("edges", [])
        colors = palette(len(self.regions))
        self.region_color = {r["id"]: colors[i] for i, r in enumerate(self.regions)}
        top = max([p["position"][2] for p in self.places] + [o["center"][2] for o in self.objects] + [0.0])
        self.pos: dict[str, np.ndarray] = {}
        self.color: dict[str, tuple] = {}
        grey = (170, 170, 170)
        for o in self.objects:
            self.pos[o["key"]] = np.asarray(o["center"], float)
            self.color[o["key"]] = self.region_color.get(o.get("region"), grey)
        for p in self.places:
            self.pos[p["key"]] = np.asarray(p["position"], float)
            self.color[p["key"]] = self.region_color.get(p.get("region"), grey)
        # each floor's regions float region_dz above that floor (one shared level for single-floor maps)
        place_z = {p["id"]: p["position"][2] for p in self.places}
        floor_top: dict = defaultdict(lambda: -np.inf)
        for r in self.regions:
            zs = [place_z[i] for i in r.get("places", []) if i in place_z] or [r["center"][2]]
            floor_top[(r.get("building"), r.get("floor"), r.get("kind", "room"))] = max(
                floor_top[(r.get("building"), r.get("floor"), r.get("kind", "room"))], max(zs))
        multi = len({k[:2] for k in floor_top}) > 1
        building_top: dict = defaultdict(lambda: -np.inf)
        for r in self.regions:
            c = np.asarray(r["center"], float)
            z = floor_top[(r.get("building"), r.get("floor"), r.get("kind", "room"))] if multi else top
            self.pos[r["key"]] = np.array([c[0], c[1], z + region_dz])
            self.color[r["key"]] = self.region_color[r["id"]]
            building_top[r.get("building")] = max(building_top[r.get("building")], z)
        for b in self.buildings:
            c = np.asarray(b["center"], float)
            z = building_top.get(b["key"], top) if multi else top
            self.pos[b["key"]] = np.array([c[0], c[1], (z if np.isfinite(z) else top) + building_dz])
            self.color[b["key"]] = (255, 255, 255)
        self.t0 = min([a["stamp"] for a in self.agents] + [o["first_seen"] for o in self.objects] + [0.0] * (not self.agents))

    def edge_segments(self) -> dict[str, list[np.ndarray]]:
        segs = defaultdict(list)
        for e in self.edges:
            a, b = self.pos.get(e["source"]), self.pos.get(e["target"])
            if a is not None and b is not None:
                segs[e["relation"]].append(np.stack([a, b]))
        return segs

    def region_text(self, r: dict) -> str:
        kind = r.get("kind", "room")
        where = (f"[{kind}]" if kind != "room" else
                 f"[{r.get('building')} F{r.get('floor')}]" if len(self.buildings) > 1 or r.get("floor") else "")
        return f"R{r['id']} {where} {r.get('label') or ''}".replace("  ", " ").strip()


def show_rerun(lay: Layout, save: str | None) -> None:
    import rerun as rr

    rr.init("topo_scene_graph", spawn=save is None)
    if save:
        rr.save(save)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    for rel, segs in lay.edge_segments().items():
        rr.log(f"world/edges/{rel}", rr.LineStrips3D(segs, colors=RELATION_COLORS.get(rel, (200, 200, 200)),
                                                     radii=0.02 if rel in ("near", "on") else 0.01), static=True)
    rr.log("world/regions", rr.Points3D([lay.pos[r["key"]] for r in lay.regions], radii=0.35,
                                        colors=[lay.color[r["key"]] for r in lay.regions],
                                        labels=[lay.region_text(r) for r in lay.regions]), static=True)
    rr.log("world/buildings", rr.Points3D([lay.pos[b["key"]] for b in lay.buildings], radii=0.5,
                                          colors=(255, 255, 255), labels=[b["key"] for b in lay.buildings]), static=True)
    for r in lay.regions:
        rr.log(f"world/regions/R{r['id']}", rr.TextDocument(
            f"# {lay.region_text(r)}\n\n{r.get('description') or ''}\n\n{r.get('reasoning') or ''}",
            media_type=rr.MediaType.MARKDOWN), static=True)

    traj = np.array([a["position"] for a in lay.agents], float)
    if len(traj) > 1:
        rr.log("world/trajectory", rr.LineStrips3D([traj], colors=(255, 255, 0), radii=0.03), static=True)

    events = [(o["first_seen"], "object", o) for o in lay.objects] + [(p["first_seen"], "place", p) for p in lay.places]
    events += [(a["stamp"], "agent", a) for a in lay.agents]
    for stamp, kind, node in sorted(events, key=lambda e: e[0]):
        rr.set_time_seconds("t", stamp - lay.t0)
        if kind == "object":
            half = np.asarray(node.get("size") or [0.3, 0.3, 0.3], float) / 2
            rr.log(f"world/objects/{node['key']}", rr.Boxes3D(
                centers=[lay.pos[node["key"]]], half_sizes=[half], colors=[lay.color[node["key"]]],
                labels=[f"{node['name']} {node['key']} R{node.get('region')}"]))
        elif kind == "place":
            rr.log(f"world/places/{node['key']}", rr.Points3D([lay.pos[node["key"]]], radii=0.12,
                                                              colors=[lay.color[node["key"]]]))
        else:
            rr.log("world/robot", rr.Points3D([node["position"]], radii=0.25, colors=(255, 0, 0)))
            region = next((p.get("region") for p in lay.places if p["id"] == node.get("place")), None)
            rr.log("status", rr.TextLog(f"robot at place{node.get('place')} region R{region}"))


def show_html(lay: Layout, path: str) -> None:
    import plotly.graph_objects as go

    def rgb(c):
        return f"rgb{tuple(c)}"

    traces = []
    for rel, segs in lay.edge_segments().items():
        xyz = np.concatenate([np.vstack([s, [np.nan] * 3]) for s in segs])
        traces.append(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines", name=f"edge: {rel}",
                                   line=dict(color=rgb(RELATION_COLORS.get(rel, (200, 200, 200))),
                                             width=4 if rel in ("near", "on", "adjacent") else 2),
                                   hoverinfo="skip", visible=True if rel not in ("reachable_from",) else "legendonly"))
    traj = np.array([a["position"] for a in lay.agents], float)
    if len(traj):
        traces.append(go.Scatter3d(x=traj[:, 0], y=traj[:, 1], z=traj[:, 2], mode="lines", name="trajectory",
                                   line=dict(color="gold", width=5),
                                   text=[f"t={a['stamp'] - lay.t0:.1f}s place{a.get('place')}" for a in lay.agents],
                                   hoverinfo="text"))
    p = np.array([lay.pos[x["key"]] for x in lay.places])
    traces.append(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", name="places",
                               marker=dict(size=4, color=[rgb(lay.color[x["key"]]) for x in lay.places]),
                               text=[f"{x['key']} R{x.get('region')} visits={x.get('visits')}" for x in lay.places],
                               hoverinfo="text"))
    o = np.array([lay.pos[x["key"]] for x in lay.objects])
    traces.append(go.Scatter3d(
        x=o[:, 0], y=o[:, 1], z=o[:, 2], mode="markers", name="objects",
        marker=dict(size=5, symbol="diamond", color=[rgb(lay.color[x["key"]]) for x in lay.objects],
                    line=dict(color="black", width=1)),
        text=[f"<b>{x['name']}</b> {x['key']}<br>R{x.get('region')} {x.get('center_source', '')}<br>"
              f"seen t={x['first_seen'] - lay.t0:.0f}-{x['last_seen'] - lay.t0:.0f}s n_obs={x.get('n_obs')}"
              for x in lay.objects], hoverinfo="text"))
    r = np.array([lay.pos[x["key"]] for x in lay.regions])
    traces.append(go.Scatter3d(
        x=r[:, 0], y=r[:, 1], z=r[:, 2], mode="markers+text", name="regions",
        marker=dict(size=10, color=[rgb(lay.color[x["key"]]) for x in lay.regions]),
        text=[lay.region_text(x) for x in lay.regions], textposition="top center",
        hovertext=[f"<b>{lay.region_text(x)}</b><br>{x.get('description') or ''}<br>"
                   f"{len(x.get('objects', []))} objects, {len(x.get('places', []))} places" for x in lay.regions],
        hoverinfo="text"))
    b = np.array([lay.pos[x["key"]] for x in lay.buildings]).reshape(-1, 3)
    traces.append(go.Scatter3d(x=b[:, 0], y=b[:, 1], z=b[:, 2], mode="markers+text", name="building",
                               marker=dict(size=12, color="black"), text=[x["key"] for x in lay.buildings]))
    fig = go.Figure(traces)
    fig.update_layout(scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                      margin=dict(l=0, r=0, t=30, b=0), title="topo scene graph (objects / places / regions / building)",
                      legend=dict(itemsizing="constant"))
    fig.write_html(path, include_plotlyjs=True)


def hierarchy_text(graph: dict, max_objects: int = 8) -> str:
    """Building → floor → region → objects as an indented tree, plus the region links."""
    layers = graph["layers"]
    objects = {o["key"]: o for o in layers.get("objects", [])}
    regions = layers.get("regions", [])
    edges = graph.get("edges", [])
    n_edges = Counter(e["relation"] for e in edges)
    lines = [f"scene graph: {len(layers.get('buildings', []))} building/area, {len(layers.get('floors', []))} floors, "
             f"{len(regions)} regions, {len(layers.get('places', []))} places, {len(objects)} objects, "
             f"{len(layers.get('agents', []))} agent keyframes; edges {dict(n_edges)}"]

    def region_line(r: dict, indent: str) -> list[str]:
        objs = [objects[k] for k in r.get("objects", []) if k in objects]
        kind = r.get("kind", "room")
        head = f"{indent}region{r['id']}" + (f" [{kind}]" if kind != "room" else "") \
            + (f" {r['label']}" if r.get("label") else "") \
            + f"  ({len(r.get('places', []))} places, {len(objs)} objects)"
        out = [head]
        counts = Counter(o["name"] for o in objs)
        if counts:
            out.append(f"{indent}   classes: " + ", ".join(f"{n}×{c}" for c, n in counts.most_common()))
        absent = [o["key"] for o in objs if o.get("status") == "absent"]
        listed = ", ".join(f"{o['key']}:{o['name']}" for o in objs[:max_objects])
        if listed:
            out.append(f"{indent}   objects: {listed}" + (" …" if len(objs) > max_objects else ""))
        if absent:
            out.append(f"{indent}   absent: {', '.join(absent)}")
        links = [f"region{x['region']}({x['kind']})" for x in r.get("links", [])]
        plain = [f"region{a}" for a in r.get("adjacent", []) if f"region{a}" not in " ".join(links)]
        if links or plain:
            out.append(f"{indent}   ↔ " + ", ".join(links + plain))
        return out

    by_building = defaultdict(list)
    for r in regions:
        by_building[r.get("building", "building0")].append(r)
    for b in layers.get("buildings", []) or [{"key": "building0", "kind": "building"}]:
        regs = by_building.pop(b["key"], [])
        extra = f", entrances {b['entrances']}" if b.get("entrances") else ""
        lines.append(f"{b['key']} [{b.get('kind', 'building')}]  floors {b.get('floors', [])}{extra}")
        if b.get("kind") == "outdoor":
            for r in regs:
                lines += region_line(r, "  ")
            continue
        floors = sorted({r.get("floor") for r in regs if r.get("kind", "room") == "room"}, key=lambda f: (f is None, f))
        for f in floors:
            lines.append(f"  floor {f}")
            for r in regs:
                if r.get("kind", "room") == "room" and r.get("floor") == f:
                    lines += region_line(r, "    ")
        for r in regs:
            if r.get("kind", "room") != "room":
                lines.append(f"  connector floors {r.get('floors')}")
                lines += region_line(r, "    ")
    for key, regs in by_building.items():
        lines.append(f"{key}")
        for r in regs:
            lines += region_line(r, "  ")
    if graph.get("transitions"):
        lines.append("transitions: " + ", ".join(f"t={t['stamp']:.0f} {t['kind']}" for t in graph["transitions"]))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="scene4d directory (topo_graph.json)")
    parser.add_argument("--tree", action="store_true", help="print the building/floor/region/object hierarchy")
    parser.add_argument("--html", default=None, help="write a plotly HTML file instead of opening rerun")
    parser.add_argument("--save", default=None, help="write a rerun .rrd file instead of spawning the viewer")
    parser.add_argument("--region_dz", type=float, default=4.0)
    parser.add_argument("--building_dz", type=float, default=7.0)
    args = parser.parse_args()

    scene_dir = Path(args.scene)
    graph = json.loads((scene_dir / "topo_graph.json").read_text(encoding="utf-8"))
    if "agents" not in graph["layers"]:
        from scene_memory.mapping.topo_graph import regraph

        graph = regraph(graph, load_scene(scene_dir).get("objects", []))
    if args.tree:
        print(hierarchy_text(graph))
        if not (args.html or args.save):
            return
    lay = Layout(graph, args.region_dz, args.building_dz)
    if args.html:
        show_html(lay, args.html)
        print(f"wrote {args.html}")
    else:
        show_rerun(lay, args.save)
        if args.save:
            print(f"wrote {args.save}; open with: rerun {args.save}")


if __name__ == "__main__":
    main()
