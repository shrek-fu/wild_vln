#!/usr/bin/env python3
"""Generate urdf/go2w.urdf + meshes/*.stl from Unitree's MuJoCo Go2-W model (unitree_mujoco/unitree_robots/go2w).

    python3 scripts/mjcf_to_urdf.py ~/00WS/00qiRobotics/18Simulator/unitree_mujoco/unitree_robots/go2w/go2w.xml

Visual-only model for Foxglove / RViz: the leg joints are baked into fixed joints at the SDK stand pose
(go2w_stand_example.cpp _targetPos_2), so robot_state_publisher publishes only static TF and needs no
/joint_states. The root is base_link = body center on the ground (cloud_preprocess base_link_mount.yaml);
the trunk sits STAND_HEIGHT above it.
"""

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

PKG = "go2w_description"
PREFIX = "go2w_"
STAND = {"hip": 0.0, "thigh": 0.67, "calf": -1.3}
MAX_FACES = 12000


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def axis_quat(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    s = math.sin(angle / 2)
    return (math.cos(angle / 2), *(axis * s))


def quat_rpy(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def quat_rotate(q, v):
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return r @ np.asarray(v, float)


def floats(text, default):
    return tuple(float(v) for v in text.split()) if text else default


def fmt(values):
    return " ".join(f"{v:.6g}" for v in values)


def origin(parent, xyz, quat):
    ET.SubElement(parent, "origin", xyz=fmt(xyz), rpy=fmt(quat_rpy(quat)))


def convert_mesh(src: Path, dst: Path):
    mesh = trimesh.load(src, force="mesh")
    if len(mesh.faces) > MAX_FACES:
        import open3d as o3d
        o3 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                       o3d.utility.Vector3iVector(mesh.faces))
        o3 = o3.simplify_quadric_decimation(target_number_of_triangles=MAX_FACES)
        mesh = trimesh.Trimesh(np.asarray(o3.vertices), np.asarray(o3.triangles))
    mesh.export(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mjcf", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()

    root = ET.parse(args.mjcf).getroot()
    meshdir = args.mjcf.parent / root.find("compiler").get("meshdir", ".")
    materials = {m.get("name"): floats(m.get("rgba"), (1, 1, 1, 1)) for m in root.iter("material")}
    meshes = {m.get("name") or Path(m.get("file")).stem: m.get("file") for m in root.iter("mesh")}
    (args.out / "meshes").mkdir(parents=True, exist_ok=True)
    (args.out / "urdf").mkdir(parents=True, exist_ok=True)

    robot = ET.Element("robot", name="go2w")
    for name, rgba in materials.items():
        ET.SubElement(ET.SubElement(robot, "material", name=name), "color", rgba=fmt(rgba))
    ET.SubElement(robot, "link", name="base_link")
    used = set()

    def add_body(body, parent_link, parent_xyz=(0, 0, 0)):
        link = PREFIX + body.get("name").replace("base_link", "trunk")
        el = ET.SubElement(robot, "link", name=link)
        for geom in body.findall("geom"):
            if geom.get("mesh") is None or geom.get("class") not in ("visual", None):
                continue
            mesh = geom.get("mesh")
            used.add(mesh)
            vis = ET.SubElement(el, "visual")
            origin(vis, floats(geom.get("pos"), (0, 0, 0)), floats(geom.get("quat"), (1, 0, 0, 0)))
            ET.SubElement(ET.SubElement(vis, "geometry"), "mesh", filename=f"package://{PKG}/meshes/{mesh}.stl")
            if geom.get("material"):
                ET.SubElement(vis, "material", name=geom.get("material"))
            else:
                ET.SubElement(ET.SubElement(vis, "material", name=f"{mesh}_color"), "color",
                              rgba=fmt(floats(geom.get("rgba"), (0.15, 0.15, 0.15, 1))))

        joint = ET.SubElement(robot, "joint", name=f"{link}_joint", type="fixed")
        ET.SubElement(joint, "parent", link=parent_link)
        ET.SubElement(joint, "child", link=link)
        quat = floats(body.get("quat"), (1, 0, 0, 0))
        mj = body.find("joint")
        if mj is not None:
            kind = mj.get("name").split("_")[1]
            if kind in STAND:
                axis = floats(mj.get("axis"), (1, 0, 0) if kind == "hip" else (0, 1, 0))
                quat = quat_mul(quat, axis_quat(axis, STAND[kind]))
        origin(joint, floats(body.get("pos"), (0, 0, 0)) if parent_link != "base_link" else parent_xyz, quat)
        for child in body.findall("body"):
            add_body(child, link)

    trunk = root.find("worldbody/body")
    stand_height = -min(trunk_leg_z(trunk)) + wheel_radius(meshdir / meshes["wheel"])
    print(f"stand height {stand_height:.3f} m")
    add_body(trunk, "base_link", (0, 0, stand_height))

    for name in sorted(used):
        convert_mesh(meshdir / meshes[name], args.out / "meshes" / f"{name}.stl")
    ET.indent(robot)
    header = ("<?xml version=\"1.0\"?>\n<!-- generated by scripts/mjcf_to_urdf.py from unitree_mujoco go2w.xml; "
              "legs fixed at the SDK stand pose -->\n")
    (args.out / "urdf" / "go2w.urdf").write_text(header + ET.tostring(robot, encoding="unicode") + "\n")
    print(f"wrote {args.out / 'urdf' / 'go2w.urdf'} and {len(used)} meshes")


def trunk_leg_z(trunk):
    """Wheel-center heights below the trunk origin at the stand pose."""
    out = []
    for hip in trunk.findall("body"):
        pos, quat = np.array(floats(hip.get("pos"), (0, 0, 0))), (1, 0, 0, 0)
        body = hip
        while body is not None:
            mj = body.find("joint")
            if body is not hip:
                pos = pos + quat_rotate(quat, floats(body.get("pos"), (0, 0, 0)))
            if mj is not None and mj.get("name").split("_")[1] in STAND:
                kind = mj.get("name").split("_")[1]
                axis = floats(mj.get("axis"), (1, 0, 0) if kind == "hip" else (0, 1, 0))
                quat = quat_mul(quat, axis_quat(axis, STAND[kind]))
            body = body.find("body")
        out.append(pos[2])
    return out


def wheel_radius(path: Path) -> float:
    extents = trimesh.load(path, force="mesh").extents
    return float(max(extents[0], extents[2]) / 2)


if __name__ == "__main__":
    main()
