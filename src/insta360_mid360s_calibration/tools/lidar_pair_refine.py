#!/usr/bin/env python3
"""Fine-tune rear_lidar -> front_lidar from the installation extrinsic.

Each lidar contributes one static frame and its built-in IMU gravity. Each
cloud is rotated so its own gravity points along -Z (roll and pitch removed).
The installation seed only carries yaw and translation. Several ICP passes
then run from that seed, wide correspondence first, then tighter.

    p_front_level = R_level_front * p_front     from the front lidar IMU
    p_rear_level  = R_level_rear  * p_rear      from the rear lidar IMU
    p_front_level = T_front_rear_level * p_rear_level

T_front_rear_level starts from results/lidar_extrinsic.initial.json.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import open3d as o3d

import extrinsic_io


def inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def mat4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(axis))
    if n < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)
    x, y, z = axis / n
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    s, c = math.sin(angle), math.cos(angle)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = np.cross(a, b)
    s = float(np.linalg.norm(c))
    d = float(np.dot(a, b))
    if s < 1e-8:
        if d > 0.0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        return rodrigues(np.cross(a, helper), math.pi)
    return rodrigues(c / s, math.atan2(s, d))


def rotation_angle_deg(R: np.ndarray) -> float:
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(c))


def apply_T(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ xyz.T).T + t


def finite_range(xyz: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    ok = np.isfinite(xyz).all(axis=1)
    r = np.linalg.norm(xyz, axis=1)
    return xyz[ok & (r > near_m) & (r < far_m)]


def voxel_down(xyz: np.ndarray, voxel: float) -> np.ndarray:
    if len(xyz) == 0:
        return xyz
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    down = pcd.voxel_down_sample(voxel)
    return np.asarray(down.points, dtype=np.float64)


def _as_pcd(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    return pcd


# Wide correspondence first, then tighter passes on the same seed chain.
# estimation, voxel_m, max_correspondence_m, iterations
STAGES = (
    ("point_to_point", 0.30, 4.00, 50),
    ("point_to_point", 0.15, 2.00, 40),
    ("point_to_plane", 0.08, 0.80, 40),
    ("point_to_plane", 0.04, 0.30, 40),
    ("point_to_plane", 0.02, 0.10, 30),
)


def _estimate(name: str):
    if name == "point_to_point":
        return o3d.pipelines.registration.TransformationEstimationPointToPoint()
    if name == "point_to_plane":
        return o3d.pipelines.registration.TransformationEstimationPointToPlane()
    raise ValueError(name)


def register_stage(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    T_init: np.ndarray,
    estimation: str,
    max_distance: float,
    iterations: int,
) -> Tuple[np.ndarray, float, float]:
    src = _as_pcd(source_xyz)
    tgt = _as_pcd(target_xyz)
    if estimation == "point_to_plane":
        radius = max(max_distance * 2.0, 0.15)
        param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40)
        src.estimate_normals(param)
        tgt.estimate_normals(param)
    reg = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        max_distance,
        T_init,
        _estimate(estimation),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations),
    )
    return np.asarray(reg.transformation, dtype=float), float(reg.fitness), float(reg.inlier_rmse)


def overlap_at(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    T: np.ndarray,
    distance: float,
) -> Tuple[float, float]:
    reg = o3d.pipelines.registration.evaluate_registration(
        _as_pcd(source_xyz),
        _as_pcd(target_xyz),
        distance,
        T,
    )
    return float(reg.fitness), float(reg.inlier_rmse)


def _down_or_raw(xyz: np.ndarray, voxel: float) -> np.ndarray:
    down = voxel_down(xyz, voxel)
    if len(down) < 200:
        return xyz
    return down


def refine_pair(
    target_xyz: np.ndarray,
    source_xyz: np.ndarray,
    T_seed: np.ndarray,
    *,
    near_m: float = 0.5,
    far_m: float = 40.0,
) -> Dict[str, Any]:
    tgt_raw = finite_range(target_xyz, near_m, far_m)
    src_raw = finite_range(source_xyz, near_m, far_m)
    if len(tgt_raw) < 200 or len(src_raw) < 200:
        raise RuntimeError(
            f"need at least 200 points after range crop, got target={len(tgt_raw)} source={len(src_raw)}"
        )
    T = np.array(T_seed, dtype=float)
    T_seed = np.array(T_seed, dtype=float)
    stages = []
    for estimation, voxel, max_distance, iterations in STAGES:
        tgt = _down_or_raw(tgt_raw, voxel)
        src = _down_or_raw(src_raw, voxel)
        T, fitness, rmse = register_stage(src, tgt, T, estimation, max_distance, iterations)
        dT = T @ inv(T_seed)
        row = {
            "estimation": estimation,
            "voxel_m": voxel,
            "max_correspondence_m": max_distance,
            "iterations": iterations,
            "fitness": fitness,
            "inlier_rmse_m": rmse,
            "target_points": int(len(tgt)),
            "source_points": int(len(src)),
            "rotation_delta_deg": rotation_angle_deg(dT[:3, :3]),
            "translation_delta_mm": (dT[:3, 3] * 1000.0).tolist(),
        }
        stages.append(row)
        print(
            f"stage {len(stages)} {estimation}  voxel={voxel:.2f}  "
            f"corr_m={max_distance:.2f}  fitness={fitness:.4f}  rmse_m={rmse:.4f}  "
            f"delta_deg={row['rotation_delta_deg']:.3f}  "
            f"delta_mm={np.round(row['translation_delta_mm'], 2).tolist()}"
        )
    tgt = _down_or_raw(tgt_raw, 0.05)
    src = _down_or_raw(src_raw, 0.05)
    fit10, rmse10 = overlap_at(src, tgt, T, 0.10)
    dT = T @ inv(T_seed)
    return {
        "T": T,
        "fitness": fit10,
        "inlier_rmse_m": rmse10,
        "fitness_0_10m": fit10,
        "rotation_delta_deg": rotation_angle_deg(dT[:3, :3]),
        "translation_delta_mm": (dT[:3, 3] * 1000.0).tolist(),
        "target_points": int(len(tgt_raw)),
        "source_points": int(len(src_raw)),
        "stages": stages,
    }


def load_ply(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points, dtype=np.float64)
    if len(xyz) == 0:
        raise SystemExit(f"no points in {path}")
    return xyz


def as_T(value: Any) -> np.ndarray:
    T = np.array(value, dtype=float)
    if T.shape == (3, 3):
        out = np.eye(4)
        out[:3, :3] = T
        return out
    if T.shape != (4, 4):
        raise SystemExit(f"transform has shape {T.shape}")
    return T


def body_from_level(pair: Dict[str, Any], T_level: np.ndarray) -> np.ndarray:
    """p_front = T_front_rear * p_rear from the leveled pose and both leveling rotations."""
    return inv(as_T(pair["R_level_front"])) @ T_level @ as_T(pair["R_level_rear"])


def initial_extrinsic_path(root: Path) -> Path:
    return root / "results" / extrinsic_io.INITIAL_FILE


def extrinsic_path(root: Path) -> Path:
    return root / "results" / extrinsic_io.EXTRINSIC_FILE


def ensure_initial_extrinsic(root: Path) -> Path:
    """Keep an installation copy that refine never overwrites.

    If the copy is missing, rebuild it from the seed stored in the working file.
    """
    dest = initial_extrinsic_path(root)
    if dest.is_file():
        return dest
    working = extrinsic_path(root)
    if not working.is_file():
        raise SystemExit(f"missing {working}; run ./calibrate.sh initial")
    _pair, _current, T_seed = load_pair(working)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    extrinsic_io.write(dest, extrinsic_io.installation_doc(extrinsic_io.load_rig(root), T_seed, now))
    print(f"wrote {dest}  {extrinsic_io.pose(T_seed)}")
    return dest


def load_pair(path: Path) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Return (doc, leveled rear-in-front pose, installation seed)."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    if "T_front_rear_level_seed" not in doc:
        raise SystemExit(f"{path} has no T_front_rear_level_seed; run ./calibrate.sh initial")
    T_seed = as_T(doc["T_front_rear_level_seed"])
    T = as_T(doc.get("T_front_rear_level", doc["T_front_rear_level_seed"]))
    doc["R_level_front"] = doc.get("T_level_front", np.eye(4).tolist())
    doc.setdefault("R_level_rear", np.eye(4).tolist())
    return doc, T, T_seed


def write_results(
    root: Path,
    pair: Dict[str, Any],
    T: np.ndarray,
    T_seed: np.ndarray,
    status: str,
    method: str,
    metrics: Dict[str, Any],
) -> Path:
    """Write results/lidar_extrinsic.json.

    T is the rear lidar in the front lidar after both are leveled.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rig = extrinsic_io.load_rig(root)
    R_front = as_T(pair["R_level_front"])
    R_rear = as_T(pair["R_level_rear"])
    T_front_rear = body_from_level(pair, T)
    T_level_rear = T @ R_rear
    dT = T @ inv(T_seed)
    leveling = dict(pair.get("leveling") or {})
    leveling["front"] = {**leveling.get("front", {}), **_level_angles(R_front)}
    leveling["rear"] = {**leveling.get("rear", {}), **_level_angles(R_rear)}
    quality = {
        "fitness_within_0_10m": round(float(metrics["fitness"]), 4),
        "inlier_rmse_mm": round(float(metrics["inlier_rmse_m"]) * 1000.0, 1),
    }
    if "scene" in metrics:
        quality["scene"] = metrics["scene"]
    out = {
        "schema": extrinsic_io.SCHEMA,
        "status": status,
        "updated_utc": now,
        "method": method,
        "lidars": extrinsic_io.lidars_block(rig),
        "convention": extrinsic_io.CONVENTION,
        "result": {
            "T_front_rear": {
                **extrinsic_io.pose(T_front_rear),
                "meaning": "rear lidar in the front lidar frame",
            },
            "T_level_front": {
                **extrinsic_io.pose(R_front),
                "meaning": f"front lidar into {extrinsic_io.LEVEL_FRAME} (gravity-aligned); apply uses it",
            },
            "T_level_rear": {
                **extrinsic_io.pose(T_level_rear),
                "meaning": f"rear lidar into {extrinsic_io.LEVEL_FRAME}; apply uses it",
            },
            "T_front_rear_level": {
                **extrinsic_io.pose(T),
                "meaning": "rear in front after both are leveled; this is what ICP refines",
            },
            "installation_seed": {
                **extrinsic_io.pose(T_seed),
                "meaning": f"T_front_rear_level from {extrinsic_io.INITIAL_FILE}",
            },
            "refined_minus_seed": {
                "rotation_deg": round(rotation_angle_deg(dT[:3, :3]), 3),
                **extrinsic_io.pose(dT),
            },
        },
        "leveling": leveling,
        "quality": quality,
        "T_front_rear": extrinsic_io.matrix(T_front_rear),
        "T_level_front": extrinsic_io.matrix(R_front),
        "T_level_rear": extrinsic_io.matrix(T_level_rear),
        "T_front_rear_level": extrinsic_io.matrix(T),
        "T_front_rear_level_seed": extrinsic_io.matrix(T_seed),
        "R_level_rear": extrinsic_io.matrix(R_rear),
    }
    path = extrinsic_path(root)
    extrinsic_io.write(path, out)
    return path


def _level_angles(R: np.ndarray) -> Dict[str, float]:
    roll, pitch, _yaw = extrinsic_io.rpy_deg(R)
    return {"roll_deg": roll, "pitch_deg": pitch}


def accept_refine(metrics: Dict[str, Any]) -> bool:
    return float(metrics["fitness"]) >= 0.5 and float(metrics["inlier_rmse_m"]) <= 0.05


def imu_path(cloud: Path) -> Path:
    return cloud.with_name(f"{cloud.stem}.imu.json")


def gravity_level(cloud: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Rotation that turns the IMU up vector of this lidar into +Z.

    The rotation axis is horizontal, so it removes roll and pitch only.
    """
    path = imu_path(cloud)
    if not path.is_file():
        raise SystemExit(
            f"missing {path}. Recapture this scene with ./calibrate.sh refine-auto <new scene>; "
            "it saves the IMU gravity next to the PCD."
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    up = np.array(doc["acc_mean"], dtype=float)
    up /= np.linalg.norm(up)
    R = rotation_between(up, np.array([0.0, 0.0, 1.0]))
    info = {
        "imu_samples": doc.get("samples"),
        "up_in_lidar": [round(float(v), 5) for v in up],
        "z_tilt_from_up_deg": round(math.degrees(math.acos(float(np.clip(up[2], -1.0, 1.0)))), 3),
    }
    return mat4(R, np.zeros(3)), info


def run_files(root: Path, front_cloud: Path, rear_cloud: Path) -> int:
    initial_path = ensure_initial_extrinsic(root)
    pair, _installed, T_seed = load_pair(initial_path)
    print(
        f"seed {initial_path.name}  rpy_deg={extrinsic_io.rpy_deg(T_seed)}  "
        f"t_mm={extrinsic_io.pose(T_seed)['t_mm']}"
    )
    R_front, g_front = gravity_level(front_cloud)
    R_rear, g_rear = gravity_level(rear_cloud)
    for name, g, R in ((front_cloud.stem, g_front, R_front), (rear_cloud.stem, g_rear, R_rear)):
        a = _level_angles(R)
        print(
            f"imu {name}  up={np.round(g['up_in_lidar'], 4).tolist()}  "
            f"z_tilt_deg={g['z_tilt_from_up_deg']:.2f}  "
            f"level roll={a['roll_deg']:.2f} pitch={a['pitch_deg']:.2f}"
        )
    pair = dict(pair)
    pair["R_level_front"] = R_front.tolist()
    pair["R_level_rear"] = R_rear.tolist()
    pair["leveling"] = {
        "source": "MID-360S built-in IMU gravity, averaged while still; roll_deg and pitch_deg are the rpy of each leveling rotation",
        "front": g_front,
        "rear": g_rear,
    }
    front_xyz = apply_T(R_front, load_ply(front_cloud))
    rear_xyz = apply_T(R_rear, load_ply(rear_cloud))
    metrics = refine_pair(front_xyz, rear_xyz, T_seed)
    metrics["scene"] = front_cloud.parent.name
    T = metrics.pop("T")
    summary = (
        f"fitness_0_10m={metrics['fitness']:.4f}  rmse_m={metrics['inlier_rmse_m']:.4f}  "
        f"delta_deg={metrics['rotation_delta_deg']:.3f}  "
        f"delta_mm={np.round(metrics['translation_delta_mm'], 2).tolist()}"
    )
    print(
        f"result T_front_rear_level rpy_deg={extrinsic_io.rpy_deg(T)}  "
        f"t_mm={extrinsic_io.pose(T)['t_mm']}"
    )
    method = (
        "IMU gravity leveling per lidar, then one static PCD per lidar with multi-stage "
        f"point-to-point and point-to-plane ICP; seed yaw and translation from {extrinsic_io.INITIAL_FILE}"
    )
    path = write_results(root, pair, T, T_seed, "auto_icp", method, metrics)
    print(f"wrote {path}  status=auto_icp  {summary}")
    if not accept_refine(metrics):
        print(
            "WARNING: fewer than half the points land within 0.10 m, or the inlier RMSE is above 0.05 m. "
            "The result was still written. The installation seed file was not changed."
        )
    return 0


def _scene(seed: int, tilt_deg: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs = np.linspace(-8.0, 8.0, 90)
    ys = np.linspace(-8.0, 8.0, 90)
    xx, yy = np.meshgrid(xs, ys)
    ground = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
    zs = np.linspace(0.0, 3.0, 25)
    ys = np.linspace(-6.0, 6.0, 70)
    zz, yy = np.meshgrid(zs, ys)
    wall_x = np.stack([np.full(zz.size, 5.0), yy.ravel(), zz.ravel()], axis=1)
    xs = np.linspace(-6.0, 6.0, 70)
    zz, xx = np.meshgrid(zs, xs)
    wall_y = np.stack([xx.ravel(), np.full(xx.size, -4.0), zz.ravel()], axis=1)
    world = np.vstack([ground, wall_x, wall_y])
    world += rng.normal(0.0, 0.005, world.shape)
    R = rodrigues(np.array([1.0, 0.0, 0.0]), math.radians(tilt_deg))
    return (R.T @ world.T).T


def _check_recovery(name: str, target: np.ndarray, T_seed: np.ndarray, T_true: np.ndarray) -> None:
    source = apply_T(inv(T_true), target)
    metrics = refine_pair(target, source, T_seed)
    err = metrics["T"] @ inv(T_true)
    rot = rotation_angle_deg(err[:3, :3])
    trans = float(np.linalg.norm(err[:3, 3]) * 1000.0)
    print(
        f"self-test {name} rot_err_deg={rot:.3f} trans_err_mm={trans:.2f} "
        f"fitness_0_10m={metrics['fitness']:.3f}"
    )
    if rot > 0.5 or trans > 20.0 or metrics["fitness"] < 0.5:
        raise SystemExit(f"self-test failed: {name}")


def self_test() -> None:
    target = _scene(1, 0.0)
    T_seed = mat4(rodrigues(np.array([0.0, 0.0, 1.0]), math.pi), np.array([-0.2475, 0.0, 0.0]))
    _check_recovery("identity", target, T_seed, T_seed)
    R = rodrigues(np.array([0.2, 0.9, 0.1]), math.radians(25.0))
    T_true = mat4(R, np.array([0.20, -0.15, 0.08])) @ T_seed
    _check_recovery("25deg", target, T_seed, T_true)
    print("self-test ok")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--scene", default="")
    ap.add_argument("--front-cloud", type=Path, default=None)
    ap.add_argument("--rear-cloud", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.front_cloud and args.rear_cloud:
        front, rear = args.front_cloud, args.rear_cloud
    elif args.scene:
        rig = extrinsic_io.load_rig(root)
        folder = extrinsic_io.clouds_dir(root) / args.scene
        front = folder / f"{rig['front']['id']}.pcd"
        rear = folder / f"{rig['rear']['id']}.pcd"
    else:
        ap.error("pass --scene, or --front-cloud and --rear-cloud, or --self-test")
    if not front.is_file() or not rear.is_file():
        raise SystemExit(f"missing clouds: {front} and {rear}")
    return run_files(root, front, rear)


if __name__ == "__main__":
    raise SystemExit(main())
