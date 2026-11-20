"""View layer (FSR-VLN's view nodes): panorama keyframes with pose and SigLIP sector embeddings.

Keyframes are taken from the session's stitched panorama topic wherever the robot moved ``spacing_m`` /
turned ``turn_deg`` / waited ``max_dt_s`` since the last one; their camera pose comes from the scene's own
trajectory (``agents`` = ``T_odom_imu`` of the mapping run) and the rig calibration, so the views live in
the scene frame (= the lidar map's frame after ``stvln.sh map`` / ``map_offline``). Each panorama is split
into ``sectors`` overlapping horizontal tiles (``sector_deg`` wide, the middle half of its height) that are
embedded with SigLIP for text retrieval (``query.view_search``).

Writes ``<scene>/keyframes/kf0000.jpg ...``, ``<scene>/keyframes.json`` and ``<scene>/view_index.npz``.

    python3 -m scene_memory.mapping.view_layer data/maps/office/office_scene --session data/sessions/office_rec_proc
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from scene_memory import paths

KEYFRAMES_FILE = "keyframes.json"
INDEX_FILE = "view_index.npz"
EQUIRECT_TOPIC = "/insta360/equirect/image/compressed"


def _quat_to_R(q) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(q).as_matrix()


class Trajectory:
    """``T_odom_imu`` at any stamp inside the scene's agent keyframes (linear + slerp)."""

    def __init__(self, agents: list[dict]):
        from scipy.spatial.transform import Rotation, Slerp

        agents = sorted(agents, key=lambda a: a["stamp"])
        self.t = np.array([a["stamp"] for a in agents], float)
        self.p = np.array([a["position"] for a in agents], float)
        self.slerp = Slerp(self.t, Rotation.from_quat([a["orientation"] for a in agents])) if len(agents) > 1 else None

    def at(self, t: float, max_gap_s: float = 2.0) -> np.ndarray | None:
        if self.slerp is None or not self.t[0] <= t <= self.t[-1]:
            return None
        i = int(np.searchsorted(self.t, t))
        if i > 0 and self.t[i] - self.t[i - 1] > max_gap_s:
            return None
        T = np.eye(4)
        T[:3, :3] = self.slerp([t]).as_matrix()[0]
        T[:3, 3] = [np.interp(t, self.t, self.p[:, k]) for k in range(3)]
        return T


def sector_boxes(width: int, height: int, sectors: int = 8, sector_deg: float = 90.0) -> list[tuple[int, int, int, int]]:
    """``(u0, v0, u1, v1)`` per sector; ``u`` may exceed the width (wraps around the seam)."""
    w = int(round(width * sector_deg / 360.0))
    return [(int(round(k * width / sectors - w / 2)), height // 4, int(round(k * width / sectors + w / 2)),
             3 * height // 4) for k in range(sectors)]


def crop_wrapped(image, box):
    """PIL crop that wraps horizontally around the panorama seam."""
    from PIL import Image

    u0, v0, u1, v1 = box
    w = image.width
    if 0 <= u0 and u1 <= w:
        return image.crop((u0, v0, u1, v1))
    out = Image.new("RGB", (u1 - u0, v1 - v0))
    left = u0 % w
    first = min(w - left, u1 - u0)
    out.paste(image.crop((left, v0, left + first, v1)), (0, 0))
    x = first
    while x < u1 - u0:
        n = min(w, u1 - u0 - x)
        out.paste(image.crop((0, v0, n, v1)), (x, 0))
        x += n
    return out


def select_keyframes(stamps: np.ndarray, traj: Trajectory, T_imu_da360: np.ndarray, spacing_m: float = 1.0,
                     turn_deg: float = 45.0, max_dt_s: float = 10.0) -> list[tuple[int, np.ndarray]]:
    """Indices of the frames to keep and their ``T_odom_da360``."""
    keep, last = [], None
    for i, t in enumerate(stamps):
        T = traj.at(float(t))
        if T is None:
            continue
        T = T @ T_imu_da360
        if last is not None:
            moved = np.linalg.norm(T[:3, 3] - last[1][:3, 3])
            turn = np.degrees(np.arccos(np.clip((np.trace(last[1][:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
            if moved < spacing_m and turn < turn_deg and t - last[0] < max_dt_s:
                continue
        keep.append((i, T))
        last = (float(t), T)
    return keep


def build(scene: Path, session: Path, encoder=None, spacing_m: float = 1.0, turn_deg: float = 45.0,
          max_dt_s: float = 10.0, sectors: int = 8, sector_deg: float = 90.0, width: int = 1920,
          calib_dir: str | Path = paths.CALIB, lio_config: str | Path = paths.LIO_CONFIG,
          topic: str = EQUIRECT_TOPIC, log=print) -> dict:
    from io import BytesIO

    from PIL import Image

    from scene_memory.common.calib_io import load_rig_calibration
    from scene_memory.mapping.scene_graph import pose_frame_to_da360

    sys.path.insert(0, str(paths.WS_ROOT / "src" / "tools" / "collection_data" / "collection_data"))
    from replay_session import bag_messages

    graph = json.loads((scene / "topo_graph.json").read_text(encoding="utf-8"))
    traj = Trajectory(graph["layers"].get("agents", []))
    T_imu_da360 = pose_frame_to_da360("lio", "", lio_config, load_rig_calibration(calib_dir))

    t0 = time.perf_counter()
    frames = [(t_ns * 1e-9, msg) for t_ns, _, msg, _ in bag_messages(session / "bag", [topic])]
    stamps = np.array([msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec for _, msg in frames])
    keep = select_keyframes(stamps, traj, T_imu_da360, spacing_m, turn_deg, max_dt_s)
    log(f"[views] {len(frames)} panoramas, {len(keep)} keyframes ({time.perf_counter() - t0:.1f} s)")

    out_dir = scene / "keyframes"
    out_dir.mkdir(exist_ok=True)
    keyframes, crops = [], []
    for k, (i, T) in enumerate(keep):
        image = Image.open(BytesIO(bytes(frames[i][1].data))).convert("RGB")
        if image.width != width:
            image = image.resize((width, width // 2))
        name = f"kf{k:04d}.jpg"
        image.save(out_dir / name, quality=88)
        boxes = sector_boxes(image.width, image.height, sectors, sector_deg)
        crops += [crop_wrapped(image, b) for b in boxes]
        keyframes.append({"id": k, "file": f"keyframes/{name}", "stamp": float(stamps[i]),
                          "position": [round(float(v), 3) for v in T[:3, 3]],
                          "T_odom_da360": np.round(T, 6).tolist(), "width": image.width, "height": image.height,
                          "sectors": [list(b) for b in boxes]})
    if encoder is None:
        from scene_memory.query.semantic_index import SiglipEncoder

        encoder = SiglipEncoder()
    t0 = time.perf_counter()
    emb = encoder.images(crops).astype(np.float32) if crops else np.zeros((0, encoder.dim), np.float32)
    log(f"[views] SigLIP on {len(crops)} sectors ({time.perf_counter() - t0:.1f} s)")
    kf = np.repeat(np.arange(len(keyframes)), sectors)
    sec = np.tile(np.arange(sectors), len(keyframes))
    np.savez(scene / INDEX_FILE, emb=emb, keyframe=kf, sector=sec)
    meta = {"frame_id": graph.get("frame_id", "odom"), "session": str(session), "topic": topic,
            "sectors": sectors, "sector_deg": sector_deg, "keyframes": keyframes}
    (scene / KEYFRAMES_FILE).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="scene dir with topo_graph.json (e.g. data/maps/<name>/<name>_scene)")
    ap.add_argument("--session", required=True, help="session whose bag holds the panoramas (the derived _proc)")
    ap.add_argument("--spacing_m", type=float, default=1.0)
    ap.add_argument("--turn_deg", type=float, default=45.0)
    ap.add_argument("--max_dt_s", type=float, default=10.0)
    ap.add_argument("--sectors", type=int, default=8)
    ap.add_argument("--sector_deg", type=float, default=90.0)
    ap.add_argument("--calib_dir", default=str(paths.CALIB))
    ap.add_argument("--lio_config", default=str(paths.LIO_CONFIG))
    args = ap.parse_args()
    meta = build(Path(args.scene), Path(args.session), spacing_m=args.spacing_m, turn_deg=args.turn_deg,
                 max_dt_s=args.max_dt_s, sectors=args.sectors, sector_deg=args.sector_deg,
                 calib_dir=args.calib_dir, lio_config=args.lio_config)
    print(f"[views] wrote {len(meta['keyframes'])} keyframes to {Path(args.scene) / KEYFRAMES_FILE}")


if __name__ == "__main__":
    main()
