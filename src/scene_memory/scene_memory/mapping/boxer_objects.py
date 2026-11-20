"""4D object memory: BoxerNet OBBs per object in odom, with time history."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scene_memory.mapping.box_limits import limit_object

# Unit-cube corner signs, ordered to match spark_dsg BoundingBox.corners()
# (bottom face 0-3, top face 4-7) so the same edge list draws both.
_CORNER_SIGNS = np.array(
    [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ],
    dtype=np.float64,
)


def backproject_depth(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    T_world_optical: np.ndarray,
    stride: int = 8,
    max_points: int = 20000,
) -> np.ndarray:
    """Sparse world points from a pinhole Z-depth image (zeros are invalid)."""
    d = np.asarray(depth_m, dtype=np.float32)[::stride, ::stride]
    vv, uu = np.nonzero(np.isfinite(d) & (d > 0))
    if vv.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = d[vv, uu].astype(np.float64)
    u = uu.astype(np.float64) * stride
    v = vv.astype(np.float64) * stride
    cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)
    if cam.shape[0] > max_points:
        cam = cam[:: int(np.ceil(cam.shape[0] / max_points))]
    T = np.asarray(T_world_optical, dtype=np.float64)
    return (cam @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


_SOURCE_RANK = {"prior": 1, "tri": 2, "lidar": 3}


def box_truncated(xyxy, width: int, height: int, margin: float = 3.0) -> bool:
    """2D box touches the image border, so the object is only partly visible."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return x1 <= margin or y1 <= margin or x2 >= width - 1 - margin or y2 >= height - 1 - margin


def fuse_label(belief: dict[int, float], cls_scores: dict[int, float], n_classes: int, accuracy: float) -> dict[int, float]:
    """One Bayesian step ``P(L|z_1:t) ∝ P(z_t|L) P(L|z_1:t-1)`` (SuperMap Eq. 10).

    The detector confusion matrix is ``accuracy`` on the diagonal and uniform
    elsewhere. ``cls_scores`` (every prompt that fired on this object, with its
    score) makes the observation soft: z is a score-weighted mix of classes.
    """
    k = max(int(n_classes), 2, len(belief) + 1, *(c + 1 for c in cls_scores))
    off = (1.0 - accuracy) / (k - 1)
    total = sum(cls_scores.values()) or 1.0
    weights = {c: s / total for c, s in cls_scores.items()}
    # Classes not in ``belief`` share the remaining mass uniformly.
    rest = max(1.0 - sum(belief.values()), 0.0)
    per_unseen = rest / (k - len(belief)) if k > len(belief) else 0.0
    post = {}
    for c in set(belief) | set(weights):
        like = sum(w * (accuracy if c == o else off) for o, w in weights.items())
        post[c] = belief.get(c, per_unseen) * like
    still_unseen = k - len(post)
    norm = sum(post.values()) + still_unseen * per_unseen * off
    floor = 1.0 / (k * 50.0)
    return {c: p / norm for c, p in post.items() if p / norm >= floor}


class BoxerObjectMap:
    """One persistent object per physical instance.

    * Track ids are aliased to objects; a new id near a hidden object whose
      label belief allows the class re-attaches to it instead of creating a duplicate.
    * Size = median of recent non-truncated observations; position = median of
      recent observations; rotation = latest.
    * Label = argmax of a Bayesian class belief over all observations, so a
      single misclassified frame does not relabel the object.
    * Unseen objects are never deleted, only marked ``visible=False``.
    * Existence is a log-odds belief (SuperMap): a detection adds ``l_hit``, a
      frame in which the object should have been visible but was not detected
      (see ``miss``) subtracts ``l_miss``. Below ``l_absent`` the object becomes
      ``status="absent"`` (removed / moved away), or ``"rejected"`` if it had
      fewer than ``confirm_obs`` detections (a false positive); a new detection
      restores it. Consecutive frames are not independent evidence, so at most
      one miss per ``miss_interval_s`` counts. Every status change is kept in
      ``events`` for 4D queries.
    * Classes in ``dynamic_classes`` (people, ...) are tracked but flagged
      ``dynamic`` so the static map and navigation can ignore them.
    """

    def __init__(
        self,
        ttl_s: float = 5.0,
        window: int = 15,
        reid_radius_m: float = 0.6,
        history_step_s: float = 1.0,
        history_step_m: float = 0.1,
        n_classes: int = 0,
        label_accuracy: float = 0.7,
        reid_min_belief: float = 0.1,
        class_names: list[str] | None = None,
        dynamic_classes: tuple[str, ...] = ("person",),
        l_hit: float = 0.85,
        l_miss: float = 0.4,
        l_min: float = -2.0,
        l_max: float = 3.5,
        l_absent: float = -0.5,
        miss_interval_s: float = 1.0,
        confirm_obs: int = 3,
    ) -> None:
        self.dynamic_classes = {c.strip().lower() for c in dynamic_classes if c.strip()}
        self.l_hit, self.l_miss, self.l_min, self.l_max, self.l_absent = l_hit, l_miss, l_min, l_max, l_absent
        self.miss_interval_s = float(miss_interval_s)
        self.confirm_obs = int(confirm_obs)
        self.ttl_s = float(ttl_s)
        self.window = int(window)
        self.reid_radius_m = float(reid_radius_m)
        self.history_step_s = float(history_step_s)
        self.history_step_m = float(history_step_m)
        self.n_classes = int(n_classes)
        self.label_accuracy = float(label_accuracy)
        self.reid_min_belief = float(reid_min_belief)
        self.objects: dict[str, dict] = {}
        self._alias: dict[int, str] = {}
        self._class_names: dict[int, str] = dict(enumerate(class_names or []))
        # Called as ``on_absorb(kept_key, removed_key)`` when consolidate merges two objects.
        self.on_absorb = None
        if not self.n_classes:
            self.n_classes = len(self._class_names)
        self._next = 0

    def fused_label(self, track_id) -> tuple[int, str] | None:
        """``(cls, name)`` of the object behind a track id, after label fusion."""
        if track_id is None:
            return None
        obj = self.objects.get(self._alias.get(int(track_id), ""))
        if obj is None or obj["cls"] < 0:
            return None
        return obj["cls"], obj["name"]

    def track_centers(self) -> list[dict]:
        """Object centres with their track ids, for the 3D-to-2D tracking prior."""
        return [
            {"key": o["key"], "center": [float(v) for v in o["center"]], "track_ids": list(o["track_ids"]),
             "visible": bool(o["visible"])}
            for o in self.objects.values()
            if o["track_ids"]
        ]

    def _new_key(self) -> str:
        self._next += 1
        return f"obj{self._next:04d}"

    def _match(self, track, cls: int, center: np.ndarray, claimed: set[str]) -> str | None:
        if track is not None and int(track) in self._alias:
            key = self._alias[int(track)]
            if key in self.objects:
                return key
        best, best_d = None, self.reid_radius_m
        for key, obj in self.objects.items():
            if key in claimed or (obj["cls"] != cls and obj["belief"].get(cls, 0.0) < self.reid_min_belief):
                continue
            # Only re-attach to objects not currently held by another live track.
            if obj["visible"] and obj["track_id"] is not None and obj["track_id"] != track:
                continue
            dist = float(np.linalg.norm(obj["center"] - center))
            if dist < best_d:
                best, best_d = key, dist
        return best

    def update(self, detections: list[dict], stamp_s: float, width: int = 0, height: int = 0) -> int:
        updated = 0
        claimed: set[str] = set()
        for det in detections:
            obb = det.get("obb")
            if not obb:
                continue
            track = det.get("track_id")
            cls = int(det.get("cls", -1))
            center = np.asarray(obb["center"], dtype=np.float64)
            size = np.asarray(obb["size"], dtype=np.float64)
            rotation = np.asarray(obb.get("rotation", np.eye(3)), dtype=np.float64)
            truncated = bool(width and height and det.get("xyxy") and box_truncated(det["xyxy"], width, height))
            key = self._match(track, cls, center, claimed)
            if key is None:
                key = self._new_key()
                self.objects[key] = {
                    "key": key,
                    "cls": cls,
                    "name": str(det.get("name", "")),
                    "first_seen": float(stamp_s),
                    "n_obs": 0,
                    "visit": 0,
                    "visible": True,
                    "track_ids": [],
                    "centers": [],
                    "sizes": [],
                    "history": [],
                    "views": [],
                    "belief": {},
                    "log_odds": 0.0,
                    "status": "present",
                    "events": [[float(stamp_s), "appeared"]],
                }
            obj = self.objects[key]
            if not obj["visible"]:
                obj["visit"] += 1
            if obj.get("status") in ("absent", "rejected"):
                obj["log_odds"] = max(obj.get("log_odds", 0.0), 0.0)
                obj["status"] = "present"
                obj.setdefault("events", []).append([float(stamp_s), "reappeared"])
            obj["log_odds"] = min(obj.get("log_odds", 0.0) + self.l_hit, self.l_max)
            if cls >= 0:
                self._class_names[cls] = str(det.get("name", self._class_names.get(cls, "")))
                cls_scores = {int(c): float(s) for c, s in (det.get("cls_scores") or {}).items()}
                cls_scores.setdefault(cls, float(det.get("score", 1.0)))
                obj["belief"] = fuse_label(obj["belief"], cls_scores, self.n_classes, self.label_accuracy)
                best = max(obj["belief"], key=obj["belief"].get)
                obj["cls"] = int(best)
                obj["name"] = self._class_names.get(int(best), obj["name"])
            claimed.add(key)
            det["memory_key"] = key
            if track is not None:
                self._alias[int(track)] = key
                if int(track) not in obj["track_ids"]:
                    obj["track_ids"].append(int(track))
            obj["track_id"] = track
            # A triangulated centre replaces class-size guesses instead of being averaged with them.
            rank = _SOURCE_RANK.get(obb.get("source"), _SOURCE_RANK["tri"])
            if rank > obj.get("center_rank", 0):
                obj["centers"] = []
                obj["center_rank"] = rank
            if rank == obj["center_rank"]:
                obj["centers"] = (obj["centers"] + [center])[-self.window :]
            obj["center_source"] = {v: k for k, v in _SOURCE_RANK.items()}[obj["center_rank"]]
            if not truncated or not obj["sizes"]:
                obj["sizes"] = (obj["sizes"] + [size])[-self.window :]
            obj["center"] = np.median(np.stack(obj["centers"]), axis=0)
            obj["size"] = np.median(np.stack(obj["sizes"]), axis=0)
            obj["rotation"] = rotation
            obj["score3d"] = float(obb.get("score3d", 0.0))
            obj["truncated"] = truncated
            obj["last_seen"] = float(stamp_s)
            obj["n_obs"] += 1
            obj["visible"] = True
            hist = obj["history"]
            if (
                not hist
                or stamp_s - hist[-1][0] >= self.history_step_s
                or np.linalg.norm(obj["center"] - np.asarray(hist[-1][1:])) >= self.history_step_m
            ):
                hist.append([float(stamp_s), *(float(v) for v in obj["center"])])
            updated += 1
        return updated

    def consolidate(self, radius_m: float | None = None) -> int:
        """Merge duplicates: compatible label, centres within ``radius_m``, not two live tracks.

        New tracks start from a class-size guess that can be off by metres, so
        they may miss the re-id radius and spawn a copy; once triangulated the
        copies converge and are folded into the older object here.
        """
        radius = self.reid_radius_m if radius_m is None else float(radius_m)
        merged = 0
        keys = sorted(self.objects, key=lambda k: self.objects[k]["first_seen"])
        for i, ka in enumerate(keys):
            a = self.objects.get(ka)
            if a is None:
                continue
            for kb in keys[i + 1 :]:
                b = self.objects.get(kb)
                if b is None or a.get("center_rank", 2) < 2 or b.get("center_rank", 2) < 2:
                    continue
                if a["cls"] != b["cls"] and a["belief"].get(b["cls"], 0.0) < self.reid_min_belief:
                    continue
                if a["visible"] and b["visible"] and a["track_id"] not in (None, b["track_id"]):
                    continue
                if np.linalg.norm(a["center"] - b["center"]) >= radius:
                    continue
                self._absorb(a, b)
                merged += 1
        return merged

    def _absorb(self, a: dict, b: dict) -> None:
        for tid in b["track_ids"]:
            self._alias[tid] = a["key"]
            if tid not in a["track_ids"]:
                a["track_ids"].append(tid)
        wa, wb = a["n_obs"], b["n_obs"]
        a["centers"] = (a["centers"] + b["centers"])[-self.window :]
        a["sizes"] = (a["sizes"] + b["sizes"])[-self.window :]
        a["center"] = np.median(np.stack(a["centers"]), axis=0)
        a["size"] = np.median(np.stack(a["sizes"]), axis=0)
        belief = {c: (wa * a["belief"].get(c, 0.0) + wb * b["belief"].get(c, 0.0)) / max(wa + wb, 1)
                  for c in set(a["belief"]) | set(b["belief"])}
        a["belief"] = belief
        if belief:
            a["cls"] = int(max(belief, key=belief.get))
            a["name"] = self._class_names.get(a["cls"], a["name"])
        a["n_obs"] = wa + wb
        a["first_seen"] = min(a["first_seen"], b["first_seen"])
        if b["last_seen"] > a["last_seen"]:
            a["last_seen"], a["visible"], a["track_id"] = b["last_seen"], b["visible"], b["track_id"]
        a["history"] = sorted(a["history"] + b["history"], key=lambda h: h[0])
        a["views"] = a.get("views", []) + b.get("views", [])
        a["log_odds"] = max(a.get("log_odds", 0.0), b.get("log_odds", 0.0))
        a["misses"] = a.get("misses", 0) + b.get("misses", 0)
        if "present" in (a.get("status"), b.get("status")):
            a["status"] = "present"
        a["events"] = sorted(a.get("events", []) + [e for e in b.get("events", []) if e[1] != "appeared"],
                             key=lambda e: e[0])
        if self.on_absorb is not None:
            self.on_absorb(a["key"], b["key"])
        del self.objects[b["key"]]

    def is_dynamic(self, obj: dict) -> bool:
        return str(obj.get("name", "")).lower() in self.dynamic_classes

    def miss(self, key: str, stamp_s: float) -> bool:
        """Negative evidence: ``key`` was in view but not detected. True if it just became absent."""
        obj = self.objects.get(key)
        if obj is None or stamp_s - obj.get("last_miss", -np.inf) < self.miss_interval_s:
            return False
        obj["last_miss"] = float(stamp_s)
        obj["log_odds"] = max(obj.get("log_odds", 0.0) - self.l_miss, self.l_min)
        obj["misses"] = obj.get("misses", 0) + 1
        if obj.get("status", "present") == "present" and obj["log_odds"] < self.l_absent:
            confirmed = obj["n_obs"] >= self.confirm_obs
            obj["status"] = "absent" if confirmed else "rejected"
            obj["visible"] = False
            obj["track_id"] = None
            obj.setdefault("events", []).append([float(stamp_s), "disappeared" if confirmed else "rejected"])
            return confirmed
        return False

    def expire(self, stamp_s: float) -> None:
        """Mark objects unseen for ``ttl_s`` as hidden; they stay in memory."""
        for obj in self.objects.values():
            if self.ttl_s > 0 and stamp_s - obj["last_seen"] > self.ttl_s:
                obj["visible"] = False
                obj["track_id"] = None

    def boxes(self, include_hidden: bool = True) -> list[dict]:
        out = []
        for obj in self.objects.values():
            if not include_hidden and not obj["visible"]:
                continue
            corners = obj["center"] + (_CORNER_SIGNS * (0.5 * obj["size"])) @ obj["rotation"].T
            out.append({**obj, "corners": corners})
        return out

    def link_dsg(self, dsg_objects: list[dict], radius_m: float = 1.0) -> None:
        """Attach each memory object to the nearest Hydra DSG object of the same class."""
        for obj in self.objects.values():
            best, best_d = None, radius_m
            for dsg in dsg_objects:
                if int(dsg["semantic_id"]) != obj["cls"] + 1:
                    continue
                dist = float(np.linalg.norm(np.asarray(dsg["center"]) - obj["center"]))
                if dist < best_d:
                    best, best_d = dsg, dist
            obj["dsg"] = (
                None
                if best is None
                else {"center": [float(v) for v in best["center"]], "dims": [float(v) for v in best["dims"]], "dist": best_d}
            )

    def to_json(self) -> dict:
        rows = []
        for obj in self.objects.values():
            rows.append(
                {
                    "key": obj["key"],
                    "name": obj["name"],
                    "cls": obj["cls"],
                    "semantic_id": obj["cls"] + 1,
                    "first_seen": obj["first_seen"],
                    "last_seen": obj["last_seen"],
                    "n_obs": obj["n_obs"],
                    "visible": bool(obj["visible"]),
                    "status": obj.get("status", "present"),
                    "existence": round(float(1.0 / (1.0 + np.exp(-obj.get("log_odds", 0.0)))), 3),
                    "misses": int(obj.get("misses", 0)),
                    "events": obj.get("events", []),
                    "dynamic": self.is_dynamic(obj),
                    "track_ids": list(obj["track_ids"]),
                    "label_belief": {
                        self._class_names.get(c, str(c)): round(p, 4)
                        for c, p in sorted(obj.get("belief", {}).items(), key=lambda kv: -kv[1])
                    },
                    "center": [float(v) for v in obj["center"]],
                    "center_source": obj.get("center_source", "tri"),
                    "size": [float(v) for v in obj["size"]],
                    "rotation": np.asarray(obj["rotation"]).tolist(),
                    "bbox3d": obj.get("bbox3d"),
                    "score3d": obj.get("score3d", 0.0),
                    "history": obj["history"],
                    "dsg": obj.get("dsg"),
                    "views": obj.get("views", []),
                    "descriptions": obj.get("descriptions", []),
                }
            )
            limit_object(rows[-1])
        return {"frame_id": "odom", "objects": rows}

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = self.to_json()
        if extra:
            body.update(extra)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(body, indent=1), encoding="utf-8")
        tmp.replace(path)
