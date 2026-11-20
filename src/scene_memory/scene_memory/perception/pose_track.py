"""BoT-SORT with a SuperMap-style 3D-to-2D motion prior.

Between two detector frames (1-2 Hz with SAM3) the camera moves too much for a
constant-velocity image model. For tracks whose object already has a 3D centre
in the 4D memory, the Kalman prediction is replaced by that centre projected
with the current VIO pose (SuperMap Eq. 5); other tracks keep the plain BoT-SORT
prediction (optionally with image-based GMC).

Track ids are made unique across views (``view_index * ID_STRIDE + local id``)
because each ultralytics tracker numbers its own tracks from 1.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

ID_STRIDE = 1_000_000


class _Dets:
    """Minimal numpy Results stand-in for the ultralytics trackers."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    @property
    def xywh(self) -> np.ndarray:
        wh = self.xyxy[:, 2:] - self.xyxy[:, :2]
        return np.concatenate([self.xyxy[:, :2] + 0.5 * wh, wh], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index) -> "_Dets":
        return _Dets(self.xyxy[index], self.conf[index], self.cls[index])


def _make_tracker(track_thresh: float, track_buffer: int, match_thresh: float, gmc: str):
    from ultralytics.trackers.bot_sort import BOTSORT

    class ProjectedBOTSORT(BOTSORT):
        priors: dict[int, tuple[float, float]] = {}

        def _pre_first_associate(self, strack_pool, unconfirmed, img, results_high) -> None:
            super()._pre_first_associate(strack_pool, unconfirmed, img, results_high)
            if not self.priors:
                return
            for track in [*strack_pool, *unconfirmed]:
                uv = self.priors.get(int(track.track_id))
                if uv is None:
                    continue
                track.mean[0], track.mean[1] = uv
                # The projection already holds the ego-motion; drop the stale image velocity.
                track.mean[4:6] = 0.0

    args = SimpleNamespace(
        tracker_type="botsort",
        track_high_thresh=float(track_thresh),
        track_low_thresh=0.1,
        new_track_thresh=float(track_thresh),
        track_buffer=int(track_buffer),
        match_thresh=float(match_thresh),
        fuse_score=True,
        gmc_method=gmc if gmc and gmc != "none" else None,
        proximity_thresh=0.5,
        appearance_thresh=0.8,
        with_reid=False,
        model="auto",
    )
    return ProjectedBOTSORT(args)


def project_centers(
    objects: list[dict],
    T_world_optical: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    min_depth: float = 0.3,
    margin: float = 0.1,
) -> dict[int, tuple[float, float]]:
    """``{global track id: (u, v)}`` for memory objects that land inside this view."""
    if not objects:
        return {}
    T_opt_world = np.linalg.inv(np.asarray(T_world_optical, dtype=np.float64))
    out: dict[int, tuple[float, float]] = {}
    for obj in objects:
        p = T_opt_world[:3, :3] @ np.asarray(obj["center"], dtype=np.float64) + T_opt_world[:3, 3]
        if p[2] < min_depth:
            continue
        u = fx * p[0] / p[2] + cx
        v = fy * p[1] / p[2] + cy
        if not (-margin * width <= u <= (1 + margin) * width and -margin * height <= v <= (1 + margin) * height):
            continue
        for tid in obj.get("track_ids", []):
            out[int(tid)] = (float(u), float(v))
    return out


class ViewPoseTracker:
    """One ProjectedBOTSORT per view. ``update`` returns confirmed tracks with the detection's mask."""

    def __init__(
        self,
        views: list[str],
        track_thresh: float = 0.25,
        track_buffer: int = 15,
        match_thresh: float = 0.8,
        gmc: str = "none",
    ) -> None:
        self._offset = {name: (i + 1) * ID_STRIDE for i, name in enumerate(views)}
        self._cfg = (track_thresh, track_buffer, match_thresh, gmc)
        self._trackers: dict[str, object] = {}
        self.n_prior = 0

    def update(
        self,
        view: str,
        detections: list[dict],
        width: int,
        height: int,
        image: np.ndarray | None = None,
        priors: dict[int, tuple[float, float]] | None = None,
    ) -> list[dict]:
        tracker = self._trackers.get(view)
        if tracker is None:
            tracker = self._trackers[view] = _make_tracker(*self._cfg)
        offset = self._offset.setdefault(view, (len(self._offset) + 1) * ID_STRIDE)
        tracker.priors = {
            gid - offset: uv for gid, uv in (priors or {}).items() if offset <= gid < offset + ID_STRIDE
        }
        self.n_prior = len(tracker.priors)
        dets = _Dets(
            [d["xyxy"] for d in detections] or np.zeros((0, 4)),
            [d["score"] for d in detections],
            [d["cls"] for d in detections],
        )
        rows = tracker.update(dets, image)
        targets = []
        for x1, y1, x2, y2, tid, score, cls, idx in np.asarray(rows).reshape(-1, 8):
            x1, x2 = float(np.clip(x1, 0, width - 1)), float(np.clip(x2, 0, width - 1))
            y1, y2 = float(np.clip(y1, 0, height - 1)), float(np.clip(y2, 0, height - 1))
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            det = detections[int(idx)] if 0 <= int(idx) < len(detections) else {}
            target = {
                "xyxy": (x1, y1, x2, y2),
                "score": float(score),
                "cls": int(cls),
                "name": det.get("name", str(int(cls))),
                "view": view,
                "track_id": offset + int(tid),
                "parts": [(x1, y1, x2, y2)],
            }
            for key in ("mask", "cls_scores"):
                if key in det:
                    target[key] = det[key]
            targets.append(target)
        targets.sort(key=lambda t: -t["score"])
        return targets
