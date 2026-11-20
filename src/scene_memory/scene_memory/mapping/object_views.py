"""Best image + mask per object visit, saved for DAM description."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scene_memory.mapping.boxer_objects import box_truncated
from scene_memory.mapping.scene_graph import _track_pixel_mask


def load_description_cache(scene_dir: str | Path) -> dict[str, dict]:
    """``descriptions.json``: view image path → ``{key, visit, stamp, text, model}``."""
    path = Path(scene_dir) / "descriptions.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_description_cache(scene_dir: str | Path, cache: dict[str, dict]) -> None:
    path = Path(scene_dir) / "descriptions.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def descriptions_by_key(cache: dict[str, dict]) -> dict[str, list[dict]]:
    by_key: dict[str, list[dict]] = {}
    for entry in cache.values():
        by_key.setdefault(entry["key"], []).append(
            {"visit": entry["visit"], "stamp": entry["stamp"], "text": entry["text"]}
        )
    for rows in by_key.values():
        rows.sort(key=lambda d: d["stamp"])
    return by_key


def view_quality(det: dict, width: int, height: int) -> float:
    """Prefer large, complete, confident views: mask area fraction × score, ÷4 when truncated."""
    mask = _track_pixel_mask(det, width, height)
    area = float(mask.mean())
    if area <= 0:
        return 0.0
    quality = min(area, 0.5) * float(det.get("score", 1.0))
    if det.get("xyxy") and box_truncated(det["xyxy"], width, height):
        quality *= 0.25
    return quality


class KeyViewStore:
    """Keeps one best view per (object, visit) under ``<scene_dir>/views``."""

    def __init__(self, scene_dir: str | Path, max_visits: int = 5, min_quality: float = 0.002) -> None:
        self.root = Path(scene_dir) / "views"
        self.max_visits = int(max_visits)
        self.min_quality = float(min_quality)

    def offer(self, obj: dict, det: dict, bgr: np.ndarray, stamp_s: float) -> bool:
        height, width = bgr.shape[:2]
        quality = view_quality(det, width, height)
        if quality < self.min_quality:
            return False
        views = obj.setdefault("views", [])
        visit = int(obj.get("visit", 0))
        slot = next((v for v in views if v["visit"] == visit), None)
        if slot is None and len(views) >= self.max_visits:
            # Out of slots: only replace the weakest stored view.
            slot = min(views, key=lambda v: v["quality"])
        if slot is not None and quality <= slot["quality"]:
            return False
        name = f"{obj['key']}_v{visit}"
        self.root.mkdir(parents=True, exist_ok=True)
        mask = _track_pixel_mask(det, width, height).astype(np.uint8) * 255
        cv2.imwrite(str(self.root / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(self.root / f"{name}_mask.png"), mask)
        record = {
            "visit": visit,
            "stamp": float(stamp_s),
            "quality": quality,
            "image": f"views/{name}.jpg",
            "mask": f"views/{name}_mask.png",
            "xyxy": [float(v) for v in det.get("xyxy", [])],
        }
        if slot is None:
            views.append(record)
        else:
            if slot["image"] != record["image"]:
                for key in ("image", "mask"):
                    stale = self.root.parent / slot[key]
                    if stale.is_file():
                        stale.unlink()
            slot.clear()
            slot.update(record)
        return True
