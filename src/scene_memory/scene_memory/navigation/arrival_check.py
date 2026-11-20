"""Did the robot really reach the target? WildOS-style check after a goal ends: find the target in the four
Insta360 views with a VLM (box), measure it with the lidar points inside the box, then decide by distance.

    verdict    meaning
    verified   seen, lidar-measured within ``reach_m``
    far        seen, but farther than ``reach_m`` (``center_base`` says where)
    seen       seen, but no lidar returns in the box (too far / thin / glass): visual only
    not_seen   in no view (moved away, wrong spot, or occluded)
    error      the VLM is unreachable

The VLM is any OpenAI-compatible endpoint (default: qwen3-vl on the .209 Ollama, ``STVLN_VLM_URL``); its
``bbox_2d`` is relative 0..1000. The lidar cloud is in ``base_link`` (``/ql/nav/cloud_in_body``, ground z=0),
the views' rotations come from the rig calibration (``insta360_intrinsics.json``), ``base_link -> insta360``
from TF.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests

from scene_memory.perception.lidar_depth import mask_depth, project_points

DEFAULT_URL = os.environ.get("STVLN_VLM_URL", "http://192.168.110.209:11434/v1")
DEFAULT_MODEL = os.environ.get("STVLN_VLM_MODEL", "qwen3-vl:8b-instruct")
LOCATE_PROMPT = (
    'The robot was told: "{instruction}". Find the target "{target}" in this camera image. '
    'Reply JSON only: {{"found": true or false, "bbox_2d": [x1, y1, x2, y2]}} with the box of the target '
    "(the one the instruction means); omit bbox_2d when it is not in the image."
)
# Rated side by side: asked one view at a time the model gives every view the same score.
PROMISE_PROMPT = (
    'A robot must "{instruction}" but cannot see the {target}. These are its camera views ({names}). Compare '
    "them and rate each direction 0-10 for how likely walking that way leads to the {target} (open corridors, "
    "doorways, signs, areas where a {target} usually is: high; walls, dead ends, clutter: low). Use the full "
    'range, give different scores. Reply JSON only, e.g. {{"{first}": 7, ...}}.'
)


def vlm_promise(frames: dict[str, bytes], target: str, instruction: str, url: str = DEFAULT_URL,
                model: str = DEFAULT_MODEL, timeout: float = 90.0) -> dict[str, float]:
    """View name -> 0..1 rating of walking that way to find ``target`` (all views in one request)."""
    names = list(frames)
    content = [{"type": "text", "text": PROMISE_PROMPT.format(instruction=instruction or f"find the {target}",
                                                              target=target, names=", ".join(names),
                                                              first=names[0])}]
    for n in names:
        content += [{"type": "text", "text": f"{n}:"}, {"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(frames[n]).decode()}}]
    r = requests.post(f"{url.rstrip('/')}/chat/completions", json={
        "model": model, "temperature": 0.0, "messages": [{"role": "user", "content": content}]}, timeout=timeout)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)
    out = json.loads(m.group(0)) if m else {}
    scores = {}
    for n in names:
        try:
            scores[n] = min(max(float(out[n]) / 10.0, 0.0), 1.0)
        except (KeyError, TypeError, ValueError):
            pass
    return scores


def vlm_locate(jpeg: bytes, target: str, instruction: str, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL,
               timeout: float = 60.0) -> list[float] | None:
    """Relative box ``[x1, y1, x2, y2]`` in 0..1 of ``target`` in the image, None when not found."""
    r = requests.post(
        f"{url.rstrip('/')}/chat/completions",
        json={"model": model, "temperature": 0.0, "messages": [{"role": "user", "content": [
            {"type": "text", "text": LOCATE_PROMPT.format(target=target, instruction=instruction or target)},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
        ]}]},
        timeout=timeout,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in VLM reply: {text[:120]!r}")
    out = json.loads(m.group(0))
    box = out.get("bbox_2d")
    if not out.get("found") or not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    x1, y1, x2, y2 = (min(max(float(v) / 1000.0, 0.0), 1.0) for v in box)
    if x2 - x1 < 0.005 or y2 - y1 < 0.005:
        return None
    return [x1, y1, x2, y2]


def pinhole_K(width: int, height: int, hfov_deg: float) -> tuple[float, float, float, float]:
    f = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
    return f, f, 0.5 * width, 0.5 * height


def measure_box(cloud_base: np.ndarray, T_base_optical: np.ndarray, K, width: int, height: int,
                box_rel: list[float], min_points: int = 8) -> dict | None:
    """Lidar-measured object inside a relative box: ``center`` (base_link), ``distance_m`` (ground plane)."""
    xyxy = [box_rel[0] * width, box_rel[1] * height, box_rel[2] * width, box_rel[3] * height]
    projected = project_points(cloud_base, T_base_optical, K, width, height)
    hit = mask_depth(projected, width, height, {"xyxy": xyxy}, min_points=min_points)
    if hit is None:
        return None
    c = np.asarray(hit["center"], dtype=np.float64)
    return {"center": c.tolist(), "distance_m": float(math.hypot(c[0], c[1])), "n_points": hit["n_points"]}


def box_bearing(T_base_optical: np.ndarray, K, width: int, height: int, box_rel: list[float]) -> float:
    """Heading (rad, base_link, CCW from +x) of the ray through the box centre."""
    fx, fy, cx, cy = K
    u, v = 0.5 * (box_rel[0] + box_rel[2]) * width, 0.5 * (box_rel[1] + box_rel[3]) * height
    d = np.asarray(T_base_optical)[:3, :3] @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    return float(math.atan2(d[1], d[0]))


def check_arrival(frames: dict[str, bytes], view_specs: dict[str, dict], cloud_base: np.ndarray | None,
                  T_base_insta360: np.ndarray | None, target: str, instruction: str, reach_m: float = 2.0,
                  url: str = DEFAULT_URL, model: str = DEFAULT_MODEL, with_promise: bool = False) -> dict:
    """``frames``: view name -> JPEG; ``view_specs``: name -> ``{R_body_optical, hfov_deg}``.
    Asks the VLM on all views in parallel, measures every hit, returns the nearest one's verdict.
    ``with_promise``: also rates the views as directions to explore (``promise``, ``view_bearing``)."""
    names = [n for n in frames if n in view_specs]
    if not names:
        return {"verdict": "error", "reason": "no camera frames"}

    def locate(name):
        try:
            return name, vlm_locate(frames[name], target, instruction, url, model), None
        except Exception as exc:  # noqa: BLE001 — one failed view must not sink the others
            return name, None, str(exc)

    def rate():
        try:
            return vlm_promise({n: frames[n] for n in names}, target, instruction, url, model)
        except Exception:  # noqa: BLE001 — exploration then uses default scores
            return {}

    with ThreadPoolExecutor(max_workers=len(names) + 1) as pool:
        rating = pool.submit(rate) if with_promise else None
        results = list(pool.map(locate, names))
        promise = rating.result() if rating else {}
    errors = [e for _, _, e in results if e]
    if len(errors) == len(names):
        return {"verdict": "error", "reason": errors[0]}

    def base_optical(name):
        T = np.eye(4)
        T[:3, :3] = view_specs[name]["R_body_optical"]
        return T_base_insta360 @ T if T_base_insta360 is not None else T

    # direction ratings for exploration: view -> 0..1, and the bearing of each view's optical axis (base_link)
    view_bearing = {n: round(box_bearing(base_optical(n), (1.0, 1.0, 0.5, 0.5), 1, 1, [0.5, 0.5, 0.5, 0.5]), 3)
                    for n in names}
    extra = {"promise": promise, "view_bearing": view_bearing}

    seen = []
    for name, box, _ in results:
        if box is None:
            continue
        hit = None
        T = base_optical(name)
        entry = {"view": name, "box": [round(v, 3) for v in box]}
        img = cv2.imdecode(np.frombuffer(frames[name], np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape[:2]
            K = pinhole_K(w, h, view_specs[name]["hfov_deg"])
            entry["bearing_rad"] = round(box_bearing(T, K, w, h, box), 3)
            if cloud_base is not None and T_base_insta360 is not None:
                hit = measure_box(cloud_base, T, K, w, h, box)
        seen.append({**entry, **(hit or {})})
    if not seen:
        return {"verdict": "not_seen", "views": names, **extra}
    measured = sorted((s for s in seen if "distance_m" in s), key=lambda s: s["distance_m"])
    if measured:
        best = measured[0]
        verdict = "verified" if best["distance_m"] <= reach_m else "far"
        return {"verdict": verdict, **best, "seen_in": [s["view"] for s in seen], **extra}
    return {"verdict": "seen", **seen[0], "seen_in": [s["view"] for s in seen], **extra}
