"""FSR-VLN slow path over the view layer: find something that is not a detected object.

1. SigLIP text -> panorama sectors (``mapping.view_layer``), best sector per keyframe, top ``k`` keyframes;
2. the vision model picks the sector that shows the request (FSR-VLN ``vlm_choose``);
3. the vision model boxes the target in that sector (0-1000 coordinates, as GLM-4V / Qwen-VL ground);
4. the box centre's ray from the panorama camera is cut with the scene's point cloud (``map_cloud.npy``):
   the nearest dense range inside a small cone around the ray is the target's 3D position.

Without a vision model the best sector's centre is used (``verified`` False).
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np

from scene_memory.common.geometry import equirect_uv_to_da360_rays
from scene_memory.mapping.view_layer import INDEX_FILE, KEYFRAMES_FILE, crop_wrapped

CHOOSE_PROMPT = ('Numbered views from a robot\'s 360° camera. The user asked for: "{request}". Which view shows '
                 'it most clearly? Reply with JSON only: {{"view": <number, 0 if none shows it>, "reason": "..."}}')
GROUND_PROMPT = ('Locate "{target}" in this image. Reply with JSON only: {{"found": true or false, '
                 '"box": [x1, y1, x2, y2]}} with coordinates normalised to 0-1000 (left, top, right, bottom).')


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def parse_box(text: str) -> list[float] | None:
    """``[x1, y1, x2, y2]`` in 0-1000 from a JSON reply, or the first four numbers of a bracketed box."""
    box = _json(text).get("box")
    if not box:
        m = re.search(r"\[\s*\[?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)",
                      text or "")
        box = [float(g) for g in m.groups()] if m else None
    if isinstance(box, list) and box and isinstance(box[0], list):
        box = box[0]
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 <= x1 or y2 <= y1 or max(x2, y2) > 1000.5:
        return None
    return [x1, y1, x2, y2]


def lift_ray(points: np.ndarray, origin: np.ndarray, direction: np.ndarray, cone_deg: float = 2.0,
             min_r: float = 0.3, max_r: float = 15.0, min_hits: int = 3) -> np.ndarray | None:
    """First surface along a ray: points within ``cone_deg`` (widened up to 3x), nearest dense range."""
    d = points - origin
    r = np.linalg.norm(d, axis=1)
    ok = (r > min_r) & (r < max_r)
    d, r, p = d[ok], r[ok], points[ok]
    cos = (d @ direction) / np.maximum(r, 1e-9)
    for widen in (1.0, 2.0, 3.0):
        hit = cos > np.cos(np.radians(cone_deg * widen))
        if hit.sum() >= min_hits:
            rr = np.sort(r[hit])
            near = rr[max(0, int(0.1 * len(rr)) - 1)]
            sel = hit & (r <= near + 0.5)
            return p[sel].mean(axis=0) if sel.any() else origin + direction * near
    return None


class ViewIndex:
    def __init__(self, scene_dir: str | Path):
        self.scene_dir = Path(scene_dir)
        meta = json.loads((self.scene_dir / KEYFRAMES_FILE).read_text(encoding="utf-8"))
        self.keyframes = meta["keyframes"]
        data = np.load(self.scene_dir / INDEX_FILE)
        self.emb, self.kf, self.sector = data["emb"], data["keyframe"], data["sector"]
        cloud = self.scene_dir / "map_cloud.npy"
        self.points = np.load(cloud).astype(np.float64) if cloud.exists() else None

    @staticmethod
    def available(scene_dir: str | Path) -> bool:
        return (Path(scene_dir) / KEYFRAMES_FILE).exists() and (Path(scene_dir) / INDEX_FILE).exists()

    def top(self, q: np.ndarray, k: int = 6) -> list[tuple[int, int, float]]:
        """``(keyframe, sector, cosine)``: best sector per keyframe, best ``k`` keyframes."""
        cos = self.emb @ q
        best: dict[int, tuple[int, float]] = {}
        for i in np.argsort(-cos):
            kf = int(self.kf[i])
            if kf not in best:
                best[kf] = (int(self.sector[i]), float(cos[i]))
            if len(best) >= k:
                break
        return [(kf, s, c) for kf, (s, c) in best.items()]

    def tile(self, kf: int, sector: int):
        from PIL import Image

        meta = self.keyframes[kf]
        image = Image.open(self.scene_dir / meta["file"]).convert("RGB")
        return crop_wrapped(image, meta["sectors"][sector])

    def ray(self, kf: int, sector: int, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """World ray through pixel ``(u, v)`` of a sector tile."""
        meta = self.keyframes[kf]
        u0, v0, _, _ = meta["sectors"][sector]
        uu, vv = (u0 + u) % meta["width"], v0 + v
        ray = equirect_uv_to_da360_rays(np.array([uu]), np.array([vv]), meta["width"], meta["height"])[0]
        T = np.asarray(meta["T_odom_da360"], float)
        return T[:3, 3].copy(), T[:3, :3] @ ray

    def locate(self, kf: int, sector: int, box01k: list[float] | None) -> dict:
        meta = self.keyframes[kf]
        u0, v0, u1, v1 = meta["sectors"][sector]
        w, h = u1 - u0, v1 - v0
        if box01k:
            x1, y1, x2, y2 = (c / 1000.0 for c in box01k)
            u, v = 0.5 * (x1 + x2) * w, 0.5 * (y1 + y2) * h
            size_deg = min(x2 - x1, y2 - y1) * w / meta["width"] * 360.0
        else:
            u, v, size_deg = 0.5 * w, 0.5 * h, 8.0
        origin, direction = self.ray(kf, sector, u, v)
        point = lift_ray(self.points, origin, direction, cone_deg=float(np.clip(0.25 * size_deg, 1.0, 4.0))) \
            if self.points is not None else None
        return {"origin": origin, "direction": direction, "point": point}


def grid_jpeg(tiles: list, cell: int = 384) -> bytes:
    from PIL import Image, ImageDraw

    cols = min(3, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * cell, rows * cell), "white")
    draw = ImageDraw.Draw(out)
    for n, t in enumerate(tiles):
        im = t.copy()
        im.thumbnail((cell, cell))
        x, y = (n % cols) * cell, (n // cols) * cell
        out.paste(im, (x, y))
        draw.rectangle([x, y, x + 34, y + 30], fill="black")
        draw.text((x + 8, y + 6), str(n + 1), fill="yellow")
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def jpeg(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def search(views: ViewIndex, q: np.ndarray, request: str, target: str, vision=None, k: int = 6) -> dict:
    """Retrieve, choose, ground, lift. Returns ``{point, keyframe, sector, score, verified, reason, ...}``."""
    cands = views.top(q, k)
    if not cands:
        return {"error": "no views"}
    chosen, reason, verified, box = 0, "", False, None
    if vision is not None:
        tiles = [views.tile(kf, s) for kf, s, _ in cands]
        reply = _json(vision(grid_jpeg(tiles), CHOOSE_PROMPT.format(request=request)))
        n = int(reply.get("view") or 0) if str(reply.get("view", "")).strip().lstrip("-").isdigit() else 0
        reason = str(reply.get("reason", ""))
        if n < 1 or n > len(cands):
            return {"error": "the vision model saw it in none of the retrieved views", "reason": reason,
                    "candidates": [{"keyframe": kf, "score": round(c, 3)} for kf, _, c in cands]}
        chosen = n - 1
        kf, s, _ = cands[chosen]
        text = vision(jpeg(tiles[chosen]), GROUND_PROMPT.format(target=target))
        found = _json(text).get("found", True)
        box = parse_box(text) if found is not False else None
        verified = box is not None
    kf, s, score = cands[chosen]
    loc = views.locate(kf, s, box)
    if loc["point"] is None:
        return {"error": "no map points along the view ray", "keyframe": kf}
    meta = views.keyframes[kf]
    return {"point": loc["point"], "camera": loc["origin"], "keyframe": kf, "sector": s, "stamp": meta["stamp"],
            "score": round(score, 3), "verified": verified, "box": box, "reason": reason}


def main() -> None:
    import argparse
    import os
    import time

    from PIL import ImageDraw

    from scene_memory.query import glm_retriever as gr
    from scene_memory.query import semantic_index as si

    parser = argparse.ArgumentParser(description="search the view layer of a map scene (no GLM agent)")
    parser.add_argument("scene")
    parser.add_argument("target", nargs="+", help="what to find, e.g. 'potted plant'")
    parser.add_argument("--request", default="", help="the user's words for the vision model (default: target)")
    parser.add_argument("--base_url", default=os.environ.get("GLM_BASE_URL", gr.DEFAULT_BASE_URL))
    parser.add_argument("--api_key_env", default="ZHIPUAI_API_KEY")
    gr.add_vision_args(parser)
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument("--siglip", default=str(si.DEFAULT_SIGLIP))
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", default="", help="write the chosen view with the box and the candidates here")
    args = parser.parse_args()
    target = " ".join(args.target)

    vision = gr.vision_from_args(args, os.environ.get(args.api_key_env))
    views = ViewIndex(args.scene)
    q = si.SiglipEncoder(args.siglip, device=args.device).texts([si.prompt(target)])[0]
    t0 = time.time()
    res = search(views, q, args.request or target, target, vision, args.k)
    print(f"{time.time() - t0:.1f} s, vision {'on' if vision else 'off'}")
    for kf, s, c in views.top(q, args.k):
        print(f"  candidate kf{kf} sector{s} cos {c:.3f}")
    print({k: (np.round(v, 2).tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()})
    if args.save and "keyframe" in res and "sector" in res:
        tile = views.tile(res["keyframe"], res["sector"])
        if res.get("box"):
            w, h = tile.size
            x1, y1, x2, y2 = res["box"]
            ImageDraw.Draw(tile).rectangle([x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000],
                                           outline=(255, 0, 0), width=4)
        tile.save(args.save)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
