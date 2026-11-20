"""Open-vocabulary object retrieval over a saved topo scene.

Each object gets one SigLIP image embedding (mean over padded crops of its key
views). A free-text query is ranked by SigLIP's sigmoid image–text score, and an
optional spatial relation (``near`` / ``on``) to an anchor query re-weights the
targets by centroid geometry, e.g. "trash can near door" or "门旁边的垃圾桶".
Objects without key views fall back to the mean image embedding of their class,
scored with a penalty.

The index is cached in ``semantic_index.npz`` next to ``objects_4d.json`` and
only objects whose view list changed are re-encoded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scene_memory import paths

DEFAULT_SIGLIP = paths.SIGLIP
INDEX_FILE = "semantic_index.npz"
NO_VIEW_PENALTY = 0.5

_NEAR_EN = r"near|next to|beside|by|close to|around"
_ON_EN = r"on top of|on"
_REL_PATTERNS = [
    (re.compile(rf"^(?P<t>.+?)\s+(?:{_NEAR_EN})\s+(?:the\s+|a\s+)?(?P<a>.+)$", re.I), "near"),
    (re.compile(rf"^(?P<t>.+?)\s+(?:{_ON_EN})\s+(?:the\s+|a\s+)?(?P<a>.+)$", re.I), "on"),
    (re.compile(r"^(?P<a>.+?)(?:旁边|附近|边上|旁)的(?P<t>.+)$"), "near"),
    (re.compile(r"^(?P<a>.+?)(?:上面|上)的(?P<t>.+)$"), "on"),
    (re.compile(r"^(?P<t>.+?)在(?P<a>.+?)(?:旁边|附近|边上|旁)$"), "near"),
    (re.compile(r"^(?P<t>.+?)在(?P<a>.+?)(?:上面|上)$"), "on"),
]


def parse_query(text: str) -> tuple[str, str | None, str | None]:
    """``"trash can near door"`` → ``("trash can", "near", "door")``; no relation → ``(text, None, None)``."""
    text = text.strip()
    for pattern, relation in _REL_PATTERNS:
        m = pattern.match(text)
        if m and m.group("t").strip() and m.group("a").strip():
            return m.group("t").strip(), relation, m.group("a").strip()
    return text, None, None


def prompt(text: str) -> str:
    # The English template also scores best for Chinese queries with the multilingual SigLIP.
    return f"This is a photo of {text}."


def crop(image, xyxy, pad: float = 0.15, min_side: int = 32):
    w, h = image.size
    x0, y0, x1, y1 = [float(v) for v in xyxy]
    bw, bh = max(x1 - x0, min_side), max(y1 - y0, min_side)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    side_w, side_h = bw * (1 + 2 * pad), bh * (1 + 2 * pad)
    box = (max(0, cx - side_w / 2), max(0, cy - side_h / 2), min(w, cx + side_w / 2), min(h, cy + side_h / 2))
    return image.crop(tuple(int(round(v)) for v in box))


class SiglipEncoder:
    def __init__(self, model: str | Path = DEFAULT_SIGLIP, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(str(model), torch_dtype=dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(str(model))
        self.dtype = dtype
        self.logit_scale = float(self.model.logit_scale.exp().item())
        self.logit_bias = float(self.model.logit_bias.item())

    def _norm(self, x) -> np.ndarray:
        x = x.float().cpu().numpy()
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

    def images(self, images: list, batch: int = 32) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(images), batch):
                inputs = self.processor(images=images[i:i + batch], return_tensors="pt")
                pixels = inputs["pixel_values"].to(self.device, self.dtype)
                out.append(self._norm(self.model.get_image_features(pixel_values=pixels)))
        return np.concatenate(out) if out else np.zeros((0, self.dim))

    def texts(self, texts: list[str]) -> np.ndarray:
        with self.torch.inference_mode():
            inputs = self.processor(text=texts, padding="max_length", max_length=64, truncation=True,
                                    return_tensors="pt").to(self.device)
            return self._norm(self.model.get_text_features(input_ids=inputs["input_ids"]))

    @property
    def dim(self) -> int:
        return int(self.model.config.vision_config.hidden_size)

    def prob(self, cos: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.logit_scale * cos + self.logit_bias)))


def _view_signature(obj: dict) -> str:
    return "|".join(v["image"] for v in obj.get("views") or [])


def build_index(scene_dir: str | Path, objects: list[dict], encoder: SiglipEncoder, rebuild: bool = False) -> dict:
    """Embed every object (cached per view list) and save ``semantic_index.npz``."""
    from PIL import Image

    scene_dir = Path(scene_dir)
    path = scene_dir / INDEX_FILE
    cached: dict[str, tuple[str, np.ndarray]] = {}
    if path.exists() and not rebuild:
        data = np.load(path, allow_pickle=False)
        for key, sig, emb in zip(data["keys"], data["signatures"], data["image_emb"]):
            cached[str(key)] = (str(sig), emb)

    keys, sigs, embs, has_view = [], [], [], []
    todo: list[int] = []
    crops: list = []
    owners: list[int] = []
    for obj in objects:
        sig = _view_signature(obj)
        keys.append(obj["key"])
        sigs.append(sig)
        has_view.append(bool(sig))
        if sig and obj["key"] in cached and cached[obj["key"]][0] == sig:
            embs.append(cached[obj["key"]][1])
            continue
        embs.append(None)
        if not sig:
            continue
        todo.append(len(keys) - 1)
        for view in obj["views"]:
            image_path = scene_dir / view["image"]
            if image_path.exists():
                crops.append(crop(Image.open(image_path).convert("RGB"), view["xyxy"]))
                owners.append(len(keys) - 1)

    feats = encoder.images(crops) if crops else np.zeros((0, encoder.dim))
    owners_arr = np.asarray(owners)
    for i in todo:
        rows = feats[owners_arr == i]
        if len(rows):
            mean = rows.mean(axis=0)
            embs[i] = mean / max(np.linalg.norm(mean), 1e-8)
        else:
            has_view[i] = False

    dim = feats.shape[1] if len(feats) else next((len(e) for e in embs if e is not None), encoder.dim)
    image_emb = np.stack([e if e is not None else np.zeros(dim, np.float32) for e in embs]) if embs else np.zeros((0, dim))
    index = {
        "keys": np.asarray(keys),
        "signatures": np.asarray(sigs),
        "image_emb": image_emb.astype(np.float32),
        "has_view": np.asarray(has_view, dtype=bool),
    }
    np.savez(path, **index)
    index["encoded"] = len(todo)
    return index


def class_prototypes(objects: list[dict], image_emb: np.ndarray, has_view: np.ndarray) -> np.ndarray:
    """Per-object fallback embedding: mean image embedding of same-class objects with views (zero if none)."""
    out = np.zeros_like(image_emb)
    for name in {o["name"] for o in objects}:
        rows = np.array([o["name"] == name for o in objects])
        seen = rows & has_view
        if seen.any():
            mean = image_emb[seen].mean(axis=0)
            out[rows] = mean / max(np.linalg.norm(mean), 1e-8)
    return out


def relation_weight(target: dict, anchor: dict, relation: str, near_m: float = 1.5, on_tol_m: float = 0.3) -> float:
    """Soft geometric support in [0, 1] for ``target <relation> anchor``."""
    ct, ca = np.asarray(target["center"], float), np.asarray(anchor["center"], float)
    if relation == "near":
        return float(np.exp(-0.5 * (np.linalg.norm(ct - ca) / near_m) ** 2))
    if relation == "on":
        st, sa = np.asarray(target["size"], float), np.asarray(anchor["size"], float)
        horiz = float(np.linalg.norm(ct[:2] - ca[:2]))
        reach = 0.5 * max(sa[0], sa[1])
        gap = (ct[2] - 0.5 * st[2]) - (ca[2] + 0.5 * sa[2])
        above = ct[2] > ca[2]
        return float(above) * float(np.exp(-0.5 * (max(0.0, horiz - reach) / 0.3) ** 2)) \
            * float(np.exp(-0.5 * (max(0.0, abs(gap) - on_tol_m) / 0.3) ** 2))
    raise ValueError(f"unknown relation {relation!r}")


@dataclass
class Hit:
    obj: dict
    score: float
    semantic: float
    anchor: dict | None = None
    anchor_score: float = 0.0
    relation_w: float = 0.0


class SemanticIndex:
    def __init__(self, scene_dir: str | Path, objects: list[dict], encoder: SiglipEncoder, rebuild: bool = False):
        self.objects = objects
        self.encoder = encoder
        index = build_index(scene_dir, objects, encoder, rebuild=rebuild)
        self.encoded = index["encoded"]
        self.image_emb = index["image_emb"]
        self.has_view = index["has_view"]
        self.fallback_emb = class_prototypes(objects, self.image_emb, self.has_view)

    def semantic_scores(self, text: str) -> np.ndarray:
        """Sigmoid image–text probability per object; objects without views use their class prototype."""
        return self.scores_for(self.encoder.texts([prompt(text)])[0])

    def scores_for(self, q: np.ndarray) -> np.ndarray:
        """``semantic_scores`` for an already encoded text embedding."""
        img = self.encoder.prob(self.image_emb @ q)
        proto = self.encoder.prob(self.fallback_emb @ q) * NO_VIEW_PENALTY
        return np.where(self.has_view, img, proto)

    def search(self, query: str, top_k: int = 5, relation: str | None = None, anchor: str | None = None,
               near_m: float = 1.5, anchor_k: int = 10) -> list[Hit]:
        if relation is None and anchor is None:
            query, relation, anchor = parse_query(query)
        target_s = self.semantic_scores(query)
        if not relation or not anchor:
            order = np.argsort(-target_s)[:top_k]
            return [Hit(self.objects[i], float(target_s[i]), float(target_s[i])) for i in order]

        anchor_s = self.semantic_scores(anchor)
        cand = [int(i) for i in np.argsort(-anchor_s)[:anchor_k] if anchor_s[i] >= 0.3 * anchor_s.max()]
        hits = []
        for i, obj in enumerate(self.objects):
            best = (0.0, None, 0.0, 0.0)
            for j in cand:
                if j == i:
                    continue
                w = relation_weight(obj, self.objects[j], relation, near_m=near_m)
                support = anchor_s[j] * w
                if support > best[0]:
                    best = (support, self.objects[j], float(anchor_s[j]), w)
            norm = best[0] / max(float(anchor_s.max()), 1e-8)
            hits.append(Hit(obj, float(target_s[i] * norm), float(target_s[i]), best[1], best[2], best[3]))
        return sorted(hits, key=lambda h: -h.score)[:top_k]


def load_graph_nodes(scene_dir: str | Path) -> tuple[dict, dict]:
    """``topo_graph.json`` → (object key → node, place id → place)."""
    path = Path(scene_dir) / "topo_graph.json"
    if not path.exists():
        return {}, {}
    graph = json.loads(path.read_text(encoding="utf-8"))
    objects = {o["key"]: o for o in graph["layers"]["objects"]}
    places = {p["id"]: p for p in graph["layers"]["places"]}
    return objects, places


def hit_to_json(hit: Hit, graph_objects: dict, places: dict) -> dict:
    obj = hit.obj
    node = graph_objects.get(obj["key"], {})
    place = places.get(node.get("place"))
    views = obj.get("views") or []
    out = {
        "key": obj["key"], "name": obj["name"], "score": round(hit.score, 4), "semantic": round(hit.semantic, 4),
        "center": [round(float(v), 3) for v in obj["center"]], "center_source": obj.get("center_source"),
        "visible": obj.get("visible"), "last_seen": obj.get("last_seen"),
        "region": node.get("region"),
        "place": place["key"] if place else None,
        "waypoint": [round(float(v), 3) for v in place["position"]] if place else None,
        "view": max(views, key=lambda v: v.get("quality", 0))["image"] if views else None,
    }
    if hit.anchor is not None:
        out["anchor"] = {"key": hit.anchor["key"], "name": hit.anchor["name"],
                         "semantic": round(hit.anchor_score, 4), "relation_w": round(hit.relation_w, 3)}
    return out
