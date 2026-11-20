"""Region (room) summaries in the style of DAAAM's ``summarize_regions.py``.

Per region, a farthest-first sample of its objects (SigLIP image embeddings when
``semantic_index.npz`` exists, else one-hot class) is given to an LLM, which
returns ``reasoning`` / ``region_label`` / ``region_description``. Without an LLM
the label falls back to the dominant classes.

Summaries are stored in ``region_summaries.json`` keyed by the region's place
set, so they survive re-numbering and are re-attached by the live mapper as long
as the region's places do not change.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

SUMMARY_FILE = "region_summaries.json"

PROMPT = """You are summarizing one region of a robot's semantic map (indoor and outdoor, several buildings / floors).
Region {key}: {where}; {n_places} places, extent {extent:.1f} m, {n_objects} objects, adjacent regions: {adjacent}.
Object class counts: {counts}
Diverse sample of objects (description, position in m):
{samples}

Infer what kind of area this is (indoor e.g. office workspace, corridor, meeting room, lab, lobby, kitchen,
stairwell, elevator hall; outdoor e.g. street, parking lot, courtyard, building entrance, sidewalk).
Reply with JSON only: {{"reasoning": "...", "region_label": "short label", "region_description": "one sentence"}}
Write region_label and region_description in {language}."""


def farthest_first(emb: np.ndarray, n: int) -> list[int]:
    """DAAAM's greedy diversity sampling: start farthest from the mean, then max-min distance."""
    if len(emb) <= n:
        return list(range(len(emb)))
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    mean = emb.mean(axis=0)
    chosen = [int(np.argmax(1 - emb @ mean))]
    while len(chosen) < n:
        sim = (emb @ emb[chosen].T).max(axis=1)
        sim[chosen] = np.inf
        chosen.append(int(np.argmin(sim)))
    return chosen


def object_embeddings(scene_dir: Path, objects: list[dict]) -> np.ndarray:
    path = scene_dir / "semantic_index.npz"
    names = sorted({o["name"] for o in objects})
    onehot = np.eye(max(len(names), 1))[[names.index(o["name"]) for o in objects]] if objects else np.zeros((0, 1))
    if not path.exists():
        return onehot
    data = np.load(path, allow_pickle=False)
    rows = {str(k): (e, bool(v)) for k, e, v in zip(data["keys"], data["image_emb"], data["has_view"])}
    out = []
    for o, oh in zip(objects, onehot):
        emb, ok = rows.get(o["key"], (None, False))
        out.append(np.concatenate([emb, 0.5 * oh]) if ok else np.concatenate([np.zeros(data["image_emb"].shape[1]), oh]))
    return np.asarray(out)


def describe_object(obj: dict) -> str:
    descs = obj.get("descriptions") or []
    return descs[-1]["text"] if descs else obj["name"]


def region_where(region: dict) -> str:
    kind = region.get("kind", "room")
    if kind == "outdoor":
        return "outdoors"
    if kind in ("stairs", "elevator", "ramp"):
        return f"{kind} of {region.get('building', 'building0')} between floors {region.get('floors')}"
    return f"indoors, {region.get('building', 'building0')} floor {region.get('floor', 0)}"


def fallback_summary(counts: Counter, kind: str = "room") -> dict:
    top = [name for name, _ in counts.most_common(3)]
    if kind in ("stairs", "elevator", "ramp"):
        return {"label": kind, "description": f"{kind.capitalize()} between floors.", "reasoning": "vertical motion"}
    if not top:
        if kind == "outdoor":
            return {"label": "outdoor passage", "description": "Outdoor area with no detected objects.",
                    "reasoning": "no objects"}
        return {"label": "passage", "description": "Traversed area with no detected objects.", "reasoning": "no objects"}
    return {"label": f"area with {', '.join(top)}",
            "description": "Area containing " + ", ".join(f"{n} {name}" for name, n in counts.most_common(5)) + ".",
            "reasoning": "dominant detected classes"}


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not data.get("region_label"):
        return None
    return {"label": str(data["region_label"]), "description": str(data.get("region_description", "")),
            "reasoning": str(data.get("reasoning", ""))}


def summarize_regions(
    graph: dict,
    objects: list[dict],
    scene_dir: str | Path,
    chat: Callable[[str], str] | None = None,
    n_samples: int = 10,
    language: str = "English",
    log: Callable[[str], None] | None = None,
) -> dict:
    """Summary per region → ``{place-id tuple: {label, description, reasoning}}`` (also saved)."""
    scene_dir = Path(scene_dir)
    by_key = {o["key"]: o for o in objects}
    places = {p["id"]: p for p in graph["layers"]["places"]}
    emb_all = object_embeddings(scene_dir, objects)
    emb_row = {o["key"]: i for i, o in enumerate(objects)}
    summaries, records = {}, []
    for region in graph["layers"]["regions"]:
        objs = [by_key[k] for k in region["objects"] if k in by_key]
        counts = Counter(o["name"] for o in objs)
        pos = np.array([places[p]["position"] for p in region["places"]])
        extent = float(np.linalg.norm(pos.max(0) - pos.min(0))) if len(pos) > 1 else 0.0
        summary = None
        if chat is not None and objs and region.get("kind", "room") in ("room", "outdoor"):
            idx = farthest_first(emb_all[[emb_row[o["key"]] for o in objs]], n_samples)
            samples = "\n".join(f"{i + 1}. {describe_object(objs[j])} (pos: {np.round(objs[j]['center'], 1).tolist()})"
                                for i, j in enumerate(idx))
            prompt = PROMPT.format(key=region["key"], where=region_where(region), n_places=len(region["places"]),
                                   extent=extent, n_objects=len(objs),
                                   adjacent=[f"region{a}" for a in region.get("adjacent", [])] or "none",
                                   counts=dict(counts.most_common()), samples=samples, language=language)
            try:
                summary = parse_json(chat(prompt))
            except Exception as exc:  # network/model errors fall back to class counts
                if log:
                    log(f"{region['key']}: LLM failed ({exc}); using class counts")
        if summary is None:
            summary = fallback_summary(counts, region.get("kind", "room"))
        summaries[tuple(sorted(region["places"]))] = summary
        records.append({"region_key": region["key"], "places": sorted(region["places"]), "num_objects": len(objs),
                        **summary})
        if log:
            log(f"{region['key']:<9} {len(objs):3d} obj  {summary['label']}: {summary['description']}")
    (scene_dir / SUMMARY_FILE).write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    return summaries


def load_summaries(scene_dir: str | Path) -> dict:
    path = Path(scene_dir) / SUMMARY_FILE
    if not path.exists():
        return {}
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {tuple(r["places"]): {k: r[k] for k in ("label", "description", "reasoning") if k in r} for r in records}
