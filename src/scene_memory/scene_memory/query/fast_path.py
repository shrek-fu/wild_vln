"""Fast path for simple "go to / find X" instructions (FSR-VLN style fast reasoning), no LLM call.

The instruction is parsed with rules into an optional level (``2楼``, ``楼上``, ``外面``), region (a region
label that appears in the text), relation (``门旁边的垃圾桶``) and a target phrase. The target phrase is mapped
to a detector class by multilingual SigLIP text–text similarity (a clear winner is required: words outside the
vocabulary such as 冰箱 / robot give a near tie), candidates of that class are filtered top-down
(level → region → class), ranked by image–text score (attributes such as 黑色) and relation support, and
equally good candidates are ordered by route length. Optionally the top candidates are checked by the vision
model in one call (FSR-VLN's slow verification); a rejection hands the instruction to the GLM agent.

Anything else — questions, counting, time, ordinals / left-right, follow-ups, negations, several targets,
unknown room words, other buildings, absent or dynamic targets — returns ``None`` with a ``reason`` and is left
to the GLM tool agent (``glm_retriever.resolve``).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from scene_memory.query import semantic_index as si

CLASS_MARGIN = 0.05            # text–text: best class must beat the runner-up by this ...
CLASS_MARGIN_HIGH = 0.025      # ... or by this when its similarity is at least CLASS_HIGH
CLASS_HIGH = 0.9
EQUIVALENT = 0.6               # candidates within this fraction of the best score are "equally good"
MIN_RELATION_SUPPORT = 0.3
VERIFY_MAX = 4

_CJK = re.compile(r"[\u4e00-\u9fff]")
_SLOW = [
    ("question", re.compile(r"[?？吗呢]|几[个把张台扇只件位次]|多少|什么|哪些|哪个|是否|有没有|怎么|为什么|"
                            r"\b(how|what|when|which|why|whether|count|is there|are there|do you|did)\b", re.I)),
    ("time", re.compile(r"刚才|刚刚|之前|以前|上次|最近一次|分钟|秒|小时|一开始|开始时|最早|最晚|昨天|今天|早上|上午|"
                        r"下午|晚上|时候|时间|\b(ago|before|after|earlier|recent(ly)?|seen|yesterday|today|minute|"
                        r"second|hour)s?\b", re.I)),
    ("ordinal / direction", re.compile(r"第[一二三四五六七八九十\d]+|最前|最后|前面|后面|前排|后排|左|右|对面|中间|"
                                       r"\b(first|second|third|last|left|right|front|behind|back|opposite|middle)\b",
                                       re.I)),
    # the fast path orders equally good candidates nearest first
    ("distance", re.compile(r"远|\b(far|farther|farthest|further|furthest|distant)\b", re.I)),
    ("follow-up", re.compile(r"另一个|另外|其他|其它|别的|它|上一个|\b(it|another|other|that one|this one)\b", re.I)),
    ("negation", re.compile(r"不是|不要|别|除了|没有|\b(not|except|without|no)\b", re.I)),
    ("several targets", re.compile(r"和|并且|然后|再去|还有|以及|或者|\b(and|then|or)\b", re.I)),
    ("other building", re.compile(r"另一栋|另一座|别的楼|号楼|栋|座楼|大楼|\bbuilding\b", re.I)),
]
_PREFIX_ZH = re.compile(r"^(请|麻烦|帮我|你|能不能|可以|能|快)*(带我|领我|陪我|帮我)?"
                        r"(去|到|找|找到|找一下|寻找|导航到|导航去|走到|走去|前往|移动到|过去|去看看|看看|看一下)+"
                        r"(一下)?")
_PREFIX_EN = re.compile(r"^(please\s+|can you\s+|could you\s+)*(go\s+(and\s+)?)?(go to|navigate to|take me to|bring me to|"
                        r"head to|move to|find|locate|search for|look for|where is|where's|show me)\s+", re.I)
_SUFFIX = re.compile(r"(在哪里|在哪儿|在哪|那里|那儿|那边|吧|啊|呀|一下|please)?\s*[。.!！]*$", re.I)
_QUANT = re.compile(r"^(那|这|一)?(个|把|张|台|扇|只|件|盏|辆|部|座)")
_ARTICLE = re.compile(r"^(the|a|an)\s+", re.I)
_NEAREST = re.compile(r"最近的|离我最近的|附近的|\b(nearest|closest)\s+", re.I)
_ZH_DIGITS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_FLOOR = re.compile(r"(地下|负|B)?\s*([0-9]+|[一二两三四五六七八九十])\s*(楼|层|F)(的|里的|上的)?|"
                    r"\b(on\s+)?(the\s+)?([0-9]+)(st|nd|rd|th)?\s+floor\b|\bfloor\s+([0-9]+)\b", re.I)
_UPDOWN = re.compile(r"(楼上|楼下|upstairs|downstairs)(的|里的)?", re.I)
_OUTDOOR = re.compile(r"(外面|室外|户外|outside|outdoors?)(的|里的)?", re.I)
_ATTRIBUTE = re.compile(r"^((黑|白|红|绿|蓝|黄|灰|棕|橙|紫|粉|银|金|彩)色?|木头?|金属|塑料|玻璃|皮|布|大|小|高|矮|长|短|"
                        r"圆|方|新|旧|开着|关着|打开|关闭|空)+(的)?$")


@dataclass
class FastResult:
    target: str | None = None
    ranked: list[str] = field(default_factory=list)
    phrase: str = ""
    cls: str = ""
    level: tuple | None = None
    region: int | None = None
    relation: str | None = None
    anchor_cls: str | None = None
    verified: bool | None = None
    reason: str = ""


_CLASS_CACHE: dict = {}


def _class_embeddings(encoder, classes: tuple[str, ...]) -> np.ndarray:
    key = (id(encoder), classes)
    if key not in _CLASS_CACHE:
        _CLASS_CACHE[key] = np.asarray(encoder.texts([si.prompt(c) for c in classes]), dtype=np.float64)
    return _CLASS_CACHE[key]


def match_class(encoder, classes: list[str], phrase: str) -> tuple[str | None, float, float, np.ndarray]:
    """Detector class for a phrase if one clearly wins → (class|None, best similarity, margin, text embedding)."""
    q = np.asarray(encoder.texts([si.prompt(phrase)])[0], dtype=np.float64)
    low = phrase.lower()
    exact = [c for c in classes if re.search(rf"\b{re.escape(c)}\b", low)]
    if len(exact) == 1:
        return exact[0], 1.0, 1.0, q
    cls = tuple(sorted(classes))
    if not cls:
        return None, 0.0, 0.0, q
    emb = _class_embeddings(encoder, cls)
    sims = emb @ q / max(float(np.linalg.norm(q)), 1e-8)
    order = np.argsort(-sims)
    best = float(sims[order[0]])
    margin = best - float(sims[order[1]]) if len(order) > 1 else 1.0
    ok = margin >= CLASS_MARGIN or (best >= CLASS_HIGH and margin >= CLASS_MARGIN_HIGH)
    return (cls[order[0]] if ok else None), best, margin, q


def _floor_number(m: re.Match) -> int:
    raw = m.group(2) or m.group(7) or m.group(9)
    n = int(raw) if raw.isdigit() else _ZH_DIGITS.get(raw, 0)
    return -n if m.group(1) else n


def parse(instruction: str, region_labels: dict[int, str]) -> dict:
    """Rule parse → ``{phrase, level_kind, level_value, region, slow}``; ``slow`` names why the agent is needed."""
    text = instruction.strip()
    if re.search(r"在哪|where", text, re.I):
        text = text.rstrip("?？ ")
    text = _PREFIX_EN.sub("", _PREFIX_ZH.sub("", text)).strip()
    text = _SUFFIX.sub("", text).strip()
    text = _NEAREST.sub("", text).strip()
    out = {"phrase": "", "level": None, "region": None, "slow": None}
    for name, pattern in _SLOW:
        if pattern.search(text):
            out["slow"] = name
            return out
    m = _FLOOR.search(text)
    if m:
        out["level"] = ("floor", _floor_number(m))
        text = (text[:m.start()] + text[m.end():]).strip()
    m = _UPDOWN.search(text)
    if m:
        out["level"] = ("relative", 1 if m.group(1).lower() in ("楼上", "upstairs") else -1)
        text = (text[:m.start()] + text[m.end():]).strip()
    m = _OUTDOOR.search(text)
    if m:
        out["level"] = ("outdoor", None)
        text = (text[:m.start()] + text[m.end():]).strip()
    for rid, label in sorted(region_labels.items(), key=lambda kv: -len(kv[1])):
        if label and len(label) >= 2 and label.lower() in text.lower():
            i = text.lower().index(label.lower())
            out["region"] = rid
            text = (text[:i] + text[i + len(label):]).strip()
            text = re.sub(r"^(里面的|里的|中的|内的|的)|^(in|at|of)\s+(the\s+)?|\s+(in|at|of)(\s+the)?\s*$", "",
                          text, flags=re.I).strip()
            break
    text = re.sub(r"^(在|in|at|on)\s*", "", text, flags=re.I).strip()
    text = _ARTICLE.sub("", _QUANT.sub("", text)).strip()
    target, relation, anchor = si.parse_query(text)
    head = target.rsplit("的", 1)[0] if "的" in target else ""
    if head and not _ATTRIBUTE.match(head + "的"):
        out["slow"] = f"unknown qualifier '{head}'"
        return out
    if re.search(r"\s(in|at|of|inside|from)\s", f" {target} ", re.I):
        out["slow"] = "unknown place qualifier"
        return out
    out.update(phrase=target.strip(), relation=relation, anchor=anchor)
    if not out["phrase"]:
        out["slow"] = "no target phrase"
    return out


_NEAR_SUFFIX = re.compile(r"(旁边|附近|边上|跟前|前面|那里|那儿|\s+(nearby|next to it))$")


def _from_views(res: FastResult, tools, instruction: str, reason: str) -> FastResult:
    """FSR-VLN fast -> slow: no detected object fits, so look for the phrase in the panorama keyframes.
    Only a vision-confirmed hit is taken; otherwise ``reason`` goes to the agent."""
    res.reason = reason
    if tools.vision is None or not res.phrase or not tools.views_available():
        return res
    found = tools.search_views(_NEAR_SUFFIX.sub("", res.phrase).strip() or res.phrase, request=instruction)
    if "error" in found or not found.get("verified"):
        res.reason = f"{reason}; view search: {found.get('error') or 'not confirmed'}"
        return res
    res.target, res.ranked, res.cls, res.verified = found["key"], [found["key"]], "view", True
    return res


def fast_resolve(instruction: str, tools, verify: bool = True) -> FastResult:
    """Pick a target without the LLM; ``FastResult.target`` is None when the agent has to take over."""
    res = FastResult()
    if tools.index is None or not getattr(tools.index, "encoder", None):
        res.reason = "no semantic index"
        return res
    labels = {rid: r.get("label") for rid, r in tools.regions.items() if r.get("label")}
    p = parse(instruction, labels)
    if p["slow"]:
        res.reason = p["slow"]
        return res
    res.phrase, res.region, res.relation = p["phrase"], p["region"], p.get("relation")
    encoder = tools.index.encoder
    cls, best, margin, q = match_class(encoder, tools.classes, p["phrase"])
    if cls is None:
        return _from_views(res, tools, instruction, f"'{p['phrase']}' matches no detector class clearly "
                                                    f"(sim {best:.2f}, margin {margin:.3f})")
    res.cls = cls

    def level_of(key):
        return tools._where(tools.graph_objects.get(key, {}).get("place"))

    level = p["level"]
    if level is not None:
        if level[0] == "outdoor":
            res.level = ("outdoor",)
        else:
            pid = (tools.planner.nearest(tools.robot_now["position"])
                   if tools.planner is not None and tools.robot_now is not None else None)
            now = tools._where(pid)
            if "floor" not in now:
                res.reason = "robot floor unknown"
                return res
            value = level[1] - 1 if level[0] == "floor" and level[1] > 0 else level[1]
            if level[0] == "relative":
                value = now["floor"] + level[1]
            res.level = (now.get("building"), value)

    def in_level(key):
        w = level_of(key)
        if res.level is None:
            return True
        if res.level == ("outdoor",):
            return w.get("building") == "outdoor"
        return (w.get("building"), w.get("floor")) == res.level

    same_class = [o for o in tools.objects if o["name"] == cls]
    cands = [o for o in same_class if tools._status(o) == "present" and in_level(o["key"])
             and (res.region is None or tools.graph_objects.get(o["key"], {}).get("region") == res.region)]
    if not cands:
        statuses = Counter(tools._status(o) for o in same_class)
        where = " and ".join(x for x in (f"level {res.level}" if res.level else "",
                                         f"region R{res.region}" if res.region is not None else "") if x)
        res.reason = (f"no present {cls} in {where} (map: {dict(statuses)})" if where else
                      f"no present {cls} (map: {dict(statuses)})")
        return res

    rows = {o["key"]: i for i, o in enumerate(tools.objects)}
    scores = tools.index.scores_for(q) if hasattr(tools.index, "scores_for") else \
        tools.index.semantic_scores(p["phrase"])
    s = np.array([float(scores[rows[o["key"]]]) for o in cands])
    s = s / max(float(s.max()), 1e-8)
    if res.relation and p.get("anchor"):
        anchor_cls, _, _, _ = match_class(encoder, tools.classes, p["anchor"])
        if anchor_cls is None:
            res.reason = f"anchor '{p['anchor']}' matches no detector class"
            return res
        res.anchor_cls = anchor_cls
        anchors = [o for o in tools.objects if o["name"] == anchor_cls and tools._status(o) == "present"]
        support = np.array([max((si.relation_weight(o, a, res.relation) for a in anchors if a is not o), default=0.0)
                            for o in cands])
        if support.max() < MIN_RELATION_SUPPORT:
            res.reason = f"no {cls} {res.relation} a {anchor_cls}"
            return res
        s = s * support
    order = np.argsort(-s)
    good = [int(i) for i in order if s[i] >= EQUIVALENT * s[order[0]]]
    rest = [int(i) for i in order if i not in good]
    lengths = {}
    for i in good:
        plan = tools.plan_to(cands[i]["key"])
        lengths[i] = plan["path_length_m"] if "error" not in plan and not plan["partial"] else float("inf")
    good.sort(key=lambda i: (lengths[i], -s[i]))
    res.ranked = [cands[i]["key"] for i in good + rest]

    if verify and tools.vision is not None:
        matches = tools.verify(res.ranked[:VERIFY_MAX], p["phrase"])
        if matches is not None:
            res.verified = bool(matches)
            hit = next((k for k in res.ranked[:VERIFY_MAX] if k in matches), None)
            if hit is None:
                return _from_views(res, tools, instruction,
                                   f"visual check rejected {res.ranked[:VERIFY_MAX]} for '{p['phrase']}'")
            res.target = hit
            return res
    res.target = res.ranked[0]
    return res


_MODE_ZH = {"stairs": "走楼梯", "elevator": "乘电梯", "ramp": "走坡道"}


def reply_text(instruction: str, res: FastResult, answer: dict, tools) -> str:
    zh = bool(_CJK.search(instruction))
    node = tools.graph_objects.get(res.target, {})
    region = tools.regions.get(node.get("region"), {})
    rname = region.get("label") or (f"区域R{node.get('region')}" if zh else f"region R{node.get('region')}")
    plan = answer.get("plan") or {}
    w = tools._where(node.get("place"))
    where = ""
    if tools.multi_level:
        if w.get("building") == "outdoor":
            where = "室外" if zh else "outdoors, "
        elif "floor" in w:
            where = (f"{w['building']} 相对 {w['floor']} 层" if zh else f"{w['building']} level {w['floor']}, ")
    steps = [s for s in plan.get("segments", []) if s["mode"] != "walk"]
    route_m = plan.get("path_length_m") if plan else None
    direct_m = tools._from_robot_now(tools.by_key[res.target]["center"])
    if zh:
        text = f"好的，去{where}{rname}的{res.phrase}（{res.target}）"
        if route_m is not None and route_m >= 2.0:
            text += f"，路线约 {route_m:.0f} 米"
        elif direct_m is not None:
            text += f"，就在附近，直线约 {direct_m:.1f} 米"
        if steps:
            text += "，途中" + "、".join(f"{_MODE_ZH.get(s['mode'], s['mode'])}从 {s.get('from_floor')} 层到 "
                                       f"{s.get('to_floor')} 层" for s in steps)
        if res.verified:
            text += "（已看图确认）"
        return text + "。"
    text = f"Going to the {res.phrase} ({res.target}) in {where}{rname}"
    if route_m is not None and route_m >= 2.0:
        text += f", route about {route_m:.0f} m"
    elif direct_m is not None:
        text += f", close by, about {direct_m:.1f} m away"
    if steps:
        text += "; " + "; ".join(s["instruction"] for s in steps)
    if res.verified:
        text += " (visually confirmed)"
    return text + "."
