"""GroundingDINO boxes + SAM2.1 masks with the YoloeEquirectDetector output format.

GroundingDINO scores every query against all prompt tokens at once; the score of a
class is the max over its tokens, so each box carries the scores of all classes in
``cls_scores`` for the label fusion in the object memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scene_memory.perception.yoloe_detector import _mask_polygons
from scene_memory.perception.sam3_detector import _box_iou

from scene_memory import paths

DEFAULT_GDINO = paths.GDINO
DEFAULT_SAM2 = paths.SAM2
# "<class>:<prompt>|<prompt>;..." in label order (label = position + 1); later classes paint over earlier ones.
DEFAULT_GROUND_CLASSES = "floor:floor|ground|carpet;grass:grass|lawn;stairs:stairs|staircase;step:step|curb"


def read_ground_spec(spec) -> list[tuple[str, list[str], set[str]]]:
    """Ground classes from a file (one ``class: prompt | prompt  @flag ...`` per line, ``#`` comments) or an
    inline ``"floor:floor|ground;stairs:stairs @terrain"`` → ``[(name, prompts, flags)]`` in label order.
    A class without ``:`` is its own prompt; flags are ``terrain`` (lidar heights, no plane) and ``blocked``."""
    spec = str(spec).strip()
    path = Path(spec).expanduser()
    if spec.endswith((".txt", ".cfg")) or (spec and ";" not in spec and ":" not in spec and path.is_file()):
        if not path.is_file():
            raise FileNotFoundError(f"ground classes file not found: {path}")
        parts = [line.split("#", 1)[0] for line in path.read_text(encoding="utf-8").splitlines()]
    else:
        parts = spec.split(";")
    out = []
    for part in parts:
        words = part.split()
        flags = {w[1:].lower() for w in words if w.startswith("@")}
        part = " ".join(w for w in words if not w.startswith("@"))
        if not part:
            continue
        name, _, prompts = part.partition(":")
        prompts = [p.strip() for p in (prompts or name).split("|") if p.strip()]
        out.append((name.strip(), prompts, flags))
    return out


def parse_ground_classes(spec) -> list[tuple[str, list[str]]]:
    """``"floor:floor|ground;stairs:stairs"`` (or a file, see ``read_ground_spec``) →
    ``[("floor", ["floor", "ground"]), ("stairs", ["stairs"])]``."""
    return [(name, prompts) for name, prompts, _ in read_ground_spec(spec)]


def class_token_map(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> np.ndarray:
    """``(n_classes, n_tokens)`` bool: token lies inside the class name span of the prompt."""
    out = np.zeros((len(spans), len(offsets)), dtype=bool)
    for t, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        for c, (s, e) in enumerate(spans):
            if a >= s and b <= e:
                out[c, t] = True
    return out


def _half_msda() -> None:
    """Run transformers' pure-PyTorch multi-scale deformable attention in fp16.

    Autocast keeps ``grid_sample`` in fp32, which makes it most of the GroundingDINO encoder time on Jetson.
    """
    import torch
    from transformers.models.grounding_dino import modeling_grounding_dino as mg

    cls = mg.MultiScaleDeformableAttention
    if getattr(cls, "_stvln_half", False):
        return
    forward = cls.forward

    def half_forward(self, value, shapes, shapes_list, level_start, locations, weights, im2col_step):
        if not (value.is_cuda and torch.is_autocast_enabled()):
            return forward(self, value, shapes, shapes_list, level_start, locations, weights, im2col_step)
        with torch.autocast("cuda", enabled=False):
            return forward(self, value.half(), shapes, shapes_list, level_start, locations.half(), weights.half(),
                           im2col_step)

    cls.forward = half_forward
    cls._stvln_half = True


def _cache_text_backbone(module) -> None:
    """The prompt is fixed, so the BERT text features are computed once per (tokens, autocast) and reused."""
    import torch

    forward, cache = module.forward, {}

    def cached(*args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        key = (tuple(ids.flatten().tolist()), torch.is_autocast_enabled())
        if key not in cache:
            cache.clear()
            cache[key] = forward(*args, **kwargs)
        return cache[key]

    module.forward = cached


def nms(xyxy: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    order = np.argsort(-scores)
    ious = _box_iou(xyxy, xyxy)
    keep: list[int] = []
    for i in order:
        if all(ious[i, k] < iou for k in keep):
            keep.append(int(i))
    return keep


class GdinoSam2ViewDetector:
    """``predict_views(images)`` like YoloeEquirectDetector: one GroundingDINO + one SAM2 pass per image."""

    nms_free = False

    def __init__(
        self,
        names: list[str],
        gdino: str | Path = DEFAULT_GDINO,
        sam2: str | Path = DEFAULT_SAM2,
        device: str = "",
        conf: float = 0.35,
        iou: float = 0.5,
        cls_floor: float = 0.5,
        half: bool = True,
        distractors: list[str] | None = None,
        ground: list[tuple[str, list[str]]] | None = None,
        ground_conf: float = 0.3,
        gdino_size: int = 800,
        sam_imgsz: int = 640,
        cache_text: bool = True,
        fast_msda: bool = True,
        trt: bool = True,
        gdino_precision: str = "fp16",
    ) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from ultralytics import SAM

        gdino = Path(gdino).expanduser()
        if not (gdino / "config.json").is_file():
            raise FileNotFoundError(
                f"GroundingDINO not found at {gdino}: "
                f"`hf download IDEA-Research/grounding-dino-tiny --local-dir {gdino}`"
            )
        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.half = bool(half) and self.device.type == "cuda"
        self.processor = AutoProcessor.from_pretrained(str(gdino))
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(str(gdino)).to(self.device).eval()
        self._sam_path = Path(sam2).expanduser()
        self.sam = SAM(str(self._sam_path))
        # GroundingDINO input: shortest side (the HF processor default is 800, which upsamples a 640 view).
        self.gdino_size = int(gdino_size)
        self.sam_imgsz = int(sam_imgsz)
        ip = self.processor.image_processor
        self._mean = torch.tensor(ip.image_mean, device=self.device).view(3, 1, 1)
        self._std = torch.tensor(ip.image_std, device=self.device).view(3, 1, 1)
        if cache_text:
            _cache_text_backbone(self.model.model.text_backbone)
        if fast_msda and self.half:
            _half_msda()

        self.names = [str(n) for n in names]
        # Prompted but never reported: they take boxes that would otherwise be
        # forced onto the closest real class (a backlit logo wall → "tv").
        self.distractors = [str(n) for n in (distractors or []) if str(n) not in self.names]
        # Ground classes ride along in the same GroundingDINO + SAM2 pass. Their raw SAM masks (holes kept) are
        # painted into a uint8 label image (class i of ``ground`` = label i + 1, later classes paint over earlier
        # ones) instead of being reported as objects. A ground prompt that is also an object name ("stairs")
        # stays an object and its object masks are painted too.
        self.ground_classes = [str(name) for name, _ in (ground or [])]
        self.ground: list[str] = []
        self._ground_label: list[int] = []
        self._object_label: dict[int, int] = {}
        for label, (_, prompts) in enumerate(ground or [], start=1):
            for prompt in prompts:
                prompt = str(prompt)
                if prompt in self.names:
                    self._object_label[self.names.index(prompt)] = label
                elif prompt not in self.distractors and prompt not in self.ground:
                    self.ground.append(prompt)
                    self._ground_label.append(label)
        self.ground_conf = float(ground_conf)
        self.ground_labels: list[np.ndarray | None] = []
        self.last_ground_labels: np.ndarray | None = None
        self.conf = float(conf)
        self.iou = float(iou)
        self.cls_floor = float(cls_floor)

        spans, text, pos = [], "", 0
        for name in self.names + self.distractors + self.ground:
            phrase = name.lower().strip()
            spans.append((pos, pos + len(phrase)))
            text += phrase + ". "
            pos = len(text)
        self.text = text.strip()
        enc = self.processor.tokenizer(self.text, return_offsets_mapping=True)
        self.token_cls = class_token_map(enc["offset_mapping"], spans)
        missing = [n for n, row in zip(self.names + self.distractors + self.ground, self.token_cls) if not row.any()]
        if missing:
            raise ValueError(f"prompts lost in tokenization: {missing}")
        self._text_inputs = self.processor.tokenizer(self.text, return_tensors="pt").to(self.device)
        self._model_dir = gdino
        self.trt = bool(trt) and self.device.type == "cuda"
        self.gdino_precision = str(gdino_precision)
        self._trt: dict[tuple[int, int, str], object] = {}

    def input_hw(self, h: int, w: int) -> tuple[int, int]:
        """GroundingDINO input size for an ``h x w`` image (HF rule: shortest side, longest side <= 1333)."""
        side = self.gdino_size or min(h, w)
        scale = min(side / min(h, w), 1333 / max(h, w))
        return int(round(h * scale)), int(round(w * scale))

    def trt_engine(self, h: int, w: int):
        """Cached TensorRT GroundingDINO for this prompt and input size, or None (stay on PyTorch)."""
        if not self.trt:
            return None
        key = (*self.input_hw(h, w), self.gdino_precision)
        if key not in self._trt:
            from scene_memory.perception.gdino_trt import TrtGdino, engine_path

            path = engine_path(self._model_dir, self.text, *key)
            self._trt[key] = TrtGdino(path, self.device) if path.is_file() else None
        return self._trt[key]

    def _pixels(self, rgb: np.ndarray):
        """GPU version of the HF GroundingDINO processor (resize, ImageNet normalize, no padding)."""
        import torch
        import torch.nn.functional as F

        h, w = rgb.shape[:2]
        x = torch.from_numpy(rgb).to(self.device, non_blocking=True).permute(2, 0, 1).float().div_(255.0)
        size = self.input_hw(h, w)
        if size != (h, w):
            x = F.interpolate(x[None], size=size, mode="bilinear", align_corners=False, antialias=size[0] < h)[0]
        return ((x - self._mean) / self._std)[None]

    def predict_views(self, images: list[np.ndarray]) -> list[list[dict]]:
        """Detections per image; with ground classes, ``self.ground_labels`` holds the matching label images."""
        out, labels = [], []
        for image in images:
            out.append(self.predict(image))
            labels.append(self.last_ground_labels)
        self.ground_labels = labels
        return out

    def _class_scores(self, logits: np.ndarray) -> np.ndarray:
        """``(queries, n_tokens)`` sigmoid logits → ``(queries, n_classes)`` max over each class's tokens."""
        n = min(logits.shape[1], self.token_cls.shape[1])
        out = np.zeros((logits.shape[0], len(self.token_cls)), dtype=np.float64)
        for c, mask in enumerate(self.token_cls[:, :n]):
            out[:, c] = logits[:, :n][:, mask].max(axis=1)
        return out

    def predict(self, bgr: np.ndarray) -> list[dict]:
        import torch

        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        engine = self.trt_engine(h, w)
        with torch.inference_mode():
            pixels = self._pixels(rgb)
            if engine is not None:
                logits, boxes = engine(pixels)
            else:
                mask = torch.ones(pixels.shape[:1] + pixels.shape[2:], dtype=torch.long, device=self.device)
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.half):
                    out = self.model(**self._text_inputs, pixel_values=pixels, pixel_mask=mask)
                logits, boxes = out.logits, out.pred_boxes
            logits = logits[0].float().sigmoid().cpu().numpy()
            boxes = boxes[0].float().cpu().numpy()

        cls_scores = self._class_scores(logits)
        n_real, n_obj = len(self.names), len(self.names) + len(self.distractors)
        self.last_ground_labels = np.zeros((h, w), np.uint8) if self.ground_classes else None

        def boxes_of(sel: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            cx, cy, bw, bh = boxes[sel].T
            xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], axis=1)
            xyxy = np.clip(xyxy, 0.0, [w - 1, h - 1, w - 1, h - 1]).astype(np.float64)
            keep = nms(xyxy, scores, self.iou) if len(sel) else []
            return xyxy[keep], sel[keep]

        # Objects and ground get separate NMS so a floor box never suppresses an object box.
        q = len(cls_scores)
        obj_scores = cls_scores[:, :n_obj].max(axis=1) if n_obj else np.zeros(q)
        g_scores = cls_scores[:, n_obj:].max(axis=1) if self.ground else np.zeros(q)
        sel = np.flatnonzero((obj_scores >= self.conf) & (obj_scores >= g_scores)) if n_obj else np.zeros(0, int)
        xyxy, sel = boxes_of(sel, obj_scores[sel])
        real = cls_scores[sel, :n_obj].argmax(axis=1) < n_real if len(sel) else np.zeros(0, bool)
        xyxy, sel = xyxy[real], sel[real]
        g_xyxy, g_sel = np.zeros((0, 4)), np.zeros(0, int)
        if self.ground:
            g_sel = np.flatnonzero((g_scores >= self.ground_conf) & (g_scores > obj_scores))
            g_xyxy, g_sel = boxes_of(g_sel, g_scores[g_sel])
        if len(sel) == 0 and len(g_xyxy) == 0:
            return []
        obj_cls = cls_scores[sel, :n_real].argmax(axis=1) if len(sel) else np.zeros(0, int)
        g_label = [self._ground_label[int(i)] for i in cls_scores[g_sel, n_obj:].argmax(axis=1)] if len(g_sel) else []
        cls_scores = cls_scores[:, :n_real]

        from scene_memory.perception import sam2_trt

        sam2_trt.use(self._sam_path if self.trt else None)
        result = self.sam.predict(bgr, bboxes=np.vstack([xyxy, g_xyxy]).tolist(), quantize=16 if self.half else 32,
                                  device=self.device, imgsz=self.sam_imgsz, verbose=False)[0]
        if self.last_ground_labels is not None and result.masks is not None:
            labels = [self._object_label.get(int(c), 0) for c in obj_cls] + g_label
            painted = [(lab, k) for k, lab in enumerate(labels) if lab and k < len(result.masks.data)]
            for lab, k in sorted(painted):
                self.last_ground_labels[result.masks.data[k].cpu().numpy()[:h, :w].astype(bool)] = lab
        if len(sel) == 0:
            return []
        polygons = _mask_polygons(result, 0)

        dets = []
        for k, q in enumerate(sel):
            row = cls_scores[q]
            cls = int(row.argmax())
            x1, y1, x2, y2 = (float(v) for v in xyxy[k])
            det = {
                "xyxy": (x1, y1, x2, y2),
                "score": float(row[cls]),
                "cls": cls,
                "name": self.names[cls],
                "cls_scores": {int(c): float(s) for c, s in enumerate(row) if s >= self.cls_floor * row[cls]},
                "parts": [(x1, y1, x2, y2)],
            }
            if k < len(polygons) and polygons[k] is not None:
                det["mask"] = polygons[k]
            dets.append(det)
        return dets
