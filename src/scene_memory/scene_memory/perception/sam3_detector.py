"""SAM3 text-prompted instance segmentation with the YoloeEquirectDetector output format.

SAM3 scores every prompt separately, so one object can come back as "chair"
and "sofa". Overlapping boxes of different prompts are merged into one
detection; the losing prompts stay in ``cls_scores`` for the label fusion
in the object memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scene_memory import paths
from scene_memory.perception.yoloe_detector import _mask_polygons

DEFAULT_SAM3 = paths.SAM3


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0.0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0.0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0)


def merge_prompts(xyxy: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float) -> list[tuple[int, dict]]:
    """Group detections of different prompts on the same object.

    Returns ``(kept index, {cls: best score})`` per object, highest score first.
    """
    order = np.argsort(-scores)
    ious = _box_iou(xyxy, xyxy) if len(xyxy) else np.zeros((0, 0))
    taken = np.zeros(len(xyxy), dtype=bool)
    groups = []
    for i in order:
        if taken[i]:
            continue
        members = np.flatnonzero(~taken & (ious[i] >= iou))
        taken[members] = True
        cls_scores: dict[int, float] = {}
        for j in members:
            c = int(classes[j])
            cls_scores[c] = max(cls_scores.get(c, 0.0), float(scores[j]))
        groups.append((int(i), cls_scores))
    return groups


class Sam3ViewDetector:
    """``predict_views(images)`` like YoloeEquirectDetector, one SAM3 pass per image."""

    nms_free = False

    def __init__(
        self,
        checkpoint: str | Path,
        names: list[str],
        device: str = "",
        conf: float = 0.4,
        iou: float = 0.5,
        merge_iou: float = 0.7,
        imgsz: int = 0,
        half: bool = True,
    ) -> None:
        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM3 weights not found at {checkpoint}. facebook/sam3 is gated: request access on "
                "huggingface.co/facebook/sam3, run `hf auth login`, then "
                f"`hf download facebook/sam3 sam3.pt --local-dir {checkpoint.parent}`"
            )
        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides = dict(
            conf=float(conf),
            iou=float(iou),
            task="segment",
            mode="predict",
            model=str(checkpoint),
            half=bool(half),
            save=False,
            verbose=False,
        )
        if device:
            overrides["device"] = device
        if imgsz > 0:
            overrides["imgsz"] = int(imgsz)
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self.names = [str(n) for n in names]
        self.merge_iou = float(merge_iou)

    def predict_views(self, images: list[np.ndarray]) -> list[list[dict]]:
        return [self.predict(image) for image in images]

    def predict(self, bgr: np.ndarray) -> list[dict]:
        self.predictor.set_image(bgr)
        result = self.predictor(text=self.names)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float64)
        scores = boxes.conf.detach().cpu().numpy().astype(np.float64)
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        polygons = _mask_polygons(result, 0)
        dets = []
        for index, cls_scores in merge_prompts(xyxy, scores, classes, self.merge_iou):
            x1, y1, x2, y2 = (float(v) for v in xyxy[index])
            cls = int(classes[index])
            det = {
                "xyxy": (x1, y1, x2, y2),
                "score": float(scores[index]),
                "cls": cls,
                "name": self.names[cls] if 0 <= cls < len(self.names) else str(cls),
                "cls_scores": cls_scores,
                "parts": [(x1, y1, x2, y2)],
            }
            if index < len(polygons) and polygons[index] is not None:
                det["mask"] = polygons[index]
            dets.append(det)
        return dets
