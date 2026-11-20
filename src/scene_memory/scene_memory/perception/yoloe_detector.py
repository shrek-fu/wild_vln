"""Text-prompt YOLOE on one global equirectangular frame."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from scene_memory.common.tiles import equirect_tiles, nms_circular, split_wrapped_box

_TEXT_WEIGHT = "mobileclip_blt.pt"


def _pad_clip_context(context_length: int = 77) -> None:
    """YOLOE-26's MobileCLIP2 script was traced at context 77.

    The vendored CLIP tokenizer shortens that context to the longest prompt.
    Pad the result back out so the scripted attention mask still matches.
    """
    import clip.clip as clip_mod

    original = clip_mod.tokenize
    if getattr(original, "_yoloe26_padded", False):
        return

    def tokenize(texts, context_length=context_length, truncate=True):
        import torch

        tokens = original(texts, context_length=context_length, truncate=truncate)
        width = int(tokens.shape[-1])
        if width < context_length:
            pad = torch.zeros(
                tokens.shape[0], context_length - width, dtype=tokens.dtype, device=tokens.device
            )
            tokens = torch.cat((tokens, pad), dim=1)
        elif width > context_length:
            tokens = tokens[:, :context_length]
        return tokens

    tokenize._yoloe26_padded = True
    clip_mod.tokenize = tokenize


def ensure_yoloe_on_path(yoloe_root: str | Path) -> Path:
    """Prefer the vendored YOLOE tree over any site-packages ultralytics."""
    root = Path(yoloe_root).expanduser().resolve()
    if not (root / "ultralytics").is_dir():
        raise FileNotFoundError(f"YOLOE checkout not found at {root}")
    extras = [
        root,
        root / "third_party" / "CLIP",
        root / "third_party" / "ml-mobileclip",
    ]
    for path in reversed(extras):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return root


def find_mobileclip(candidates: list[Path]) -> Path:
    for folder in candidates:
        weight = folder / _TEXT_WEIGHT
        if weight.is_file():
            return weight
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"missing {_TEXT_WEIGHT}. Download "
        "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt "
        f"into one of: {searched}"
    )


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class YoloeEquirectDetector:
    """Run YOLOE on the full panorama or on overlapping square tiles, then merge across the seam."""

    def __init__(
        self,
        checkpoint: str | Path,
        yoloe_root: str | Path,
        names: list[str],
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 0,
        overlap: float = 0.25,
        mode: str = "tiles",
        mobileclip: str | Path | None = None,
    ) -> None:
        if mode not in {"tiles", "full"}:
            raise ValueError("mode must be 'tiles' or 'full'")
        if not names:
            raise ValueError("at least one text prompt is required")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"missing {self.checkpoint}. Download with scripts/download_weights.sh "
                "(yoloe-26s-seg.pt, or yoloe-v8s-seg.pt)."
            )
        self.names = [str(n) for n in names]
        self.device = device
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.overlap = float(overlap)
        self.mode = mode
        # YOLOE-26 keeps a one-to-one head. nms=False selects it and skips NMS.
        self.nms_free = "yoloe-26" in self.checkpoint.name
        os.environ.setdefault("YOLO_AUTOINSTALL", "False")
        os.environ.setdefault("YOLO_OFFLINE", "True")
        if self.nms_free:
            self._load_yoloe26(yoloe_root)
        else:
            self._load_vendored(yoloe_root, mobileclip)
        self.set_device(self.device)

    def _load_yoloe26(self, yoloe_root: str | Path) -> None:
        """Official Ultralytics YOLOE-26. Class names are encoded once at startup."""
        import torch

        clip_root = Path(yoloe_root).expanduser().resolve() / "third_party" / "CLIP"
        if clip_root.is_dir():
            entry = str(clip_root)
            if entry not in sys.path:
                sys.path.insert(0, entry)
        _pad_clip_context(77)
        for key in list(sys.modules):
            if key == "ultralytics" or key.startswith("ultralytics."):
                del sys.modules[key]
        from ultralytics import YOLOE

        self.model = YOLOE(str(self.checkpoint))
        self.model.to("cpu")
        text_encoder = self.checkpoint.parent / "mobileclip2_b.ts"
        if not text_encoder.is_file():
            raise FileNotFoundError(
                f"missing {text_encoder}. Download with scripts/download_weights.sh yoloe-26s-seg.pt"
            )
        with _working_directory(self.checkpoint.parent):
            self.model.set_classes(self.names)
        inner = getattr(self.model, "model", None)
        pe = getattr(inner, "pe", None)
        if torch.is_tensor(pe):
            inner.pe = torch.from_numpy(np.ascontiguousarray(pe.detach().float().cpu().numpy()))

    def _load_vendored(self, yoloe_root: str | Path, mobileclip: str | Path | None) -> None:
        self.yoloe_root = ensure_yoloe_on_path(yoloe_root)
        clip_dirs = [self.checkpoint.parent, self.yoloe_root, Path.cwd()]
        if mobileclip:
            clip_path = Path(mobileclip).expanduser().resolve()
            clip_dirs.insert(0, clip_path if clip_path.is_dir() else clip_path.parent)
        self.mobileclip = find_mobileclip(clip_dirs)
        from ultralytics import YOLOE

        self.model = YOLOE(str(self.checkpoint))
        # MobileCLIP reparameterize deep-copies the text tower. Encode once on CPU,
        # then copy the embedding out of inference mode so later `.to()` / forward can use it.
        self.model.to("cpu")
        with _working_directory(self.mobileclip.parent):
            embeddings = self.model.get_text_pe(self.names)
        import torch

        embeddings = torch.from_numpy(np.ascontiguousarray(embeddings.detach().float().cpu().numpy()))
        self.model.set_classes(self.names, embeddings)

    def set_device(self, device: str) -> None:
        """Move the detector, including the text embedding stored outside ``nn.Parameter``."""
        import gc

        import torch

        self.device = device
        if device == "cpu":
            # The predictor keeps the last CUDA batch alive. Drop it before emptying the cache.
            self.model.predictor = None
        if device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
        self.model.to(device)
        inner = getattr(self.model, "model", None)
        pe = getattr(inner, "pe", None)
        if torch.is_tensor(pe):
            # Prompt embedding is not an nn.Parameter. Keep a detached leaf so later
            # `.to()` / fuse does not try to flip requires_grad on a computed tensor.
            inner.pe = pe.detach().to(device=device, dtype=torch.float32)
        gc.collect()
        if device == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, bgr: np.ndarray) -> list[dict]:
        """Return detections in the original equirect pixel frame."""
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"expected BGR image, got {bgr.shape}")
        height, width = bgr.shape[:2]
        if self.mode == "full":
            windows = [(0, bgr)]
        else:
            windows = equirect_tiles(bgr, self.overlap)
        crops = [crop for _, crop in windows]
        results = self._forward_images(crops)
        raw: list[dict] = []
        for (origin, _), result in zip(windows, results):
            raw.extend(self._read_result(result, origin, width, height))
        merged = nms_circular(raw, self.iou, width, agnostic=False)
        for det in merged:
            det["parts"] = split_wrapped_box(det["xyxy"], width, height)
        return merged

    def predict_views(self, images: list[np.ndarray]) -> list[list[dict]]:
        """One batched pass. Each returned list is in that image's pixel frame."""
        if not images:
            return []
        results = self._forward_images(images)
        grouped: list[list[dict]] = []
        for image, result in zip(images, results):
            height, width = image.shape[:2]
            dets = self._read_result(result, 0, width, height)
            for det in dets:
                x1, y1, x2, y2 = det["xyxy"]
                det["parts"] = [(x1, y1, x2, y2)]
            grouped.append(dets)
        return grouped

    def _forward_images(self, images: list[np.ndarray]):
        # imgsz 0 keeps each split crop at its own resolution. A smaller value
        # letterboxes the crop and is what made the boxes worse.
        if self.imgsz > 0:
            imgsz = self.imgsz
        else:
            side = max(max(int(image.shape[0]), int(image.shape[1])) for image in images)
            imgsz = max(32, ((side + 31) // 32) * 32)
        predict_args = dict(
            conf=self.conf,
            iou=self.iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
            batch=max(1, len(images)),
        )
        if self.nms_free:
            predict_args["nms"] = False
        return self.model.predict(images, **predict_args)

    def _read_result(self, result, origin: int, width: int, height: int) -> list[dict]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        polygons = _mask_polygons(result, origin)
        raw = []
        for index, ((x1, y1, x2, y2), score, cls) in enumerate(zip(xyxy, scores, classes)):
            name = self.names[int(cls)] if 0 <= int(cls) < len(self.names) else str(int(cls))
            det = {
                "xyxy": (float(x1) + origin, float(y1), float(x2) + origin, float(y2)),
                "score": float(score),
                "cls": int(cls),
                "name": name,
            }
            if index < len(polygons) and polygons[index] is not None:
                det["mask"] = polygons[index]
            raw.append(det)
        return raw


def _mask_polygons(result, origin: int) -> list:
    """Segmentation contours in the original image, at most 48 vertices."""
    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "xy", None) is None:
        return []
    polygons = []
    for poly in masks.xy:
        arr = np.asarray(poly, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 3:
            polygons.append(None)
            continue
        arr = arr.copy()
        arr[:, 0] += float(origin)
        if arr.shape[0] > 48:
            index = np.linspace(0, arr.shape[0] - 1, 48).astype(int)
            arr = arr[index]
        polygons.append([(float(x), float(y)) for x, y in arr])
    return polygons
