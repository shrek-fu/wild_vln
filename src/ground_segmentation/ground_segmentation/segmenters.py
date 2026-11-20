"""Image ground segmenters. Each is ``seg(bgr) -> (H, W) uint8`` label image over ``seg.classes``
(pixel = class index + 1, 0 = not ground), with the classes of a ``DEFAULT_GROUND_CLASSES``-style spec.

* ``GdinoSam2Ground``: own GroundingDINO + SAM2 instance prompted with the ground classes only.
* ``DetectorGround``: offline twin of the live ``--segmenter detector`` (the semantic detector's pass with its
  object prompts plus the ground classes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scene_memory import paths
from scene_memory.perception.gdino_sam2_detector import DEFAULT_GROUND_CLASSES, parse_ground_classes, read_ground_spec

from ground_segmentation.mapper import TERRAIN_CLASSES


def class_names(spec: str = DEFAULT_GROUND_CLASSES) -> list[str]:
    return [name for name, _ in parse_ground_classes(spec)]


def terrain_classes(spec: str = DEFAULT_GROUND_CLASSES, override: str = "") -> list[str]:
    """``override`` (comma list) if given, else the ``@terrain`` classes of a spec that uses flags, else
    ``TERRAIN_CLASSES``."""
    if override.strip():
        return [c.strip() for c in override.split(",") if c.strip()]
    entries = read_ground_spec(spec)
    if any(flags for _, _, flags in entries):
        return [name for name, _, flags in entries if "terrain" in flags]
    return list(TERRAIN_CLASSES)


def blocked_classes(spec: str = DEFAULT_GROUND_CLASSES, extra: str = "") -> list[str]:
    """``@blocked`` classes of the spec plus the comma list ``extra``."""
    out = [name for name, _, flags in read_ground_spec(spec) if "blocked" in flags]
    return out + [c.strip() for c in extra.split(",") if c.strip() and c.strip() not in out]


class _DetectorLabels:
    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        self.last_detections = self.det.predict(bgr)
        labels = self.det.last_ground_labels
        return np.zeros(bgr.shape[:2], np.uint8) if labels is None else labels


class GdinoSam2Ground(_DetectorLabels):
    name = "gdino_sam2"

    def __init__(self, classes: str = DEFAULT_GROUND_CLASSES, ground_conf: float = 0.3, device: str = "",
                 half: bool = True) -> None:
        from scene_memory.perception.gdino_sam2_detector import GdinoSam2ViewDetector

        self.det = GdinoSam2ViewDetector([], gdino=paths.GDINO, sam2=paths.SAM2, device=device, half=half,
                                         ground=parse_ground_classes(classes), ground_conf=ground_conf)
        self.classes = self.det.ground_classes


class DetectorGround(_DetectorLabels):
    name = "detector"

    def __init__(self, classes: str = DEFAULT_GROUND_CLASSES, conf: float = 0.35, ground_conf: float = 0.3,
                 device: str = "", names_file: str | Path = paths.CONFIG / "prompts.txt",
                 distractors_file: str | Path = paths.CONFIG / "distractors.txt") -> None:
        from scene_memory.nodes.detector_node import load_names
        from scene_memory.perception.gdino_sam2_detector import GdinoSam2ViewDetector

        self.det = GdinoSam2ViewDetector(load_names("", str(names_file)), gdino=paths.GDINO, sam2=paths.SAM2,
                                         device=device, conf=conf, distractors=load_names("", str(distractors_file)),
                                         ground=parse_ground_classes(classes), ground_conf=ground_conf)
        self.classes = self.det.ground_classes
        self.last_detections: list[dict] = []


def make_segmenter(kind: str, classes: str = DEFAULT_GROUND_CLASSES, device: str = "", **kwargs):
    if kind == "gdino_sam2":
        return GdinoSam2Ground(classes, device=device, **kwargs)
    if kind == "detector":
        return DetectorGround(classes, device=device, **kwargs)
    raise ValueError(f"unknown segmenter {kind!r} (gdino_sam2 | detector)")
