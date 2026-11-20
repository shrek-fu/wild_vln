"""DAAAM-style semantic labelspace from ``config/prompts.txt``.

DAAAM maps each Hydra pixel to a semantic id via ``labels_pseudo.yaml`` /
``labels_pseudo.csv``. Here id ``0`` is background; class ``i`` in YOLOE
prompts becomes semantic id ``i + 1``.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabelSpace:
    names: list[str]
    """Index 0 unused; names[semantic_id] for id >= 1."""

    colors_bgr: list[tuple[int, int, int]]
    """Parallel to ``names``; index 0 is black."""

    @property
    def num_labels(self) -> int:
        return max(0, len(self.names) - 1)

    def semantic_id(self, cls: int) -> int:
        sid = int(cls) + 1
        if sid < 1 or sid >= len(self.names):
            return 0
        return sid

    def name(self, semantic_id: int) -> str:
        if 0 <= semantic_id < len(self.names):
            return self.names[semantic_id]
        return "unknown"

    def color_bgr(self, semantic_id: int) -> tuple[int, int, int]:
        if 0 <= semantic_id < len(self.colors_bgr):
            return self.colors_bgr[semantic_id]
        return (128, 128, 128)

    def to_yaml_dict(self) -> dict:
        return {
            "total_semantic_labels": self.num_labels,
            "labels": {i: self.names[i] for i in range(1, len(self.names))},
        }

    def to_hydra_label_config(self) -> dict:
        """Hydra ``label_spaces/*_label_space.yaml`` fields for YOLOE prompt ids."""
        ids = list(range(1, len(self.names)))
        return {
            "total_semantic_labels": self.num_labels,
            "dynamic_labels": [],
            "invalid_labels": [0],
            "object_labels": ids,
            "surface_places_labels": [],
            "label_names": [{"label": i, "name": self.names[i]} for i in range(len(self.names))],
        }


def _color_for_id(semantic_id: int) -> tuple[int, int, int]:
    if semantic_id <= 0:
        return (0, 0, 0)
    hue = (semantic_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
    # OpenCV BGR
    return int(b * 255), int(g * 255), int(r * 255)


def load_labelspace(prompts_path: str | Path) -> LabelSpace:
    path = Path(prompts_path)
    names = ["background"]
    colors = [(0, 0, 0)]
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            names.append(name)
            colors.append(_color_for_id(len(names) - 1))
    return LabelSpace(names=names, colors_bgr=colors)


def write_labelspace_files(labelspace: LabelSpace, out_dir: str | Path) -> tuple[Path, Path]:
    """Write DAAAM-compatible ``labels_pseudo.yaml`` and ``.csv``."""
    import csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    yaml_path = out / "labels_pseudo.yaml"
    csv_path = out / "labels_pseudo.csv"
    lines = [f"total_semantic_labels: {labelspace.num_labels}", "labels:"]
    for i in range(1, len(labelspace.names)):
        lines.append(f'  {i}: "{labelspace.names[i]}"')
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "name", "b", "g", "r"])
        for i in range(len(labelspace.names)):
            b, g, r = labelspace.colors_bgr[i]
            writer.writerow([i, labelspace.names[i], b, g, r])
    return yaml_path, csv_path
