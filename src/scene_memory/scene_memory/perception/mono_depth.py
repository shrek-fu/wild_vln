"""Metric monocular depth for the four Insta360 pinhole views.

Uses Depth-Anything-V2 Metric Hypersim ViT-S (indoor, max 20 m). Near range is
the strong regime; far depth is coarser but still usable for Hydra when no
lidar is present. The four virtual views share one optical centre, so they
cannot form stereo — each view is estimated independently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from scene_memory import paths

THIRD_PARTY = paths.PKG / "third_party"
DEFAULT_WEIGHTS = paths.MONO_DEPTH

_MODEL_CFGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _ensure_path() -> None:
    root = str(THIRD_PARTY)
    if root not in sys.path:
        sys.path.insert(0, root)


class MetricMonoDepth:
    """BGR uint8 → float32 metres, same HxW."""

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        encoder: str = "vits",
        max_depth: float = 20.0,
        input_size: int = 518,
        device: str = "",
    ) -> None:
        _ensure_path()
        from depth_anything_v2.dpt import DepthAnythingV2

        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(
                f"metric depth weights missing: {path}. "
                "Download Depth-Anything-V2-Metric-Hypersim-Small into models/depth_anything_v2/."
            )
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = {**_MODEL_CFGS[encoder], "max_depth": float(max_depth)}
        model = DepthAnythingV2(**cfg)
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self.model = model.eval().to(self.device)
        self.input_size = int(input_size)
        self.max_depth = float(max_depth)

    @torch.inference_mode()
    def infer_bgr(self, bgr: np.ndarray) -> np.ndarray:
        """Return HxW float32 metres. Invalid / behind stay 0 after ReLU."""
        if bgr is None or bgr.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        # Official infer_image always moves the tensor to a global DEVICE; call
        # the pieces here so the tensor lands on the same device as the weights.
        image, (h, w) = self.model.image2tensor(bgr, self.input_size)
        image = image.to(self.device, non_blocking=True)
        depth = self.model.forward(image)
        depth = torch.nn.functional.interpolate(
            depth[:, None], (h, w), mode="bilinear", align_corners=True
        )[0, 0]
        out = depth.float().cpu().numpy().astype(np.float32)
        # Keep Hydra's invalid convention for absurd values.
        out[~np.isfinite(out)] = 0.0
        out[out < 0.05] = 0.0
        out[out > self.max_depth] = 0.0
        return out

    def infer_rgb(self, rgb: np.ndarray) -> np.ndarray:
        return self.infer_bgr(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
