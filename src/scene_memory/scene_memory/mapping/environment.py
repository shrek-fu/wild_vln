"""Indoor / outdoor from the 360° image: SigLIP zero-shot over four pinhole views, with hysteresis.

On the indoor session 20261107_112406 every frame scores p(outdoor) <= 0.007; street images score >= 0.98,
so the thresholds below leave a wide margin. The EMA + hysteresis keep a doorway or a window from flipping
the state.
"""

from __future__ import annotations

import math

import numpy as np

INDOOR = ["a photo taken indoors", "an indoor room", "a corridor inside a building", "an office interior",
          "a staircase inside a building", "inside an elevator"]
OUTDOOR = ["a photo taken outdoors", "a street outside", "outside a building with the sky", "a parking lot",
           "a park with trees", "a campus road between buildings"]


def pinhole_maps(eq_w: int, eq_h: int, size: int, hfov_deg: float, yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """cv2.remap maps for a level pinhole view of an equirect (X forward, Y left, Z up, yaw positive left)."""
    f = 0.5 * size / math.tan(0.5 * math.radians(hfov_deg))
    yaw = math.radians(yaw_deg)
    grid = (np.arange(size) + 0.5 - 0.5 * size) / f
    a, b = np.meshgrid(grid, grid)
    dx = math.cos(yaw) + a * math.sin(yaw)
    dy = math.sin(yaw) - a * math.cos(yaw)
    dz = -b
    lon = np.arctan2(dy, dx)
    lat = np.arctan2(dz, np.hypot(dx, dy))
    map_u = (eq_w * (0.5 - lon / (2 * math.pi)) - 0.5).astype(np.float32)
    map_v = np.clip(eq_h * (0.5 - lat / math.pi) - 0.5, 0, eq_h - 1).astype(np.float32)
    return map_u, map_v


class EnvironmentClassifier:
    def __init__(self, encoder=None, enter_outdoor: float = 0.7, enter_indoor: float = 0.3, alpha: float = 0.5,
                 view_size: int = 256, hfov_deg: float = 100.0, state: str = "indoor"):
        self._encoder = encoder
        self.enter_outdoor, self.enter_indoor, self.alpha = enter_outdoor, enter_indoor, alpha
        self.view_size, self.hfov_deg = view_size, hfov_deg
        self.state = state
        self.p = 1.0 if state == "outdoor" else 0.0
        self._maps: dict[tuple, list] = {}
        self._text = None

    @property
    def encoder(self):
        if self._encoder is None:
            from scene_memory.query.semantic_index import DEFAULT_SIGLIP, SiglipEncoder

            self._encoder = SiglipEncoder(DEFAULT_SIGLIP)
        return self._encoder

    def p_outdoor(self, equirect_bgr: np.ndarray) -> float:
        import cv2
        from PIL import Image

        h, w = equirect_bgr.shape[:2]
        maps = self._maps.get((w, h))
        if maps is None:
            maps = self._maps[(w, h)] = [pinhole_maps(w, h, self.view_size, self.hfov_deg, yaw)
                                         for yaw in (0.0, 90.0, 180.0, -90.0)]
        views = [Image.fromarray(cv2.cvtColor(cv2.remap(equirect_bgr, u, v, cv2.INTER_LINEAR,
                                                        borderMode=cv2.BORDER_WRAP), cv2.COLOR_BGR2RGB))
                 for u, v in maps]
        enc = self.encoder
        if self._text is None:
            self._text = enc.texts(INDOOR + OUTDOOR)
        logits = enc.logit_scale * enc.images(views) @ self._text.T
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        p_out = e[:, len(INDOOR):].sum(1) / e.sum(1)
        return float(np.mean(p_out))

    def update(self, equirect_bgr: np.ndarray) -> str:
        self.p = (1 - self.alpha) * self.p + self.alpha * self.p_outdoor(equirect_bgr)
        if self.state == "indoor" and self.p > self.enter_outdoor:
            self.state = "outdoor"
        elif self.state == "outdoor" and self.p < self.enter_indoor:
            self.state = "indoor"
        return self.state
