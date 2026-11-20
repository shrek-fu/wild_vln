"""DA360 inference wrapper for live / offline equirect frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

import networks
from checkpoint import checkpoint_meta, load_checkpoint, state_dict_from_checkpoint


class DA360Runner:
    def __init__(self, model_path: str | Path, device: str | None = None) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = load_checkpoint(str(model_path), map_location="cpu")
        meta = checkpoint_meta(ckpt)
        Net = getattr(networks, meta["net"])
        self.height = int(meta["height"])
        self.width = int(meta["width"])
        self.model = Net(
            self.height, self.width, dinov2_encoder=meta["dinov2_encoder"]
        )
        state = state_dict_from_checkpoint(ckpt, self.model.state_dict())
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @torch.inference_mode()
    def predict_depth(
        self, bgr_or_rgb: np.ndarray, *, is_bgr: bool = True, out_size: tuple[int, int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(depth_hw float32, rgb_out_hw3 float 0..1)``.

        Depth is relative (scale-invariant). ``out_size`` is (width, height); default
        keeps the network resolution.
        """
        if is_bgr:
            rgb = cv2.cvtColor(bgr_or_rgb, cv2.COLOR_BGR2RGB)
        else:
            rgb = bgr_or_rgb
        resized = cv2.resize(
            rgb, dsize=(self.width, self.height), interpolation=cv2.INTER_CUBIC
        )
        tensor = self.normalize(self.to_tensor(resized.copy())).unsqueeze(0).to(self.device)
        outputs = self.model(tensor)
        disp = outputs["pred_disp"].detach().float().cpu().numpy()[0, 0]
        depth = 1.0 / np.maximum(disp, 1e-6)
        depth = depth / max(float(depth.min()), 1e-6)
        rgb_f = resized.astype(np.float32) / 255.0
        if out_size is not None:
            ow, oh = out_size
            depth = cv2.resize(depth, dsize=(ow, oh), interpolation=cv2.INTER_LINEAR)
            rgb_f = cv2.resize(rgb_f, dsize=(ow, oh), interpolation=cv2.INTER_LINEAR)
            # Keep relative scale consistent after resize: re-normalize by min.
            depth = depth / max(float(depth[depth > 0].min()), 1e-6)
        return depth.astype(np.float32), rgb_f
