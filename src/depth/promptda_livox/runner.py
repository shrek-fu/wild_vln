"""PromptDA model wrapper: RGB + lidar prompt_depth → metric depth (metres)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Vendor PromptDA + local dinov2 from the RoboSense refer tree (or a clone next to this package).
_HERE = Path(__file__).resolve().parent
_WS = _HERE.parents[2]  # .../10SpationTemporalSystemVLN
_REF = _WS / "refer" / "robosense_ac_perception" / "python" / "PromptDA"
_LOCAL = _HERE / "PromptDA"
PROMPTDA_ROOT = _LOCAL if (_LOCAL / "promptda").is_dir() else _REF


def _ensure_promptda_path() -> Path:
    if not (PROMPTDA_ROOT / "promptda" / "promptda.py").is_file():
        raise FileNotFoundError(
            f"PromptDA sources not found at {PROMPTDA_ROOT}. Expected "
            "refer/robosense_ac_perception/python/PromptDA or src/depth/promptda_livox/PromptDA")
    root = str(PROMPTDA_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROMPTDA_ROOT


class PromptDARunner:
    """Loads Prompt-Depth-Anything (default: ViT-S for 8 GB GPUs)."""

    def __init__(self, ckpt: str | Path, encoder: str = "vits", device: str = "") -> None:
        _ensure_promptda_path()
        from promptda.promptda import PromptDA

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = Path(ckpt).expanduser()
        if not ckpt.is_file():
            raise FileNotFoundError(f"PromptDA checkpoint missing: {ckpt}")
        # PromptDA.__init__ loads dinov2 from the package's torchhub/ and then the ckpt.
        self.model = PromptDA(encoder=encoder, ckpt_path=str(ckpt)).to(self.device).eval()
        self.encoder = encoder

    @torch.inference_mode()
    def predict(self, bgr: np.ndarray, prompt_depth: np.ndarray) -> np.ndarray:
        """``bgr`` HxWx3 uint8, ``prompt_depth`` HxW metres (0 = unknown). Returns HxW metres."""
        if prompt_depth.shape[:2] != bgr.shape[:2]:
            prompt_depth = cv2.resize(prompt_depth, (bgr.shape[1], bgr.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
        if not (prompt_depth > 0).any():
            raise ValueError("prompt_depth has no lidar hits; cannot run PromptDA")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # PromptDA expects multiples of patch_size 14.
        h, w = rgb.shape[:2]
        nh, nw = (h // 14) * 14, (w // 14) * 14
        if (nh, nw) != (h, w):
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            prompt_depth = cv2.resize(prompt_depth, (nw, nh), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        prompt = torch.from_numpy(prompt_depth.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        depth = self.model.predict(image, prompt)
        out = depth.squeeze().detach().float().cpu().numpy()
        if out.shape != bgr.shape[:2]:
            out = cv2.resize(out, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        return out.astype(np.float32)


def default_ckpt(encoder: str = "vits") -> Path:
    name = {"vits": "prompt-depth-anything-vits.ckpt",
            "vitl": "prompt-depth-anything-vitl.ckpt"}.get(encoder, f"prompt-depth-anything-{encoder}.ckpt")
    return Path(__file__).resolve().parent / "checkpoints" / name
