"""DepthART metric depth (TinyViM-S, 448) as a TensorRT engine for the pinhole views.

DepthART (refer/DepthART, CC BY-NC parts: research use only) predicts metric depth from RGB + intrinsics. The
intrinsics are baked into the ONNX graph, so an engine belongs to one (domain, input size, K, TensorRT version,
GPU); every 640x640 / 90 deg Insta360 view shares one. The TinyViM selective scan is a custom TensorRT plugin
(``com.depthart::SelectiveScan``) built on the target from refer/DepthART/deploy/shared/selective_scan.

Weights: ``models/depthart/depthart_metric_{indoor,outdoor}_s_448.pth`` (HF Fengxue93/DepthART). Steps::

    # PC (needs onnx, einops, timm): export the graph for the view intrinsics
    python3 src/scene_memory/scripts/depthart_trt.py --onnx_only
    # target: plugin + engine (+ PyTorch vs TensorRT check on a recorded view)
    python3 src/scene_memory/scripts/depthart_trt.py --build_plugin --check_image <view.jpg>
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from scene_memory import paths
from scene_memory.perception.trt_runtime import TrtEngine, build_engine_api, engine_suffix

EXPORT_VERSION = 2
DOMAINS = ("indoor", "outdoor")
MAX_DEPTH = {"indoor": 10.0, "outdoor": 80.0}
INPUT = 448
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SCAN_ROOT = paths.DEPTHART_ROOT / "deploy" / "shared" / "selective_scan"


def checkpoint(domain: str) -> Path:
    return paths.DEPTHART / f"depthart_metric_{domain}_s_448.pth"


def input_size(width: int, height: int) -> tuple[int, int]:
    """DepthART's lower-bound resize: keep the aspect ratio, both sides >= 448 and multiples of 32."""
    scale = max(INPUT / width, INPUT / height)

    def side(value: float) -> int:
        rounded = int(round(value / 32) * 32)
        return rounded if rounded >= INPUT else int(np.ceil(value / 32) * 32)

    return side(scale * width), side(scale * height)


def view_K(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Intrinsics of a ``width x height`` view (same convention as ``boxer_lift.pinhole_rays``), scaled to the
    network input."""
    fx = 0.5 * width / np.tan(np.deg2rad(hfov_deg) * 0.5)
    fy = 0.5 * height / np.tan(np.deg2rad(hfov_deg) * 0.5)
    in_w, in_h = input_size(width, height)
    sx, sy = in_w / width, in_h / height
    return np.array([[fx * sx, 0, (width - 1) * 0.5 * sx], [0, fy * sy, (height - 1) * 0.5 * sy], [0, 0, 1]],
                    dtype=np.float32)


def engine_key(domain: str, K: np.ndarray, in_w: int, in_h: int) -> str:
    k = ",".join(f"{v:.2f}" for v in (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    digest = hashlib.sha1(f"{domain}|{in_h}x{in_w}|{k}|v{EXPORT_VERSION}".encode()).hexdigest()[:10]
    return f"metric_s_{domain}_{digest}_{in_h}x{in_w}"


def onnx_path(domain: str, K: np.ndarray, in_w: int, in_h: int) -> Path:
    return paths.DEPTHART / "trt" / f"{engine_key(domain, K, in_w, in_h)}.onnx"


def engine_path(domain: str, K: np.ndarray, in_w: int, in_h: int) -> Path:
    return paths.DEPTHART / "trt" / f"{engine_key(domain, K, in_w, in_h)}_{engine_suffix()}.engine"


def plugin_path() -> Path:
    return paths.DEPTHART / "trt" / f"libdepthart_selective_scan_{engine_suffix()}.so"


def load_plugin() -> None:
    path = plugin_path()
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run scripts/depthart_trt.py --build_plugin on this device")
    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def build_plugin() -> Path:
    """Compile the SelectiveScan TensorRT plugin in refer/DepthART and copy it next to the engines."""
    import torch

    major, minor = torch.cuda.get_device_capability(0)
    env = {**os.environ, "TORCH_CUDA_ARCH_LIST": f"{major}.{minor}", "MAX_JOBS": os.environ.get("MAX_JOBS", "4")}
    subprocess.run([sys.executable, "setup_trt_plugin.py", "build_ext", "--inplace"], cwd=SCAN_ROOT, env=env,
                   check=True)
    built = next(SCAN_ROOT.glob("depthart_selective_scan_trt_plugin*.so"))
    out = plugin_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, out)
    return out


def _depthart_modules():
    """refer/DepthART's metric package uses top-level module names (``model``, ``network``): import on demand."""
    for sub in ("metric", "deploy/shared/selective_scan", "deploy/shared"):
        p = str(paths.DEPTHART_ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import depthart_selective_scan as scan
    from export_helpers import install_exportable_sdpa
    from model import load_model
    from network import tvimblock

    return load_model, tvimblock, scan, install_exportable_sdpa


def load_torch(domain: str, device: str):
    load_model, tvimblock, scan, _ = _depthart_modules()
    model = load_model(checkpoint(domain), "S", domain, device)
    scan.install_depthart(tvimblock)
    return model


def export_onnx(domain: str, K: np.ndarray, in_w: int, in_h: int, path: Path, opset: int = 17) -> int:
    """Export on CPU with K as a constant; returns the node count."""
    import onnx
    import torch

    load_model, tvimblock, scan, install_exportable_sdpa = _depthart_modules()
    model = load_model(checkpoint(domain), "S", domain, "cpu")
    scan.install_depthart(tvimblock)
    from depthart_selective_scan import onnx as scan_onnx

    def scan_symbolic(g, u, *rest):
        # Without a static output type every shape after the scan is unknown and squeeze(dim) exports as an
        # ``If`` whose branch shapes TensorRT 10.3 rejects. The scan output has u's shape, in float32.
        return scan_onnx._symbolic(g, u, *rest).setType(u.type().with_dtype(torch.float32))

    torch.onnx.register_custom_op_symbolic("depthart::selective_scan", scan_symbolic, opset)
    # Also disables the MultiheadAttention fast path (refer/DepthART's train-mode MHA export is not needed).
    install_exportable_sdpa()

    class FixedK(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.register_buffer("K", torch.from_numpy(K)[None])

        def forward(self, image):
            return self.model(image, self.K)[:, None]

    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(FixedK().eval(), (torch.randn(1, 3, in_h, in_w),), str(path), input_names=["image"],
                          output_names=["depth"], opset_version=opset, do_constant_folding=True,
                          dynamo=False)
    graph = onnx.load(str(path))
    try:
        import onnx_graphsurgeon as gs

        g = gs.import_onnx(graph)
        g.fold_constants().cleanup().toposort()
        graph = gs.export_onnx(g)
        onnx.save(graph, str(path))
    except ImportError:
        pass
    ops = {n.op_type for n in graph.graph.node}
    if "SelectiveScan" not in ops:
        raise RuntimeError("export contains no SelectiveScan plugin nodes")
    if "If" in ops:
        raise RuntimeError("export contains If nodes (TensorRT rejects their branch shapes)")
    return len(graph.graph.node)


def keep_fp32(layer) -> bool:
    """Normalizations and the final depth convolutions overflow in fp16 (refer/DepthART build_trt_engine.py)."""
    import tensorrt as trt

    return layer.type == trt.LayerType.NORMALIZATION or "/depth_head/output_conv" in layer.name


class DepthArtMetric:
    """BGR uint8 view -> float32 metres (same HxW, 0 = invalid) through the TensorRT engine for its K."""

    def __init__(self, width: int, height: int, hfov_deg: float, domain: str = "outdoor", device: str = "cuda",
                 max_depth: float = 0.0) -> None:
        import torch

        self.device = torch.device(device)
        self.width, self.height = int(width), int(height)
        self.in_w, self.in_h = input_size(self.width, self.height)
        K = view_K(self.width, self.height, hfov_deg)
        self.engine_file = engine_path(domain, K, self.in_w, self.in_h)
        if not self.engine_file.is_file():
            raise FileNotFoundError(f"{self.engine_file} missing: run scripts/depthart_trt.py --domain {domain}")
        load_plugin()
        self.engine = TrtEngine(self.engine_file, self.device)
        self.max_depth = float(max_depth) or MAX_DEPTH[domain]
        self.mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1) * 255.0
        self.std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1) * 255.0

    def preprocess(self, bgr: np.ndarray):
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).to(self.device).permute(2, 0, 1)[None].float()
        x = F.interpolate(x, (self.in_h, self.in_w), mode="bicubic", align_corners=False)
        return (x - self.mean) / self.std

    def infer_bgr(self, bgr: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        if bgr is None or bgr.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        h, w = bgr.shape[:2]
        with torch.inference_mode():
            depth = self.engine.run(self.preprocess(bgr))["depth"].float()
            depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)[0, 0]
            out = depth.cpu().numpy()
        out[~np.isfinite(out) | (out < 0.05) | (out > self.max_depth)] = 0.0
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calib_dir", default=str(paths.CALIB))
    parser.add_argument("--view", default="front", help="calibrated view whose size / hfov defines K")
    parser.add_argument("--domain", default="all", choices=(*DOMAINS, "all"))
    parser.add_argument("--onnx_only", action="store_true", help="export the ONNX graph only (PC)")
    parser.add_argument("--build_plugin", action="store_true", help="compile the SelectiveScan plugin first")
    parser.add_argument("--check_image", default="", help="compare PyTorch and TensorRT depth on this BGR image")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from scene_memory.perception.boxer_lift import load_pinhole_views

    spec = next(v for v in load_pinhole_views(args.calib_dir) if v["name"] == args.view)
    width, height, hfov = spec["width"], spec["height"], spec["hfov_deg"]
    in_w, in_h = input_size(width, height)
    K = view_K(width, height, hfov)
    print(f"view {args.view} {width}x{height} hfov {hfov} -> input {in_w}x{in_h} K fx={K[0, 0]:.2f} cx={K[0, 2]:.2f}")
    if args.build_plugin and not args.onnx_only:
        print("plugin:", build_plugin())
    for domain in DOMAINS if args.domain == "all" else (args.domain,):
        onnx_file = onnx_path(domain, K, in_w, in_h)
        if args.force or not onnx_file.is_file():
            t = time.time()
            nodes = export_onnx(domain, K, in_w, in_h, onnx_file)
            print(f"{domain}: exported {onnx_file.name} ({nodes} nodes, {time.time() - t:.0f} s)")
        if args.onnx_only:
            continue
        engine_file = engine_path(domain, K, in_w, in_h)
        if args.force or not engine_file.is_file():
            load_plugin()
            t = time.time()
            info = build_engine_api(onnx_file, engine_file, fp16=True, keep_fp32=keep_fp32)
            print(f"{domain}: built {engine_file.name} {info} ({time.time() - t:.0f} s)")
        if args.check_image:
            check(args.check_image, width, height, hfov, domain)


def check(image: str, width: int, height: int, hfov: float, domain: str) -> None:
    import cv2
    import torch

    bgr = cv2.resize(cv2.imread(image), (width, height), interpolation=cv2.INTER_AREA)
    trt_model = DepthArtMetric(width, height, hfov, domain)
    for _ in range(5):
        trt_model.infer_bgr(bgr)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        d_trt = trt_model.infer_bgr(bgr)
    ms = (time.time() - t) / 20 * 1e3
    print(f"{domain}: TensorRT {ms:.1f} ms/frame (preprocess + copy back included)", flush=True)
    # CPU runs the reference scan; the CUDA path needs refer/DepthART's separate PyTorch extension.
    torch_model = load_torch(domain, "cpu")
    K = torch.from_numpy(view_K(width, height, hfov))[None]
    with torch.inference_mode():
        d = torch_model(trt_model.preprocess(bgr).cpu(), K)[:, None]
        d = torch.nn.functional.interpolate(d, (height, width), mode="bilinear", align_corners=True)[0, 0]
    d_ref = d.float().cpu().numpy()
    valid = (d_trt > 0) & (d_ref > 0.05)
    rel = np.abs(d_trt[valid] - d_ref[valid]) / d_ref[valid]
    print(f"{domain}: TensorRT vs PyTorch fp32 AbsRel {rel.mean():.4f} p99 {np.percentile(rel, 99):.4f}; "
          f"median depth {np.median(d_ref):.2f} m")
    np.save(paths.DEPTHART / "trt" / f"check_{domain}.npy", np.stack([d_trt, d_ref]))


if __name__ == "__main__":
    main()
