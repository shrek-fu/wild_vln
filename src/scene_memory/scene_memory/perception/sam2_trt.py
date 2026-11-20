"""SAM2 image encoder (Hiera backbone + FPN neck) as a TensorRT engine for GdinoSam2ViewDetector.

Two engines per (weights, square input size), cached under ``<sam2 dir>/trt/``: ``enc`` replaces
``SAM2Predictor.get_im_features`` and ``dec`` (prompt encoder + mask decoder, 1..MAX_BOXES boxes) replaces the
box-prompt case of ``_inference_features``; points, mask prompts or multimask output stay in PyTorch. Build on the target (export on a PC with ``--onnx_only`` if ``onnx`` is missing there)::

    python3 src/scene_memory/scripts/sam2_trt.py --imgsz 640
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from scene_memory.perception.trt_runtime import TrtEngine, build_engine, engine_suffix

EXPORT_VERSION = 1
OUTPUTS = ("high_res_0", "high_res_1", "image_embed")
# Box prompts per image the decoder engine accepts; more fall back to PyTorch.
MAX_BOXES = 64


def onnx_path(sam: Path, imgsz: int, part: str = "enc") -> Path:
    sam = Path(sam).expanduser()
    return sam.parent / "trt" / f"{sam.stem}_{part}{imgsz}_v{EXPORT_VERSION}.onnx"


def engine_path(sam: Path, imgsz: int, part: str = "enc") -> Path:
    onnx = onnx_path(sam, imgsz, part)
    return onnx.with_name(f"{onnx.stem}_{engine_suffix()}.engine")


def feat_sizes(imgsz: int) -> list[tuple[int, int]]:
    return [(imgsz // s, imgsz // s) for s in (4, 8, 16)]


def export_decoder_onnx(model, imgsz: int, path: Path) -> None:
    """fp32 ONNX of the box-prompt path of ``SAM2Predictor._inference_features``:
    ``(image_embed, high_res_0, high_res_1, boxes [N,2,2] in input pixels) -> (masks [N,1,h,w], scores [N,1])``.
    The prompt embedding is written out for boxes (corner labels 2/3 plus the padding point) because the generic
    label-mask and per-axis coordinate assignments export as ScatterND, and ``repeat_interleave`` of the batch-1
    positional encoding becomes ``expand`` (TensorRT rejects the OneHot it exports to)."""
    import torch

    model.set_imgsz((imgsz, imgsz))
    pe = model.sam_prompt_encoder

    class Decoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, image_embed, high_res_0, high_res_1, boxes):
            n = boxes.shape[0]
            h, w = pe.input_image_size
            corners = pe.pe_layer._pe_encoding((boxes + 0.5) / boxes.new_tensor([w, h]))
            corners = corners + torch.cat([pe.point_embeddings[2].weight, pe.point_embeddings[3].weight])[None]
            sparse = torch.cat([corners, pe.not_a_point_embed.weight[None].expand(n, 1, -1)], dim=1)
            dense = pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(n, -1, *pe.image_embedding_size)
            masks, scores, _, _ = self.model.sam_mask_decoder(
                image_embeddings=image_embed.expand(n, -1, -1, -1), image_pe=pe.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False,
                repeat_image=False, high_res_features=[high_res_0, high_res_1])
            return masks, scores

    s0, s1, s2 = feat_sizes(imgsz)
    device = next(model.parameters()).device
    dummy = (torch.zeros((1, 256, *s2), device=device), torch.zeros((1, 32, *s0), device=device),
             torch.zeros((1, 64, *s1), device=device), torch.tensor([[[10.0, 20.0], [300.0, 400.0]]] * 3, device=device))
    from ultralytics.nn.modules.transformer import LayerNorm2d

    def layer_norm_2d(self, x):
        # x.shape[1:2] becomes a traced tensor once the batch is dynamic.
        return torch.nn.functional.layer_norm(x.permute(0, 2, 3, 1), self.weight.shape, self.weight, self.bias,
                                              self.eps).permute(0, 3, 1, 2)

    repeat_interleave = torch.repeat_interleave

    def repeat_batch1(x, repeats=None, dim=None, **kwargs):
        if dim == 0 and torch.is_tensor(x) and x.shape[0] == 1:
            return x.expand(repeats, *[-1] * (x.dim() - 1))
        return repeat_interleave(x, repeats, dim=dim, **kwargs)

    path.parent.mkdir(parents=True, exist_ok=True)
    original = LayerNorm2d.forward
    LayerNorm2d.forward = layer_norm_2d
    torch.repeat_interleave = repeat_batch1
    try:
        with torch.inference_mode():
            torch.onnx.export(Decoder().eval(), dummy, str(path),
                              input_names=["image_embed", "high_res_0", "high_res_1", "boxes"],
                              output_names=["masks", "scores"],
                              dynamic_axes={"boxes": {0: "n"}, "masks": {0: "n"}, "scores": {0: "n"}},
                              opset_version=17, do_constant_folding=True, dynamo=False)
    finally:
        LayerNorm2d.forward = original
        torch.repeat_interleave = repeat_interleave


def export_onnx(model, imgsz: int, path: Path) -> None:
    """fp32 ONNX of ``image -> (high_res_0, high_res_1, image_embed)``, same as ``SAM2Predictor.get_im_features``."""
    import torch

    model.set_imgsz((imgsz, imgsz))
    sizes = feat_sizes(imgsz)

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, image):
            _, feats, _, _ = self.model._prepare_backbone_features(self.model.forward_image(image))
            if self.model.directly_add_no_mem_embed:
                feats[-1] = feats[-1] + self.model.no_mem_embed
            return tuple(f.permute(1, 2, 0).reshape(1, -1, *s) for f, s in zip(feats, sizes))

    path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    with torch.inference_mode():
        torch.onnx.export(Encoder().eval(), (torch.zeros((1, 3, imgsz, imgsz), device=device),), str(path),
                          input_names=["image"], output_names=list(OUTPUTS), opset_version=17,
                          do_constant_folding=True, dynamo=False)


class TrtSam2Encoder(TrtEngine):
    def __call__(self, image, dtype):
        out = self.run(image)
        return {"image_embed": out["image_embed"].to(dtype),
                "high_res_feats": [out["high_res_0"].to(dtype), out["high_res_1"].to(dtype)]}


class TrtSam2Decoder(TrtEngine):
    def __call__(self, features, boxes, dtype):
        high = features["high_res_feats"]
        out = self.run(image_embed=features["image_embed"], high_res_0=high[0], high_res_1=high[1], boxes=boxes)
        return out["masks"].to(dtype).flatten(0, 1), out["scores"].to(dtype).flatten(0, 1)


_STATE: dict = {"sam": None, "engines": {}}


def use(sam: Path | None) -> None:
    """Route the SAM2 image encoder and the box-prompt decoder through TensorRT engines for these weights
    (None: PyTorch). Whatever has no engine for the predictor's input size stays on PyTorch."""
    from ultralytics.models.sam.predict import SAM2Predictor

    _STATE["sam"] = Path(sam).expanduser() if sam else None
    if getattr(SAM2Predictor, "_stvln_trt", False):
        return
    get_im_features, inference_features = SAM2Predictor.get_im_features, SAM2Predictor._inference_features

    def trt_get_im_features(self, im):
        engine = _engine(self.imgsz, "enc", im.device)
        if engine is None:
            return get_im_features(self, im)
        return engine(im, next(self.model.parameters()).dtype)

    def trt_inference_features(self, features, points=None, labels=None, masks=None, multimask_output=False,
                               img_idx=-1):
        engine = _engine(self.imgsz, "dec", self.device)
        boxes_only = (
            engine is not None and isinstance(features, dict) and masks is None and not multimask_output
            and points is not None and points.ndim == 3 and points.shape[1:] == (2, 2)
            and 0 < points.shape[0] <= MAX_BOXES and labels is not None
            and bool((labels == labels.new_tensor([2, 3])).all())
            and features["image_embed"].shape[0] == 1
        )
        if not boxes_only:
            return inference_features(self, features, points, labels, masks, multimask_output, img_idx)
        return engine(features, points, next(self.model.parameters()).dtype)

    SAM2Predictor.get_im_features = trt_get_im_features
    SAM2Predictor._inference_features = trt_inference_features
    SAM2Predictor._stvln_trt = True


def _engine(imgsz, part: str, device):
    if _STATE["sam"] is None or len(set(imgsz)) != 1:
        return None
    key = (_STATE["sam"], imgsz[0], part)
    if key not in _STATE["engines"]:
        path = engine_path(_STATE["sam"], imgsz[0], part)
        cls = TrtSam2Encoder if part == "enc" else TrtSam2Decoder
        _STATE["engines"][key] = cls(path, device) if path.is_file() else None
    return _STATE["engines"][key]


def has_engine(sam: Path, imgsz: int) -> bool:
    return engine_path(sam, imgsz).is_file()


def main() -> None:
    from scene_memory import paths

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sam2_model", default=str(paths.SAM2))
    p.add_argument("--imgsz", type=int, default=640, help="the detector's --sam_imgsz")
    p.add_argument("--onnx_only", action="store_true", help="export the ONNX and stop (e.g. on a PC)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    model = None
    shapes = {"enc": [], "dec": ["--minShapes=boxes:1x2x2", "--optShapes=boxes:8x2x2",
                                 f"--maxShapes=boxes:{MAX_BOXES}x2x2"]}
    for part, export in (("enc", export_onnx), ("dec", export_decoder_onnx)):
        onnx = onnx_path(args.sam2_model, args.imgsz, part)
        if args.force or not onnx.is_file():
            from ultralytics import SAM

            if model is None:
                # CPU: the decoder's repeat_interleave traces with a CPU index tensor.
                model = SAM(str(Path(args.sam2_model).expanduser())).model.cpu().float().eval()
            t0 = time.time()
            export(model, args.imgsz, onnx)
            print(f"exported {onnx} in {time.time() - t0:.0f} s ({onnx.stat().st_size / 1e6:.0f} MB)")
        if args.onnx_only:
            continue
        engine = engine_path(args.sam2_model, args.imgsz, part)
        if engine.is_file() and not args.force:
            print(f"engine exists: {engine}")
            continue
        print(f"building {engine} (fp16; log {engine.with_suffix('.log')})")
        t0 = time.time()
        build_engine(onnx, engine, extra=shapes[part])
        print(f"built in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
