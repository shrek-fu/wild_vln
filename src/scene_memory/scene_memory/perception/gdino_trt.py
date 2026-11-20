"""GroundingDINO as a TensorRT engine for GdinoSam2ViewDetector.

The prompt is baked in: the ONNX graph takes only ``pixel_values`` and the BERT output is a constant, so an
engine belongs to one (prompt text, input size, TensorRT version, GPU). Engines are cached next to the model
under ``trt/<key>.engine``; the detector loads a matching engine and otherwise stays on PyTorch.

Build on the target (Jetson) for the prompts/ground classes the pipeline uses::

    python3 src/scene_memory/scripts/gdino_trt.py --ground_classes src/scene_memory/config/ground_classes.txt

The ONNX export needs the ``onnx`` package; without it, export on a PC with the same prompts
(``--onnx_only``) and copy ``trt/<key>.onnx`` next to the model on the target, then run the build there.
Deformable attention is exported as TensorRT's ``MultiscaleDeformableAttnPlugin_TRT`` (in libnvinfer_plugin), so
the ONNX only runs in TensorRT. With ``onnx_graphsurgeon`` importable the export folds the constant subgraphs
(~7% faster engine). ``--precision int8`` calibrates in PyTorch during the export and writes ``trt/<key>_qdq.onnx``.

Orin NX 16GB, 800x800, 34 classes: fp16 161 ms, mixed 177 ms, int8 157 ms (recall vs fp32 PyTorch 0.98 / 0.99 /
0.93): the model is bound by deformable attention and memory-bound layers, so INT8 GEMMs barely help.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from scene_memory.perception.trt_runtime import TrtEngine, build_engine, build_engine_api, engine_suffix

# Bump when the exported graph changes so stale engines are not picked up.
EXPORT_VERSION = 3
# fp16: everything fp16 (trtexec). mixed: fp16 with the overflow-prone layers kept fp32 (see keep_fp32).
# int8: explicit Q/DQ on the Linear/Conv layers of quantize_module (scales calibrated in PyTorch on recorded
# views, exported in their own ONNX), fp16 elsewhere. TensorRT 10.3's implicit INT8 calibration fails on this graph.
PRECISIONS = ("fp16", "mixed", "int8")
# Module name parts kept in float by int8: patch embedding, the constant text path, text-image fusion, the
# deformable-attention sampling offsets/weights and the box/class heads.
NO_QUANT = ("embeddings", "text", "fusion_layer", "sampling_offsets", "attention_weights", "reference_points",
            "bbox_embed", "class_embed", "enc_output", "encoder_output", "query_position")


def engine_key(text: str, height: int, width: int) -> str:
    """ONNX depends on (prompt, size); the engine also on TensorRT and the GPU, kept in the file name."""
    digest = hashlib.sha1(f"{text}|{height}x{width}|v{EXPORT_VERSION}".encode()).hexdigest()[:12]
    return f"{digest}_{height}x{width}"


def engine_path(model_dir: Path, text: str, height: int, width: int, precision: str = "fp16") -> Path:
    tag = "" if precision == "fp16" else f"_{precision}"
    return Path(model_dir) / "trt" / f"{engine_key(text, height, width)}{tag}_{engine_suffix()}.engine"


def quantize_module(name: str, module) -> bool:
    import torch

    return isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)) and not any(k in name for k in NO_QUANT)


def calibrate(model, text_inputs: dict, batches, percentile: float = 99.99) -> dict[str, float]:
    """Input amax per quantized module: the largest per-image ``percentile`` of |input| over the batches."""
    import torch

    amax: dict[str, float] = {}
    hooks = []

    def hook(name):
        def record(module, args):
            x = args[0].detach().abs().flatten().float()
            k = max(1, min(x.numel(), int(np.ceil(x.numel() * percentile / 100.0))))
            amax[name] = max(amax.get(name, 0.0), float(x.kthvalue(k).values))

        return record

    for name, module in model.named_modules():
        if quantize_module(name, module):
            hooks.append(module.register_forward_pre_hook(hook(name)))
    try:
        with torch.inference_mode():
            for pixels in batches:
                mask = torch.ones(pixels.shape[:1] + pixels.shape[2:], dtype=torch.long, device=pixels.device)
                model(pixel_values=pixels, pixel_mask=mask, **text_inputs)
    finally:
        for h in hooks:
            h.remove()
    return amax


def _fake_quant(module, amax: float):
    """Forward with int8 fake quantization (per-tensor input, per-output-channel weight); exports as Q/DQ."""
    import torch
    import torch.nn.functional as F

    def forward(x):
        x = torch.fake_quantize_per_tensor_affine(x, max(amax, 1e-8) / 127.0, 0, -128, 127)
        w = module.weight
        scale = (w.detach().abs().flatten(1).amax(dim=1).clamp(min=1e-8) / 127.0).float()
        w = torch.fake_quantize_per_channel_affine(w, scale, torch.zeros_like(scale, dtype=torch.int32), 0, -128, 127)
        if isinstance(module, torch.nn.Linear):
            return F.linear(x, w, module.bias)
        return module._conv_forward(x, w, module.bias)

    return forward


def keep_fp32(layer) -> bool:
    """Layers left in fp32 by ``mixed``/``int8``: layer norms, the text-image fusion (bi-attention logits reach
    the fp16 range) and the class/box heads with the sigmoid/inverse-sigmoid box refinement."""
    import tensorrt as trt

    if layer.type == trt.LayerType.NORMALIZATION:
        return True
    name = layer.name
    if any(k in name for k in ("fusion_layer", "class_embed", "bbox_embed", "enc_output", "reference_points_head")):
        return True
    return name.rsplit("/", 1)[-1].startswith(("Log", "Sigmoid"))


def onnx_path(model_dir: Path, text: str, height: int, width: int, precision: str = "fp16") -> Path:
    tag = "_qdq" if precision == "int8" else ""
    return Path(model_dir) / "trt" / f"{engine_key(text, height, width)}{tag}.onnx"


def export_onnx(model, text_inputs: dict, height: int, width: int, path: Path,
                amax: dict[str, float] | None = None) -> None:
    """fp32 ONNX of ``model(pixel_values) -> (logits, pred_boxes)`` with the text inputs as constants;
    with ``amax`` (from ``calibrate``) those modules carry int8 Q/DQ."""
    import torch

    class ImageOnly(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.text = {k: v.clone() for k, v in text_inputs.items()}

        def forward(self, pixel_values):
            mask = torch.ones((1, height, width), dtype=torch.long, device=pixel_values.device)
            out = self.model(pixel_values=pixel_values, pixel_mask=mask, **self.text)
            return out.logits, out.pred_boxes

    from transformers.models.grounding_dino import modeling_grounding_dino as mg

    def contrastive_pad(self, vision_hidden_state, text_hidden_state, text_token_mask):
        # Same as HF, but pads with F.pad: the slice assignment exports as a ScatterND TensorRT 10.3 rejects.
        out = vision_hidden_state @ text_hidden_state.transpose(-1, -2)
        out = out.masked_fill(~text_token_mask[:, None, :], float("-inf"))
        return torch.nn.functional.pad(out, (0, self.max_text_len - out.shape[-1]), value=float("-inf"))

    msda = mg.MultiScaleDeformableAttention
    msda_forward = msda.forward

    class Plugin(torch.autograd.Function):
        """PyTorch result while tracing; exported as TensorRT's MultiscaleDeformableAttnPlugin_TRT
        (value [B,S,M,D], shapes [L,2] i32, level start [L] i32, locations [B,Q,M,L,P,2] in 0..1,
        weights [B,Q,M,L,P] -> [B,Q,M,D]). TensorRT lowers GridSample to slow gather chains."""

        @staticmethod
        def forward(ctx, value, shapes, level_start, locations, weights):
            b, _, m, d = value.shape
            shapes_list = [tuple(s) for s in shapes.tolist()]
            out = msda_forward(None, value, shapes, shapes_list, level_start, locations, weights, 64)
            return out.view(b, -1, m, d)

        @staticmethod
        def symbolic(g, value, shapes, level_start, locations, weights):
            int32 = 6
            return g.op("MultiscaleDeformableAttnPlugin_TRT", value, g.op("Cast", shapes, to_i=int32),
                        g.op("Cast", level_start, to_i=int32), locations, weights)

    def msda_plugin(self, value, shapes, shapes_list, level_start, locations, weights, im2col_step):
        return Plugin.apply(value, shapes, level_start, locations, weights).flatten(2)

    path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    dummy = torch.zeros((1, 3, height, width), device=device)
    # The prompt is fixed: run BERT once and export its output as a constant, so the graph holds no text
    # backbone (~50k TensorRT layers otherwise).
    text_backbone = model.model.text_backbone
    text_forward, patched = text_backbone.forward, "forward" in text_backbone.__dict__
    captured = {}

    def capture(*args, **kwargs):
        captured["out"] = text_forward(*args, **kwargs)
        return captured["out"]

    original = mg.GroundingDinoContrastiveEmbedding.forward
    mg.GroundingDinoContrastiveEmbedding.forward = contrastive_pad
    msda.forward = msda_plugin
    text_backbone.forward = capture
    quantized = [m for name, m in model.named_modules() if name in (amax or {})]
    try:
        with torch.inference_mode():
            ImageOnly().eval()(dummy)
            text_backbone.forward = lambda *args, **kwargs: captured["out"]
            for name, module in model.named_modules():
                if name in (amax or {}):
                    module.forward = _fake_quant(module, amax[name])
            torch.onnx.export(ImageOnly().eval(), (dummy,), str(path), input_names=["pixel_values"],
                              output_names=["logits", "pred_boxes"], opset_version=17, do_constant_folding=True,
                              dynamo=False)
    finally:
        mg.GroundingDinoContrastiveEmbedding.forward = original
        msda.forward = msda_forward
        if patched:
            text_backbone.forward = text_forward
        else:
            del text_backbone.forward
        for module in quantized:
            module.__dict__.pop("forward", None)


def fix_for_trt(path: Path) -> None:
    """Fold constant subgraphs (Swin window masks, text masks) with onnx-graphsurgeon if installed, then turn bool
    constants into int32 + Cast: TensorRT 10.3 rejects Constant layers with bool output."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(path))
    try:
        import onnx_graphsurgeon as gs

        before = len(model.graph.node)
        graph = gs.import_onnx(model)
        # Folding a weight's DequantizeLinear would turn it back into a float constant.
        graph.fold_constants(should_exclude_node=lambda n: n.op in ("QuantizeLinear", "DequantizeLinear"))
        graph.cleanup().toposort()
        model = gs.export_onnx(graph)
        print(f"folded constants: {before} -> {len(model.graph.node)} nodes")
    except ImportError:
        print("onnx_graphsurgeon not installed: constants left to TensorRT")
    graph = model.graph
    nodes = []
    for node in graph.node:
        value = next((a for a in node.attribute if a.name == "value"), None) if node.op_type == "Constant" else None
        if value is not None and value.t.data_type == TensorProto.BOOL:
            out = node.output[0]
            arr = numpy_helper.to_array(value.t).astype(np.int32)
            nodes.append(helper.make_node("Constant", [], [out + "_i32"],
                                          value=numpy_helper.from_array(arr, out + "_i32")))
            nodes.append(helper.make_node("Cast", [out + "_i32"], [out], to=TensorProto.BOOL))
        else:
            nodes.append(node)
    for init in list(graph.initializer):
        if init.data_type == TensorProto.BOOL:
            arr = numpy_helper.to_array(init).astype(np.int32)
            graph.initializer.remove(init)
            graph.initializer.append(numpy_helper.from_array(arr, init.name + "_i32"))
            nodes.insert(0, helper.make_node("Cast", [init.name + "_i32"], [init.name], to=TensorProto.BOOL))
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.save(model, str(path))


class TrtGdino(TrtEngine):
    """``__call__(pixel_values) -> (logits, pred_boxes)`` as torch CUDA tensors (reused buffers)."""

    def __call__(self, pixel_values):
        out = self.run(pixel_values)
        return out["logits"], out["pred_boxes"]


def main() -> None:
    from scene_memory import paths
    from scene_memory.nodes.detector_node import load_names
    from scene_memory.perception.gdino_sam2_detector import GdinoSam2ViewDetector, parse_ground_classes

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gdino_model", default=str(paths.GDINO))
    p.add_argument("--names_file", default=str(paths.CONFIG / "prompts.txt"))
    p.add_argument("--distractors_file", default=str(paths.CONFIG / "distractors.txt"))
    p.add_argument("--ground_classes", default="", help="same value as the detector's --ground_classes")
    p.add_argument("--view_size", default="640x640", help="pinhole view WxH the detector sees")
    p.add_argument("--gdino_size", type=int, default=800)
    p.add_argument("--precision", default="fp16", choices=PRECISIONS)
    p.add_argument("--calib_bag", default=str(paths.SESSIONS / "depthtest_20261110" / "raw_pano"),
                   help="int8: session bag with /insta360/equirect/image/compressed")
    p.add_argument("--calib_views", default="left,rear,right",
                   help="int8: pinhole views to calibrate on (front is what bench_detector scores)")
    p.add_argument("--calib_n", type=int, default=24, help="int8: frames per view")
    p.add_argument("--onnx_only", action="store_true", help="export the ONNX and stop (e.g. on a PC)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    det = GdinoSam2ViewDetector(
        load_names("", args.names_file), args.gdino_model, paths.SAM2,
        distractors=load_names("", args.distractors_file) if args.distractors_file else [],
        ground=parse_ground_classes(args.ground_classes) if args.ground_classes else None,
        gdino_size=args.gdino_size, cache_text=False, trt=False,
    )
    vw, vh = (int(x) for x in args.view_size.lower().split("x"))
    h, w = det.input_hw(vh, vw)
    onnx = onnx_path(args.gdino_model, det.text, h, w, args.precision)
    print(f"prompt '{det.text}'\ninput {w}x{h} -> {onnx}")
    if args.force or not onnx.is_file():
        t0 = time.time()
        model = det.model.float()
        amax = None
        if args.precision == "int8":
            from scene_memory.perception.bench_detector import load_views

            batches = [det._pixels(np.ascontiguousarray(img[..., ::-1])).float()
                       for view in args.calib_views.split(",")
                       for img in load_views(Path(args.calib_bag), "/insta360/equirect/image/compressed", view,
                                             args.calib_n, 8)]
            amax = calibrate(model, det._text_inputs, batches)
            print(f"calibrated {len(amax)} Linear/Conv inputs on {len(batches)} views ({args.calib_views})")
        export_onnx(model, det._text_inputs, h, w, onnx, amax)
        fix_for_trt(onnx)
        print(f"exported ONNX in {time.time() - t0:.0f} s ({onnx.stat().st_size / 1e6:.0f} MB)")
    if args.onnx_only:
        return
    engine = engine_path(args.gdino_model, det.text, h, w, args.precision)
    if engine.is_file() and not args.force:
        print(f"engine exists: {engine}")
        return
    t0 = time.time()
    if args.precision in ("fp16", "int8"):
        print(f"building {engine} ({args.precision}, takes minutes; log {engine.with_suffix('.log')})")
        build_engine(onnx, engine, extra=["--int8"] if args.precision == "int8" else None)
        print(f"built in {time.time() - t0:.0f} s")
        return

    import gc

    import torch

    # The builder needs the GPU memory more than the PyTorch models.
    del det
    gc.collect()
    torch.cuda.empty_cache()
    print(f"building {engine} ({args.precision}, takes minutes)")
    stats = build_engine_api(onnx, engine, keep_fp32=keep_fp32)
    print(f"built in {time.time() - t0:.0f} s ({stats['fp32_pinned']}/{stats['layers']} layers pinned to fp32)")


if __name__ == "__main__":
    main()
