"""Shared TensorRT helpers: trtexec engine builds and a torch-buffer runner for static-shape engines."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

TRTEXEC = Path("/usr/src/tensorrt/bin/trtexec")


def trt_version() -> str:
    try:
        import tensorrt

        return tensorrt.__version__
    except ImportError:
        return "none"


def engine_suffix() -> str:
    """Engines only load on the TensorRT version and GPU they were built with."""
    import torch

    gpu = torch.cuda.get_device_name(0).replace(" ", "_") if torch.cuda.is_available() else "cpu"
    return f"trt{trt_version()}_{gpu}"


def build_engine(onnx: Path, engine: Path, fp16: bool = True, log: Path | None = None,
                 extra: list[str] | None = None) -> None:
    if not TRTEXEC.is_file():
        raise FileNotFoundError(f"{TRTEXEC} not found (TensorRT samples)")
    tmp = engine.with_suffix(".building")
    cmd = [str(TRTEXEC), f"--onnx={onnx}", f"--saveEngine={tmp}", "--memPoolSize=workspace:2048M"]
    if fp16:
        cmd.append("--fp16")
    cmd += extra or []
    with open(log or engine.with_suffix(".log"), "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    tmp.rename(engine)


def build_engine_api(onnx: Path, engine: Path, fp16: bool = True, keep_fp32=None, workspace_mb: int = 2048) -> dict:
    """Build with the TensorRT Python API: ``keep_fp32(layer) -> bool`` pins layers to fp32 (preferred, not
    forced). Returns layer counts for the log."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    pinned = 0
    if keep_fp32 is not None:
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        # Shape/constant layers and int tensors have fixed types; only float compute layers can be pinned.
        fixed = {trt.LayerType.SHAPE, trt.LayerType.CONSTANT, trt.LayerType.PLUGIN_V2, trt.LayerType.CAST}
        floats = (trt.float32, trt.float16)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            outs = [layer.get_output(j) for j in range(layer.num_outputs)]
            if layer.type in fixed or not outs or any(t.dtype not in floats for t in outs) or not keep_fp32(layer):
                continue
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            pinned += 1
    tmp = engine.with_suffix(".building")
    data = builder.build_serialized_network(network, config)
    if data is None:
        raise RuntimeError(f"TensorRT build failed for {onnx}")
    tmp.write_bytes(bytes(data))
    tmp.rename(engine)
    return {"layers": network.num_layers, "fp32_pinned": pinned}


class TrtEngine:
    """``run(x)`` or ``run(name=x, ...) -> {output name: torch CUDA tensor}``; dynamic input dims follow the
    given tensors. Output buffers are reused while the shapes stay the same: consume them before the next call."""

    def __init__(self, engine: Path, device) -> None:
        import tensorrt as trt
        import torch

        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(Path(engine).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"cannot load TensorRT engine {engine}")
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=device)
        self.inputs: dict[str, tuple] = {}
        self.dtypes = {}
        self.outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.dtypes[name] = torch.from_numpy(np.zeros(0, trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = tuple(self.engine.get_tensor_shape(name))
            else:
                self.outputs[name] = None
        self.input_name, self.input_shape = next(iter(self.inputs.items()))

    def run(self, x=None, **inputs):
        import torch

        if x is not None:
            inputs[self.input_name] = x
        keep = []
        for name, t in inputs.items():
            t = t.to(self.dtypes[name]).contiguous()
            expect = self.inputs[name]
            if len(t.shape) != len(expect) or any(e not in (-1, s) for e, s in zip(expect, t.shape)):
                raise ValueError(f"engine input {name} {expect}, got {tuple(t.shape)}")
            if -1 in expect:
                self.ctx.set_input_shape(name, tuple(t.shape))
            self.ctx.set_tensor_address(name, t.data_ptr())
            keep.append(t)
        for name, buf in self.outputs.items():
            shape = tuple(self.ctx.get_tensor_shape(name))
            if buf is None or tuple(buf.shape) != shape:
                self.outputs[name] = buf = torch.empty(shape, dtype=self.dtypes[name], device=self.device)
            self.ctx.set_tensor_address(name, buf.data_ptr())
        # TensorRT adds extra syncs on the legacy default stream.
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        caller.wait_stream(self.stream)
        for t in keep:
            t.record_stream(self.stream)
        return self.outputs
