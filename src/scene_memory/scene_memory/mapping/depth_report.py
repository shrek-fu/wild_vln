"""Depth estimation report (architecture, per-stage latency, accuracy) as HTML + PDF from the benchmark outputs.

Inputs in ``<session>``: ``depth_bench/bench.json`` (+ ``view_*.npz``) from ``depth_stage_bench``,
``da360_eval_orin_{small,base}/report.json`` from ``pano_depth_bag``, ``pano_depth_raw/report.json`` (PC run)
and one ``pano_depth_raw/frame_*.jpg`` montage. The PDF is printed by headless Chrome::

    python3 -m scene_memory.mapping.depth_report <session> --out_dir <session>/depth_report
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BIN_NAMES = ("0-5", "5-10", "10-30", "30-inf")
VMAX = 30.0


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _jpg(img: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()).decode()


def depth_color(depth: np.ndarray) -> np.ndarray:
    level = np.clip(np.sqrt(np.clip(depth, 0, None) / VMAX) * 255.0, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(level, cv2.COLORMAP_TURBO)
    out[~(depth > 0)] = 0
    return out


def tile(img: np.ndarray, text: str, size: int = 320) -> np.ndarray:
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (size, 24), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def view_grid(bench_dir: Path, names: list[str]) -> np.ndarray:
    rows = []
    for name in names:
        z = np.load(bench_dir / name)
        lidar = cv2.dilate(z["lidar"].astype(np.float32), np.ones((3, 3), np.uint8))
        rows.append(np.hstack([
            tile(z["rgb"], f"front view ({name[5:9]})"),
            tile(depth_color(lidar), "MID-360 lidar (truth)"),
            tile(depth_color(z["da360_live"].astype(np.float32)), "DA360 live (Hydra input)"),
            tile(depth_color(z["depthart_outdoor_global"].astype(np.float32)), "DepthART outdoor x scale"),
            tile(depth_color(z["depthart_indoor_raw"].astype(np.float32)), "DepthART indoor raw"),
        ]))
    return np.vstack(rows)


def colorbar() -> str:
    fig, ax = plt.subplots(figsize=(7.5, 0.35))
    x = np.linspace(0, VMAX, 512)
    lut = cv2.applyColorMap(np.clip(np.sqrt(x / VMAX) * 255, 0, 255).astype(np.uint8)[None], cv2.COLORMAP_TURBO)
    ax.imshow(lut[..., ::-1], aspect="auto", extent=(0, VMAX, 0, 1))
    ax.set_yticks([])
    ax.set_xticks([0, 1, 2, 5, 10, 15, 20, 30])
    ax.set_xlabel("depth [m] (sqrt scale, black = no value)", fontsize=8)
    ax.tick_params(labelsize=7)
    return _png(fig)


def latency_chart(st: dict, pc_net: float, base_net: float) -> str:
    rows = [
        ("DA360-small live\n(per 360° frame)", [("JPEG decode", st["decode_equirect"]), ("preprocess", st["da360_preprocess"]),
                                               ("network fp32", st["da360_net_fp32"]), ("postprocess+resize", st["da360_postprocess"]),
                                               ("lidar global scale", st["da360_global_scale"])]),
        ("Hydra: DA360 -> front view\n(hydra_views=front)", [("equirect->pinhole x1", st["da360_to_pinhole_view"])]),
        ("DepthART-S TRT\n(4 views)", [("render view x4", 4 * st["render_pinhole_rgb"]), ("TRT fp16 x4", 4 * st["depthart_trt"])]),
        ("Lidar view_depth\n(4 views)", [("projection x4", 4 * st["lidar_view_depth"])]),
    ]
    colors = {"JPEG decode": "#8c8c8c", "preprocess": "#c7b07a", "network fp32": "#d9534f", "postprocess+resize": "#f0ad4e",
              "lidar global scale": "#5bc0de", "equirect->pinhole x1": "#9b59b6", "render view x4": "#c7b07a",
              "TRT fp16 x4": "#2e86c1", "projection x4": "#27ae60"}
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    seen = set()
    for i, (label, parts) in enumerate(rows):
        left = 0.0
        for name, ms in parts:
            ax.barh(i, ms, left=left, color=colors[name], edgecolor="white", label=None if name in seen else name)
            seen.add(name)
            left += ms
        ax.text(left + 8, i, f"{left:.0f} ms", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("median latency on Jetson Orin NX 16GB [ms]", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(0, 760)
    ax.legend(fontsize=7, ncol=3, loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(f"DA360 network for reference: small on RTX 5060 {pc_net:.0f} ms, base on Orin {base_net:.0f} ms",
                 fontsize=8)
    return _png(fig)


def bins_chart(acc: dict, keys: list[tuple[str, str, str]], title: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.7))
    x = np.arange(3)
    width = 0.8 / len(keys)
    for j, (metric, ylabel) in enumerate((("rel_mean", "mean |err| / truth"), ("abs_med", "median |err| [m]"))):
        ax = axes[j]
        for i, (key, label, color) in enumerate(keys):
            vals = [acc[key]["bins"][b][metric] for b in range(3)]
            ax.bar(x + (i - (len(keys) - 1) / 2) * width, vals, width, label=label, color=color)
        ax.set_xticks(x, [f"{b} m" for b in BIN_NAMES[:3]], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return _png(fig)


def _f(v, nd=3) -> str:
    return f"{v:.{nd}f}"


def acc_rows(acc: dict, keys: list[tuple[str, str]]) -> str:
    out = []
    for key, label in keys:
        a = acc[key]
        bins = "".join(f"<td>{_f(b['rel_mean'])}<br><span class=s>{_f(b['abs_med'], 2)} m</span></td>" for b in a["bins"][:3])
        out.append(f"<tr><td class=l>{label}</td><td>{_f(a['abs_rel'])}</td><td>{_f(a['rmse'], 2)}</td>"
                   f"<td>{_f(a['delta1'])}</td>{bins}</tr>")
    return "\n".join(out)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--chrome", default="google-chrome")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    s = args.session
    out = args.out_dir or s / "depth_report"
    out.mkdir(parents=True, exist_ok=True)
    bench = json.loads((s / "depth_bench" / "bench.json").read_text())
    st = {k: v["median"] for k, v in bench["stages"].items()}
    p90 = {k: v["p90"] for k, v in bench["stages"].items()}
    acc = bench["accuracy"]
    orin = {m: json.loads((s / f"da360_eval_orin_{m}" / "report.json").read_text())["summary"] for m in ("small", "base")}
    pc = json.loads((s / "pano_depth_raw" / "report.json").read_text())["summary"]
    hemi = {f"{m}_{k}": {"bins": orin[m][k]} for m in ("small", "base") for k in ("global", "local")}

    live_ms = sum(st[k] for k in ("decode_equirect", "da360_preprocess", "da360_net_fp32", "da360_postprocess",
                                  "da360_global_scale"))
    hydra_ms = st["da360_to_pinhole_view"]
    dart_view = st["render_pinhole_rgb"] + st["depthart_trt"]
    lidar_ms = st["lidar_view_depth"]
    n_frames = bench["frames"]
    n_truth = acc["da360_live"]["n"]

    views = sorted(p.name for p in (s / "depth_bench").glob("view_*.npz"))
    pick = [views[i] for i in sorted({0, len(views) // 2, len(views) - 1})]
    grid = _jpg(view_grid(s / "depth_bench", pick))
    montages = sorted((s / "da360_eval_orin_small").glob("frame_*.jpg"))
    pano = cv2.imread(str(montages[len(montages) // 2]))

    front_keys = [("da360_live", "DA360-small live", "#d9534f"), ("depthart_outdoor_global", "DepthART outdoor x scale", "#2e86c1"),
                  ("depthart_indoor_global", "DepthART indoor x scale", "#85c1e9"), ("depthart_indoor_raw", "DepthART indoor raw", "#aaaaaa")]
    hemi_keys = [("small_global", "DA360-small global scale", "#f1948a"), ("small_local", "DA360-small local align", "#d9534f"),
                 ("base_global", "DA360-base global scale", "#bb8fce"), ("base_local", "DA360-base local align", "#7d3c98")]

    lat_rows = [
        ("A", "Insta360 equirect JPEG 解码 / decode (3840×1920, cv2 CPU)", "decode_equirect"),
        ("A", "DA360 预处理 / preprocess (BGR→RGB, resize 1036×518, normalize, H2D)", "da360_preprocess"),
        ("A", "DA360-small 网络 / network (PyTorch fp32, ViT-S)", "da360_net_fp32"),
        ("A", "DA360 后处理 / postprocess (D2H, 1/disp, resize 回 3840×1920)", "da360_postprocess"),
        ("A", "雷达全局尺度 / lidar global scale (median ratio)", "da360_global_scale"),
        ("A′", "Hydra: 全景深度→针孔视角 / equirect→pinhole (每视角 per view)", "da360_to_pinhole_view"),
        ("B", "全景→针孔 RGB 视角 / render pinhole view (cv2.remap, 每视角)", "render_pinhole_rgb"),
        ("B", "DepthART-S TRT fp16 (448², 含 GPU 预处理与回传, 每视角)", "depthart_trt"),
        ("C", "雷达投影到视角 / lidar→view projection (view_depth, 每视角)", "lidar_view_depth"),
    ]
    lat_html = "\n".join(f"<tr><td>{p}</td><td class=l>{html.escape(t)}</td><td>{st[k]:.1f}</td><td>{p90[k]:.1f}</td></tr>"
                         for p, t, k in lat_rows)

    html_doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Depth estimation report</title>
<style>
@page {{ size: A4; margin: 14mm 13mm; }}
body {{ font-family: "Noto Sans CJK SC", "Noto Sans CJK TC", "Droid Sans Fallback", sans-serif; font-size: 10pt;
       color: #222; line-height: 1.45; }}
h1 {{ font-size: 18pt; margin: 0 0 4px; }} h2 {{ font-size: 13pt; border-bottom: 2px solid #2e86c1; padding-bottom: 2px;
       margin-top: 18px; }} h3 {{ font-size: 11pt; margin: 12px 0 4px; }}
.meta {{ color: #555; font-size: 9pt; }}
.en {{ color: #555; font-size: 9pt; font-style: italic; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: 6px 0; }}
th, td {{ border: 1px solid #ccc; padding: 3px 5px; text-align: center; vertical-align: middle; }}
th {{ background: #eef4fa; }} td.l {{ text-align: left; }} .s {{ color: #777; font-size: 7.6pt; }}
.box {{ background: #f7f9fb; border-left: 4px solid #2e86c1; padding: 6px 10px; margin: 8px 0; }}
.warn {{ border-left-color: #f0ad4e; background: #fdf8ef; }}
img {{ max-width: 100%; }} .pb {{ page-break-before: always; }}
.flow {{ display: flex; align-items: center; flex-wrap: wrap; gap: 4px; margin: 4px 0 2px; font-size: 8.4pt; }}
.flow .n {{ border: 1px solid #888; border-radius: 4px; padding: 3px 6px; background: #fff; }}
.flow .g {{ background: #fde2e1; }} .flow .t {{ background: #dbeaf7; }} .flow .c {{ background: #dff3e4; }}
.flow .a {{ color: #888; }}
.lane {{ border: 1px dashed #bbb; border-radius: 6px; padding: 6px 8px; margin: 6px 0; }}
.lane b {{ font-size: 9pt; }}
</style></head><body>

<h1>深度估计技术报告 / Depth Estimation Report</h1>
<div class=meta>项目 STVLN scene_memory · 平台 Jetson Orin NX 16GB (L4T R36.4.4 / JetPack 6.2, MAXN_SUPER, TensorRT 10.3) ·
数据 session <code>depthtest_20261110/raw_pano</code>（室内大厅，Insta360 3840×1920 全景 + Livox MID-360）·
{n_frames} 帧 / frames · 生成于 2026-11-12</div>

<h2>1. 结论摘要 / Summary</h2>
<div class=box>
<b>Hydra 模式下的学习型深度是全景深度。</b> Hydra 建图（<code>mapper:=hydra</code>，默认 <code>depth_source=da360</code>）使用 DA360 对整张
360° equirect 全景图推理相对深度，再用 MID-360 雷达拟合一个全局尺度得到米制深度，最后采样到 Hydra 使用的针孔视角
（默认 <code>hydra_views=front</code>，只有前视 90° 送入 <code>pipeline.step</code>）。
DepthART 是单视角针孔模型，只作为备选；默认的拓扑建图（<code>mapper:=topo</code>）根本不用学习型深度，只把雷达点投影到视角。
<div class=en>In Hydra mode the learned depth is panoramic: DA360 runs on the full 360° equirect, one lidar median scale makes it metric,
Hydra samples it into four 90° pinhole views. DepthART is a single-pinhole alternative; the default topo mapper uses lidar
projection only.</div>
</div>
<div class="box warn"><b>部署现状：</b>Orin 上目前没有安装 Hydra（<code>hydra_python</code> 不存在），线上运行的是默认 <code>mapper:=topo</code>，
因此 DA360 也不会启动（<code>start_da360=auto</code> 只在 hydra 模式下开启）。Hydra 只在 PC 上用过（<code>refer/hydra_ws</code>，2026-11-06/07 的
<code>data/scene4d</code> 产出 dsg.json）。本报告的 A/B 路径数据为离线 bag 上的 Orin 实测，用于评估将 Hydra 移植到 Orin 的可行性。
<div class=en>Status: Hydra is not installed on the Orin; the live pipeline runs the default topo mapper, so DA360 is not started
either. Hydra has only run on the PC. Paths A/B here are Orin measurements on a recorded bag, to judge a Hydra port.</div></div>
<table>
<tr><th>路径 / path</th><th>覆盖 / coverage</th><th>Orin 时延 / latency</th><th>可达帧率 / max rate</th>
<th>前视精度 AbsRel / δ1 (同像素 same pixels)</th></tr>
<tr><td class=l>A. DA360-small + 雷达全局尺度 (线上 live)</td><td>360°×180° 一次推理</td><td>{live_ms:.0f} ms / 全景帧<br>
<span class=s>+ Hydra 采样 {hydra_ms:.0f} ms (前视)</span></td><td>{1000 / live_ms:.2f} Hz<br><span class=s>(节点上限 da360_max_hz=2)</span></td>
<td><b>{_f(acc['da360_live']['abs_rel'])} / {_f(acc['da360_live']['delta1'])}</b></td></tr>
<tr><td class=l>B. DepthART-S metric, TRT fp16</td><td>每次 1 个 90° 视角</td><td>{dart_view:.0f} ms / 视角<br>
<span class=s>{4 * dart_view:.0f} ms / 4 视角</span></td><td>{1000 / dart_view:.1f} Hz 单视角<br><span class=s>{1000 / (4 * dart_view):.1f} Hz 4 视角</span></td>
<td>raw {_f(acc['depthart_outdoor_raw']['abs_rel'])}/{_f(acc['depthart_outdoor_raw']['delta1'])} (outdoor),
{_f(acc['depthart_indoor_raw']['abs_rel'])}/{_f(acc['depthart_indoor_raw']['delta1'])} (indoor)<br>
×雷达尺度 {_f(acc['depthart_outdoor_global']['abs_rel'])}/{_f(acc['depthart_outdoor_global']['delta1'])}</td></tr>
<tr><td class=l>C. 雷达投影 view_depth (topo 默认)</td><td>雷达视场内稀疏点</td><td>{lidar_ms:.1f} ms / 视角</td><td>&gt;100 Hz</td>
<td>真值本身（稀疏）/ ground truth itself (sparse)</td></tr>
</table>
<ul>
<li><b>精度：</b>同一前视、同一批留出雷达点上，DA360 线上输出 AbsRel {_f(acc['da360_live']['abs_rel'])}，约为 DepthART 最佳配置
（{_f(acc['depthart_outdoor_global']['abs_rel'])}）的一半；DepthART 原生米制输出不可直接用（室内权重尺度偏大约 1.4×，室外权重偏大约 6.5×）。
<span class=en>DA360 halves DepthART's error on identical pixels; DepthART's own metric scale is unusable here.</span></li>
<li><b>速度：</b>DepthART TRT 每视角 {st['depthart_trt']:.0f} ms，比 DA360 一帧全景 {live_ms:.0f} ms 快很多，但需要 4 次推理覆盖水平 360°。
DA360 的瓶颈是未加速的 PyTorch fp32 网络（{st['da360_net_fp32']:.0f} ms，占 {100 * st['da360_net_fp32'] / live_ms:.0f}%）。
<span class=en>DepthART is fast; DA360 is bound by its un-accelerated PyTorch network.</span></li>
<li><b>建议：</b>保留 DA360 作为 Hydra 深度来源；优先做 DA360 TensorRT 化、缓存全景→针孔映射表、把离线已验证的局部对齐（local align）接入线上（见第 6 节）。
<span class=en>Keep DA360; accelerate it with TensorRT, cache remap tables, bring local alignment online.</span></li>
</ul>

<h2>2. 技术架构 / Architecture</h2>
<div class=lane><b>A. 全景路径（Hydra 默认）/ panoramic path — <code>depth/da360_depth/live_ros.py</code> → <code>hydra_scene_node.py</code></b>
<div class=flow><span class=n>Insta360 双鱼眼→equirect<br>3840×1920 JPEG</span><span class=a>→</span><span class=n>解码 decode</span><span class=a>→</span>
<span class="n g">DA360-small (DINOv2 ViT-S)<br>输入 1036×518, 360° 一次推理</span><span class=a>→</span><span class=n>相对深度 relative<br>1/disp, resize 回全分辨率</span>
<span class=a>→</span><span class="n c">MID-360 点投影到 equirect<br>median(lidar/pred) = 全局尺度</span><span class=a>→</span><span class=n>/da360/depth_metric<br>(equirect, 米 m)</span>
<span class=a>→</span><span class=n>Hydra: equirect_depth_to_pinhole<br>默认仅 front 640² 90°</span><span class=a>→</span><span class=n>3D 场景图 scene graph</span></div>
<div class=en>Rate-capped at da360_max_hz=2. One network pass covers the full sphere; lidar fixes a single scale per frame.</div></div>
<div class=lane><b>B. 针孔路径（备选 <code>depth_source:=depthart</code>）/ pinhole path — <code>perception/depthart_trt.py</code></b>
<div class=flow><span class=n>针孔视角 640² 90°<br>(相机节点或全景重投影)</span><span class=a>→</span><span class=n>GPU bicubic → 448²</span><span class=a>→</span>
<span class="n t">DepthART TinyViM-S metric<br>TRT fp16 + SelectiveScan 插件<br>K 固化进 ONNX (fx=224)</span><span class=a>→</span><span class=n>米制深度 metric<br>(无雷达 no lidar)</span><span class=a>→</span>
<span class=n>Hydra</span></div>
<div class=en>Indoor (≤10 m) and outdoor (≤80 m) weights; one inference per view; licence CC BY-NC (research only).</div></div>
<div class=lane><b>C. 雷达路径（topo 默认，无学习型深度）/ lidar path — <code>nodes/view_depth_node.py</code></b>
<div class=flow><span class=n>MID-360 /LIO 点云</span><span class=a>→</span><span class=n>位姿 + 外参<br>T_odom_optical</span><span class=a>→</span>
<span class="n c">投影到每个视角 view_depth<br>稀疏, 最近点优先</span><span class=a>→</span><span class=n>/insta360/&lt;view&gt;/depth<br>检测框 3D 提升</span></div>
<div class=en>Pure geometry; sparse but exact where the lidar sees.</div></div>
<p>启动示例 / launch: <code>ros2 launch scene_memory pipeline.launch.py mapper:=hydra depth_source:=da360</code>
（或 <code>depth_source:=depthart depthart_domain:=indoor</code>）；默认 <code>mapper:=topo</code> 不启动 DA360。</p>

<h2>3. 各环节时延 / Per-stage latency (Orin NX)</h2>
<p>测量方式：<code>scene_memory.mapping.depth_stage_bench</code> 在 Orin 上按线上节点的调用方式逐环节计时（每段前后 <code>cuda.synchronize</code>），
{bench['stages']['da360_net_fp32']['n']} 帧，3 帧预热，单进程独占 GPU（未与检测器并发）。
<span class=en>Each stage timed with CUDA sync, {bench['stages']['da360_net_fp32']['n']} frames, GPU not shared.</span></p>
<table><tr><th>路径</th><th>环节 / stage</th><th>中位数 median [ms]</th><th>p90 [ms]</th></tr>
{lat_html}
<tr><th colspan=2 class=l>A 合计（每全景帧）/ total per 360° frame</th><th>{live_ms:.0f}</th><th>→ {1000 / live_ms:.2f} Hz</th></tr>
<tr><th colspan=2 class=l>B 合计（4 视角）/ total for 4 views</th><th>{4 * dart_view:.0f}</th><th>→ {1000 / (4 * dart_view):.1f} Hz</th></tr>
</table>
<img src="data:image/png;base64,{latency_chart(st, pc['net_ms']['median'], orin['base']['net_ms']['median'])}">
<div class="box warn"><b>观察 / observations</b>
<ul>
<li>DA360 网络占线上全景路径的 {100 * st['da360_net_fp32'] / live_ms:.0f}%。外部 <code>torch.autocast(fp16)</code> 实测无效（{st['da360_net_fp16']:.0f} ms），
因为 DA360 内部用 <code>autocast(enabled=self.mixed_precision)</code> 覆盖了精度；需要导出 ONNX/TensorRT 才能加速。
DA360-base 在 Orin 上为 {orin['base']['net_ms']['median']:.0f} ms；DA360-small 在 PC RTX 5060 上为 {pc['net_ms']['median']:.0f} ms。</li>
<li>Hydra 的 <code>equirect_depth_to_pinhole</code> 每次调用都重新计算映射表（反三角函数 409,600 个像素），每视角 {st['da360_to_pinhole_view']:.0f} ms；
缓存后应与 RGB 重投影相当（约 {st['render_pinhole_rgb']:.0f} ms）。</li>
<li>JPEG 解码 {st['decode_equirect']:.0f} ms 与后处理 resize 回 3840×1920 的 {st['da360_postprocess']:.0f} ms 都在 CPU 上，可改 nvJPEG / 保持网络分辨率。</li>
</ul>
<div class=en>The DA360 network dominates; outer fp16 autocast has no effect (the model sets its own), so TensorRT export is
the lever. Hydra recomputes the equirect→pinhole map every call (84 ms/view), which a cache removes. Decode and the
full-res resize are CPU-bound.</div></div>

<h2>4. 深度精度对比 / Accuracy</h2>
<h3>4.1 评测协议 / protocol</h3>
<p>真值来自 MID-360 原始扫描（0.3 s 窗口，只用相机-雷达外参投影，不依赖里程计；该 session 的 LIO 发散，故不用 LIO 点云），
剔除相机视角下被遮挡的雷达点，随机留出 10% 作为真值，其余 90% 用于拟合尺度（“×scale / global”）。
指标：AbsRel = mean(|d−d*|/d*)，RMSE [m]，δ1 = 比例 max(d/d*, d*/d) &lt; 1.25；按距离分段 0–5 / 5–10 / 10–30 m。
<span class=en>Ground truth = raw MID-360 returns (0.3 s, extrinsic only), occlusion-filtered, 10% held out; the scale fit
uses the other 90%.</span></p>

<h3>4.2 同一前视、同一像素对比 / same front view, same pixels</h3>
<p>前视 640×640、90°，{n_frames} 帧，共 {n_truth:,} 个留出真值点。DA360 live 即 Hydra 实际拿到的深度（全景全局尺度后再重投影）。
<span class=en>Front 640² 90° view, {n_truth:,} held-out returns. "DA360 live" is exactly what Hydra consumes.</span></p>
<table><tr><th>方法 / method</th><th>AbsRel ↓</th><th>RMSE [m] ↓</th><th>δ1 ↑</th><th>0–5 m<br><span class=s>rel / 中位绝对误差</span></th>
<th>5–10 m</th><th>10–30 m</th></tr>
{acc_rows(acc, [("da360_live", "DA360-small live（全景全局尺度）"), ("da360_global", "DA360-small ×视角尺度 / per-view scale"),
                ("depthart_outdoor_global", "DepthART outdoor ×雷达尺度"), ("depthart_indoor_global", "DepthART indoor ×雷达尺度"),
                ("depthart_indoor_raw", "DepthART indoor 原生米制 raw"), ("depthart_outdoor_raw", "DepthART outdoor 原生米制 raw")])}
</table>
<img src="data:image/png;base64,{bins_chart(acc, front_keys, 'Front view, identical held-out lidar pixels')}">

<h3>4.3 DA360 前半球（180°）：全局尺度 vs 局部对齐 / hemisphere, global vs local alignment</h3>
<p><code>pano_depth_bag</code>（Orin），前半球 518×518，{orin['small']['frames']} 帧。“local align”（<code>align_local</code>，雷达稠密区直接用雷达、
8 px 网格局部尺度、远处回退全局尺度）目前只在离线评测中使用，线上节点仍是全局尺度。其额外耗时约 {orin['small']['align_ms']['median']:.0f} ms。
<span class=en>Local alignment is offline-only today; it costs ~{orin['small']['align_ms']['median']:.0f} ms and cuts near-range error ~3×.</span></p>
<table><tr><th>方法</th><th>0–5 m</th><th>5–10 m</th><th>10–30 m</th></tr>
{"".join(f"<tr><td class=l>{lab}</td>" + "".join(f"<td>{_f(b['rel_mean'])}<br><span class=s>{_f(b['abs_med'], 2)} m</span></td>" for b in hemi[k]['bins'][:3]) + "</tr>" for k, lab, _ in hemi_keys)}
</table>
<img src="data:image/png;base64,{bins_chart(hemi, hemi_keys, 'DA360 front hemisphere (180 deg), Orin')}">
<p class=en>前半球包含大量近处地面和机身两侧，所以 0–5 m 误差高于 4.2 的前视。The hemisphere includes much near floor and the rig's
sides, hence the higher 0–5 m error than in 4.2.</p>

<h2 class=pb>5. 可视化 / Qualitative</h2>
<p>同一色标 0–30 m（平方根刻度）。从左到右：前视 RGB、雷达真值（稀疏）、DA360 live、DepthART outdoor ×雷达尺度、DepthART indoor 原生输出。
<span class=en>Shared 0–30 m colour scale. DepthART is visibly over-smoothed; its raw indoor output is offset in scale.</span></p>
<img src="data:image/png;base64,{colorbar()}">
<img src="data:image/jpeg;base64,{grid}">
<h3>DA360 前半球 / hemisphere (rgb | lidar | aligned depth | source: white=lidar, green=local, red=global fallback)</h3>
<img src="data:image/jpeg;base64,{_jpg(pano)}">

<h2>6. 局限与改进 / Limitations and next steps</h2>
<ul>
<li>数据只有一段室内大厅（{n_frames} 帧），30 m 以外真值仅 {acc['da360_live']['bins'][3]['n']} 点（多为穿窗/反射），不计入结论；室外、楼梯、坡道、草地待补测。
<span class=en>Single indoor session; &gt;30 m has too few returns to judge; outdoor/stairs/ramps/grass pending.</span></li>
<li>雷达真值稀疏且只覆盖 MID-360 视场（约 −7°~52° 俯仰），天花板高处与近地盲区没有真值。<span class=en>Truth is sparse and limited to the MID-360 FOV.</span></li>
<li>时延在 GPU 独占时测得；与 GroundingDINO+SAM2 检测器（约 219 ms/帧）并发时会变长；整机功耗/内存峰值与此前的复位问题需另行记录。
<span class=en>Timings without GPU contention; concurrent detector load will slow both.</span></li>
<li>DepthART 许可证为 CC BY-NC，仅限研究/内部使用。<span class=en>DepthART licence: research/internal only.</span></li>
</ul>
<table><tr><th>改进 / improvement</th><th>预期 / expected</th><th>工作量 / effort</th></tr>
<tr><td class=l>缓存 equirect→pinhole 映射表（Hydra）</td><td>{st['da360_to_pinhole_view']:.0f} → ~{st['render_pinhole_rgb']:.0f} ms / 视角</td><td>小 small</td></tr>
<tr><td class=l>DA360-small 导出 ONNX → TensorRT fp16</td><td>网络 {st['da360_net_fp32']:.0f} ms 预计降到 100–150 ms（待验证）</td><td>中 medium</td></tr>
<tr><td class=l>线上接入 align_local（局部对齐）</td><td>+{orin['small']['align_ms']['median']:.0f} ms；近处误差约降 3×（4.3）</td><td>小 small</td></tr>
<tr><td class=l>nvJPEG 解码 + 不再上采样到 3840×1920</td><td>约 −{st['decode_equirect'] + st['da360_postprocess'] - 15:.0f} ms</td><td>小 small</td></tr>
</table>

<h2>附录 / Reproduce</h2>
<pre style="font-size:7.6pt;white-space:pre-wrap">S=data/sessions/depthtest_20261110
python3 -m scene_memory.mapping.depth_stage_bench $S/raw_pano --calib_dir $S/calib --model_path models/da360/DA360_small.pth --frames 200 --out_dir $S/depth_bench
python3 -m scene_memory.mapping.pano_depth_bag $S/raw_pano --calib_dir $S/calib --model_path models/da360/DA360_small.pth --out_dir $S/da360_eval_orin_small
python3 -m scene_memory.mapping.depthart_bag_eval $S/raw_pano --from_equirect --calib_dir $S/calib --out_dir $S/depthart_eval_raw
python3 -m scene_memory.mapping.depth_report $S</pre>
</body></html>"""
    (out / "depth_report.html").write_text(html_doc, encoding="utf-8")
    chrome = shutil.which(args.chrome)
    if chrome is None:
        print(f"wrote {out / 'depth_report.html'} (no {args.chrome} for the PDF)")
        return
    pdf = out / "depth_estimation_report.pdf"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    (out / "depth_report.html").resolve().as_uri()], check=True, capture_output=True)
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
