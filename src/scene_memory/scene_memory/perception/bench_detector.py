"""Stage timing and output agreement of GdinoSam2ViewDetector on recorded panoramas.

Frames come from a session bag (``/insta360/equirect/image/compressed``) and are rendered to a pinhole view with
the rig calibration, as in the live pipeline. Every ``--configs`` entry runs on the same frames; detections are
compared with the first entry (same class, box IoU >= 0.5)::

    python3 src/scene_memory/scripts/bench_detector.py --configs 800/1024,640/1024,640/640
    python3 src/scene_memory/scripts/bench_detector.py --no_cache_text --no_fast_msda   # old behaviour

A config is ``<gdino_size>/<sam_imgsz>[/torch|torch32|trt[/fp16|mixed|int8]]``: trt = the engines from
scripts/gdino_trt.py (that precision) and scripts/sam2_trt.py if they exist, torch32 = PyTorch without fp16
(the accuracy reference)::

    python3 src/scene_memory/scripts/bench_detector.py --configs 800/640/torch32,800/640/trt,800/640/trt/int8

``--no_cache_text`` / ``--no_fast_msda`` patch the model, so they apply to the whole run.
"""

from __future__ import annotations

import argparse
import statistics as st
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from scene_memory import paths

STAGES = ("g.backbone", "g.text", "g.encoder", "g.decoder", "g.trt", "sam2", "total")
# Parts of a stage above (not added to the total).
SUBSTAGES = ("s.pre", "s.inf", "s.enc", "s.post")


def load_views(bag: Path, topic: str, view: str, n: int, stride: int) -> list[np.ndarray]:
    from scene_memory.common.calib_io import load_rig_calibration
    from scene_memory.mapping.remap_bag import read_bag
    from scene_memory.perception.boxer_lift import equirect_remap, load_pinhole_views

    calib = load_rig_calibration(paths.CALIB)
    v = next(v for v in load_pinhole_views(calib.source_dir) if v["name"] == view)
    out, maps = [], None
    for i, (_, data) in enumerate(read_bag(bag, [topic], raw=frozenset())):
        if i % stride:
            continue
        eq = cv2.imdecode(np.frombuffer(bytes(data.data), np.uint8), cv2.IMREAD_COLOR)
        if maps is None:
            maps = equirect_remap(eq.shape[0], eq.shape[1], v["R_body_optical"], v["height"], v["width"],
                                  v["hfov_deg"])[:2]
        out.append(cv2.remap(eq, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP))
        if len(out) >= n:
            break
    return out


def _poly_iou(a, b, shape) -> float:
    ma, mb = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    cv2.fillPoly(ma, [np.round(np.asarray(a)).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(np.asarray(b)).astype(np.int32)], 1)
    union = int((ma | mb).sum())
    return int((ma & mb).sum()) / union if union else 1.0


def agreement(ref: list[list[dict]], other: list[list[dict]], shape) -> tuple[float, float, float]:
    """(recall of ref detections, precision of other detections, median mask polygon IoU of the matches);
    a match has the same class and box IoU >= 0.5."""
    from scene_memory.perception.sam3_detector import _box_iou

    hit_ref = hit_other = n_ref = n_other = 0
    mask_iou = []
    for a, b in zip(ref, other):
        n_ref, n_other = n_ref + len(a), n_other + len(b)
        if not a or not b:
            continue
        iou = _box_iou(np.array([d["xyxy"] for d in a]), np.array([d["xyxy"] for d in b]))
        same = np.array([[da["cls"] == db["cls"] for db in b] for da in a])
        ok = (iou >= 0.5) & same
        hit_ref += int(ok.any(axis=1).sum())
        hit_other += int(ok.any(axis=0).sum())
        for i in np.flatnonzero(ok.any(axis=1)):
            j = int(np.argmax(np.where(ok[i], iou[i], -1.0)))
            if a[i].get("mask") and b[j].get("mask"):
                mask_iou.append(_poly_iou(a[i]["mask"], b[j]["mask"], shape))
    return hit_ref / max(n_ref, 1), hit_other / max(n_other, 1), float(np.median(mask_iou)) if mask_iou else 1.0


def label_agreement(ref: list, other: list) -> float:
    """Fraction of pixels labelled ground in either image that carry the same label in both."""
    same = total = 0
    for a, b in zip(ref, other):
        if a is None or b is None:
            continue
        any_ = (a > 0) | (b > 0)
        total += int(any_.sum())
        same += int((any_ & (a == b)).sum())
    return same / total if total else 1.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", default=str(paths.SESSIONS / "depthtest_20261110" / "raw_pano"))
    p.add_argument("--topic", default="/insta360/equirect/image/compressed")
    p.add_argument("--view", default="front")
    p.add_argument("--n", type=int, default=20, help="frames")
    p.add_argument("--stride", type=int, default=8, help="use every k-th bag frame")
    p.add_argument("--configs", default="800/640/torch,800/640/trt",
                   help="comma list of <gdino_size>/<sam_imgsz>[/torch|trt]")
    p.add_argument("--no_cache_text", action="store_true")
    p.add_argument("--no_fast_msda", action="store_true")
    p.add_argument("--no_trt", action="store_true", help="PyTorch GroundingDINO even if an engine exists")
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()

    import torch

    from scene_memory.nodes.detector_node import load_names
    from scene_memory.perception.gdino_sam2_detector import GdinoSam2ViewDetector, parse_ground_classes

    images = load_views(Path(args.bag), args.topic, args.view, args.n, args.stride)
    print(f"{len(images)} {args.view} views {images[0].shape[1]}x{images[0].shape[0]} from {args.bag}")
    det = GdinoSam2ViewDetector(
        load_names("", str(paths.CONFIG / "prompts.txt")), paths.GDINO, paths.SAM2, conf=0.35,
        distractors=load_names("", str(paths.CONFIG / "distractors.txt")),
        ground=parse_ground_classes(str(paths.CONFIG / "ground_classes.txt")), ground_conf=0.3,
        cache_text=not args.no_cache_text, fast_msda=not args.no_fast_msda, trt=not args.no_trt,
    )
    times: dict[str, list[float]] = defaultdict(list)

    def now() -> float:
        torch.cuda.synchronize()
        return time.perf_counter()

    def timed(module, name):
        module.register_forward_pre_hook(lambda m, a: setattr(m, "_t0", now()))
        module.register_forward_hook(lambda m, a, o: times[name].append(1e3 * (now() - m._t0)))

    m = det.model.model
    for module, name in ((m.backbone, "g.backbone"), (m.text_backbone, "g.text"), (m.encoder, "g.encoder"),
                         (m.decoder, "g.decoder")):
        timed(module, name)
    sam_predict = det.sam.predict

    def sam_timed(*a, **k):
        t0 = now()
        r = sam_predict(*a, **k)
        times["sam2"].append(1e3 * (now() - t0))
        return r

    det.sam.predict = sam_timed
    from scene_memory.perception.gdino_trt import TrtGdino

    trt_call = TrtGdino.__call__

    def trt_timed(self, x):
        t0 = now()
        r = trt_call(self, x)
        times["g.trt"].append(1e3 * (now() - t0))
        return r

    TrtGdino.__call__ = trt_timed
    from ultralytics.models.sam.predict import SAM2Predictor

    from scene_memory.perception import sam2_trt

    sam2_trt.use(None)
    sam_features = SAM2Predictor.get_im_features

    def sam_features_timed(self, im):
        t0 = now()
        r = sam_features(self, im)
        times["s.enc"].append(1e3 * (now() - t0))
        return r

    SAM2Predictor.get_im_features = sam_features_timed

    def stage_timed(name, fn):
        def wrapped(*a, **k):
            t0 = now()
            r = fn(*a, **k)
            times[name].append(1e3 * (now() - t0))
            return r

        return wrapped

    for attr, name in (("preprocess", "s.pre"), ("inference", "s.inf"), ("postprocess", "s.post")):
        setattr(SAM2Predictor, attr, stage_timed(name, getattr(SAM2Predictor, attr)))
    print(f"cache_text={not args.no_cache_text} fast_msda={not args.no_fast_msda} "
          f"prompt tokens={det.token_cls.shape[1]} classes={len(det.token_cls)}")
    ref = ref_labels = None
    for config in args.configs.split(","):
        parts = config.split("/")
        gsize, simg = int(parts[0]), int(parts[1])
        backend = parts[2] if len(parts) > 2 else "trt"
        det.gdino_size, det.sam_imgsz = gsize, simg
        det.trt = not args.no_trt and backend == "trt"
        det.half = backend != "torch32"
        det.gdino_precision = parts[3] if len(parts) > 3 else "fp16"
        for img in images[: args.warmup]:
            det.predict(img)
        times.clear()
        results, labels = [], []
        for img in images:
            t0 = now()
            results.append(det.predict(img))
            times["total"].append(1e3 * (now() - t0))
            labels.append(det.last_ground_labels)
        med = {k: st.median(times[k]) if times[k] else 0.0 for k in STAGES + SUBSTAGES}
        other = med["total"] - sum(med[k] for k in STAGES[:-1])
        line = " ".join(f"{k}={med[k]:.0f}" for k in STAGES[:-1] + SUBSTAGES + STAGES[-1:])
        n = sum(len(r) for r in results) / len(results)
        agree = ""
        if ref is None:
            ref, ref_labels = results, labels
        else:
            rec, prec, miou = agreement(ref, results, images[0].shape[:2])
            agree = (f" | vs first: recall {rec:.2f} precision {prec:.2f} mask IoU {miou:.2f} "
                     f"ground labels {label_agreement(ref_labels, labels):.2f}")
        trt = f"trt {det.gdino_precision}" if det.trt_engine(*images[0].shape[:2]) is not None else backend
        print(f"[{gsize}/{simg} {trt}] ms {line} other={other:.0f} | {1e3 / med['total']:.2f} fps | dets/frame {n:.1f}{agree}")


if __name__ == "__main__":
    main()
