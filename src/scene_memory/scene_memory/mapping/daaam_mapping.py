"""DAAAM-aligned semantic frame prep and Hydra ``pipeline.step`` bridge.

Mirrors MIT-SPARK DAAAM:

* ``orchestrator._create_output_images`` — track-id mask → semantic label + color
* ``HydraIntegration.process_frame`` — depth m→mm, pose ``[x,y,z,qx,qy,qz,qw]``
  → Hydra ``[qw,qx,qy,qz]``, then ``pipeline.step``

Uses packaged Hydra dataset/labelspace configs via ``hydra_python.pipeline.load_pipeline``
(default: ``habitat`` + ``ade20k_mp3d``). Source ``refer/hydra_ws/setup_hydra_env.bash``
so the native bindings resolve.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from scene_memory.common.labelspace import LabelSpace


def clip_depth_meters(depth_m: np.ndarray, depth_lb: float = 0.25, depth_ub: float = 20.0) -> np.ndarray:
    """DAAAM ``depth.depth_lb`` / ``depth_ub``: out-of-range → 0 (invalid)."""
    out = np.asarray(depth_m, dtype=np.float32).copy()
    invalid = ~np.isfinite(out) | (out < float(depth_lb)) | (out > float(depth_ub))
    out[invalid] = 0.0
    return out


def depth_edge_mask(depth_m: np.ndarray, ratio: float = 0.08, radius: int = 1) -> np.ndarray:
    """Pixels on a depth discontinuity (flying pixels): local max-min > ``ratio * depth``."""
    d = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    if ratio <= 0 or radius <= 0 or not np.any(valid):
        return np.zeros(d.shape, dtype=bool)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    hi = cv2.dilate(np.where(valid, d, 0.0).astype(np.float32), kernel)
    lo = cv2.erode(np.where(valid, d, np.float32(1e6)).astype(np.float32), kernel)
    return valid & ((hi - lo) > float(ratio) * d)


def erode_label_edges(labels: np.ndarray, radius: int = 3) -> np.ndarray:
    """Zero labels within ``radius`` px of any label boundary so masks don't bleed onto background."""
    out = np.asarray(labels).copy()
    if radius <= 0 or not np.any(out):
        return out
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    f = out.astype(np.float32)
    out[cv2.dilate(f, kernel) != cv2.erode(f, kernel)] = 0
    return out


def create_label_and_color(
    width: int,
    height: int,
    tracks: list[dict],
    labelspace: LabelSpace,
) -> tuple[np.ndarray, np.ndarray]:
    """DAAAM ``_create_output_images``: paint one slot per track, lookup semantic id.

    Returns
    -------
    label_image : uint16 HxW (Hydra / DAAAM cast to int32 at step)
    color_image : uint8 HxWx3 BGR for visualization
    """
    if not tracks:
        return (
            np.zeros((height, width), dtype=np.uint16),
            np.zeros((height, width, 3), dtype=np.uint8),
        )

    combined = np.zeros((height, width), dtype=np.int32)
    # Higher score last so it wins overlapping pixels (same idea as DAAAM order).
    # Slots, not track ids, index the lookup: ids can be large (per-view offsets).
    ordered = sorted(tracks, key=lambda t: float(t.get("score", 0.0)))
    valid_ids: list[int] = []
    max_tid = 0
    for slot, det in enumerate(ordered, start=1):
        tid = slot
        painted = False
        mask = det.get("mask")
        if mask:
            polygon = np.asarray(mask, dtype=np.int32).reshape(-1, 2)
            if polygon.shape[0] >= 3:
                cv2.fillPoly(combined, [polygon], int(tid))
                painted = True
        if not painted:
            xyxy = det.get("xyxy")
            if not xyxy:
                continue
            x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
            x1 = min(max(x1, 0), width - 1)
            x2 = min(max(x2, 0), width)
            y1 = min(max(y1, 0), height - 1)
            y2 = min(max(y2, 0), height)
            if x2 > x1 and y2 > y1:
                combined[y1:y2, x1:x2] = tid
                painted = True
        if painted:
            valid_ids.append(tid)
            max_tid = max(max_tid, tid)

    if not valid_ids:
        return (
            np.zeros((height, width), dtype=np.uint16),
            np.zeros((height, width, 3), dtype=np.uint8),
        )

    semantic_lookup = np.zeros(max_tid + 1, dtype=np.uint16)
    color_lookup = np.zeros((max_tid + 1, 3), dtype=np.uint8)
    for tid, det in enumerate(ordered, start=1):
        if tid > max_tid:
            continue
        sid = labelspace.semantic_id(int(det.get("cls", -1)))
        semantic_lookup[tid] = sid
        color_lookup[tid] = labelspace.color_bgr(sid)

    label_image = semantic_lookup[combined]
    color_image = color_lookup[combined]
    zero = combined == 0
    label_image[zero] = 0
    color_image[zero] = 0
    return label_image.astype(np.uint16), color_image


class DaaamHydraBridge:
    """Subset of DAAAM ``HydraIntegration`` used for live semantic mapping."""

    def __init__(
        self,
        config_path: str = "",
        labelspace_path: str = "",
        depth_lb: float = 0.25,
        depth_ub: float = 20.0,
        dataset: str = "habitat",
        labelspace: str = "ade20k_mp3d",
        output_path: str = "",
        update_separation_s: float = 0.2,
        hydra_label_config: dict | None = None,
    ) -> None:
        self.config_path = str(config_path or "")
        self.labelspace_path = str(labelspace_path or "")
        self.depth_lb = float(depth_lb)
        self.depth_ub = float(depth_ub)
        self.dataset = str(dataset or "habitat")
        self.labelspace = str(labelspace or "ade20k_mp3d")
        self.output_path = str(output_path or "")
        self.update_separation_s = float(update_separation_s)
        self.hydra_label_config = dict(hydra_label_config) if hydra_label_config else None
        self.pipeline = None
        self.camera = None
        self.frames = 0
        self.attempts = 0
        self.reason = ""
        self._last_error = ""
        try:
            import hydra_python
            from hydra_python import HydraPipeline, make_camera, set_glog_level
            from hydra_python.pipeline import load_configs, update_nested
        except ImportError as exc:
            self.reason = f"hydra_python is not installed ({exc})"
            return
        self._hydra = hydra_python
        self._HydraPipeline = HydraPipeline
        self._make_camera = make_camera
        self._set_glog_level = set_glog_level
        self._load_configs = load_configs
        self._update_nested = update_nested
        self.reason = ""

    @property
    def available(self) -> bool:
        return self.reason == ""

    def _build_pipeline(self, camera):
        """Build ``HydraPipeline`` from packaged dataset configs (or a YAML file)."""
        self._set_glog_level(0, 0)
        config_file = Path(self.config_path) if self.config_path else None
        if config_file is not None and config_file.is_file():
            with config_file.open("r", encoding="utf-8") as fin:
                contents = yaml.safe_load(fin.read()) or {}
            label_file = Path(self.labelspace_path) if self.labelspace_path else None
            if label_file is not None and label_file.is_file():
                with label_file.open("r", encoding="utf-8") as fin:
                    self._update_nested(contents, yaml.safe_load(fin.read()) or {})
        else:
            # Packaged hydra config dir (share/hydra/config or source tree).
            pkg_config = None
            if config_file is not None and config_file.is_dir():
                pkg_config = config_file
            contents = self._load_configs(
                self.dataset,
                labelspace=self.labelspace,
                config_path=pkg_config,
            )
            if not contents:
                raise RuntimeError(
                    f"failed to load Hydra configs dataset={self.dataset} "
                    f"labelspace={self.labelspace}"
                )

        # Live RGB-D: allow sub-second updates (habitat default is 1.0s).
        self._update_nested(
            contents,
            {
                "frontend": {"type": "GraphBuilder"},
                "backend": {"type": "BackendModule"},
                "reconstruction": {
                    "type": "ReconstructionModule",
                    "show_stats": False,
                    "pose_graphs": {"make_pose_graph": True},
                },
                "active_window": {"full_update_separation_s": self.update_separation_s},
                "lcd": {"lcd_use_bow_vectors": False},
            },
        )
        # YOLOE prompt ids override packaged ade20k names when provided.
        if self.hydra_label_config:
            self._update_nested(contents, self.hydra_label_config)
        if self.output_path:
            out = Path(self.output_path)
            out.mkdir(parents=True, exist_ok=True)
            self._update_nested(contents, {"log_path": str(out)})

        return self._HydraPipeline.from_config(
            yaml.safe_dump(contents),
            camera,
            robot_id=0,
            config_verbosity=0,
            use_step_mode=True,
            zmq_url="",
        )

    def process_frame(
        self,
        timestamp: float,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        semantic_labels: np.ndarray,
        transform7: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> bool:
        """DAAAM ``HydraIntegration.process_frame`` contract."""
        depth = clip_depth_meters(depth_image, self.depth_lb, self.depth_ub)
        if int(np.count_nonzero(depth)) == 0:
            return False
        if not self.available:
            return False
        height, width = depth.shape[:2]
        try:
            if self.pipeline is None:
                self.camera = self._make_camera(
                    fx=float(fx),
                    fy=float(fy),
                    cx=float(cx),
                    cy=float(cy),
                    width=int(width),
                    height=int(height),
                    min_range=self.depth_lb,
                    max_range=self.depth_ub,
                    name="front_rgbd",
                )
                self.pipeline = self._build_pipeline(self.camera)
            pose = np.asarray(transform7, dtype=np.float64).reshape(-1)
            translation = pose[:3]
            # DAAAM: Hydra wants [qw, qx, qy, qz]
            quaternion = np.array([pose[6], pose[3], pose[4], pose[5]], dtype=np.float64)
            depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
            rgb = np.ascontiguousarray(rgb_image)
            labels = np.ascontiguousarray(semantic_labels, dtype=np.int32)
            self.attempts += 1
            ok = bool(
                self.pipeline.step(
                    int(timestamp * 1e9),
                    translation,
                    quaternion,
                    depth_mm,
                    labels,
                    rgb,
                )
            )
            if ok:
                self.frames += 1
            return ok
        except Exception as exc:  # noqa: BLE001 — keep ROS node alive on bad frames
            self._last_error = f"{type(exc).__name__}: {exc}"
            return False

    def save_graph(self, path: str | Path, include_mesh: bool = True) -> bool:
        """Write the current Hydra scene graph (spark_dsg; ``.json`` or ``.sparkdsg``)."""
        if self.pipeline is None:
            return False
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.pipeline.save_graph(str(path), include_mesh=include_mesh)
            return True
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"save_graph: {type(exc).__name__}: {exc}"
            return False

    def layer_counts(self) -> dict[str, int]:
        """Node count per named DSG layer (objects, places, rooms, ...)."""
        if self.pipeline is None:
            return {}
        counts: dict[str, int] = {}
        try:
            graph = self.pipeline.graph
            for name in graph.layer_names:
                try:
                    num = graph.get_layer(name).num_nodes
                    counts[str(name)] = int(num() if callable(num) else num)
                except Exception:  # noqa: BLE001
                    continue
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"layers: {type(exc).__name__}: {exc}"
        return counts

    def iter_object_boxes(self) -> list[dict]:
        """Read Hydra DSG OBJECTS layer → center/dims/label for MarkerArray."""
        out: list[dict] = []
        if self.pipeline is None:
            return out
        try:
            from spark_dsg import DsgLayers
            from spark_dsg import BoundingBoxType
        except ImportError:
            return out
        try:
            graph = self.pipeline.graph
            if graph is None or not graph.has_layer(DsgLayers.OBJECTS):
                return out
            layer = graph.get_layer(DsgLayers.OBJECTS)
            for node in layer.nodes:
                attrs = node.attributes
                bb = getattr(attrs, "bounding_box", None)
                if bb is None:
                    continue
                try:
                    valid = bool(bb.is_valid()) if callable(bb.is_valid) else bool(bb.is_valid)
                except Exception:  # noqa: BLE001
                    valid = bb.type != BoundingBoxType.INVALID
                if not valid or bb.type == BoundingBoxType.INVALID:
                    continue
                center = np.asarray(bb.world_P_center, dtype=np.float64).reshape(3)
                dims = np.asarray(bb.dimensions, dtype=np.float64).reshape(3)
                if not np.all(np.isfinite(center)) or not np.all(np.isfinite(dims)):
                    continue
                if float(np.max(dims)) < 1e-3:
                    continue
                sid = int(getattr(attrs, "semantic_label", 0) or 0)
                try:
                    corners = np.asarray(bb.corners(), dtype=np.float64).reshape(-1, 3)
                except Exception:  # noqa: BLE001
                    corners = None
                out.append(
                    {
                        "id": len(out),
                        "semantic_id": sid,
                        "name": str(getattr(attrs, "name", "") or ""),
                        "center": center,
                        "dims": dims,
                        "corners": corners,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"objects: {type(exc).__name__}: {exc}"
        return out
