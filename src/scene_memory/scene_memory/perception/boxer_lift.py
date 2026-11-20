"""Run BoxerNet on pinhole views cut from the Insta360 equirect."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from scene_memory import paths
from scene_memory.common.calib_io import R_BODY_DA360, rotation_to_quaternion_xyzw
from scene_memory.common.geometry import equirect_uv_to_da360_rays

BOXER_ROOT = paths.REFER / "boxer"
DEFAULT_CKPT = BOXER_ROOT / "ckpts" / "boxernet_hw960in2x6d768-c88128f8.ckpt"


def load_pinhole_views(calib_dir: str | Path) -> list[dict]:
    """Four calibrated perspective views stored in ``insta360_intrinsics.json``."""
    path = Path(calib_dir) / "insta360_intrinsics.json"
    views = json.loads(path.read_text(encoding="utf-8"))["views"]
    loaded = []
    for name, view in views.items():
        loaded.append(
            {
                "name": name,
                "hfov_deg": float(view["hfov_deg"]),
                "R_body_optical": np.asarray(view["R_body_optical"], dtype=np.float64),
                "image_topic": str(view.get("image_topic", f"/insta360/{name}/image/compressed")),
                "width": int(view.get("width", 640)),
                "height": int(view.get("height", 640)),
            }
        )
    return loaded


def _optical_from_da360(R_body_optical: np.ndarray) -> np.ndarray:
    """``p_optical = R @ p_da360``."""
    return R_body_optical.T @ R_BODY_DA360


def _da360_from_optical(R_body_optical: np.ndarray) -> np.ndarray:
    """``p_da360 = R @ p_optical``."""
    return R_BODY_DA360.T @ R_body_optical


def pinhole_rays(height: int, width: int, hfov_deg: float) -> tuple[np.ndarray, float, float, float, float]:
    """Unit rays in the optical frame for every pixel, plus ``fx, fy, cx, cy``."""
    fx = 0.5 * width / np.tan(np.deg2rad(hfov_deg) * 0.5)
    fy = 0.5 * height / np.tan(np.deg2rad(hfov_deg) * 0.5)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    rays = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays, float(fx), float(fy), float(cx), float(cy)


def equirect_remap(
    equirect_h: int,
    equirect_w: int,
    R_body_optical: np.ndarray,
    view_h: int,
    view_w: int,
    hfov_deg: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Maps a pinhole view back onto the equirect. Returns ``map_x, map_y, fx, fy, cx, cy``."""
    rays, fx, fy, cx, cy = pinhole_rays(view_h, view_w, hfov_deg)
    da360 = rays @ _da360_from_optical(R_body_optical).T
    x, y, z = da360[..., 0], da360[..., 1], da360[..., 2]
    depth = np.linalg.norm(da360, axis=-1)
    theta = np.arccos(np.clip(y / depth, -1.0, 1.0))
    phi = np.arctan2(x, z)
    map_y = (np.pi - theta) * equirect_h / np.pi - 0.5
    map_x = (phi + np.pi) * equirect_w / (2.0 * np.pi) - 0.5
    return map_x.astype(np.float32), map_y.astype(np.float32), fx, fy, cx, cy


def equirect_depth_to_pinhole(
    depth_equirect: np.ndarray,
    R_body_optical: np.ndarray,
    width: int,
    height: int,
    hfov_deg: float,
) -> np.ndarray:
    """Sample DA360 equirect radial depth into pinhole optical-Z depth (metres).

    ``depth_equirect`` is ``||p||`` in the DA360 camera frame (same as
    ``/da360/depth`` / ``/da360/depth_metric``). Shared optical centre ⇒
    ``Z_opt = radial * ray_opt_z``.
    """
    eq = np.asarray(depth_equirect, dtype=np.float32)
    if eq.ndim != 2 or eq.size == 0:
        return np.zeros((height, width), dtype=np.float32)
    eh, ew = eq.shape
    map_x, map_y, _, _, _, _ = equirect_remap(
        eh, ew, R_body_optical, height, width, float(hfov_deg)
    )
    # Horizontal wrap for 360° continuity. Nearest, not linear: blending
    # foreground/background depth at object edges creates flying pixels.
    map_x = np.mod(map_x, float(ew))
    radial = cv2.remap(
        eq,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    rays, _, _, _, _ = pinhole_rays(height, width, float(hfov_deg))
    z = radial * rays[..., 2].astype(np.float32)
    bad = ~(np.isfinite(radial) & (radial > 0) & (rays[..., 2] > 1e-6))
    z[bad] = 0.0
    return z.astype(np.float32)


def project_equirect_box(
    parts: list[tuple[float, float, float, float]],
    equirect_w: int,
    equirect_h: int,
    R_body_optical: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    view_w: int,
    view_h: int,
) -> tuple[float, float, float, float] | None:
    """Equirect box to pinhole ``(x1, x2, y1, y2)``, the order BoxerNet queries use."""
    samples_u: list[float] = []
    samples_v: list[float] = []
    for x1, y1, x2, y2 in parts:
        for step in np.linspace(0.0, 1.0, 8):
            samples_u.extend([x1 + (x2 - x1) * step, x1 + (x2 - x1) * step, x1, x2])
            samples_v.extend([y1, y2, y1 + (y2 - y1) * step, y1 + (y2 - y1) * step])
    if not samples_u:
        return None
    rays = equirect_uv_to_da360_rays(
        np.asarray(samples_u), np.asarray(samples_v), equirect_w, equirect_h
    )
    optical = rays @ _optical_from_da360(R_body_optical).T
    front = optical[:, 2] > 0.2
    if int(front.sum()) < 4:
        return None
    uu = fx * optical[front, 0] / optical[front, 2] + cx
    vv = fy * optical[front, 1] / optical[front, 2] + cy
    px1, px2 = float(np.clip(uu.min(), 0, view_w - 1)), float(np.clip(uu.max(), 0, view_w - 1))
    py1, py2 = float(np.clip(vv.min(), 0, view_h - 1)), float(np.clip(vv.max(), 0, view_h - 1))
    if px2 - px1 < 4.0 or py2 - py1 < 4.0:
        return None
    return px1, px2, py1, py2


def view_center_ray_level(
    u: float,
    v: float,
    width: int,
    height: int,
    hfov_deg: float,
    R_body_optical: np.ndarray,
    T_level_da360: np.ndarray,
) -> np.ndarray:
    """Unit ray in ``lidar_level`` through one pixel of a calibrated pinhole view."""
    fx = 0.5 * width / np.tan(np.deg2rad(hfov_deg) * 0.5)
    fy = 0.5 * height / np.tan(np.deg2rad(hfov_deg) * 0.5)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    optical = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    optical /= np.linalg.norm(optical)
    da360 = _da360_from_optical(R_body_optical) @ optical
    rotation = np.asarray(T_level_da360, dtype=np.float64)[:3, :3]
    level = rotation @ da360
    return level / np.linalg.norm(level)


def _best_view(
    detection: dict,
    views: list[dict],
    equirect_w: int,
    equirect_h: int,
    intrinsics: dict[str, tuple[float, float, float, float]],
    view_size: int,
) -> tuple[str, tuple[float, float, float, float]] | None:
    from scene_memory.common.tiles import box_center_uv

    u, v = box_center_uv(detection["xyxy"], equirect_w)
    ray = equirect_uv_to_da360_rays(np.array([u]), np.array([v]), equirect_w, equirect_h)[0]
    best_name = None
    best_z = 0.2
    best_box = None
    for view in views:
        optical = _optical_from_da360(view["R_body_optical"]) @ ray
        if optical[2] <= best_z:
            continue
        fx, fy, cx, cy = intrinsics[view["name"]]
        box = project_equirect_box(
            detection.get("parts") or [],
            equirect_w,
            equirect_h,
            view["R_body_optical"],
            fx,
            fy,
            cx,
            cy,
            view_size,
            view_size,
        )
        if box is None:
            continue
        best_z = float(optical[2])
        best_name = view["name"]
        best_box = box
    if best_name is None or best_box is None:
        return None
    return best_name, best_box


class BoxerLifter:
    """YOLOE boxes on the equirect, 3D boxes from BoxerNet."""

    def __init__(self, calib_dir: str | Path, checkpoint: str | Path, device: str, thresh: float = 0.3, image_size: int | None = None) -> None:
        self.checkpoint = Path(checkpoint).expanduser()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"missing BoxerNet checkpoint {self.checkpoint}. "
                "Download boxernet and dinov3 into refer/boxer/ckpts "
                "(see refer/boxer/scripts/download_ckpts.sh)."
            )
        dino = BOXER_ROOT / "ckpts" / "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
        if not dino.is_file():
            raise FileNotFoundError(f"missing DINOv3 weights {dino}")
        root = str(BOXER_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        import torch
        from boxernet.boxernet import BoxerNet

        self.torch = torch
        self.device = device
        self.thresh = float(thresh)
        self.model = BoxerNet.load_from_checkpoint(str(self.checkpoint), device=device)
        hw = self.model.hw
        trained = int(hw[0] if isinstance(hw, (list, tuple)) else hw)
        self.size = int(image_size or trained)
        if self.size % 16 != 0:
            raise ValueError(f"BoxerNet image size {self.size} is not a multiple of 16")
        self.views = load_pinhole_views(calib_dir)
        self._intrinsics: dict[str, tuple[float, float, float, float]] = {}
        self._maps: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._prepared: dict[str, tuple] = {}
        self._batch_failed = False
        self._batch_error: Exception | None = None
        for view in self.views:
            _, fx, fy, cx, cy = pinhole_rays(self.size, self.size, view["hfov_deg"])
            self._intrinsics[view["name"]] = (fx, fy, cx, cy)

    def _remap(self, equirect: np.ndarray, view: dict) -> np.ndarray:
        height, width = equirect.shape[:2]
        key = (view["name"], height, width)
        if key not in self._maps:
            map_x, map_y, _, _, _, _ = equirect_remap(
                height, width, view["R_body_optical"], self.size, self.size, view["hfov_deg"]
            )
            self._maps[key] = (map_x, map_y)
        map_x, map_y = self._maps[key]
        return cv2.remap(equirect, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def _pose_and_camera(self, view: dict, T_level_da360: np.ndarray):
        from utils.tw.camera import CameraTW
        from utils.tw.pose import PoseTW

        torch = self.torch
        R_level_da360 = np.asarray(T_level_da360, dtype=np.float64)[:3, :3]
        origin = np.asarray(T_level_da360, dtype=np.float64)[:3, 3]
        R_level_opt = R_level_da360 @ _da360_from_optical(view["R_body_optical"])
        fx, fy, cx, cy = self._intrinsics[view["name"]]
        camera = CameraTW.from_surreal(
            width=torch.tensor([float(self.size)]),
            height=torch.tensor([float(self.size)]),
            type_str="pinhole",
            params=torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32),
        )
        pose = PoseTW.from_Rt(
            torch.tensor(R_level_opt, dtype=torch.float32).unsqueeze(0),
            torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
        )
        return camera, pose

    def _camera_pose(self, view: dict, T_level_da360: np.ndarray):
        key = np.asarray(T_level_da360, dtype=np.float64).tobytes()
        cached = self._prepared.get(view["name"])
        if cached is None or cached[0] != key:
            cached = (key, self._pose_and_camera(view, T_level_da360))
            self._prepared[view["name"]] = cached
        return cached[1]

    def lift_views(
        self,
        frames: dict[str, np.ndarray],
        grouped: dict[str, list[dict]],
        points_level: np.ndarray | None,
        T_level_da360: np.ndarray,
    ) -> int:
        """BoxerNet on the driver's pinhole views. Boxes are in each view's pixels."""
        points = np.zeros((0, 3), dtype=np.float32)
        if points_level is not None and len(points_level) > 0:
            cloud = np.asarray(points_level, dtype=np.float32).reshape(-1, 3)
            if cloud.shape[0] > 20000:
                step = int(np.ceil(cloud.shape[0] / 20000))
                cloud = cloud[::step]
            points = cloud
        views = {view["name"]: view for view in self.views}
        samples = []
        for name, dets in grouped.items():
            image = frames.get(name)
            view = views.get(name)
            if image is None or view is None or not dets:
                continue
            height, width = image.shape[:2]
            resized = image
            if width != self.size or height != self.size:
                resized = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            sx = self.size / float(width)
            sy = self.size / float(height)
            boxes = []
            for det in dets:
                x1, y1, x2, y2 = det["xyxy"]
                boxes.append((x1 * sx, x2 * sx, y1 * sy, y2 * sy))
            samples.append((resized, dets, boxes, view))
        if not samples:
            return 0
        if not self._batch_failed:
            try:
                return self._forward_batch(samples, points, T_level_da360)
            except Exception as exc:
                self._batch_failed = True
                self._batch_error = exc
        fitted = 0
        for resized, dets, boxes, view in samples:
            fitted += self._forward(resized, dets, boxes, view, points, T_level_da360)
        return fitted

    def lift(
        self,
        bgr: np.ndarray,
        detections: list[dict],
        points_level: np.ndarray | None,
        T_level_da360: np.ndarray,
    ) -> int:
        """Write BoxerNet boxes into ``det['obb']``. Returns how many passed the threshold."""
        for det in detections:
            det.pop("obb", None)
        if not detections:
            return 0
        height, width = bgr.shape[:2]
        grouped: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = {}
        for index, det in enumerate(detections):
            assigned = _best_view(det, self.views, width, height, self._intrinsics, self.size)
            if assigned is None:
                continue
            name, box = assigned
            grouped.setdefault(name, []).append((index, box))

        points = np.zeros((0, 3), dtype=np.float32)
        if points_level is not None and len(points_level) > 0:
            cloud = np.asarray(points_level, dtype=np.float32).reshape(-1, 3)
            if cloud.shape[0] > 20000:
                step = int(np.ceil(cloud.shape[0] / 20000))
                cloud = cloud[::step]
            points = cloud

        fitted = 0
        views = {view["name"]: view for view in self.views}
        for name, items in grouped.items():
            fitted += self._infer_view(bgr, detections, views[name], items, points, T_level_da360)
        return fitted

    def _infer_view(self, bgr, detections, view, items, points, T_level_da360) -> int:
        image = self._remap(bgr, view)
        boxes = [box for _, box in items]
        indexed = [detections[index] for index, _ in items]
        return self._forward(image, indexed, boxes, view, points, T_level_da360)

    def _image_tensor(self, bgr):
        torch = self.torch
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0

    def _run(self, datum):
        """Run BoxerNet and copy the head to CPU before inference mode ends."""
        torch = self.torch
        from contextlib import ExitStack

        contexts = [torch.inference_mode()]
        if self.device.startswith("cuda") and torch.cuda.is_bf16_supported():
            contexts.append(torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        with ExitStack() as stack:
            for ctx in contexts:
                stack.enter_context(ctx)
            outputs = self.model.forward(datum)
            return self._read_obbs(outputs)

    def _run_inputs(self, inputs):
        """Same as ``_run``, but the batch dict is already on device."""
        torch = self.torch
        from contextlib import ExitStack

        contexts = [torch.inference_mode()]
        if self.device.startswith("cuda") and torch.cuda.is_bf16_supported():
            contexts.append(torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        with ExitStack() as stack:
            for ctx in contexts:
                stack.enter_context(ctx)
            encoded = self.model.encode(inputs)
            outputs = self.model.query(inputs, encoded)
            return self._read_obbs(outputs)

    def _read_obbs(self, outputs):
        obbs = outputs["obbs_pr_w"].cpu()
        return (
            np.asarray(obbs.prob.detach()),
            np.asarray(obbs.T_world_object.t.detach()),
            np.asarray(obbs.T_world_object.R.detach()),
            np.asarray(obbs.bb3_object.detach()),
        )

    def _write_obbs(self, detections, scores, centers, rotations, extents, view_name: str) -> int:
        scores = np.reshape(scores, (len(detections), -1))[:, 0]
        if centers.ndim == 1:
            centers = centers.reshape(1, 3)
            rotations = rotations.reshape(1, 3, 3)
            extents = extents.reshape(1, 6)
        kept = 0
        for slot, det in enumerate(detections):
            score = float(scores[slot])
            if score < self.thresh:
                continue
            extent = extents[slot]
            size = np.array(
                [extent[1] - extent[0], extent[3] - extent[2], extent[5] - extent[4]],
                dtype=np.float64,
            )
            if not np.isfinite(size).all() or float(np.max(size)) > 8.0 or float(np.max(size)) < 0.05:
                continue
            quat = rotation_to_quaternion_xyzw(rotations[slot])
            yaw = float(np.arctan2(rotations[slot, 1, 0], rotations[slot, 0, 0]))
            mid = 0.5 * np.array([extent[0] + extent[1], extent[2] + extent[3], extent[4] + extent[5]])
            center = np.asarray(centers[slot], dtype=np.float64) + np.asarray(rotations[slot], dtype=np.float64) @ mid
            det["obb"] = {
                "center": [float(v) for v in center],
                "size": [float(v) for v in size],
                "yaw": yaw,
                "quat": [float(v) for v in quat],
                "rotation": np.asarray(rotations[slot], dtype=np.float64).tolist(),
                "score3d": score,
                "source": "boxer",
                "view": view_name,
            }
            kept += 1
        return kept

    def _forward_batch(self, samples, points, T_level_da360) -> int:
        """One BoxerNet forward for every view in this frame."""
        torch = self.torch
        from utils.tw.camera import CameraTW
        from utils.tw.pose import PoseTW

        max_count = max(len(boxes) for _, _, boxes, _ in samples)
        images = []
        box_rows = []
        cam_rows = []
        pose_rows = []
        for bgr, _dets, boxes, view in samples:
            images.append(self._image_tensor(bgr))
            padded = list(boxes)
            if len(padded) < max_count:
                padded.extend([padded[-1]] * (max_count - len(padded)))
            box_rows.append(padded)
            camera, pose = self._camera_pose(view, T_level_da360)
            cam_rows.append(camera._data if camera._data.ndim > 1 else camera._data.reshape(1, -1))
            pose_rows.append(pose._data if pose._data.ndim > 1 else pose._data.reshape(1, -1))
        batch = len(samples)
        cloud = torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32)).view(1, -1, 3)
        inputs = self.model.process_camera(
            {
                "img0": torch.stack(images, 0),
                "cam0": CameraTW(torch.cat(cam_rows, 0)),
                "T_world_rig0": PoseTW(torch.cat(pose_rows, 0)),
            }
        )
        # prepare_inputs only accepts an unbatched (N, 3) cloud, so the batched
        # depth and boxes are attached after the cameras are packed.
        inputs["sdp_w_padded"] = cloud.expand(batch, -1, 3).contiguous()
        inputs["bb2d"] = torch.tensor(box_rows, dtype=torch.float32)
        device = self.model.device
        for key in inputs:
            inputs[key] = inputs[key].to(device).float()
        scores, centers, rotations, extents = self._run_inputs(inputs)
        if centers.ndim == 2:
            centers = centers.reshape(1, -1, 3)
            rotations = rotations.reshape(1, -1, 3, 3)
            extents = extents.reshape(1, -1, 6)
            scores = scores.reshape(1, -1)
        kept = 0
        for index, (_bgr, dets, _boxes, view) in enumerate(samples):
            count = len(dets)
            kept += self._write_obbs(
                dets,
                scores[index, :count],
                centers[index, :count],
                rotations[index, :count],
                extents[index, :count],
                view["name"],
            )
        return kept

    def _forward(self, bgr, detections, boxes, view, points, T_level_da360) -> int:
        torch = self.torch
        camera, pose = self._camera_pose(view, T_level_da360)
        datum = {
            "img0": self._image_tensor(bgr),
            "cam0": camera,
            "T_world_rig0": pose,
            "sdp_w": torch.from_numpy(np.ascontiguousarray(points)),
            "bb2d": torch.tensor(boxes, dtype=torch.float32),
        }
        scores, centers, rotations, extents = self._run(datum)
        # A single view comes back with a leading batch dimension.
        if centers.ndim == 3:
            centers = centers[0]
            rotations = rotations[0]
            extents = extents[0]
            scores = scores[0]
        return self._write_obbs(detections, scores, centers, rotations, extents, view["name"])
