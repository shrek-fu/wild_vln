# PromptDA + Livox → Insta360 depth

Prompt-Depth-Anything style depth for this robot: **Livox** points are projected into an
**Insta360 pinhole view** as a metric `prompt_depth`, then PromptDA turns the RGB + prompt into a
dense depth map in metres.

PromptDA is a **pinhole** model. We use the calibrated views (`front` / `left` / `rear` / `right`),
not the equirect panorama. For full 360° equirect depth, keep using `da360_depth/` (relative +
optional lidar scale).

## Pipeline

```
Livox / lidar_level cloud
        │  project with K, T_level_optical (scale 1/7)
        ▼
 sparse depth ──KNN pool──► prompt_depth (metres)
                                  │
Insta360 /insta360/<view>/image ──┤
                                  ▼
                         PromptDA (DINOv2 + DPT)
                                  │
                                  ▼
                         /promptda/<view>/depth  (32FC1, metres)
```

## Setup

```bash
# weights (ViT-S fits RTX 5060 8 GB; use vitl if you have more VRAM)
src/depth/promptda_livox/scripts/download_weights.sh vits

# PromptDA sources: already under refer/robosense_ac_perception/python/PromptDA
# Needs torch + the same CUDA env as scene_memory / da360_depth
```

Calib: same as DA360 (`src/depth/da360_depth/calib/`).

## Offline

```bash
# one-shot from live ROS topics
python3 src/depth/promptda_livox/offline.py --ros --view front --save /tmp/promptda

# or files: RGB + Nx3 .npy in lidar_level
python3 src/depth/promptda_livox/offline.py --image front.jpg --cloud cloud.npy --view front
```

## Live

```bash
# camera + a lidar_level cloud (e.g. from the calibration bringup / extrinsic publisher)
python3 src/depth/promptda_livox/live_ros.py --views front --encoder vits --max_hz 1.0
# Foxglove: /promptda/front/depth_viz , /promptda/front/depth , /promptda/front/prompt
```

`--views front,left,rear,right` runs every view (heavier on GPU).

## Notes

* Lidar topic default `/extrinsic/lidar_fused` must be in **lidar_level** (same as `da360_depth/live_ros.py`).
* If the prompt has too few hits in a view (side/rear of MID-360), that view is skipped until the cloud moves.
* ViT-L: `download_weights.sh vitl` and `--encoder vitl --ckpt ...vitl.ckpt`.
