# Spatio-Temporal System VLN

Insta360 X4 Air 360° camera + equirect VIO → open-vocabulary detection and tracking →
layered 4D topo scene graph (objects / places / regions / building, DAAAM-compatible DSG) →
semantic retrieval and spatio-temporal QA with GLM.

## Layout

```
.
├── src/                         ROS 2 workspace sources (colcon builds only this), one dir per layer
│   ├── sensors/
│   │   ├── insta360_views/        camera node: equirect + 4 pinhole views (C++, ROS pkg insta360_views)
│   │   └── drivers/               Insta360 Camera/Media SDK, Livox SDK + driver
│   ├── localization/
│   │   ├── super_lio/             Super-LIO (MID-360S lidar-inertial odometry, from QiNav), ROS pkgs basic + super_lio
│   │   └── equirect_vio/          equirect visual-inertial odometry, OpenVIO inside (own ros2_ws)
│   ├── depth/
│   │   └── da360_depth/           DA360 360° depth (upstream repo + live ROS node, own .venv)
│   ├── scene_memory/            4D scene memory (ROS pkg + Python module scene_memory)
│   │   ├── scene_memory/
│   │   │   ├── paths.py           single source of every default model/data path
│   │   │   ├── common/            calib_io, geometry, obb, labelspace, tiles
│   │   │   ├── perception/        yoloe / gdino_sam2 / sam3 detectors, byte_track, pose_track (BoT-SORT),
│   │   │   │                      mono_depth, boxer_lift, lift, infer_image
│   │   │   ├── mapping/           centroid_lift, object_map, boxer_objects, object_views, describe_objects,
│   │   │   │                      scene_graph, daaam_mapping, topo_graph, dsg_export
│   │   │   ├── query/             query_4d, semantic_index, region_summary, glm_retriever
│   │   │   ├── navigation/        planner (place-graph route + goal pose facing the target)
│   │   │   ├── viz/               annotate, dsg_viz
│   │   │   └── nodes/             detector_node, topo_map_node, goal_server_node, hydra_scene_node,
│   │   │                          daaam_live_node
│   │   ├── launch/                pipeline.launch.py (full system), lio.launch.py (MID-360S + Super-LIO), live.launch.py
│   │   ├── config/                prompts, distractors, DAAAM config
│   │   ├── scripts/               source-tree node runners (used by pipeline.launch.py), download_weights.sh
│   │   └── test/
│   ├── ground_segmentation/     camera traversable ground by class (GroundingDINO + SAM2 + lidar), obstacles_node
│   ├── mapping/                 3D point-cloud mapping (DLIO map accumulation), map.launch.py, save_map.py
│   ├── localization/scripts/    auto_initial_pose.py (last pose / global search -> /initialpose)
│   ├── planning/                DDDMR global + CMU local planner, qi_move_base, Go2-W control, nav.launch.py
│   ├── scripts/                 stvln.sh (one entry point), env.sh
│   └── tools/
│       └── collection_data/       record / replay / export sessions (ROS pkg collection_data)
├── models/                      all weights (see models/README.md)
├── data/
│   ├── calib/                   rig calibration JSON (+ export/ projection previews)
│   ├── sessions/<time>/         recordings: raw/, bag/, config/, scene4d/<run>/
│   └── scene4d/<time>/          live mapping runs
├── refer/                       upstream reference repos (DAAAM, boxer, ByteTrack, hydra_ws, ...)
├── paper/  jsons/               papers, camera config reference
└── build/ install/ log/         colcon output
```

A scene directory (`data/scene4d/<time>` or `data/sessions/<s>/scene4d/<run>`) holds
`objects_4d.json`, `topo_graph.json`, `views/`, `semantic_index.npz`, `region_summaries.json`,
`clustered_dsg_with_summaries.json` (spark_dsg) and the 3D viewers `topo_graph_3d.html` / `topo_graph.rrd`.

Environment overrides: `STVLN_ROOT`, `STVLN_MODELS`, `STVLN_DATA`, `STVLN_REFER`.

## Build

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select insta360_views collection_data scene_memory
# Super-LIO needs livox_ros_driver2 (CustomMsg) from the calibration repo install
source src/insta360_mid360s_calibration/install/setup.bash
colcon build --packages-select basic super_lio --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

Jetson (Orin NX, JetPack 6.2): the GPU nodes run from `<ws>/.venv` (`--system-site-packages`, jetson-ai-lab
torch 2.9.1 + torchvision 0.24.1, cuDSS 0.7 linked into `torch/lib`, ultralytics, transformers, Pillow 11);
`pipeline.launch.py` uses that interpreter whenever `<ws>/.venv` exists. `detector:=auto` picks gdino_sam2 as on
the PC (`detector:=yoloe` is the fast option on the Orin), and `insta360_views` stitches with its cuda backend
(no Media SDK).

## Run

`src/scripts/stvln.sh` wraps the commands below (environment from `src/scripts/env.sh`, launch files run from
`src`, `--bg` = detached with `log/stvln_<cmd>.log`; `stvln.sh help` lists everything):

```bash
src/scripts/stvln.sh live hydra_fake --bg      # pipeline, odom:=lio; then: stvln.sh check / status / stop
src/scripts/stvln.sh record_eval room1         # + gt_objects.yaml -> stvln.sh eval room1 (topo vs hydra_fake)
src/scripts/stvln.sh replay 20261107_112406    # offline re-run of a session
```

With QiNav: `stvln.sh build qinav_msgs` once (builds QiNav's `qi_nav_msgs` into `qinav_msgs/install`, sourced by
`env.sh`), start the QiNav 3D stack with its map (api 1007), then `stvln.sh live hydra_fake odom:=qinav --bg`.
Objects are then mapped in QiNav's `map` frame (pose `/ql/nav/localization`, cloud `/ql/nav/cloud_in_map`; our
driver/LIO stay off since QiNav runs its own), and every answered prompt is sent to `/slam/navigation/execute`
(api 1001, goal `Waypoint` in `map`; `--qinav_route` on the goal server also sends the place-graph route).
Progress: `/scene_memory/nav_status` (send result + mirrored `/ql/nav/task_status`).

```bash
# record a session (camera accepts one USB client: stop other camera nodes first)
ros2 launch collection_data record.launch.py
# live mapping (topo mapper, gdino_sam2 detector) -> data/scene4d/<time>
ros2 launch scene_memory pipeline.launch.py
# offline replay -> data/sessions/<s>/scene4d/<time>
ros2 launch scene_memory pipeline.launch.py session:=data/sessions/20261107_112406
```

Odometry (`odom:=lio`, default): the MID-360S driver (`xfer_format 1`, not together with the
calibration bringup) and Super-LIO give `/lio/odom` (`odom` → `livox_frame`, the lidar IMU) and the deskewed
scan `/lio/cloud_world`. Camera poses chain `T_odom_livox · T_imu_lidar` (`config/super_lio_mid360.yaml`)
`· T_front_insta360` (`data/calib`). Objects whose mask holds ≥ 12 lidar returns (last 1 s of scans,
nearest depth cluster) are placed at the measured 3D point (`source` lidar, preferred over
triangulation); `view_depth_node` publishes `/insta360/<view>/depth_lidar` (`32FC1`, 0 = no return) and,
with `view_depth_overlay:=true`, `/insta360/<view>/depth_lidar/overlay/compressed` to check extrinsics.
`odom:=vio` keeps the equirect VIO; a recorded session without `/lio/odom` falls back to VIO.

```bash
ros2 launch scene_memory lio.launch.py                                   # driver + Super-LIO + camera TF only
ros2 launch scene_memory pipeline.launch.py view_depth_overlay:=true
ros2 launch scene_memory pipeline.launch.py odom:=vio
```

## 3D mapping and navigation on the Go2-W (without QiNav)

STVLN's own copy of QiNav's point-cloud mapping, localization, planners and Go2-W control (`src/mapping`,
`src/localization`, `src/planning`), isolated from QiNav: own topic names (`/stvln/cmd_vel`,
`/stvln/unitree/posture_cmd`, ...), and `map` / `nav` refuse to start while QiNav's driver / LIO /
localization / planner / controller processes run (qi-nav-server itself may keep running). Each map gets
its own folder, `data/maps/<name>/` (`data/maps/demo/demo_map.pcd`, `demo/demo_scene/`, ...). Older flat maps
(`data/maps/<name>_*`) still load; `src/scripts/stvln.sh map_migrate <name>` (or `--all`) moves them into
their folders.

On the Orin (`192.168.110.56`):

```bash
cd ~/stvln && source src/scripts/env.sh          # ROS_DOMAIN_ID=10, CycloneDDS on lo, .venv python
src/scripts/stvln.sh build planning              # once / after pulling: planners, localization, DLIO, Go2-W control
sudo nvpmodel -q                                 # full semantic stack: use MAXN (25 W hard-rebooted the board)
```

Foxglove: robot bridge `ws://192.168.110.56:8766`, display frame `map` (mapping: `odom`), URDF layer from
`/robot_description`.

### 1. Map

```bash
src/scripts/stvln.sh map office --bg             # Super-LIO + linefit + DLIO map + semantic pipeline
#   semantic:=false     point cloud only (no camera / detector / topo graph; GPU free)
#   ground:=linefit     planning ground from linefit instead of the camera ground segmentation
# walk the area slowly (loops help), watch /stvln/laser_map, /topo/markers, /ground/markers in Foxglove
src/scripts/stvln.sh map_save office             # while mapping: PCDs + traversable + semantic files
src/scripts/stvln.sh stop map
src/scripts/stvln.sh map_save office --semantic_only   # final semantic export after stop (optional)
```

`map_save` writes into `data/maps/<name>/`:
- `<name>_map.pcd` (localization), `_ground.pcd` / `_nonground.pcd` / `_poses.pcd`;
- `<name>_traversable.pcd` (global planner graph, intensity 0.25 = lethal);
- `<name>_ground_sem.pcd` / `_nonground_sem.pcd` (camera ground, with `ground:=semantic`);
- `<name>_topo_graph.json`, `_objects_4d.json`, `_semantic.json` + the scene dir `<name>_scene/`.

The planning ground defaults to the camera ground segmentation (`--ground auto`), because linefit's ground
follows ramps and uneven floors badly. Re-export only the planner graph from saved PCDs:

```bash
src/scripts/stvln.sh map_save office --traversable_only --trav_leaf 0.2 --obstacle_half 0.3
```

| `map_save` option | default | meaning |
|---|---|---|
| `--ground` | auto | auto / semantic / linefit planning ground |
| `--blocked_classes` | – | ground classes that are never traversable, on top of `@blocked` in `config/ground_classes.txt` |
| `--trav_leaf` | 0.2 | m, planner graph vertex spacing |
| `--obstacle_half` | 0.3 | m, half size of the box around a vertex searched for obstacle points |
| `--leaf` | 0.1 | m, PCD voxel size |

#### Offline mapping on the PC

Use this when the full semantic stack is too heavy for the board. Record raw data on the robot, then build the
point-cloud map and the semantic map on the PC and copy them back. `map_offline` replays the derived
session (panorama and views) together with the raw lidar, and runs the same chain as live mapping
(Super-LIO, linefit, DLIO, semantic pipeline). The PCD map and the semantic map therefore come from one
Super-LIO run and share one frame.

```bash
# robot: raw sensors only
src/scripts/stvln.sh record office_rec --bg      # walk the area, then: src/scripts/stvln.sh stop record
# PC: data dir on a big disk, then copy the session over
export STVLN_DATA=/media/foo/50B3920F3AB07555/stvln_data
rsync -a user@192.168.110.56:stvln/data/sessions/office_rec $STVLN_DATA/sessions/
src/scripts/stvln.sh derive office_rec          # -> office_rec_proc (stitched panorama + Super-LIO odometry)
src/scripts/stvln.sh map_offline office_rec office   # replay, map_save, stop map, map_save --semantic_only
src/scripts/stvln.sh map_push office            # rsync $STVLN_DATA/maps/office/ to user@192.168.110.56:stvln/data/maps/office/
# robot
src/scripts/stvln.sh nav office --bg            # loads office_map.pcd + the office semantic layer / VLN goals
```

`map_offline` accepts `map` options (`semantic:=false`, `ground:=linefit`, ...) plus `replay_rate:=1.0` and
`--drain_s=20` (seconds the semantic nodes get after the replay ends). The replay runs at bag pace; when the PC
falls behind, the run just takes longer, because every node uses the recorded stamps.

After the replay, `map_offline` adds two FSR-VLN layers to `<name>_scene` (skip them with `--no_rooms` /
`--no_views`):

- **Rooms from the lidar map** (`scene_memory.mapping.pcd_rooms`):
  1. Find the floor and ceiling heights.
  2. Mark as wall every cell with points in at least half of the 0.1 m slices between 0.5 m and 2.4 m
     above the floor, so desks and ceiling lamps are not walls.
  3. Treat everything outside the map's footprint as outside.
  4. Seed rooms from the distance transform of the free cells (Otsu threshold), then grow the seeds at
     equal speed, so rooms split in the middle of doorways.

  Places and objects get `room_pcd`, and regions follow those rooms. Door cuts and the radius /
  clearance rules no longer apply. `rooms.png` shows the result.
- **View layer** (`scene_memory.mapping.view_layer`): a panorama keyframe of the derived session every
  1 m or 45°, posed from the scene's trajectory and the rig calibration. Each keyframe is split into
  8 sectors, and each sector gets a SigLIP embedding. The files are `keyframes/`, `keyframes.json` and
  `view_index.npz`.
- **Region labels:** `dsg_export --summarize` labels the new regions. With `--glm` it uses GLM;
  otherwise the labels are class counts.

The view layer gives the VLN agent the tool `search_views`, which is FSR-VLN's slow path. It is meant for
things outside the detector classes, or cases where `find_objects` finds nothing that fits. It runs in four
steps:
1. SigLIP retrieves the best sector of each of the top 6 keyframes.
2. The vision model picks the sector that shows the request.
3. The vision model boxes the target in that sector.
4. The ray through the box centre is cut with `map_cloud.npy` to get a 3D point, which becomes a temporary
   object key for `route` / `answer`.

Without a vision model, the best SigLIP sector's centre is used, and the result is marked unverified.

**Local vision model (Ollama).** The vision steps (`look_at`, the choose and ground steps of `search_views`, and
the fast path's check) can run on a local Qwen3-VL instead of GLM-4V. They can use a separate endpoint from the
GLM text agent, so they work without a Zhipu key. Ollama, the models and the logs all live on the data disk
(no sudo needed):

```bash
O=/media/foo/50B3920F3AB07555/stvln_data/ollama   # ollama-linux-amd64.tar.zst from the GitHub releases, unpacked here
OLLAMA_MODELS=$O/models OLLAMA_HOST=127.0.0.1:11434 OLLAMA_KEEP_ALIVE=30m nohup $O/bin/ollama serve > $O/serve.log 2>&1 &
$O/bin/ollama pull qwen3-vl:4b-instruct            # ~3.2 GB, ~4 GB VRAM; the 8b needs ~7 GB
export VISION_BASE_URL=http://127.0.0.1:11434/v1 VISION_MODEL=qwen3-vl:4b-instruct
python3 -m scene_memory.query.view_search $STVLN_DATA/maps/office_loop8_off/office_loop8_off_scene potted plant --save /tmp/view.jpg
```

`--vision_url` and `--vision_model` (environment variables `VISION_BASE_URL` and `VISION_MODEL`) are accepted by
`glm_retriever`, `goal_server_node` and `view_search`. A remote GPU machine works the same way: run
`ollama serve` there with `OLLAMA_HOST=0.0.0.0:11434`, and point `VISION_BASE_URL` at `http://<host>:11434/v1`.

The desktop at 192.168.110.209 (RTX 3090 24 GB) serves `qwen3-vl:8b-instruct` this way:
- Ollama is in `~/ollama`, run as the systemd user service `ollama` (linger is on, so it starts at boot without a
  login).
- The first request after 30 min idle loads the model (~30 s); later requests take ~1 s.

```bash
export VISION_BASE_URL=http://192.168.110.209:11434/v1 VISION_MODEL=qwen3-vl:8b-instruct
# on .209: systemctl --user status|restart|stop ollama ; journalctl --user -u ollama -f
```

Qwen3-VL also calls tools, so it can replace GLM as the text agent. Point `GLM_BASE_URL` / `GLM_MODEL` (or
`--base_url` / `--model`) at it, and no Zhipu key is needed. `stvln.sh ... --vlm` sets all four variables for
.209; override them with `STVLN_VLM_URL` / `STVLN_VLM_MODEL`:

```bash
src/scripts/stvln.sh nav office_loop8_off --bg --vlm          # on the robot; then e.g.
src/scripts/stvln.sh query "去窗户那边"
```

Maps that went through `map_offline` already have the rooms and the view layer, so they need no new run.

When the fast path finds no detector class for the phrase, or the vision check rejects every candidate, it tries
`search_views` itself before handing over to the agent. A small model can't be relied on to call that tool,
so the code makes the call. Only a vision-confirmed hit is taken.

If `ollama pull` fails with "redirect target not allowed ... resolves to non-public 198.18.x.x", the cause is a
proxy that hands out fake IPs, and Ollama refuses to follow the redirect. In that case, fetch the manifest
`https://registry.ollama.ai/v2/library/qwen3-vl/manifests/4b-instruct` and its blobs with curl, and store them as
`$O/models/blobs/sha256-<digest>` and `$O/models/manifests/registry.ollama.ai/library/qwen3-vl/4b-instruct`.

**Object boxes.** Each object's `bbox3d` comes from the lidar voxels of its masks. To stop background at the
mask edges from inflating it, only voxels inside the class's largest plausible box around the object's anchor
(its fused `center`) count. That box is `centroid_lift.class_max_size`: the class's typical size ×1.5 + 0.4 m
horizontally, ×1.3 + 0.3 m vertically. On export, `size` and `bbox3d` are capped the same way. Scenes saved
before this change can be fixed in place (then `map_save --semantic_only` / `map_push`):

```bash
python3 -m scene_memory.mapping.box_limits $STVLN_DATA/maps/office/office_scene --save
```

Both layers can be rebuilt on an existing map scene:

```bash
python3 -m scene_memory.mapping.pcd_rooms $STVLN_DATA/maps/office/office_scene --pcd $STVLN_DATA/maps/office/office_map.pcd --plot /tmp/rooms.png --save
python3 -m scene_memory.mapping.view_layer $STVLN_DATA/maps/office/office_scene --session $STVLN_DATA/sessions/office_rec_proc
python3 -m scene_memory.mapping.dsg_export $STVLN_DATA/maps/office/office_scene --summarize --language Chinese
src/scripts/stvln.sh map_save office --semantic_only
```

### 2. Navigate

```bash
src/scripts/stvln.sh nav office --bg                         # planners only, the dog does not move
src/scripts/stvln.sh nav office control:=true --glm --bg     # drive the Go2-W, GLM for free-form VLN
src/scripts/stvln.sh stop nav
```

On start:
- **Localization.** QiNav's ICP localization on `<map>_map.pcd` publishes TF `map → odom` and
  `/localization`. `auto_initial_pose` publishes `/initialpose` itself. It first tries the last saved pose
  (`data/loc/<map>_last_pose.json`, written every 5 s while localized). Otherwise it runs a global search of
  the accumulated LIO cloud over the map's traversable cells (gn10-pointcloud-localization style). It
  checks the result against `/localize/icp_score` and searches again on `localization_failed`. Once
  localized it only tracks; it does not relocalize by itself. To relocalize by hand, publish
  `/auto_initial_pose/trigger` or use Foxglove "2D Pose Estimate" (`/initialpose`).
- **Semantic layer.** `<map>_scene/topo_graph.json` is published on `/topo/markers` in `map` (object boxes and
  labels, places, rooms).
- **VLN.** `goal_server` runs on the same scene and sends goals to `/p2p_move_base`.

With `control:=true`, the Go2-W stands up (StandUp + speed level 1) by itself when a point goal
(`/clicked_point`) or a semantic goal (`/scene_memory/goal`) arrives, or on the first motion command. If it
already stands, nothing is sent, so a new goal while walking does not stop it. With `auto_prepare:=false`
it only stands up by hand. Posture commands:

```bash
ros2 topic pub --once /stvln/unitree/posture_cmd std_msgs/msg/Int32 "{data: 1}"   # StandUp
# 0 StopMove (soft e-stop), 2 StandDown, 3 Damp, 21 / 22 / 23 speed level
```

The node only knows postures it commanded itself. If the dog was laid down with the remote, send `2`
(StandDown) or `1` once, so the next goal stands it up again.

Goals:

```bash
# point goal (Foxglove "Publish point" on /clicked_point, or):
ros2 topic pub --once /clicked_point geometry_msgs/msg/PointStamped \
    "{header: {frame_id: map}, point: {x: 2.0, y: 0.5, z: 0.0}}"
# language goal on the saved semantic map:
ros2 topic pub --once /scene_memory/query std_msgs/msg/String "{data: '去电视那里'}"
ros2 topic echo /scene_memory/answer       # reply, target, plan
ros2 topic echo /scene_memory/nav_status   # p2p_move_base accepted / finished
```

Goals on a lethal or off-graph point snap to the nearest free graph vertex (within
`free_node_search_radius`, 1 m). Goals above the ground are searched down to 1 m, so place heights work.
Do not publish on `/goal_pose_3d`: `clicked2goal` reads `PoseStamped` there while `p2p_move_base` writes
`PointStamped`.

**Occupied goals.** The global planner only sees the static map. When a person or a box stands on the goal,
the local planner stops in front of it, and without a check the action would never end. `p2p_move_base`
therefore has a stall watchdog (`qi_move_base.yaml`):
- Within `near_goal_accept_dis` (1.5 m), if the distance to the goal does not shrink by `stall_progress`
  (0.2 m) for `stall_patience` (8 s), the robot turns to the goal heading and succeeds.
- Farther out, if the robot does not move for `stall_abort_patience` (30 s), the goal aborts.

`goal_server` then tells the outcomes apart on `nav_status` (and, with a reply, on `answer`):

| event | meaning |
|---|---|
| `arrived` | the goal was reached |
| `arrived_near` | the robot stopped within `--near_goal_m` 1.5 m |
| `retry` | after an abort farther out, the next free approach pose around the target, away from the blocked ones (up to `--goal_retries` 2) |
| `alternative` | the next alternative target (up to `--alt_targets` 1) |
| `failed` | nothing worked |

| `nav` argument | default | meaning |
|---|---|---|
| `control` | false | `robot_controller` + `go2_cmd_vel_node` (`/stvln/cmd_vel` → SportClient on eth0) |
| `auto_prepare` | true | StandUp + speed level 1 on a point / semantic goal or the first motion command, if not standing |
| `auto_init` | true | `auto_initial_pose` (last pose, then global search) |
| `semantic_viz` | true | saved semantic layer → `/topo/markers` |
| `vln` | true | `goal_server` on `<map>_scene` → `/p2p_move_base`; `--glm` loads the GLM key (else fast path only) |
| `local_obstacles` | freespace | `freespace`: local planner obstacles from the camera freespace (GPU, MAXN); `linefit`: QiNav's linefit split |
| `freespace_views` / `freespace_hz` | front / 0.5 | segmented views and frames per second per view |
| `freespace_window` | 30.0 | m, side of the freespace square kept around the robot |
| `ground_blocked_classes` | – | extra classes that are always obstacles (e.g. `grass`) |
| `semantic_local` | false | `local_obstacles:=linefit` only: add the camera ground's obstacle cells |
| `max_speed` | 0.5 | m/s |
| `goal_tolerance_xy` / `goal_tolerance_yaw` | planner default | arrival tolerances |
| `camera` / `battery` / `model` | true | Insta360 views, Go2-W battery, Go2-W URDF |

**Local obstacles (freespace).** The local planner does not use linefit by default: its per-scan ground
line splits ramps, kerbs and low objects wrongly. Instead:
1. `ground_node` segments the ground classes (`config/ground_classes.txt`, GroundingDINO + SAM2, front view,
   about 1 s per frame on the Orin) and lifts them with the lidar: the image decides what is ground, the
   lidar where it is. Only the 30 × 30 m square around the robot is kept (`freespace_window`), so no global
   freespace is built during navigation. Output: `/ground/traversable_cloud` (odom), `/ground/markers`.
2. `freespace_obstacles` measures every LIO scan (10 Hz, last 0.3 s) against that ground: each point's
   height above the nearest freespace cell (within 1 m), else above the ground under the robot. Points
   0.15–1.2 m above it go to `/stvln/local_obstacles` (map), the planner's `/terrain_map`. It never waits
   for a camera frame, so people and boxes show up at lidar rate. `@blocked` / `ground_blocked_classes`
   cells within 4 m go to `/no_entry_obstacle_pc`, the planner's `/added_obstacles`. Points within 0.45 m
   of the robot are skipped.

Without camera images (`camera:=false` and no `insta360_views`), the ground under the robot is the only
reference: fine on flat floors, not on ramps. `local_obstacles:=linefit` restores linefit +
`/non_ground_in_map` (with `semantic_local:=true` adding the camera's obstacle cells on top).

Troubleshooting:
- **"Waiting for initial pose":** see `/auto_initial_pose/status`. Publish `/auto_initial_pose/trigger`
  (`std_msgs/msg/Empty`), or set the pose by hand.
- **"No path found":** the start or goal has no free vertex nearby. Re-export the traversable map with a
  smaller `--obstacle_half`, or check `/ql/nav/global_path`.
- **"motion requested but robot is not prepared":** `auto_prepare:=false`, or StandUp failed (see the
  `prepare (...)` log line). Send posture `1` (StandUp).

## Query a scene

```bash
S=data/sessions/20261107_112406/scene4d/latest
python3 -m scene_memory.query.query_4d $S summary
python3 -m scene_memory.query.query_4d $S find "trash can near door"
python3 -m scene_memory.mapping.dsg_export $S --summarize --language Chinese   # regions + DAAAM DSG
python3 -m scene_memory.viz.dsg_viz $S --html $S/topo_graph_3d.html          # 3D viewer
export ZHIPUAI_API_KEY=$(cat ~/.zhipuai_key)
python3 -m scene_memory.query.glm_retriever $S -i                             # spatio-temporal QA
```

## Rooms after mapping: revisits + free space (offline)

Mapping cuts regions at detected doors and then splits them every 4 m along the path, which over-segments
open areas (22 regions on the example session). It also has no loop closure. If you walk back through a
room and VIO has drifted, or the return path doesn't overlap the first pass, the same room appears twice as
separate place chains and therefore separate rooms. On the example session, the outbound and return paths
through the same office and lab are 3.5–7.5 m apart. The steps below fix both problems. No remapping is
needed: they reuse the saved `topo_graph.json` / `objects_4d.json`.

1. **Revisits (loop closure).** Each panorama in `equirect/` gets a SigLIP fingerprint made of 8
   horizontal sectors, compared at every circular shift, so walking the other way still matches.
   Candidate frame pairs must:
   - be at least 30 s apart and look alike;
   - be within plausible drift (3 m + 10% of the distance walked in between);
   - be a real shortcut in the graph.

   At least 3 neighbouring pairs must agree on the offset, which guards against repetitive offices.
   Accepted pairs become `revisit` edges, and the two places are always put in the same room. A
   translation-only pose graph then pulls them together. Places, agents and objects (centre and
   history, by their stamps) follow the correction, and the originals are kept as `position_odom` /
   `center_odom` plus the backups `topo_graph_odom.json` / `objects_4d_odom.json`. `frame_id` stays `odom`;
   the graph gets `loop_corrected: true` and the list of `revisits`.
2. **Horizontal profiles.** For every frame, DA360 predicts 360° depth. The floor plane is fitted, which
   gives gravity and camera height, and the nearest obstacle per azimuth is stored in camera heights (no
   lidar or metric scale needed). The result goes to `<session>/equirect_profiles.json`.
3. **Clearance and rooms (Hydra RoomFinder style).** Each frame is matched to the trajectory by stamp, and
   nearby places get its free distance, skipping the operator sector behind the camera. A threshold sweep
   keeps the room count that stays stable longest; those components seed the rooms, and every other place
   joins the nearest seed in the graph. Door cuts and revisit links still apply.
4. **Labels and DSG.** Regions are renumbered, so the old LLM labels no longer match; regenerate them.
   `clearance` and revisit edges stay in the graph, so later regraphs keep the new rooms.
5. **Check.** Print the tree, and rewrite the 3D page (an existing HTML file still shows the old graph).

Run everything from the workspace root, because the paths are relative to it:

```bash
cd ~/00WS/00qiRobotics/10SpationTemporalSystemVLN && source install/setup.bash
SESSION=data/sessions/20261107_112406
S=$SESSION/scene4d/latest

# 1. revisits: compare first (prints loops, drift correction, regions, writes a before/after PNG), then save
python3 -m scene_memory.mapping.revisit $S --plot /tmp/revisit.png
python3 -m scene_memory.mapping.revisit $S --save        # backups: *_odom.json; --no_correct = edges only

# 2. profiles (DA360 venv, GPU)
src/depth/da360_depth/.venv/bin/python src/depth/da360_depth/scripts/horizontal_profile.py \
    --session $SESSION --model_path src/depth/da360_depth/checkpoints/DA360_large.pth

# 3. compare radius vs clearance regions (prints stats + timing, writes a top-down PNG), then save
python3 -m scene_memory.mapping.clearance $S --profiles $SESSION/equirect_profiles.json --plot /tmp/rooms.png
cp $S/topo_graph.json $S/topo_graph_radius.json          # optional backup before the clearance rooms
python3 -m scene_memory.mapping.clearance $S --profiles $SESSION/equirect_profiles.json --save

# 4. region labels + DAAAM DSG (falls back to class counts without the key)
export ZHIPUAI_API_KEY=$(cat ~/.zhipuai_key)
python3 -m scene_memory.mapping.dsg_export $S --summarize --language Chinese

# 5. check
python3 -m scene_memory.viz.dsg_viz $S --tree
python3 -m scene_memory.viz.dsg_viz $S --html $S/topo_graph_3d.html && xdg-open $S/topo_graph_3d.html
```

Undo:
- **Drift correction and revisits:** `cp $S/topo_graph_odom.json $S/topo_graph.json; cp $S/objects_4d_odom.json $S/objects_4d.json`.
- **Clearance rooms:** `cp $S/topo_graph_radius.json $S/topo_graph.json`.

`revisit` refuses to run twice on the same graph; restore the `_odom` backups first.

| option | default | meaning |
|---|---|---|
| `revisit --sim_min` | 0.86 | SigLIP similarity for a frame pair (same room ≈ 0.87–0.93, other rooms ≤ 0.85 here) |
| `revisit --min_support` | 3 | neighbouring frame pairs that must agree on the offset |
| `revisit --drift_rate` | 0.1 | allowed drift per metre walked between the two visits |
| `revisit --no_correct` | off | only add revisit edges (rooms merge), keep VIO positions |
| `horizontal_profile.py --model_path` | `DA360_base.pth` | small / base / large: 132 / 251 / 678 ms per frame (RTX 5060); large fits the floor best |
| `horizontal_profile.py --min_r` | 0.6 | ignore points closer than this, in camera heights (operator, rig) |
| `clearance --ignore_sector` | 60 240 | azimuth range to skip, in degrees (0 = camera forward, 90 = right); where a hand-held operator stands |
| `clearance --percentile` | 30 | percentile of the free distances around a place; lower means closer to Hydra's minimum distance (use it when no operator blocks the view) |
| `clearance --camera_height` | – | metres; without it, clearance is in camera heights (the sweep does not depend on units) |
| `clearance --min_room_places` | 4 | minimum component size that counts as a room seed |

On the example session:
- **Revisits:** SigLIP takes 5–9 s for 113 frames (cached in `<session>/equirect_siglip8.npz`), matching
  0.2 s, correction plus regraph 0.3 s. It finds 13 revisit links (VIO offsets 3–7.7 m), shifts places by
  up to 4.5 m (2.4 m on average), and cuts the median route from the start to an object from 64.8 m to
  28.0 m.
- **Profiles:** about 80 s with DA360_large (about 30 s with base).
- **Clearance:** the clearance step and regraph take about 30 ms.
- **Regions:** 22 with the radius rule, 4 with clearance (bounded by the detected doors), and 3 with
  revisits plus clearance (outbound and return pass through the same space become one region). The
  graph's `segmentation` field records the method, threshold and number of seeds.

## 4D mapping → prompt → navigation goal

`topo_map_node` (default mapper) keeps one object per physical instance:

| step | method | source |
|---|---|---|
| 3D position | ray triangulation over a track's VIO poses, class-size prior until there is parallax | WayTrace / nebula triangulator |
| tracking | BoT-SORT + projected 3D centres as motion prior | SuperMap |
| label | Bayesian fusion of soft detector scores | SuperMap Eq. 10 |
| existence (4D) | log-odds: +hit when detected, −miss when the object projects well inside a processed view (< `miss_range`, large enough, not behind a nearer detection) but is not detected → `status` present / absent, `events` appeared / disappeared / reappeared | SuperMap, canopy `checkAbsence` |
| dynamics | `dynamic_classes` (person) tracked but kept out of the static graph / DSG / navigation | CausalNav |
| layers | places (driven keyframes) → regions (cut at doors) → floors → buildings + outdoor area, agents = trajectory | Hydra / DAAAM |
| indoor / outdoor | SigLIP zero-shot on 4 pinhole crops of the equirect every 2 s, EMA + hysteresis (`environment:=auto`, or force `indoor` / `outdoor`) | — |
| floors | VIO height: a settled change ≥ 2 m indoors = floor change (`floor_height` 3 m per storey); elevator (no horizontal motion) / stairs (slope ≥ 0.35) / ramp; places created meanwhile become connector places; floors count from the start / entrance floor (level 0) | — |
| buildings | indoor places connected without going outdoors, pieces with overlapping footprints merged; regions never span floors, indoor/outdoor or connectors; region `links` kind entrance / stairs / elevator / ramp | — |

With `ZHIPUAI_API_KEY` set, `pipeline.launch.py` also starts `goal_server_node`:

```bash
ros2 topic pub --once /scene_memory/query std_msgs/String "{data: '去门旁边的垃圾桶'}"
ros2 topic echo /scene_memory/answer      # JSON: reply, target, status, plan
# /scene_memory/goal (PoseStamped), /scene_memory/path (nav_msgs/Path), /scene_memory/goal_markers
ros2 launch scene_memory pipeline.launch.py nav2:=through_poses   # also forward to Nav2
```

Query = fast path, then agent (FSR-VLN style fast / slow reasoning, `query/fast_path.py`):

* fast (~0.1–0.3 s on CPU, no LLM): simple "go to / find X" instructions, optionally with a level (`2楼`,
  `楼上`, `外面`), a labelled region (`办公室里的…`, needs region summaries) or a relation (`门旁边的垃圾桶`).
  The phrase is mapped to a detector class by SigLIP text similarity (a clear winner is required, so
  `冰箱` / `机器人` outside the vocabulary are left to the agent), candidates are filtered
  level → region → class, ranked by image–text score (attributes like 黑色) and relation, ties broken by
  route length.
* slow (GLM tool agent): questions, counting, time, ordinals / left-right, follow-ups, negations, several
  targets, unlabelled room words, absent / dynamic targets, or when the fast path is unsure.
* verification: with the vision model, the chosen target (fast: top 4 in one call; agent: any final target
  it did not already inspect with `look_at`) is checked on its saved crops. Fast-path rejections go to the
  agent with a note; agent rejections are returned to it (up to 2 times). `answer.path` / `verified` /
  `fast_reason` show what happened; `--no_fast` / `--no_verify` turn either off. Without
  `ZHIPUAI_API_KEY` the goal server still answers fast-path instructions.

GLM only picks object keys; `navigation/planner.py` computes the geometry: Dijkstra over the
traversable place graph (space the robot already drove through) to the place from which the target was
seen (ABot-N0), final heading facing the object. `plan.partial` = target area not connected to the
robot's place (odometry gap). One GLM request runs at a time; a query while busy gets `{"error": "busy"}`.

Across floors / buildings: edges cost walk ×1, stairs ×3 + 5 m, elevator +25 m (`allow_stairs:=false` /
`allow_elevator:=false` remove them). The goal place must be on the target's floor (or outdoors). The plan
is cut into `segments` (walk / stairs / elevator / ramp, each with `instruction`, e.g. "take the elevator
up from floor 1 to floor 2 (building0)"). The goal server merges consecutive walking segments into legs,
publishes the current leg on `/scene_memory/route_leg`, sends only walking legs to Nav2 (one map per floor),
and continues after a stairs / elevator leg when something publishes `std_msgs/Empty` on
`/scene_memory/next_leg`. GLM tools report `building` / `floor` per object, buildings and floors in
`region_information`, `objects_on_floor`, and the route instructions; "二楼 / 楼下 / 另一栋楼 / 外面" map to
these relative levels.

Tune the mapper without the GPU: record `/detector/detections_json` + `/equirect_vio/odometry` during a run,
then rebuild in seconds with any `topo_map_node` option:

```bash
ros2 bag record -o /tmp/det_bag /detector/detections_json /equirect_vio/odometry
python3 -m scene_memory.mapping.remap_bag /tmp/det_bag --scene_dir /tmp/remap --miss_range 4
# LIO + lidar depth
ros2 bag record -o /tmp/det_bag /detector/detections_json /lio/odom /lio/cloud_world
python3 -m scene_memory.mapping.remap_bag /tmp/det_bag --scene_dir /tmp/remap \
    --pose_source lio --odom_topic /lio/odom --cloud_topic /lio/cloud_world
```
