#!/usr/bin/env bash
# MID-360S (one, or front + rear) + Insta360 X4 Air calibration entry.
# The number of lidars in config/rig.yaml selects the single- or dual-lidar workflow.
# Livox driver and SDK are under drivers/. Build them with ./scripts/build_driver.sh.
set -euo pipefail

readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Capture data lives under the workspace, not in src/ (keeps the package small for git).
readonly WS_ROOT="$(cd -- "${PROJECT_DIR}/../.." && pwd)"
readonly DATA_ROOT="${CALIB_DATA_ROOT:-${WS_ROOT}/data/calib_capture}"
readonly LIDAR_BAGS_DIR="${DATA_ROOT}/lidar/bags"
readonly LIDAR_CLOUDS_DIR="${DATA_ROOT}/lidar/clouds"
readonly CAM_BAGS_DIR="${DATA_ROOT}/camera/bags"
readonly CAM_CALIB_DIR="${DATA_ROOT}/camera/runs"
readonly INSTA360_STILLS_DIR="${DATA_ROOT}/camera/stills"
# Driver and SDK copied under drivers/. install/ is produced by scripts/build_driver.sh.
readonly LIVOX_WS="${LIVOX_WS:-${PROJECT_DIR}}"
readonly RUN_DIR="${PROJECT_DIR}/.run"
readonly PID_FILE="${RUN_DIR}/livox.pid"
readonly EXTRINSIC_PID_FILE="${RUN_DIR}/extrinsic.pid"
readonly CAMERA_PID_FILE="${RUN_DIR}/insta360.pid"
# Local Media SDK build (x86-64, ./scripts/build_insta360.sh), else the workspace package
# src/sensors/insta360_views (colcon), which stitches with CUDA on Jetson.
if [[ -x "${PROJECT_DIR}/build/insta360_views/insta360_views" ]]; then
  readonly CAMERA_BIN="${PROJECT_DIR}/build/insta360_views/insta360_views"
else
  readonly CAMERA_BIN="${PROJECT_DIR}/../../install/insta360_views/lib/insta360_views/insta360_views"
fi
readonly MEDIA_SDK_DIR="${INSTA360_MEDIA_SDK_ROOT:-${WS_ROOT}/insta360_driver/insta360_media_sdk}"
readonly CONFIG_JSON="${PROJECT_DIR}/config/MID360s_dual.json"
# The workspace .venv (OpenVINO) has no PyYAML, numpy, or rclpy.
readonly PYTHON="${CALIB_PYTHON:-/usr/bin/python3}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
say() { printf '[calib] %s\n' "$*"; }

# True when config/rig.yaml lists one lidar.
is_single() {
  [[ "$("${PYTHON}" -c "import yaml; print(len(yaml.safe_load(open('${PROJECT_DIR}/config/rig.yaml'))['lidars']))")" == 1 ]]
}

resolve_ros() {
  if [[ -n "${ROS_SETUP:-}" && -f "${ROS_SETUP}" ]]; then
    printf '%s' "${ROS_SETUP}"
    return 0
  fi
  if [[ -f /opt/ros/humble/setup.bash ]]; then
    printf '%s' /opt/ros/humble/setup.bash
    return 0
  fi
  if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    printf '%s' /opt/ros/jazzy/setup.bash
    return 0
  fi
  return 1
}

source_ros() {
  local setup
  setup="$(resolve_ros)" || die "ROS 2 setup.bash not found"
  set +u
  # shellcheck disable=SC1090
  source "${setup}"
  if [[ -f "${LIVOX_WS}/install/setup.bash" ]]; then
    # shellcheck disable=SC1090
    source "${LIVOX_WS}/install/setup.bash"
  fi
  set -u
  export LD_LIBRARY_PATH="${PROJECT_DIR}/drivers/livox_lidar_sdk/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

lidar_ips() {
  "${PYTHON}" - "${PROJECT_DIR}/config/rig.yaml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for lidar in cfg["lidars"]:
    print(lidar["ip"], lidar["id"])
PY
}

# Position-named topics, /livox/<id>/points and /livox/<id>/imu.
topics() {
  (cd "${PROJECT_DIR}/tools" && "${PYTHON}" -c '
from pathlib import Path
import extrinsic_io
rig = extrinsic_io.load_rig(Path(".."))
for role in extrinsic_io.roles(rig):
    print(rig[role]["points"])
    print(rig[role]["imu"])
')
}

# The driver names topics by IP. Remap them to the position names.
driver_remaps() {
  (cd "${PROJECT_DIR}/tools" && "${PYTHON}" -c '
from pathlib import Path
import extrinsic_io
rig = extrinsic_io.load_rig(Path(".."))
for role in extrinsic_io.roles(rig):
    l = rig[role]
    print("-r")
    print(l["driver_points"] + ":=" + l["points"])
    print("-r")
    print(l["driver_imu"] + ":=" + l["imu"])
')
}

cmd_status() {
  say "host $("${PYTHON}" -c "import yaml; print(yaml.safe_load(open('${PROJECT_DIR}/config/rig.yaml'))['rig']['host_ip'])")"
  while read -r ip id; do
    if ping -c1 -W1 "${ip}" >/dev/null 2>&1; then
      say "lidar ${id} ${ip} ping ok"
    else
      say "lidar ${id} ${ip} ping failed"
    fi
  done < <(lidar_ips)
  "${PYTHON}" "${PROJECT_DIR}/tools/check_insta360_usb.py" || true
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    say "livox driver pid $(cat "${PID_FILE}")"
  else
    say "livox driver not started by this script"
  fi
  if camera_running; then
    say "insta360 views pid $(cat "${CAMERA_PID_FILE}")"
  else
    say "insta360 views not running"
  fi
}

stop_pid_file() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "${file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" || true
    say "stopped pid ${pid}"
  fi
  rm -f "${file}"
}

cmd_apply() {
  if is_single; then
    [[ -s "${PROJECT_DIR}/results/lidar_extrinsic.json" ]] || { say "no lidar extrinsic yet: run ./calibrate.sh level <scene>; apply not started"; return 0; }
    grep -q '"T_level_rear"' "${PROJECT_DIR}/results/lidar_extrinsic.json" && \
      { say "results/lidar_extrinsic.json is from the dual rig: run ./calibrate.sh level <scene>; apply not started"; return 0; }
  fi
  [[ -s "${PROJECT_DIR}/results/lidar_extrinsic.json" ]] || die "missing extrinsic; run ./calibrate.sh initial"
  if pgrep -f "${PROJECT_DIR}/tools/apply_lidar_extrinsic.py" >/dev/null 2>&1; then
    say "extrinsic apply already running"
    return 0
  fi
  source_ros
  mkdir -p "${RUN_DIR}"
  nohup "${PYTHON}" "${PROJECT_DIR}/tools/apply_lidar_extrinsic.py" \
    >"${RUN_DIR}/extrinsic.log" 2>&1 &
  echo $! >"${EXTRINSIC_PID_FILE}"
  say "applying the leveling transform(s) from results/lidar_extrinsic.json"
  say "fixed frame: lidar_level (gravity-aligned, origin at the front lidar)"
  if is_single; then
    say "topics: /extrinsic/lidar_fused (front lidar leveled)  /extrinsic/lidar_fused_front (frame front_lidar)"
  else
    say "topics: /extrinsic/front_lidar  /extrinsic/rear_lidar  /extrinsic/lidar_fused  /extrinsic/lidar_fused_front (frame front_lidar)"
  fi
}

camera_running() {
  [[ -f "${CAMERA_PID_FILE}" ]] && kill -0 "$(cat "${CAMERA_PID_FILE}")" 2>/dev/null
}

# Insta360 front/left/rear/right views. A missing build or camera does not stop the lidars.
cmd_camera() {
  if camera_running; then
    say "insta360 views already running pid $(cat "${CAMERA_PID_FILE}")"
    return 0
  fi
  if [[ ! -x "${CAMERA_BIN}" ]]; then
    say "insta360 views not built; run ./scripts/build_insta360.sh (x86-64) or colcon build insta360_views (Jetson). Lidars continue without the camera."
    return 0
  fi
  "${PYTHON}" "${PROJECT_DIR}/tools/check_insta360_usb.py" || say "the node waits and connects when the camera is plugged in"
  "${PYTHON}" "${PROJECT_DIR}/tools/write_insta360_intrinsics.py"
  "${PYTHON}" "${PROJECT_DIR}/tools/export_results.py" >/dev/null
  source_ros
  mkdir -p "${RUN_DIR}"
  nohup "${CAMERA_BIN}" --ros-args \
    --params-file "${PROJECT_DIR}/config/insta360_views.yaml" \
    -p model_dir:="${MEDIA_SDK_DIR}/bin/models" \
    >"${RUN_DIR}/insta360.log" 2>&1 &
  echo $! >"${CAMERA_PID_FILE}"
  say "insta360 views pid $(cat "${CAMERA_PID_FILE}") log ${RUN_DIR}/insta360.log"
  say "topics: /insta360/{front,left,rear,right}/image/compressed  /insta360/equirect/image/compressed"
}

# True when every lidar in rig.yaml answers ping.
lidars_reachable() {
  local ip id
  while read -r ip id; do
    ping -c1 -W1 "${ip}" >/dev/null 2>&1 || return 1
  done < <(lidar_ips)
  return 0
}

# Camera-only bringup for benches without MID-360S (DA360 relative depth, no lidar scale).
cmd_bringup_camera() {
  say "camera-only bringup (no Livox driver)"
  source_ros
  mkdir -p "${RUN_DIR}"
  cmd_camera
  if camera_running; then
    say "DA360 can use /insta360/equirect/image/compressed (relative depth; no /extrinsic/lidar_fused)"
  else
    die "camera did not start; check USB Android mode and .run/insta360.log"
  fi
}

cmd_bringup() {
  local background="${1:-0}"
  local camera_only="${2:-${CAMERA_ONLY:-0}}"
  local with_camera="${3:-1}"

  if [[ "${camera_only}" == "1" || "${camera_only}" == "camera" ]]; then
    cmd_bringup_camera
    return 0
  fi

  if ! lidars_reachable; then
    [[ "${with_camera}" == "1" ]] || die "MID-360S did not answer ping; check config/rig.yaml and the network"
    say "warning: one or more MID-360S did not answer ping; starting camera only"
    say "hint: full stack later with lidars up, or force camera-only: ./calibrate.sh bringup-camera"
    cmd_bringup_camera
    return 0
  fi

  [[ -x "${LIVOX_WS}/install/livox_ros_driver2/lib/livox_ros_driver2/livox_ros_driver2_node" ]] || \
    die "local livox driver is not built. Run ./scripts/build_driver.sh"
  "${PYTHON}" "${PROJECT_DIR}/tools/write_livox_config.py"
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    say "already running pid $(cat "${PID_FILE}")"
    cmd_apply
    [[ "${with_camera}" == "1" ]] && cmd_camera
    return 0
  fi
  local -a remaps=()
  while read -r arg; do
    remaps+=("${arg}")
  done < <(driver_remaps)
  source_ros
  mkdir -p "${RUN_DIR}"
  local freq
  freq="$("${PYTHON}" -c "import yaml; print(yaml.safe_load(open('${PROJECT_DIR}/config/rig.yaml'))['livox']['publish_freq'])")"
  say "starting $(lidar_ips | wc -l) lidar(s), multi_topic, PointCloud2, ${freq} Hz"
  say "topics: $(topics | tr '\n' ' ')"
  say "config ${CONFIG_JSON}"
  cmd_apply
  if [[ "${with_camera}" == "1" ]]; then
    cmd_camera
  else
    say "camera not started (bringup-lidar)"
  fi
  if [[ "${background}" == "1" ]]; then
    # shellcheck disable=SC2091
    nohup ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \
      -p xfer_format:=0 \
      -p multi_topic:=1 \
      -p data_src:=0 \
      -p publish_freq:="${freq}" \
      -p frame_id:=livox_frame \
      -p user_config_path:="${CONFIG_JSON}" \
      -p cmdline_input_bd_code:=livox0000000001 \
      "${remaps[@]}" \
      >"${RUN_DIR}/livox.log" 2>&1 &
    echo $! >"${PID_FILE}"
    say "background pid $(cat "${PID_FILE}") log ${RUN_DIR}/livox.log"
  else
    trap 'stop_pid_file "${EXTRINSIC_PID_FILE}"; stop_pid_file "${CAMERA_PID_FILE}"' EXIT INT TERM
    ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \
      -p xfer_format:=0 \
      -p multi_topic:=1 \
      -p data_src:=0 \
      -p publish_freq:="${freq}" \
      -p frame_id:=livox_frame \
      -p user_config_path:="${CONFIG_JSON}" \
      -p cmdline_input_bd_code:=livox0000000001 \
      "${remaps[@]}"
  fi
}

cmd_stop() {
  stop_pid_file "${EXTRINSIC_PID_FILE}"
  stop_pid_file "${CAMERA_PID_FILE}"
  if [[ -f "${PID_FILE}" ]]; then
    stop_pid_file "${PID_FILE}"
  else
    say "no livox pid file"
  fi
}

cmd_record() {
  local scene="${1:-}"
  [[ -n "${scene}" ]] || die "record needs a scene name"
  source_ros
  local seconds out
  seconds="$("${PYTHON}" -c "import yaml; print(int(yaml.safe_load(open('${PROJECT_DIR}/config/rig.yaml'))['recording']['seconds']))")"
  out="${LIDAR_BAGS_DIR}/${scene}"
  [[ ! -e "${out}" ]] || die "refusing to overwrite ${out}"
  mkdir -p "${LIDAR_BAGS_DIR}"
  local -a topic_args=()
  while read -r topic; do
    topic_args+=("${topic}")
  done < <(topics)
  say "recording ${seconds}s -> ${out}"
  printf '  %s\n' "${topic_args[@]}"
  timeout --signal=INT --kill-after=10 "$((seconds + 2))" \
    ros2 bag record --storage mcap -o "${out}" "${topic_args[@]}" || true
  [[ -d "${out}" ]] || die "bag was not created"
  "${PYTHON}" - "${out}/capture_manifest.json" "${scene}" "${PROJECT_DIR}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import yaml
root = Path(sys.argv[3])
cfg = yaml.safe_load((root / "config" / "rig.yaml").read_text())
Path(sys.argv[1]).write_text(json.dumps({
    "scene": sys.argv[2],
    "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "rig_id": cfg["rig"]["id"],
    "lidars": cfg["lidars"],
    "insta360": "capture a still into this scene directory with ./calibrate.sh insta360 " + sys.argv[2],
}, indent=2) + "\n")
PY
  say "lidar bag ready. Keep the rig still and run: ./calibrate.sh insta360 ${scene}"
}

cmd_view() {
  say "view is the same as apply: both lidars in the gravity-leveled frame"
  cmd_apply
}

cmd_initial() {
  is_single && die "single lidar: there is no lidar-lidar seed. Run ./calibrate.sh level <scene>"
  "${PYTHON}" "${PROJECT_DIR}/tools/build_cad_extrinsic.py"
  "${PYTHON}" "${PROJECT_DIR}/tools/export_results.py"
}

cmd_capture() {
  local scene="${1:-live}"
  source_ros
  say "capturing 10 frames per lidar into ${LIDAR_CLOUDS_DIR}/${scene}/"
  "${PYTHON}" "${PROJECT_DIR}/tools/capture_lidar_pair.py" --scene "${scene}"
}

# Single lidar: 5 static frames + 2 s of IMU, then gravity leveling -> results/lidar_extrinsic.json.
cmd_level() {
  local scene="${1:-}"
  [[ -n "${scene}" ]] || die "level needs a scene name"
  is_single || die "config/rig.yaml has two lidars; use ./calibrate.sh refine-auto"
  local front dir
  front="$("${PYTHON}" -c "import yaml; print(yaml.safe_load(open('${PROJECT_DIR}/config/rig.yaml'))['lidars'][0]['id'])")"
  dir="${LIDAR_CLOUDS_DIR}/${scene}"
  if [[ ! -s "${dir}/${front}.pcd" || ! -s "${dir}/${front}.imu.json" ]]; then
    source_ros
    say "keep the rig still: capturing 5 frames and 2 s of IMU into ${dir}"
    "${PYTHON}" "${PROJECT_DIR}/tools/capture_lidar_pair.py" \
      --scene "${scene}" --frames 5 --voxel 0 --format pcd
  else
    say "using ${dir}"
  fi
  "${PYTHON}" "${PROJECT_DIR}/tools/lidar_level_single.py" --scene "${scene}"
  "${PYTHON}" "${PROJECT_DIR}/tools/export_results.py" >/dev/null
  say "restart apply to publish it: kill \"\$(cat .run/extrinsic.pid)\"; ./calibrate.sh apply"
}

cmd_refine_auto() {
  local scene="${1:-live}"
  is_single && die "single lidar: use ./calibrate.sh level <scene> (no lidar-lidar registration)"
  local front rear dir
  front="$("${PYTHON}" -c "import yaml; c=yaml.safe_load(open('${PROJECT_DIR}/config/rig.yaml')); print(next(x['id'] for x in c['lidars'] if x['primary']))")"
  rear="$("${PYTHON}" -c "import yaml; c=yaml.safe_load(open('${PROJECT_DIR}/config/rig.yaml')); print(next(x['id'] for x in c['lidars'] if not x['primary']))")"
  dir="${LIDAR_CLOUDS_DIR}/${scene}"
  if [[ ! -s "${dir}/${front}.pcd" || ! -s "${dir}/${rear}.pcd" \
     || ! -s "${dir}/${front}.imu.json" || ! -s "${dir}/${rear}.imu.json" ]]; then
    source_ros
    say "capturing 1 static frame and 2 s of IMU from each lidar into ${dir}"
    "${PYTHON}" "${PROJECT_DIR}/tools/capture_lidar_pair.py" \
      --scene "${scene}" --frames 1 --voxel 0 --format pcd
  else
    say "using ${dir}"
  fi
  "${PYTHON}" "${PROJECT_DIR}/tools/lidar_pair_refine.py" --scene "${scene}"
  "${PYTHON}" "${PROJECT_DIR}/tools/export_results.py"
  say "wrote results/lidar_extrinsic.json"
  say "restart ./calibrate.sh apply to publish the new extrinsic"
}

cmd_refine_manual() {
  is_single && die "single lidar: there is no lidar pair to refine"
  [[ -s "${PROJECT_DIR}/results/lidar_extrinsic.json" ]] || die "missing seed; run ./calibrate.sh initial"
  source_ros
  say "keyboard refine. Foxglove fixed frame lidar_level, topic /calib/lidar_fused"
  local status=0
  "${PYTHON}" "${PROJECT_DIR}/tools/lidar_pair_manual.py" || status=$?
  "${PYTHON}" "${PROJECT_DIR}/tools/export_results.py" >/dev/null
  return "${status}"
}

cmd_solve_lidar() {
  local scene="${1:-}"
  [[ -n "${scene}" ]] || die "solve-lidar needs a scene name. It runs the same refine as refine-auto."
  cmd_refine_auto "${scene}"
}

cmd_insta360() {
  local scene="${1:-still}"
  camera_running && die "insta360 views holds the camera; run ./calibrate.sh camera-stop first"
  "${PYTHON}" "${PROJECT_DIR}/tools/check_insta360_usb.py"
  local sdk="${INSTA360_SDK_ROOT:-${WS_ROOT}/insta360_driver/insta360_sdk}"
  [[ -d "${sdk}/include" && -d "${sdk}/lib" ]] || die "Insta360 Camera SDK is not in ${sdk}. Put include/ and lib/ there (see <ws>/insta360_driver/README.md)."
  local out="${INSTA360_STILLS_DIR}/${scene}"
  mkdir -p "${out}" "${PROJECT_DIR}/build"
  local bin="${PROJECT_DIR}/build/insta360_capture"
  c++ -std=c++17 -O2 \
    -I"${sdk}/include" \
    "${PROJECT_DIR}/tools/insta360_capture.cpp" \
    -L"${sdk}/lib" -lCameraSDK \
    -Wl,-rpath,"${sdk}/lib" \
    -o "${bin}"
  say "taking a still into ${out}"
  "${bin}" "${out}"
}

rig_value() {
  "${PYTHON}" - "${PROJECT_DIR}/config/rig.yaml" "$1" <<'PY'
import sys, yaml
v = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for key in sys.argv[2].split("."):
    v = v[key]
print(v)
PY
}

cam_points_topic() {
  if is_single || [[ "$(rig_value camera_lidar.cloud)" == "front" ]]; then
    printf '%s' /livox/front_lidar/points
  else
    printf '%s' /extrinsic/lidar_fused_front
  fi
}

readonly EQUIRECT_TOPIC=/insta360/equirect/image/compressed

wait_topic() {
  local topic="$1" field="$2"
  timeout 8 ros2 topic echo --once "${topic}" --field "${field}" >/dev/null 2>&1
}

# Static capture of the panorama, the four views and the lidars for camera-lidar calibration.
cmd_cam_record() {
  local scene="${1:-}"
  [[ -n "${scene}" ]] || die "cam-record needs a scene name"
  local out="${CAM_BAGS_DIR}/${scene}"
  [[ ! -e "${out}" ]] || die "refusing to overwrite ${out}"
  camera_running || die "insta360 views not running; start ./calibrate.sh bringup or ./calibrate.sh camera"
  source_ros
  local points seconds
  points="$(cam_points_topic)"
  seconds="$(rig_value camera_lidar.record_seconds)"
  wait_topic "${EQUIRECT_TOPIC}" format || die "no panorama on ${EQUIRECT_TOPIC}; check .run/insta360.log and Android mode on the camera"
  wait_topic "${points}" header.frame_id || die "no cloud on ${points}; start the lidar (./calibrate.sh bringup) or restart ./calibrate.sh apply so it publishes /extrinsic/lidar_fused_front"
  local -a topic_args=("${EQUIRECT_TOPIC}" "${points}")
  local view
  for view in front left rear right; do
    topic_args+=("/insta360/${view}/image/compressed" "/insta360/${view}/camera_info")
  done
  while read -r topic; do
    topic_args+=("${topic}")
  done < <(topics)
  mkdir -p "${CAM_BAGS_DIR}"
  say "keep the rig still for ${seconds} s. Recording -> ${out}"
  printf '  %s\n' "${topic_args[@]}"
  timeout --signal=INT --kill-after=10 "$((seconds + 2))" \
    ros2 bag record --storage mcap -o "${out}" "${topic_args[@]}" </dev/null >/dev/null || true
  [[ -f "${out}/metadata.yaml" ]] || die "bag was not created"
  "${PYTHON}" - "${out}/metadata.yaml" "${EQUIRECT_TOPIC}" "${points}" <<'PY'
import sys, yaml
info = yaml.safe_load(open(sys.argv[1]))["rosbag2_bagfile_information"]
counts = {t["topic_metadata"]["name"]: t["message_count"] for t in info["topics_with_message_count"]}
for topic in sys.argv[2:]:
    n = counts.get(topic, 0)
    print(f"  {topic}: {n} messages")
    if n == 0:
        raise SystemExit(f"ERROR: {topic} has no messages")
PY
  say "recorded ${scene}. Move the rig to a new place and pose, then record the next scene (3-5 in total)"
}

SOLVER_SETUP="${PROJECT_DIR}/opt/solver_ws/install/setup.bash"

cam_docker_gui_args() {
  [[ -f "${SOLVER_SETUP}" ]] && return
  CAM_DOCKER_ARGS=(run --rm --network host --ipc host -e HOME=/tmp
    -e "HOST_UID=$(id -u)" -e "HOST_GID=$(id -g)"
    -v "${DATA_ROOT}:/data:rw")
  if [[ -n "${DISPLAY:-}" ]]; then
    CAM_DOCKER_ARGS+=(-e "DISPLAY=${DISPLAY}" -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:ro)
    if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
      CAM_DOCKER_ARGS+=(-e XAUTHORITY=/tmp/host.Xauthority -v "${XAUTHORITY}:/tmp/host.Xauthority:ro")
    fi
  fi
  if docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia'; then
    CAM_DOCKER_ARGS+=(--gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display)
  else
    CAM_DOCKER_ARGS+=(-e LIBGL_ALWAYS_SOFTWARE=1)
  fi
  CAM_DOCKER_ARGS+=("$(rig_value camera_lidar.solver_image)")
}

# Run a command in the solver image and hand the run directory back to the host user.
# With a native build (scripts/build_solver_native.sh) run it directly; /data/ maps to DATA_ROOT.
cam_docker() {
  local run_rel="$1"
  shift
  if [[ -f "${SOLVER_SETUP}" ]]; then
    (
      set +u
      # shellcheck disable=SC1090
      source "${SOLVER_SETUP}"
      export LD_LIBRARY_PATH="${PROJECT_DIR}/opt/lib:${LD_LIBRARY_PATH:-}"
      "${@/#\/data\//${DATA_ROOT}/}"
    )
    return
  fi
  docker "${CAM_DOCKER_ARGS[@]}" bash -c '
    status=0
    "$@" || status=$?
    chown -R "${HOST_UID}:${HOST_GID}" "/data/'"${run_rel}"'" 2>/dev/null || true
    exit "${status}"
  ' solver "$@"
}

cam_run_dir() {
  local name="${1:-latest}"
  local run="${CAM_CALIB_DIR}/${name}"
  [[ -d "${run}/preprocessed" ]] || die "no preprocessed run at ${run}; run ./calibrate.sh cam-calib first"
  (cd "${run}" && pwd -P)
}

cmd_cam_calib() {
  local -a scenes=("$@")
  if ((${#scenes[@]} == 0)); then
    local d
    for d in "${CAM_BAGS_DIR}"/*/; do
      [[ -f "${d}metadata.yaml" ]] && scenes+=("$(basename "${d}")")
    done
  fi
  ((${#scenes[@]} > 0)) || die "no scenes in ${CAM_BAGS_DIR}; record with ./calibrate.sh cam-record <scene>"
  ((${#scenes[@]} >= 3)) || say "warning: ${#scenes[@]} scene(s); 3-5 scenes at different places give a steadier result"
  local name run run_rel points s
  name="$(date -u +%Y%m%dT%H%M%SZ)"
  run="${CAM_CALIB_DIR}/${name}"
  run_rel="camera/runs/${name}"
  points="$(cam_points_topic)"
  mkdir -p "${run}/bags"
  for s in "${scenes[@]}"; do
    [[ -f "${CAM_BAGS_DIR}/${s}/metadata.yaml" ]] || die "missing bag ${CAM_BAGS_DIR}/${s}"
    ln -s "../../../bags/${s}" "${run}/bags/${s}"
  done
  "${PYTHON}" - "${run}/run.json" "${points}" "${PROJECT_DIR}/config/insta360_views.yaml" "${scenes[@]}" <<'PY'
import json, sys, yaml
params = yaml.safe_load(open(sys.argv[3]))["insta360_views"]["ros__parameters"]
json.dump({"points_topic": sys.argv[2], "stitch_type": params["stitch_type"],
           "equirect_width": params["equirect_width"], "scenes": sys.argv[4:]},
          open(sys.argv[1], "w"), indent=2)
PY
  say "run ${run} with ${#scenes[@]} scene(s): ${scenes[*]}"
  say "preprocess (panorama ${EQUIRECT_TOPIC}, cloud ${points})"
  cam_docker_gui_args
  cam_docker "${run_rel}" ros2 run direct_visual_lidar_calibration preprocess \
    "/data/${run_rel}/bags" "/data/${run_rel}/preprocessed" \
    --image_topic "${EQUIRECT_TOPIC}" --points_topic "${points}" \
    --intensity_channel intensity --camera_model equirectangular
  [[ -s "${run}/preprocessed/calib.json" ]] || die "preprocess did not write calib.json"
  ln -sfn "${name}" "${CAM_CALIB_DIR}/latest"
  "${PYTHON}" "${PROJECT_DIR}/tools/insta360_lidar_calib.py" initial "${run}"
  cmd_cam_solve "${name}"
}

cmd_cam_solve() {
  local run run_rel
  run="$(cam_run_dir "${1:-latest}")"
  run_rel="camera/runs/$(basename "${run}")"
  say "NID refinement in ${run}"
  cam_docker_gui_args
  cam_docker "${run_rel}" ros2 run direct_visual_lidar_calibration calibrate "/data/${run_rel}/preprocessed" \
    --background --auto_quit --nid_bins "$(rig_value camera_lidar.nid_bins)"
  source_ros
  "${PYTHON}" "${PROJECT_DIR}/tools/insta360_lidar_calib.py" import "${run}"
  "${PYTHON}" "${PROJECT_DIR}/tools/export_results.py"
  say "check the overlays in ${run}/overlay before using results/insta360_extrinsic.json"
}

cmd_cam_initial_manual() {
  local run run_rel
  run="$(cam_run_dir "${1:-latest}")"
  run_rel="camera/runs/$(basename "${run}")"
  [[ -n "${DISPLAY:-}" ]] || die "initial_guess_manual needs a display"
  say "right-click a point in the cloud and the same spot in the image, add 4+ pairs, Estimate, Save, close"
  cam_docker_gui_args
  cam_docker "${run_rel}" ros2 run direct_visual_lidar_calibration initial_guess_manual "/data/${run_rel}/preprocessed"
  say "then run ./calibrate.sh cam-solve"
}

# Copy results/export into the folders semantic mapping (data/calib) and DA360 read; the files
# already there are moved to <dest>/archive/<time>/ first. New sessions snapshot data/calib.
cmd_publish() {
  local ws stamp dest f
  ws="$(cd "${PROJECT_DIR}/../.." && pwd)"
  stamp="$(date +%Y%m%d_%H%M%S)"
  "${PYTHON}" "${PROJECT_DIR}/tools/export_results.py" >/dev/null
  for dest in "${ws}/data/calib" "${ws}/src/depth/da360_depth/calib"; do
    mkdir -p "${dest}/archive/${stamp}"
    for f in "${dest}"/*.json; do
      [[ -e "${f}" ]] && mv "${f}" "${dest}/archive/${stamp}/"
    done
    for f in "${PROJECT_DIR}"/results/export/*.json; do
      cp -a "${f}" "${dest}/"
    done
    cp -a "${PROJECT_DIR}/config/rig.yaml" "${dest}/rig.yaml"
    (cd "${ws}/src/depth/da360_depth" && "${PYTHON}" -c "
from insta_live.calib_io import load_rig_calibration, write_merged_summary
c = load_rig_calibration('${dest}')
write_merged_summary(c, '${dest}/rig_calib.json')
print(c.summary())")
    say "published to ${dest} (previous files in archive/${stamp})"
  done
}

usage() {
  cat <<EOF
MID-360S + Insta360 X4 Air. One lidar in config/rig.yaml = single-lidar workflow:
  bringup -> level <scene> -> cam-record <scene> (x3-5) -> cam-calib
Two lidars = dual workflow: bringup -> initial -> refine-auto <scene> -> cam-record -> cam-calib

  ./calibrate.sh status
  ./calibrate.sh bringup              # lidars + apply + camera; if lidars unreachable, camera only
  ./calibrate.sh bringup-bg
  ./calibrate.sh bringup-lidar        # lidars + apply in the background, camera untouched
  ./calibrate.sh level <scene>        # single lidar: IMU gravity leveling -> results/lidar_extrinsic.json
  ./calibrate.sh bringup-camera       # camera only (no MID-360S); for DA360 relative-depth tests
  ./calibrate.sh apply                # /extrinsic/lidar_fused in the gravity-leveled frame lidar_level
  ./calibrate.sh camera               # Insta360 /insta360/{front,left,rear,right}/image/compressed
  ./calibrate.sh camera-stop
  ./calibrate.sh intrinsics           # four view intrinsics -> results/insta360_intrinsics.json
  ./calibrate.sh cam-record <scene>   # -> <ws>/data/calib_capture/camera/bags/<scene>
  ./calibrate.sh cam-calib [scenes]   # preprocess, initial guess, NID -> results/insta360_extrinsic.json
  ./calibrate.sh cam-solve [run]      # rerun NID on calib_capture/camera/runs/<run> (default latest)
  ./calibrate.sh cam-initial-manual [run]  # GUI point pairs for the initial guess, then cam-solve
  ./calibrate.sh stop
  ./calibrate.sh record <scene>       # 15 s static bag -> calib_capture/lidar/bags/<scene>
  ./calibrate.sh initial              # installation seed -> results/lidar_extrinsic*.json
  ./calibrate.sh view                 # same as apply
  ./calibrate.sh capture <scene>      # 10 static frames -> calib_capture/lidar/clouds/<scene>/<id>.ply
  ./calibrate.sh refine-auto [scene]  # 1 PCD + IMU gravity each, level, ICP from yaw 180 + [-247.5,0,0] mm
  ./calibrate.sh refine-manual        # keyboard q/a w/s e/d r/f t/g y/h, z saves
  ./calibrate.sh solve-lidar <scene>  # same as refine-auto
  ./calibrate.sh insta360 <scene>     # USB check; still capture if INSTA360_SDK_ROOT is set
  ./calibrate.sh export [--scene S]   # copy lidar + insta360 results and one projection scene into results/export/
  ./calibrate.sh publish              # results/export -> data/calib (semantic mapping) and DA360 calib, old files archived

Lidars: front_lidar (primary) and, on the dual rig, rear_lidar. IPs are set only in config/rig.yaml
(config/rig_dual_mid360s.yaml keeps the dual layout). Host IP 192.168.1.5 (enp129s0).
Insta360 X4 Air: USB 3.0, choose Android mode on the camera. Build the views once with ./scripts/build_insta360.sh
(x86-64, Media SDK) or, on Jetson, colcon build --base-paths src/sensors --packages-select insta360_views.
EOF
}

main() {
  local cmd="${1:-help}"
  shift || true
  case "${cmd}" in
    status) cmd_status ;;
    bringup) cmd_bringup 0 "${1:-}" ;;
    bringup-bg) cmd_bringup 1 "${1:-}" ;;
    bringup-camera) cmd_bringup_camera ;;
    bringup-lidar) cmd_bringup 1 0 0 ;;
    level) cmd_level "${1:-}" ;;
    apply) cmd_apply ;;
    camera) cmd_camera ;;
    camera-stop) stop_pid_file "${CAMERA_PID_FILE}" ;;
    intrinsics) "${PYTHON}" "${PROJECT_DIR}/tools/write_insta360_intrinsics.py" && "${PYTHON}" "${PROJECT_DIR}/tools/export_results.py" ;;
    cam-record) cmd_cam_record "${1:-}" ;;
    cam-calib) cmd_cam_calib "$@" ;;
    cam-solve) cmd_cam_solve "${1:-latest}" ;;
    cam-initial-manual) cmd_cam_initial_manual "${1:-latest}" ;;
    stop) cmd_stop ;;
    record) cmd_record "${1:-}" ;;
    initial) cmd_initial ;;
    view) cmd_view ;;
    capture) cmd_capture "${1:-live}" ;;
    refine-auto) cmd_refine_auto "${1:-live}" ;;
    refine-manual) cmd_refine_manual ;;
    solve-lidar) cmd_solve_lidar "${1:-}" ;;
    insta360) cmd_insta360 "${1:-still}" ;;
    export) "${PYTHON}" "${PROJECT_DIR}/tools/export_results.py" "$@" ;;
    publish) cmd_publish ;;
    help|-h|--help) usage ;;
    *) usage >&2; die "unknown command: ${cmd}" ;;
  esac
}

main "$@"
