#!/usr/bin/env bash
# One entry point for the STVLN system (camera + MID-360S/Super-LIO + detector + 4D topo mapper + goal server).
#
#   src/scripts/stvln.sh <command> [args]          extra key:=value args go straight to ros2 launch
#
# Live (on the robot)
#   live [topo|hydra_fake|hydra] [--bg] [--glm] [k:=v ...]   full pipeline -> data/scene4d/<time>
#        odom:=qinav  map in QiNav's localized map frame (QiNav 3D stack running) and send goals to QiNav
#   qinav_bridge [topo|hydra_fake] [--bg] [--maps_dir DIR] [k:=v ...]   follow QiNav 2D/3D mapping: its
#        start (api 1031) starts `live odom:=qinav_mapping`, each map save writes <map>_topo_graph.json,
#        <map>_objects_4d.json, ... next to QiNav's map files (~/.ros/nav2d/maps or ~/.ros/nav3d/maps, QiNav
#        map frame), mapping end stops it (final graph written again)
#   map <name> [--bg] [semantic:=false] [k:=v ...]            own point-cloud + semantic mapping without QiNav
#        (src/mapping): Super-LIO + linefit + DLIO map accumulation + pipeline odom:=lio; refuses while QiNav runs
#   map_save <name> [--semantic_only]                         while mapping: data/maps/<name>/<name>_map/_ground/
#        _nonground/_traversable.pcd + semantic files (--semantic_only after `stop map`: final semantic export)
#   map_offline <session> <name> [--glm] [--no_rooms] [--no_views] [k:=v ...]   the same mapping on a recording
#        (e.g. on the PC): replays <session>_proc (derive) + the raw bag's lidar, map_save, stop map, then rooms
#        from the lidar map (pcd_rooms), panorama view layer (view_layer), region labels, map_save --semantic_only
#   map_push <name> [user@host] [remote dir]                  rsync data/maps/<name>/ to the robot's
#        <remote dir>/data/maps/<name>/ (.56, ~/stvln)
#   map_migrate <name>|--all                                  move older flat data/maps/<name>_* into data/maps/<name>/
#   nav <map> [--bg] [--glm|--vlm] [control:=true] [k:=v ...] own 3D nav stack without QiNav (src/planning):
#        Super-LIO + QiNav localization on data/maps/<map>/<map>_map.pcd (else ~/.ros/nav3d/maps) + auto initial pose
#        + DDDMR/CMU planners -> /stvln/cmd_vel + the map's semantic layer (/topo/markers) and VLN goal server;
#        goals on /clicked_point or /scene_memory/query; control:=true drives the Go2-W (stands up by itself
#        on a goal; auto_prepare:=false: posture 1 on /stvln/unitree/posture_cmd); local obstacles from the
#        camera freespace in a 30 m window (local_obstacles:=linefit: QiNav's linefit); refuses while QiNav's
#        nav/control stack runs
#   camera [--bg] [-p k:=v ...]                               Insta360 only: panorama /insta360/equirect/image/compressed
#        + views (nav starts it itself unless camera:=false; refuses while another insta360_views runs)
#   janus [--instruction="..."] [--bg] [--execute] [--brain=URL] [k:=v ...]
#        front JPEG → JanusVLN brain on .87 (:8088) → /janus/action; --execute also guarded /stvln/cmd_vel steps
#   janus_nav [--instruction="..."] [--bg] [--brain=URL] [control:=true] [k:=v ...]   map-free JanusVLN walking:
#        LIO + body cloud + camera + janus client; each action one 0.25 m / 15° step closed on /lio/base/odom,
#        MOVE_FORWARD vetoed by the lidar corridor guard; control:=true drives the Go2-W; no instruction: idle
#   janus_go "<instruction>" | stop                           new instruction for the running janus client
#        (/janus/instruction): aborts the current step, new brain episode; stop halts and idles
#   battery [--bg] [-p k:=v ...]                              Go2-W battery (rt/lowstate on eth0) -> /go2w/battery,
#        /go2w/diagnostics (WARN <= 30 %, ERROR <= 15 %); nav starts it itself unless battery:=false
#   model [--bg]                                              Go2-W model on /robot_description under base_link
#        (nav starts it itself); Foxglove (robot bridge 8766): URDF layer from topic /robot_description
#   lio [--bg] [k:=v ...]                                     MID-360S driver + Super-LIO + camera TF only
#        live / lio / record reuse a livox driver / super_lio_node that is already running (QiNav mapping or
#        localization); start_driver:=force / start_lio:=force start ours anyway
#   record [name] [--bg] [--no_camera_check] [k:=v ...]       raw session -> data/sessions/<name|time>
#        refuses to start unless the Insta360 is in Android mode (2e1a:0002); --bg waits for the fisheye stream
#        sensor data only (fisheye stream in raw/, camera IMU, MID-360S + IMU); raw_only:=false adds LIO/VIO/stitched
#   record_eval <name> [--no_equirect]                        mapper-input bag next to `live` -> data/sessions/<name>/topo_bag
#   foxglove [--bg] [port]                                    own foxglove bridge (default 8767, all topics)
#   check                                                     rates of the sensor / LIO / detector / map topics
#   query "<prompt>"                                          ask the goal server, print the answer JSON
#   plan [scene] [--targets tv,door | --all | --query "<prompt>"] [--qinav]   check goal routes offline
#                                                             (default scene data/scene4d/latest; PNGs + report)
#   stop [map|nav|live|lio|record|foxglove|...]               stop a --bg run (SIGINT to its process group)
#   status                                                    --bg runs and the newest scene dirs
# Offline
#   derive <session> [--every 2] [--no_lio] [--keep_lidar]    raw-only session -> <session>_proc (stitched
#                                                             equirect + Super-LIO re-run, real time) for replay
#   replay <session> [topo|hydra_fake|hydra] [k:=v ...]       re-run the pipeline on a recorded session
#   vln [session|scene] [--episodes yaml] [--glm] [--only ids]   offline VLN: goal server + simulated QiNav on the
#                                                             replayed map, instructions scored (SR / SPL, views);
#                                                             starts the battery monitor (robot domain, stays up)
#                                                             when on the Go2-W network and not running yet;
#                                                             Go2-W model on /robot_description; view it with
#                                                             ROS_DOMAIN_ID=78 stvln.sh foxglove 8768 (own domain)
#   eval <session|bag> [--gt file] [--mappers topo,hydra_fake] [topo_eval args]   localisation-error report
#   build [qinav_msgs|planning]                               colcon build of the workspace packages
#                                                             (qinav_msgs: QiNav's interfaces -> qinav_msgs/install;
#                                                             planning: src/planning global/local planner + Go2-W control,
#                                                             src/localization/qinav_localization, src/mapping DLIO)
#
# --bg runs detached (survives ssh logout): log in log/stvln_<cmd>.log, pid in log/stvln_<cmd>.pid.
# --glm exports ZHIPUAI_API_KEY from ~/.zhipuai_key when it is not set, so the goal server starts.
set -eo pipefail
# shellcheck source=env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
set -u
cd "$STVLN_ROOT"
LOG=$STVLN_ROOT/log
DATA=${STVLN_DATA:-$STVLN_ROOT/data}
MAPPERS="topo hydra_fake hydra"
# Launch files straight from src: they locate their siblings and scripts relative to themselves, and a stale
# install/share (launch files are copied, not symlinked) would otherwise run old code.
PIPELINE=$STVLN_ROOT/src/scene_memory/launch/pipeline.launch.py
LIO=$STVLN_ROOT/src/scene_memory/launch/lio.launch.py
RECORD=$STVLN_ROOT/src/tools/collection_data/launch/record.launch.py
NAV=$STVLN_ROOT/src/planning/launch/nav.launch.py
MAP=$STVLN_ROOT/src/mapping/launch/map.launch.py

die() { echo "[stvln] $*" >&2; exit 1; }
usage() { sed -n '2,/^set -eo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit "${1:-0}"; }

# run <name> <bg 0|1> cmd...: foreground, or detached in its own session with a pid file.
run() {
    local name=$1 bg=$2
    shift 2
    if ((!bg)); then
        echo "[stvln] $*"
        exec "$@"
    fi
    mkdir -p "$LOG"
    local pidf=$LOG/stvln_$name.pid logf=$LOG/stvln_$name.log
    if [[ -f $pidf ]] && group_alive "$(cat "$pidf")"; then
        die "$name already running (process group $(cat "$pidf")); src/scripts/stvln.sh stop $name"
    fi
    # bash starts background jobs with SIGINT ignored and exec keeps that, so ros2 launch would never see the
    # SIGINT from `stop`; restore the default before exec
    setsid python3 -c 'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execvp(sys.argv[1], sys.argv[1:])' \
        "$@" >"$logf" 2>&1 </dev/null &
    echo $! >"$pidf"
    echo "[stvln] $name started in background, pid $!"
    echo "        log:  tail -f $logf"
    echo "        stop: src/scripts/stvln.sh stop $name"
}

# split_args: --bg / --glm flags out of "$@" into BG / GLM, the rest into REST; --vlm applies at once.
split_args() {
    BG=0 GLM=0 REST=()
    local a
    for a in "$@"; do
        case $a in
            --bg) BG=1 ;;
            --glm) GLM=1 ;;
            --vlm) use_vlm ;;
            *) REST+=("$a") ;;
        esac
    done
}

# --vlm: the VLN agent and its vision steps on a self-hosted Ollama (no Zhipu key), default the .209 desktop.
use_vlm() {
    local url=${STVLN_VLM_URL:-http://192.168.110.209:11434/v1}
    export GLM_BASE_URL=$url VISION_BASE_URL=$url
    export GLM_MODEL=${STVLN_VLM_MODEL:-qwen3-vl:8b-instruct} VISION_MODEL=${STVLN_VLM_MODEL:-qwen3-vl:8b-instruct}
    if curl -s -m 3 "${url%/v1}/api/version" >/dev/null; then
        echo "[stvln] --vlm: $GLM_MODEL on $url"
    else
        echo "[stvln] --vlm: $url not reachable (ollama serve running there?); instructions needing it will fail" >&2
    fi
}

load_glm_key() {
    [[ -n ${ZHIPUAI_API_KEY:-} ]] && return 0
    if [[ -r ~/.zhipuai_key ]]; then
        ZHIPUAI_API_KEY=$(<~/.zhipuai_key)
        export ZHIPUAI_API_KEY
    else
        echo "[stvln] --glm: ZHIPUAI_API_KEY not set and ~/.zhipuai_key missing; goal server stays off" >&2
    fi
}

session_dir() {
    local s=$1
    [[ -d $s ]] && { cd "$s" && pwd; return; }
    [[ -d $DATA/sessions/$s ]] && { echo "$DATA/sessions/$s"; return; }
    die "no session '$s' (neither a directory nor $DATA/sessions/$s)"
}

# map <name> lives in its own folder: data/maps/<name>/<name>_map.pcd, <name>_scene/, ...
map_dir() { echo "${STVLN_MAPS:-$DATA/maps}/$1"; }
MAP_SUFFIXES="map.pcd ground.pcd ground_sem.pcd nonground.pcd nonground_raw.pcd nonground_sem.pcd poses.pcd
traversable.pcd scene semantic.json topo_graph.json objects_4d.json dsg.json descriptions.json map_cloud.npy"

cmd_live() {
    split_args "$@"
    local mapper=topo
    if ((${#REST[@]})) && [[ " $MAPPERS " == *" ${REST[0]} "* ]]; then
        mapper=${REST[0]}
        REST=("${REST[@]:1}")
    fi
    ((GLM)) && load_glm_key
    local odom=(odom:=lio)
    [[ " ${REST[*]} " == *" odom:="* ]] && odom=()
    run live "$BG" ros2 launch "$PIPELINE" "${odom[@]}" mapper:="$mapper" "${REST[@]}"
}

cmd_qinav_bridge() {
    split_args "$@"
    local mapper=topo
    if ((${#REST[@]})) && [[ " $MAPPERS " == *" ${REST[0]} "* ]]; then
        mapper=${REST[0]}
        REST=("${REST[@]:1}")
    fi
    ((GLM)) && load_glm_key
    local opts=() launch=() a
    for a in "${REST[@]}"; do
        if [[ $a == *:=* ]]; then launch+=("$a"); else opts+=("$a"); fi
    done
    run qinav_bridge "$BG" "$STVLN_PY" -m scene_memory.nodes.qinav_mapping_bridge_node --mapper "$mapper" \
        "${opts[@]}" -- "${launch[@]}"
}

cmd_map() {
    split_args "$@"
    ((${#REST[@]})) && [[ ${REST[0]} != *:=* ]] || die "usage: map <name> [--bg] [semantic:=false] [k:=v ...]"
    local name=${REST[0]}
    run map "$BG" ros2 launch "$MAP" name:="$name" "${REST[@]:1}"
}

cmd_map_save() {
    (($#)) || die "usage: map_save <name> [--semantic_only] [--leaf 0.1]"
    "$STVLN_PY" "$STVLN_ROOT/src/mapping/scripts/save_map.py" "$@"
}

cmd_map_offline() {
    split_args "$@"
    ((${#REST[@]} >= 2)) && [[ ${REST[1]} != *:=* ]] ||
        die "usage: map_offline <session> <name> [--glm] [--drain_s=20] [--no_rooms] [--no_views] [k:=v ...]"
    local raw name=${REST[1]} drain=20 extra=() a rooms=1 views=1
    raw=$(session_dir "${REST[0]}")
    raw=${raw%_proc}
    for a in "${REST[@]:2}"; do
        case $a in
            --drain_s=*) drain=${a#*=} ;;
            --no_rooms) rooms=0 ;;
            --no_views) views=0 ;;
            *) extra+=("$a") ;;
        esac
    done
    ((GLM)) && load_glm_key
    [[ -f $raw/bag/metadata.yaml ]] || die "no raw bag in $raw (map_offline needs the recording with the MID-360S)"
    [[ -f ${raw}_proc/bag/metadata.yaml ]] ||
        die "no ${raw}_proc: run 'src/scripts/stvln.sh derive $(basename "$raw")' first"
    local logf=$LOG/stvln_map.log
    run map 1 ros2 launch "$MAP" name:="$name" session:="${raw}_proc" raw_session:="$raw" \
        ${extra[@]+"${extra[@]}"}
    trap 'echo "[stvln] interrupted"; cmd_stop map; exit 130' INT
    echo "[stvln] waiting for the replay to finish ($(basename "$raw"))"
    until grep -q "replay done" "$logf" 2>/dev/null; do
        group_alive "$(cat "$LOG/stvln_map.pid")" || die "map exited before the replay finished, see $logf"
        sleep 5
    done
    # the semantic nodes are still working through the last frames
    echo "[stvln] replay done, draining ${drain} s"
    sleep "$drain"
    cmd_map_save "$name"
    cmd_stop map
    trap - INT
    local dir scene
    dir=$(map_dir "$name")
    scene=$dir/${name}_scene
    if [[ -f $scene/topo_graph.json ]]; then
        # FSR-VLN style rooms from the lidar map, panorama view layer, then region labels for the new rooms
        ((rooms)) && { "$STVLN_PY" -m scene_memory.mapping.pcd_rooms "$scene" --pcd "$dir/${name}_map.pcd" \
            --plot "$scene/rooms.png" --save || echo "[stvln] warning: pcd_rooms failed, keeping the mapping rooms"; }
        ((views)) && { "$STVLN_PY" -m scene_memory.mapping.view_layer "$scene" --session "${raw}_proc" ||
            echo "[stvln] warning: view_layer failed, no search_views on this map"; }
        "$STVLN_PY" -m scene_memory.mapping.dsg_export "$scene" --summarize --language Chinese ||
            echo "[stvln] warning: region labels failed"
    fi
    cmd_map_save "$name" --semantic_only
    echo "[stvln] offline map done; copy it to the robot: src/scripts/stvln.sh map_push $name [user@host]"
}

cmd_map_migrate() {
    (($#)) || die "usage: map_migrate <name>|--all"
    local maps=${STVLN_MAPS:-$DATA/maps} names=() name f s files
    if [[ $1 == --all ]]; then
        for f in "$maps"/*_map.pcd; do [[ -f $f ]] && names+=("$(basename "${f%_map.pcd}")"); done
    else
        names=("$1")
    fi
    for name in ${names[@]+"${names[@]}"}; do
        [[ -f $maps/${name}_map.pcd ]] || { echo "[stvln] $name: no flat $maps/${name}_map.pcd, skipped"; continue; }
        files=()
        for s in $MAP_SUFFIXES; do
            f=$maps/${name}_$s
            [[ -e $f ]] || continue
            [[ -e $maps/$name/$(basename "$f") ]] && die "$maps/$name/$(basename "$f") exists, not overwriting"
            files+=("$f")
        done
        mkdir -p "$maps/$name"
        mv "${files[@]}" "$maps/$name/"
        echo "[stvln] $name: ${#files[@]} files -> $maps/$name/"
    done
}

cmd_map_push() {
    (($#)) || die "usage: map_push <name> [user@host (user@192.168.110.56)] [remote stvln dir (stvln)]"
    local name=$1 host=${2:-user@192.168.110.56} rdir=${3:-stvln} dir
    dir=$(map_dir "$name")
    [[ -f $dir/${name}_map.pcd ]] ||
        die "no $dir/${name}_map.pcd (older flat map? src/scripts/stvln.sh map_migrate $name)"
    echo "[stvln] $dir/ -> $host:$rdir/data/maps/$name/"
    rsync -a --mkpath --info=stats1 "$dir/" "$host:$rdir/data/maps/$name/"
    echo "[stvln] on the robot: src/scripts/stvln.sh nav $name --bg"
}

cmd_nav() {
    split_args "$@"
    ((${#REST[@]})) && [[ ${REST[0]} != *:=* ]] ||
        die "usage: nav <map name|prefix> [--bg] [--glm|--vlm] [control:=true] [k:=v ...]"
    ((GLM)) && load_glm_key
    local map=${REST[0]}
    run nav "$BG" ros2 launch "$NAV" map:="$map" "${REST[@]:1}"
}

# Insta360 only (panorama + views); nav starts it itself unless camera:=false.
cmd_camera() {
    split_args "$@"
    if pgrep -x insta360_views >/dev/null; then
        die "insta360_views already running (pid $(pgrep -x insta360_views | tr '\n' ' ')); the camera takes one client"
    fi
    run camera "$BG" "$(ros2 pkg prefix insta360_views)/lib/insta360_views/insta360_views" --ros-args \
        --params-file "$STVLN_ROOT/src/sensors/insta360_views/config/insta360_views.yaml" \
        -p publish_equirect:=true -p publish_vio_topics:=false "${REST[@]}"
}

# JanusVLN only knows English: non-ASCII instructions go through Zhipu GLM (ZHIPUAI_API_KEY / ~/.zhipuai_key).
janus_english() {
    local text=$1 out rc=0
    if [[ -z $text ]] || LC_ALL=C grep -qv '[^ -~]' <<<"$text"; then
        echo "$text"
        return
    fi
    load_glm_key 2>/dev/null
    out=$(python3 "$STVLN_ROOT/refer/JanusVLN/deploy/translate_instruction.py" "$text") || rc=$?
    case $rc in
        0) echo "[stvln] instruction → English: $out" >&2; echo "$out" ;;
        2) die "\"$text\" is not English and ZHIPUAI_API_KEY / ~/.zhipuai_key is missing; give it in English" ;;
        *) die "translating \"$text\" failed; give it in English" ;;
    esac
}

# Front view → remote JanusVLN brain (.87). Needs camera (or nav) already publishing front.
cmd_janus() {
    split_args "$@"
    local brain="${STVLN_JANUS_URL:-http://192.168.110.87:8088}" instruction="" execute=0
    local client="$STVLN_ROOT/refer/JanusVLN/deploy/dog_front_client.py"
    local args=()
    for a in "${REST[@]}"; do
        case $a in
            --brain=*) brain=${a#--brain=} ;;
            --brain) die "use --brain=URL" ;;
            --instruction=*) instruction=${a#--instruction=} ;;
            --instruction) die "use --instruction=\"...\"" ;;
            --execute) execute=1 ;;
            *) args+=("$a") ;;
        esac
    done
    [[ -f $client ]] || die "missing $client (rsync refer/JanusVLN/deploy to the robot)"
    python3 -c "import requests" 2>/dev/null || die "pip install requests (on the Orin)"
    curl -sf --max-time 3 "$brain/health" >/dev/null || \
        echo "[stvln] WARN: $brain/health not reachable; client will retry" >&2
    instruction=$(janus_english "$instruction") || exit 1
    local extra=(--brain "$brain" --instruction "$instruction")
    ((execute)) && extra+=(--execute)
    run janus "$BG" python3 "$client" "${extra[@]}" "${args[@]}"
}

# Map-free JanusVLN walking (src/planning/launch/janus_nav.launch.py): no map name, works off the map.
cmd_janus_nav() {
    split_args "$@"
    local brain="${STVLN_JANUS_URL:-http://192.168.110.87:8088}" instruction=""
    local args=()
    for a in "${REST[@]}"; do
        case $a in
            --brain=*) brain=${a#--brain=} ;;
            --instruction=*) instruction=${a#--instruction=} ;;
            --brain | --instruction) die "use $a=..." ;;
            *) args+=("$a") ;;
        esac
    done
    python3 -c "import requests" 2>/dev/null || die "pip install requests (on the Orin)"
    curl -sf --max-time 3 "$brain/health" >/dev/null || \
        echo "[stvln] WARN: $brain/health not reachable; client will retry" >&2
    instruction=$(janus_english "$instruction") || exit 1
    # ros2 launch rejects an empty 'instruction:='
    [[ -n $instruction ]] && args=(instruction:="$instruction" "${args[@]}")
    run janus_nav "$BG" ros2 launch "$STVLN_ROOT/src/planning/launch/janus_nav.launch.py" \
        brain:="$brain" "${args[@]}"
}

# New instruction for a running janus / janus_nav client ("stop" halts it); the brain starts a new episode.
cmd_janus_go() {
    (($#)) || die 'usage: janus_go "<instruction>" | stop'
    pgrep -f dog_front_client.py >/dev/null ||
        die "no janus client running: start it first (src/scripts/stvln.sh janus_nav control:=true, or janus --execute)"
    local text="$*" msg
    case ${text,,} in
        stop | cancel | 停 | 停止) ;;
        *) text=$(janus_english "$text") || exit 1 ;;
    esac
    msg=$(python3 -c 'import json, sys; print(json.dumps({"data": sys.argv[1]}, ensure_ascii=False))' "$text")
    timeout 8 ros2 topic echo --once /janus/status std_msgs/msg/String 2>/dev/null | sed -n 's/^data: /[janus] /p' &
    local echo_pid=$!
    sleep 1.5
    timeout 10 ros2 topic pub --once -w 1 /janus/instruction std_msgs/msg/String "$msg" >/dev/null ||
        die "no janus client on /janus/instruction (janus / janus_nav running, same ROS_DOMAIN_ID?)"
    echo "[stvln] sent to /janus/instruction: $text"
    wait "$echo_pid" || true
}

# Go2-W battery only (read-only rt/lowstate); nav starts it itself unless battery:=false.
cmd_battery() {
    split_args "$@"
    pgrep -x go2w_battery_no >/dev/null && die "go2w_battery_node already running (pid $(pgrep -x go2w_battery_no))"
    run battery "$BG" "$(ros2 pkg prefix go2w_keyboard_control)/lib/go2w_keyboard_control/go2w_battery_node" \
        --ros-args "${REST[@]}"
}

# Battery monitor in the robot's domain (the 8766 bridge's), detached so it outlives the caller; skipped off the
# Go2-W network (192.168.123.x, e.g. on the PC) or when one already runs.
battery_monitor() {
    pgrep -x go2w_battery_no >/dev/null && return 0
    if ! ip -4 addr show | grep -q "inet 192\.168\.123\."; then
        echo "[stvln] not on the Go2-W network (192.168.123.x): battery monitor not started"
        return 0
    fi
    if ! ros2 pkg prefix go2w_keyboard_control >/dev/null 2>&1; then
        echo "[stvln] go2w_keyboard_control not built (build planning): battery monitor not started"
        return 0
    fi
    (cmd_battery --bg) || echo "[stvln] battery monitor failed to start (see $LOG/stvln_battery.log)"
}

# Go2-W model only, for runs where something else (QiNav, live with odom:=qinav*) provides base_link.
cmd_model() {
    split_args "$@"
    run model "$BG" ros2 launch go2w_description description.launch.py "${REST[@]}"
}

cmd_lio() {
    split_args "$@"
    run lio "$BG" ros2 launch "$LIO" "${REST[@]}"
}

# camera_usb: Insta360 USB product id (0002 = Android mode, streams; 1005 = U-disk, cannot stream), or empty.
camera_usb() { lsusb -d 2e1a: 2>/dev/null | grep -o '2e1a:[0-9a-f]*' | head -1 | cut -d: -f2 || true; }

cmd_record() {
    split_args "$@"
    local extra=() sess check=1 a rest=()
    for a in "${REST[@]}"; do
        [[ $a == --no_camera_check ]] && check=0 || rest+=("$a")
    done
    REST=("${rest[@]}")
    if ((${#REST[@]})) && [[ ${REST[0]} != *:=* ]]; then
        sess=$DATA/sessions/${REST[0]}
        extra=(session_dir:="$sess")
        REST=("${REST[@]:1}")
    else
        sess=$DATA/sessions/$(date +%Y%m%d_%H%M%S)
        extra=(session_dir:="$sess")
    fi
    [[ -e $sess/bag ]] && die "$sess/bag already exists; pick a new name"
    if ((check)); then
        case $(camera_usb) in
            0002) ;;
            1005) die "Insta360 is in U-disk mode (2e1a:1005): set USB mode to Android on the camera and replug" \
                      "(lsusb must show 2e1a:0002); --no_camera_check records without the camera" ;;
            "") die "no Insta360 on USB (cable, camera powered on?); --no_camera_check records without the camera" ;;
            *) echo "[stvln] Insta360 2e1a:$(camera_usb): unexpected USB mode, trying anyway" >&2 ;;
        esac
    fi
    run record "$BG" ros2 launch "$RECORD" "${extra[@]}" "${REST[@]}"
    ((check)) || return 0
    # --bg only (foreground exec'd above): wait until the fisheye stream actually lands in raw/
    local kb=0
    for _ in $(seq 30); do
        sleep 1
        kb=$(du -sk "$sess/raw" 2>/dev/null | cut -f1)
        ((kb > 2048)) && break
        group_alive "$(cat "$LOG/stvln_record.pid")" || die "record exited; see $LOG/stvln_record.log"
    done
    if ((kb > 2048)); then
        echo "[stvln] camera streaming: raw/ ${kb} KB after start; OK to drive"
    else
        echo "[stvln] WARNING: no fisheye stream in $sess/raw after 30 s (camera hung?);" \
             "check: grep insta360_views $LOG/stvln_record.log | tail; stop, power-cycle the camera, retry" >&2
    fi
}

# Separate from the robot's foxglove-bridge service (port 8766): decodes this workspace's message types and
# serves another ROS domain (vln: ROS_DOMAIN_ID=78). The 8766 service sees the workspace topics and the Go2-W
# meshes through /etc/systemd/system/foxglove-bridge.service.d/stvln-go2w-model.conf.
cmd_foxglove() {
    split_args "$@"
    local port=${REST[0]:-8767}
    run foxglove "$BG" "$(ros2 pkg prefix foxglove_bridge)/lib/foxglove_bridge/foxglove_bridge" --ros-args \
        -p port:="$port" -p address:=0.0.0.0 -p send_buffer_limit:=20000000 -p max_qos_depth:=2 \
        -p "capabilities:=[clientPublish,connectionGraph,services,assets]"
}

cmd_record_eval() {
    (($#)) || die "usage: record_eval <name> [--no_equirect]"
    exec "$STVLN_ROOT/src/scene_memory/scripts/record_topo_eval.sh" "$@"
}

cmd_replay() {
    (($#)) || die "usage: replay <session> [topo|hydra_fake|hydra] [k:=v ...]"
    local s mapper=topo
    s=$(session_dir "$1")
    shift
    if (($#)) && [[ " $MAPPERS " == *" $1 "* ]]; then
        mapper=$1
        shift
    fi
    run replay 0 ros2 launch "$PIPELINE" session:="$s" mapper:="$mapper" "$@"
}

cmd_eval() {
    (($#)) || die "usage: eval <session|bag> [--gt file] [--mappers topo,hydra_fake] [topo_eval args]"
    local target=$1 bag sess gt="" mappers="topo,hydra_fake" extra=()
    shift
    while (($#)); do
        case $1 in
            --gt) gt=$2; shift 2 ;;
            --mappers) mappers=$2; shift 2 ;;
            *) extra+=("$1"); shift ;;
        esac
    done
    if [[ -f $target/metadata.yaml ]]; then
        bag=$(cd "$target" && pwd)
        sess=$(dirname "$bag")
    else
        sess=$(session_dir "$target")
        bag=$sess/topo_bag
        [[ -f $bag/metadata.yaml ]] || die "no topo_bag in $sess (record with: stvln.sh record_eval <name>)"
    fi
    [[ -z $gt && -f $sess/gt_objects.yaml ]] && gt=$sess/gt_objects.yaml
    [[ -n $gt ]] && extra+=(--gt "$gt") || echo "[stvln] no gt_objects.yaml: ground-truth-free metrics only"
    local m outs=()
    for m in ${mappers//,/ }; do
        echo "[stvln] topo_eval $m -> $sess/eval_$m"
        python3 -m scene_memory.mapping.topo_eval "$bag" --mapper "$m" --out_dir "$sess/eval_$m" "${extra[@]}"
        outs+=("$sess/eval_$m")
    done
    python3 - "${outs[@]}" <<'EOF'
import json, sys
from pathlib import Path
rows = []
for d in map(Path, sys.argv[1:]):
    r = json.loads((d / "report.json").read_text())
    s, g = r.get("summary", {}), r.get("gt") or {}
    e = g.get("err_xy") or {}
    f = lambda v, n=3: "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))
    rows.append([d.name.removeprefix("eval_"), f(s.get("objects")), f(len(s.get("duplicates", []))),
                 f"{g.get('matched', '-')}/{g.get('gt_objects', '-')}" if g else "-",
                 f(e.get("median")), f(e.get("p90")), f(e.get("within_0.3m"), 2), f(s.get("revisit_shift_med"))])
head = ["mapper", "objects", "dups", "matched", "err_med_m", "err_p90_m", "<=0.3m", "revisit_shift_m"]
w = [max(len(str(x)) for x in col) for col in zip(head, *rows)]
for row in [head, *rows]:
    print("  ".join(str(x).ljust(n) for x, n in zip(row, w)))
print("maps:", *(str(Path(o) / "map.png") for o in sys.argv[1:]))
EOF
}

cmd_check() {
    local topics=(/insta360/front/image/compressed /insta360/equirect/image/compressed
                  /livox/front_lidar/custom /livox/front_lidar/imu /livox/lidar /livox/imu /lio/odom
                  /lio/cloud_world /detector/detections_json /topo/graph /ql/nav/localization
                  /ql/nav/cloud_in_map /ql/nav/task_status)
    local tmp t
    tmp=$(mktemp -d)
    (cd "$STVLN_ROOT/src/scene_memory" && "$STVLN_ROOT/.venv/bin/python" -m scene_memory.common.livox_probe) |
        sed 's/^/[stvln] /' || true
    echo "[stvln] ROS_DOMAIN_ID=$ROS_DOMAIN_ID RMW=${RMW_IMPLEMENTATION:-default}, sampling 8 s ..."
    for t in "${topics[@]}"; do
        (timeout -s INT 8 ros2 topic hz "$t" >"$tmp/${t//\//_}.hz" 2>&1 || true) &
        (timeout 8 ros2 topic info "$t" >"$tmp/${t//\//_}.info" 2>&1 || true) &
    done
    wait
    for t in "${topics[@]}"; do
        local rate pubs
        rate=$(grep -o 'average rate: [0-9.]*' "$tmp/${t//\//_}.hz" | tail -1 | awk '{print $3}' || true)
        pubs=$(grep -o 'Publisher count: [0-9]*' "$tmp/${t//\//_}.info" | awk '{print $3}' || true)
        if [[ -n $rate ]]; then
            printf '  %-42s %s Hz\n' "$t" "$rate"
        elif [[ ${pubs:-0} -gt 0 ]]; then
            printf '  %-42s <2 msgs in 8 s (%s publisher(s): slow, or publisher up but silent)\n' "$t" "$pubs"
        else
            printf '  %-42s NO DATA (no publisher)\n' "$t"
        fi
    done
    rm -f "$tmp"/_* && rmdir "$tmp"
}

cmd_query() {
    (($#)) || die 'usage: query "<prompt>"'
    local msg
    msg=$(python3 -c 'import json, sys; print(json.dumps({"data": " ".join(sys.argv[1:])}, ensure_ascii=False))' "$@")
    timeout 60 ros2 topic echo --once --full-length /scene_memory/answer std_msgs/msg/String &
    local echo_pid=$!
    sleep 1.5
    ros2 topic pub --once /scene_memory/query std_msgs/msg/String "$msg" >/dev/null
    wait "$echo_pid" || die "no answer within 60 s (goal server running? needs ZHIPUAI_API_KEY or start with live --glm)"
}

cmd_derive() {
    (($#)) || die "usage: derive <session> [derive_session args]"
    local s
    s=$(session_dir "$1")
    shift
    "$STVLN_PY" "$STVLN_ROOT/src/tools/collection_data/collection_data/derive_session.py" "$s" "$@"
}
VLN_PIDS=()
vln_cleanup() {
    ((${#VLN_PIDS[@]})) || return 0
    kill -TERM "${VLN_PIDS[@]}" 2>/dev/null || true
    wait "${VLN_PIDS[@]}" 2>/dev/null || true
}
cmd_vln() {
    split_args "$@"
    set -- ${REST[@]+"${REST[@]}"}
    ((GLM)) && load_glm_key
    local scene session="" episodes="" target=""
    if (($#)) && [[ $1 != --* ]]; then
        target=$1
        shift
    fi
    if [[ -z $target ]]; then
        scene=$(cd "$DATA/scene4d/latest" && pwd -P)
    elif [[ -f $target/topo_graph.json ]]; then
        scene=$(cd "$target" && pwd -P)
    else
        session=$(session_dir "$target")
        scene=$(cd "$session/scene4d/latest" && pwd -P) || die "no scene4d/latest in $session (replay it first)"
    fi
    [[ -f $scene/topo_graph.json && -f $scene/map_cloud.npy ]] || die "$scene needs topo_graph.json and map_cloud.npy"
    [[ -n $session ]] && episodes=$STVLN_ROOT/src/scene_memory/config/vln_episodes/$(basename "$session" _proc).yaml
    local -a ev=()
    while (($#)); do
        case $1 in
            --episodes) episodes=$2; shift 2 ;;
            --session) session=$(session_dir "$2"); shift 2 ;;
            *) ev+=("$1"); shift ;;
        esac
    done
    [[ -f $episodes ]] || die "no episodes file '$episodes' (--episodes <yaml>)"
    local frame
    frame=$("$STVLN_PY" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("frame_id", "map"))' "$scene/topo_graph.json")
    battery_monitor
    # own domain: the simulated /ql/nav/* must not meet a real QiNav or a live pipeline
    export ROS_DOMAIN_ID=${STVLN_VLN_DOMAIN:-78}
    [[ -n ${ZHIPUAI_API_KEY:-} ]] || echo "[stvln] ZHIPUAI_API_KEY not set: fast-path instructions only (--glm loads ~/.zhipuai_key)"
    echo "[stvln] vln on $scene (frame $frame, ROS_DOMAIN_ID=$ROS_DOMAIN_ID), logs $LOG/vln_*.log"
    trap vln_cleanup EXIT
    "$STVLN_PY" -m scene_memory.navigation.qinav_sim_node --scene_dir "$scene" >"$LOG/vln_qinav_sim.log" 2>&1 </dev/null &
    VLN_PIDS+=($!)
    if ros2 pkg prefix go2w_description >/dev/null 2>&1; then
        ros2 launch go2w_description description.launch.py >"$LOG/vln_model.log" 2>&1 </dev/null &
        VLN_PIDS+=($!)
    else
        echo "[stvln] go2w_description not built (build planning): no robot model"
    fi
    "$STVLN_PY" -m scene_memory.nodes.goal_server_node --scene_dir "$scene" --odom_topic /ql/nav/localization \
        --qinav --qinav_frame "$frame" >"$LOG/vln_goal_server.log" 2>&1 </dev/null &
    VLN_PIDS+=($!)
    "$STVLN_PY" -m scene_memory.navigation.vln_eval "$scene" --episodes "$episodes" ${session:+--session "$session"} \
        ${ev[@]+"${ev[@]}"}
}
cmd_plan() {
    local scene=$DATA/scene4d/latest
    if (($#)) && [[ $1 != --* ]]; then
        scene=$1
        shift
    fi
    [[ -f $scene/topo_graph.json ]] || die "no topo_graph.json in $scene"
    (($#)) || set -- --all
    "$STVLN_PY" -m scene_memory.navigation.plan_check "$scene" "$@"
}

# group_alive <pgid>: any process left in the group (the launch parent may exit before its children).
group_alive() { pgrep -g "$1" >/dev/null 2>&1; }

cmd_stop() {
    local names=("$@") n pid stopped=0
    # live before qinav_bridge: the bridge files the final semantic map once the pipeline has exited
    ((${#names[@]})) || names=(map nav camera battery model live lio record foxglove qinav_bridge)
    # Ctrl+C while waiting must not leave a half-stopped group behind
    trap 'echo "[stvln] interrupted; run stop again to finish"; exit 130' INT
    for n in "${names[@]}"; do
        local pidf=$LOG/stvln_$n.pid
        [[ -f $pidf ]] || continue
        pid=$(cat "$pidf")
        if group_alive "$pid"; then
            echo "[stvln] stopping $n (process group $pid, up to ~45 s)"
            # SIGINT to the launch process only: it shuts its children down and no longer respawns them; a
            # group-wide SIGINT lets children exit first and launch respawns them (respawn=True nodes)
            kill -INT "$pid" 2>/dev/null || kill -INT -- "-$pid" 2>/dev/null || true
            for _ in $(seq 30); do group_alive "$pid" || break; sleep 1; done
            if group_alive "$pid"; then
                echo "[stvln] $n still running after 30 s, sending SIGTERM"
                kill -TERM -- "-$pid" 2>/dev/null || true
                for _ in $(seq 10); do group_alive "$pid" || break; sleep 1; done
            fi
            if group_alive "$pid"; then
                echo "[stvln] $n still running, sending SIGKILL to: $(pgrep -g "$pid" -d ' ')"
                kill -KILL -- "-$pid" 2>/dev/null || true
                sleep 1
            fi
            stopped=1
        fi
        group_alive "$pid" || rm -f "$pidf"
    done
    trap - INT
    ((stopped)) || echo "[stvln] nothing started with --bg is running"
}

cmd_status() {
    local n pidf
    for n in map nav camera battery model live lio record foxglove qinav_bridge; do
        pidf=$LOG/stvln_$n.pid
        if [[ -f $pidf ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
            echo "  $n: running (pid $(cat "$pidf"), log $LOG/stvln_$n.log)"
        else
            echo "  $n: not running"
        fi
    done
    echo "newest scene dirs:"
    ls -1dt "$DATA"/scene4d/*/ "$DATA"/sessions/*/scene4d/*/ 2>/dev/null | head -5 | sed 's/^/  /' || true
}

cmd_build() {
    if [[ ${1:-} == qinav_msgs ]]; then
        local src=${STVLN_QINAV_MSGS_SRC:-$HOME/xjy/QiNav_ws/src/qi_servers/qi_nav_msgs}
        [[ -f $src/package.xml ]] || die "no qi_nav_msgs at $src (set STVLN_QINAV_MSGS_SRC)"
        colcon build --base-paths "$src" --build-base "$STVLN_ROOT/qinav_msgs/build" \
            --install-base "$STVLN_ROOT/qinav_msgs/install" --packages-select qi_nav_msgs
        return
    fi
    if [[ ${1:-} == planning ]]; then
        # QiNav's DDDMR global planner + qi_move_base, CMU local planner, Go2-W control (unitree_sdk2) and the
        # localization they need (localization_cpp, linefit, cloud_preprocess)
        colcon build --symlink-install --base-paths src/planning src/localization/qinav_localization src/mapping \
            --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF
        return
    fi
    # by path: the calibration repo carries its own insta360_views (duplicate package name)
    colcon build --symlink-install --paths src/sensors/insta360_views src/tools/collection_data src/scene_memory
    colcon build --packages-select basic super_lio --cmake-args -DCMAKE_BUILD_TYPE=Release
}

(($#)) || usage 1
cmd=$1
shift
case $cmd in
    live | lio | map | map_save | map_offline | map_migrate | map_push | nav | camera | battery | model | qinav_bridge | record | foxglove | record_eval | replay | derive | vln | eval | check | query | plan | stop | status | build | janus | janus_nav | janus_go) "cmd_$cmd" "$@" ;;
    -h | --help | help) usage 0 ;;
    *) echo "[stvln] unknown command '$cmd'" >&2; usage 1 ;;
esac
