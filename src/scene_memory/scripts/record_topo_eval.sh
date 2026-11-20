#!/usr/bin/env bash
# Record a topo localisation-accuracy run next to a live pipeline (start pipeline.launch.py with odom:=lio first).
# Mapper inputs (detections, LIO odom + cloud, resets) make the run replayable offline by
# scene_memory.mapping.topo_eval; raw lidar / IMU / front view / equirect allow re-running LIO or the detector.
#
#   scripts/record_topo_eval.sh <name> [--no_equirect]     # Ctrl-C to stop
#   -> data/sessions/<name>/topo_bag  (+ put gt_objects.yaml in data/sessions/<name>/)
set -euo pipefail
name=${1:?usage: record_topo_eval.sh <name> [--no_equirect]}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../../.." && pwd)
session=${STVLN_DATA:-$root/data}/sessions/$name
qos=$root/src/tools/collection_data/config/qos_overrides.yaml
[[ -e $session/topo_bag ]] && { echo "$session/topo_bag exists" >&2; exit 1; }
mkdir -p "$session"
topics=(/detector/detections_json /lio/odom /lio/cloud_world /lio/reset /tf_static
        /livox/front_lidar/custom /livox/front_lidar/imu /insta360/front/image/compressed /topo/graph)
[[ ${2:-} == --no_equirect ]] || topics+=(/insta360/equirect/image/compressed)
echo "recording ${#topics[@]} topics to $session/topo_bag"
exec ros2 bag record -o "$session/topo_bag" --qos-profile-overrides-path "$qos" --include-hidden-topics "${topics[@]}"
