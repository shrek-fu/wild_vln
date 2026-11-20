# Source me: ROS 2 Humble + livox driver (calibration repo) + workspace overlay for the STVLN pipeline.
#   source src/scripts/env.sh
# Works from any checkout location (PC workspace or ~/stvln on the Orin).
#   ROS_DOMAIN_ID  kept if already set, else 10 (same domain as the robot's foxglove bridge)
#   STVLN_PY       GPU interpreter: <ws>/.venv/bin/python when present (Jetson), else python3

_stvln_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export STVLN_ROOT=${STVLN_ROOT:-$(cd "$_stvln_here/../.." && pwd)}
unset _stvln_here

_stvln_source() {
    # ROS setup scripts read unset variables, so never source them under `set -u`.
    local had_u=0
    [[ $- == *u* ]] && had_u=1 && set +u
    # shellcheck disable=SC1090
    source "$1"
    ((had_u)) && set -u
    return 0
}

if [[ -f /opt/ros/humble/setup.bash ]]; then
    _stvln_source /opt/ros/humble/setup.bash
else
    echo "[stvln] /opt/ros/humble/setup.bash not found" >&2
fi
for _s in "$STVLN_ROOT/src/insta360_mid360s_calibration/install/setup.bash" "$STVLN_ROOT/install/setup.bash"; do
    if [[ -f $_s ]]; then
        _stvln_source "$_s"
    else
        echo "[stvln] missing $_s (build first: src/scripts/stvln.sh build)" >&2
    fi
done
unset _s
# qi_nav_msgs for the goal server's QiNav client (odom:=qinav), built from QiNav's source into our own
# overlay (src/scripts/stvln.sh build qinav_msgs) so QiNav_ws itself is never touched.
[[ -f $STVLN_ROOT/qinav_msgs/install/setup.bash ]] && _stvln_source "$STVLN_ROOT/qinav_msgs/install/setup.bash"

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-10}
# The robot's services (foxglove bridge, QiNav) run CycloneDDS with this config; a shell without it (ssh,
# cron) would otherwise start FastDDS and neither side discovers the other.
STVLN_DDS_URI=${STVLN_DDS_URI:-/opt/qinav_ws/dds_configs/cyclonedds.xml}
if [[ -z ${RMW_IMPLEMENTATION:-} && -f $STVLN_DDS_URI ]]; then
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI=${CYCLONEDDS_URI:-file://$STVLN_DDS_URI}
fi
case ":${PYTHONPATH:-}:" in
    *":$STVLN_ROOT/src/scene_memory:"*) ;;
    *) export PYTHONPATH="$STVLN_ROOT/src/scene_memory:$STVLN_ROOT/src/hydra_fake${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
if [[ -x $STVLN_ROOT/.venv/bin/python ]]; then
    export STVLN_PY=$STVLN_ROOT/.venv/bin/python
else
    export STVLN_PY=python3
fi
