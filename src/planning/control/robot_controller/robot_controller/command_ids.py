"""Platform 1043 control_id values and RobotCommand strings."""

CMD_STAND = 'stand'
CMD_LIE = 'lie'
CMD_EMERGENCY = 'emergency'
CMD_STOP = 'stop'
CMD_PREPARE = 'prepare'
CMD_ACTION = 'action'
CMD_WAYPOINT_ACTION = 'waypoint_action'

COMMANDS = (
    CMD_STAND,
    CMD_LIE,
    CMD_EMERGENCY,
    CMD_STOP,
    CMD_PREPARE,
    CMD_ACTION,
    CMD_WAYPOINT_ACTION,
)

# 1043 robot_control_id
ACTION_SPEED_1 = 1
ACTION_SPEED_2 = 2
ACTION_SPEED_3 = 3
ACTION_DAMP = 101
ACTION_STAND = 102
ACTION_LIE = 103
ACTION_WAVE = 104
ACTION_PITCH = 105
ACTION_HEIGHT = 106
ACTION_TURN_LEFT = 107
ACTION_TURN_RIGHT = 108

SPEED_RATIOS = {
    ACTION_SPEED_1: 1.0,
    ACTION_SPEED_2: 1.5,
    ACTION_SPEED_3: 2.0,
}

# Unitree C++ posture_cmd Int32 values (shared by b2 / go2 / go2w)
POSTURE_STOP = 0
POSTURE_STAND = 1
POSTURE_LIE = 2
POSTURE_DAMP = 3
POSTURE_HELLO = 4
POSTURE_SPEED_1 = 21
POSTURE_SPEED_2 = 22
POSTURE_SPEED_3 = 23

ROBOT_TYPES = (
    'chitu',
    'zsl1',
    'ysc_lite3',
    'unitree_b2',
    'unitree_go2',
    'unitree_go2w',
)

CONFIG_FILES = {
    'chitu': 'chitu.yaml',
    'zsl1': 'zsl1.yaml',
    'ysc_lite3': 'ysc_lite3.yaml',
    'unitree_b2': 'unitree_b2.yaml',
    'unitree_go2': 'unitree_go2.yaml',
    'unitree_go2w': 'unitree_go2w.yaml',
}


def normalize_robot_type(robot_type: str) -> str:
    key = str(robot_type or '').strip().lower()
    if key not in ROBOT_TYPES:
        supported = ', '.join(ROBOT_TYPES)
        raise ValueError(f'Unsupported robot_type: {robot_type}. Supported: {supported}')
    return key


def is_unitree(robot_type: str) -> bool:
    return normalize_robot_type(robot_type).startswith('unitree_')
