import importlib.util
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_controller.adapters.chitu import ChituAdapter
from robot_controller.adapters.unitree import UnitreeAdapter
from robot_controller.adapters.ysc_lite3 import CMD_CRAWL, CMD_MANUAL_MODE, CMD_MOVE_MODE, CMD_STAND, YscLite3Adapter


def _load_source_module(module_name: str, relative: str):
    # Nuitka *.so next to the source shadows the .py; load the file directly.
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_command_ids_src = _load_source_module(
    '_robot_controller_command_ids_src',
    'robot_controller/command_ids.py',
)
_orig_command_ids = sys.modules.get('robot_controller.command_ids')
sys.modules['robot_controller.command_ids'] = _command_ids_src
try:
    _zsl1_src = _load_source_module(
        '_robot_controller_zsl1_src',
        'robot_controller/adapters/zsl1.py',
    )
finally:
    if _orig_command_ids is not None:
        sys.modules['robot_controller.command_ids'] = _orig_command_ids

ACTION_DAMP = _command_ids_src.ACTION_DAMP
ACTION_HEIGHT = _command_ids_src.ACTION_HEIGHT
ACTION_LIE = _command_ids_src.ACTION_LIE
ACTION_PITCH = _command_ids_src.ACTION_PITCH
ACTION_STAND = _command_ids_src.ACTION_STAND
ACTION_TURN_LEFT = _command_ids_src.ACTION_TURN_LEFT
ACTION_TURN_RIGHT = _command_ids_src.ACTION_TURN_RIGHT
ACTION_WAVE = _command_ids_src.ACTION_WAVE
POSTURE_DAMP = _command_ids_src.POSTURE_DAMP
POSTURE_HELLO = _command_ids_src.POSTURE_HELLO
POSTURE_LIE = _command_ids_src.POSTURE_LIE
POSTURE_STAND = _command_ids_src.POSTURE_STAND
normalize_robot_type = _command_ids_src.normalize_robot_type
Zsl1Adapter = _zsl1_src.Zsl1Adapter
value_limit = _zsl1_src.value_limit


class FakeLogger:
    def info(self, *_a, **_k):
        return None

    def warn(self, *_a, **_k):
        return None

    def error(self, *_a, **_k):
        return None


class FakePublisher:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


class FakeNode:
    def __init__(self, udp=None):
        self.publisher = FakePublisher()
        self.udp = udp
        self.subscriptions = []

    def get_logger(self):
        return FakeLogger()

    def create_publisher(self, *_a, **_k):
        return self.publisher

    def create_subscription(self, *_a, **_k):
        self.subscriptions.append((_a, _k))
        return None

    def pump_lite3_state(self, wait=False):
        if self.udp is None:
            return None
        if self.udp.available:
            return self.udp.available.pop(0)
        if wait and self.udp.states:
            return self.udp.states.pop(0)
        return None


class FakeUdp:
    def __init__(self, states=None, available=None):
        self.simple = []
        self.complex = []
        self.states = list(states or [])
        self.available = list(available or [])
        self.recv_waits = []

    def send_simple(self, code, param1=0, param2=0):
        self.simple.append(code)

    def send_complex(self, code, value):
        self.complex.append((code, value))

    def recv_basic_state(self, extra_socks, wait_s, until_packet=False):
        self.recv_waits.append(wait_s)
        if self.available:
            return self.available.pop(0)
        if wait_s > 0.0 and self.states:
            return self.states.pop(0)
        return None

    def poll_state(self, extra_socks):
        latest = None
        while self.available:
            latest = self.available.pop(0)
        return latest

    def close(self):
        return None


def test_normalize_robot_type_canonical_names():
    assert normalize_robot_type('unitree_b2') == 'unitree_b2'
    assert normalize_robot_type('UNITREE_GO2W') == 'unitree_go2w'
    assert normalize_robot_type('ysc_lite3') == 'ysc_lite3'
    assert normalize_robot_type('unitree_go2') == 'unitree_go2'
    for alias in ('b2', 'go2', 'go2w', 'lite3'):
        try:
            normalize_robot_type(alias)
        except ValueError:
            continue
        raise AssertionError(f'{alias} should not be accepted')


def test_chitu_has_no_posture_and_republishes_twist():
    node = FakeNode()
    adapter = ChituAdapter(node, {'output_cmd_vel_topic': '/rl_cmd_vel'})
    assert adapter.supports_posture() is False
    assert adapter.is_standing() is True
    stand = adapter.stand()
    lie = adapter.lie()
    assert stand.success is True
    assert lie.success is True
    adapter.apply_velocity(0.2, -0.1, 0.3)
    assert len(node.publisher.msgs) >= 1
    last = node.publisher.msgs[-1]
    assert abs(last.linear.x - 0.2) < 1e-9
    assert abs(last.linear.y - (-0.1)) < 1e-9
    assert abs(last.angular.z - 0.3) < 1e-9


def test_chitu_emergency_only_zeroes_velocity():
    node = FakeNode()
    adapter = ChituAdapter(node, {})
    result = adapter.emergency()
    assert result.success is True
    last = node.publisher.msgs[-1]
    assert last.linear.x == 0.0
    assert last.angular.z == 0.0


def test_zsl1_stand_then_move_and_limit():
    node = FakeNode()
    cmd_q = queue.Queue()
    adapter = Zsl1Adapter(
        node,
        {'start_process': False, 'cmd_queue': cmd_q, 'max_linear_x': 1.5},
    )
    assert adapter.is_standing() is False
    adapter.apply_velocity(0.5, 0.0, 0.0)
    assert cmd_q.empty()
    assert adapter.stand().success is True
    assert adapter.is_standing() is True
    assert cmd_q.get_nowait()[0] == 'standUp'
    adapter.apply_velocity(3.0, 0.0, 0.0)
    cmd, args = cmd_q.get_nowait()
    assert cmd == 'move'
    assert args[0] == 1.5


def test_zsl1_value_limit_deadband():
    assert value_limit(0.01, 1.5, 0.05, 0.05) == 0.0
    assert value_limit(0.06, 1.5, 0.05, 0.05) == 0.06


def test_zsl1_dog_version_velocity_limits():
    adapter, cmd_q = _zsl1_adapter(dog_version='old')
    assert adapter.dog_version == 'old'
    adapter.stand()
    cmd_q.get_nowait()
    adapter.apply_velocity(0.15, 0.0, 0.0)
    assert cmd_q.get_nowait() == ('move', (0.0, 0.0, 0.0))

    adapter, cmd_q = _zsl1_posture_adapter(dog_version='new')
    assert adapter.dog_version == 'new'
    new_limits = (
        (adapter.vx_min, adapter.vx_zero),
        (adapter.vy_min, adapter.vy_zero),
        (adapter.wz_min, adapter.wz_zero),
    )
    adapter.stand()
    cmd_q.get_nowait()
    adapter.apply_velocity(0.15, 0.0, 0.0)
    cmd, args = cmd_q.get_nowait()
    assert cmd == 'move'
    assert abs(args[0] - 0.15) < 1e-9

    adapter, cmd_q = _zsl1_posture_adapter(dog_version='new-v2')
    assert adapter.dog_version == 'new-v2'
    assert (
        (adapter.vx_min, adapter.vx_zero),
        (adapter.vy_min, adapter.vy_zero),
        (adapter.wz_min, adapter.wz_zero),
    ) == new_limits
    adapter.stand()
    cmd_q.get_nowait()
    adapter.apply_velocity(0.15, 0.0, 0.0)
    cmd, args = cmd_q.get_nowait()
    assert cmd == 'move'
    assert abs(args[0] - 0.15) < 1e-9
    adapter.apply_velocity(0.03, 0.0, 0.0)
    assert cmd_q.get_nowait() == ('move', (0.0, 0.0, 0.0))


def test_new_v2_sdk_maps_limited_velocity_to_stick():
    from robot_controller.adapters.zsl1_v2 import (
        CMD_BALANCE_STAND_MODE,
        CMD_SLOW_SPEED,
        CMD_MOVE_MODE,
        CMD_SIT_DOWN,
        CMD_STAND_UP,
        CM_MOVE_MODE,
        CM_SIT_DOWN,
        CM_STAND_UP,
        SL_SLOW,
        ZsibotApp,
    )

    class FakeEx:
        def __init__(self):
            self.cmds = []
            self.remotes = []
            self.mode = CM_STAND_UP
            self.level = 2

        def set_cmd(self, code):
            self.cmds.append(int(code))
            if code == CMD_SIT_DOWN:
                self.mode = CM_SIT_DOWN
            elif code == CMD_STAND_UP:
                self.mode = CM_STAND_UP
            elif code == CMD_MOVE_MODE:
                self.mode = CM_MOVE_MODE
                self.level = 2
            elif code == CMD_SLOW_SPEED:
                self.level = SL_SLOW

        def set_remote(self, joy, button):
            self.remotes.append((tuple(joy), tuple(button)))

        def control_mode(self):
            return self.mode

        def speed_level(self):
            return self.level

    ex = FakeEx()
    app = ZsibotApp(ex, (1.5, 1.0, 3.0), sleep=lambda _s: None, repeat=False)
    assert app.standUp() == 0
    assert ex.cmds == [CMD_STAND_UP, CMD_MOVE_MODE, CMD_SLOW_SPEED]
    app.move(0.6, -0.3, 0.8)
    joy, button = ex.remotes[-1]
    assert joy == (0.6, -0.3, 0.8, 0.0)
    app.move(1.5, -0.5, 1.5)
    joy, button = ex.remotes[-1]
    assert joy == (1.0, -0.5, 1.0, 0.0)
    assert button == (0.0,) * 14
    assert ex.cmds.count(CMD_MOVE_MODE) == 1
    assert ex.cmds.count(CMD_SLOW_SPEED) == 1

    # 105 puts pitch_rate in the second argument; 106 uses the height argument.
    app.attitudeControl(0.0, 0.25, 0.0, 0.4)
    assert ex.cmds[-1] == CMD_BALANCE_STAND_MODE
    joy, button = ex.remotes[-1]
    assert joy == (0.0, 0.0, 0.0, 0.75)
    assert button[7] == 1.0
    assert button[4] == 0.0

    app.attitudeControl(0.0, -0.25, 0.0, -0.4)
    joy, button = ex.remotes[-1]
    assert joy[3] == -0.75
    assert button[4] == 1.0
    assert button[7] == 0.0

    before = len(ex.cmds)
    app.move(0.2, 0.0, 0.0)
    assert ex.cmds[before:] == [CMD_MOVE_MODE, CMD_SLOW_SPEED]

    assert app.lieDown() == 0
    assert ex.cmds[-1] == CMD_SIT_DOWN


def test_new_v2_passive_is_sent_once():
    from robot_controller.adapters.zsl1_v2 import (
        CMD_DISABLE_AUTO_STAND_UP,
        CMD_EMERGENCY_STOP,
        CMD_SIT_DOWN,
        CM_EMERGENCY_STOP,
        CM_SIT_DOWN,
        ZsibotApp,
    )

    class FakeEx:
        def __init__(self):
            self.cmds = []
            self.mode = 0

        def set_cmd(self, code):
            self.cmds.append(int(code))
            if code == CMD_EMERGENCY_STOP:
                self.mode = CM_EMERGENCY_STOP
            elif code == CMD_SIT_DOWN:
                self.mode = CM_SIT_DOWN

        def set_remote(self, joy, button):
            return None

        def control_mode(self):
            return self.mode

    ex = FakeEx()
    app = ZsibotApp(ex, sleep=lambda _s: None, repeat=False)
    assert app.passive() == 0
    assert ex.cmds == [CMD_DISABLE_AUTO_STAND_UP, CMD_EMERGENCY_STOP]
    assert app.move(0.0, 0.0, 0.0) == 0
    assert ex.cmds == [CMD_DISABLE_AUTO_STAND_UP, CMD_EMERGENCY_STOP]
    assert app.lieDown() == 0
    assert ex.cmds[-1] == CMD_SIT_DOWN
    app._passive = True
    assert app.passive() == 0
    assert ex.cmds[-1] == CMD_SIT_DOWN


def test_new_v2_repeats_turn_and_pitch_while_held():
    from robot_controller.adapters.zsl1_v2 import CMD_BALANCE_STAND_MODE, ZsibotApp

    class FakeEx:
        def __init__(self):
            self.cmds = []
            self.remotes = []

        def set_cmd(self, code):
            self.cmds.append(int(code))

        def set_remote(self, joy, button):
            self.remotes.append((tuple(joy), tuple(button)))

    ex = FakeEx()
    app = ZsibotApp(ex, (1.5, 1.0, 3.0), repeat=True)
    try:
        app.move(0.0, 0.0, 0.8)
        time.sleep(0.16)
        yaw = [remote[0][2] for remote in ex.remotes]
        assert len(yaw) >= 3
        assert abs(yaw[-1] - 0.8) < 1e-9

        before = len(ex.remotes)
        app.attitudeControl(0.0, 0.25, 0.0, 0.0)
        time.sleep(0.16)
        assert ex.cmds[-1] == CMD_BALANCE_STAND_MODE
        pitch = [remote[0][3] for remote in ex.remotes[before:]]
        assert len(pitch) >= 3
        assert abs(pitch[-1] - 0.75) < 1e-9
    finally:
        app.close()


def _zsl1_adapter(**overrides):
    node = FakeNode()
    cmd_q = queue.Queue()
    params = {'start_process': False, 'cmd_queue': cmd_q}
    params.update(overrides)
    adapter = Zsl1Adapter(node, params)
    return adapter, cmd_q


def test_zsl1_lie_and_emergency_use_passive():
    adapter, cmd_q = _zsl1_adapter()
    assert adapter.lie().success is True
    assert cmd_q.get_nowait()[0] == 'passive'
    assert adapter.is_standing() is False
    assert adapter.lie().message == 'zsl1: already lying'
    assert _drain(cmd_q) == []
    adapter.stand()
    cmd_q.get_nowait()
    cmd_q.put(('move', (0.0, 0.0, 0.0)))
    lied = adapter.lie()
    assert lied.success is True
    assert lied.message == 'zsl1: passive queued'
    assert cmd_q.get_nowait()[0] == 'passive'
    assert _drain(cmd_q) == []

    adapter, cmd_q = _zsl1_adapter()
    adapter.stand()
    cmd_q.get_nowait()
    assert adapter.emergency().success is True
    assert cmd_q.get_nowait()[0] == 'passive'
    assert _drain(cmd_q) == []
    assert adapter.is_standing() is False


def test_zsl1_remote_103_stays_down():
    adapter, cmd_q = _zsl1_adapter()
    result = adapter.execute_action(ACTION_LIE)
    assert result.success is True
    assert 'lieDown' in result.message
    assert cmd_q.get_nowait()[0] == 'lieDown'
    assert _drain(cmd_q) == []
    assert adapter.is_standing() is False
    adapter.apply_velocity(0.5, 0.0, 0.0)
    assert _drain(cmd_q) == []


def test_zsl1_waypoint_103_queues_pose_not_remote_lie():
    adapter, cmd_q = _zsl1_adapter()
    result = adapter.execute_waypoint_action(ACTION_LIE)
    assert result.success is True
    cmd, args = cmd_q.get_nowait()
    assert cmd == 'waypoint_pose'
    assert args[0] == ACTION_LIE
    assert _drain(cmd_q) == []
    assert adapter._action_busy is False


def test_sdk_worker_waypoint_liedown_stops_retries_then_stands():
    calls = []

    class App:
        def __init__(self):
            self._tries = 0

        def initRobot(self, *_a):
            return None

        def checkConnect(self):
            return True

        def move(self, *args):
            calls.append(('move', args))

        def lieDown(self):
            self._tries += 1
            calls.append(('lieDown', self._tries))
            return 0 if self._tries > 1 else 3

        def passive(self):
            return None

        def standUp(self):
            calls.append(('standUp',))
            return 0

    old_settle = _zsl1_src._LIE_SETTLE_SEC
    old_hold = _zsl1_src._LIE_HOLD_SEC
    _zsl1_src._LIE_SETTLE_SEC = 0.0
    _zsl1_src._LIE_HOLD_SEC = 0.0
    cmd_q = queue.Queue()
    result_q = queue.Queue()
    stop = threading.Event()
    worker = threading.Thread(
        target=_zsl1_src.sdk_worker,
        args=(cmd_q, '127.0.0.1', 1, '127.0.0.1', result_q, 0.0, stop),
        kwargs={'app_factory': App},
        daemon=True,
    )
    worker.start()
    assert result_q.get(timeout=2.0)[0] == 'ready'
    cmd_q.put(('waypoint_pose', (ACTION_LIE, 0.0, 0.0, 0.0, 1, 0.0, 0.0)))
    cmd, rc = result_q.get(timeout=2.0)
    assert cmd == 'waypoint_pose'
    assert rc == 0
    assert calls[0] == ('move', (0.0, 0.0, 0.0))
    lie_calls = [item for item in calls if item[0] == 'lieDown']
    assert lie_calls == [('lieDown', 1), ('lieDown', 2)]
    assert calls[-1] == ('standUp',)
    stop.set()
    cmd_q.put(('exit', ()))
    worker.join(timeout=2.0)
    _zsl1_src._LIE_SETTLE_SEC = old_settle
    _zsl1_src._LIE_HOLD_SEC = old_hold


def test_sdk_worker_waypoint_turn_waits_out_running_before_stand():
    """107 move() leaves the running gait; standUp is refused until it settles."""
    calls = []

    class App:
        def initRobot(self, *_a):
            return None

        def checkConnect(self):
            return True

        def move(self, *args):
            calls.append(('move', args))

        def standUp(self):
            calls.append(('standUp',))
            # First call is still "running"; the retry after settle succeeds.
            return 0 if calls.count(('standUp',)) > 1 else 1

    old_settle = _zsl1_src._LIE_SETTLE_SEC
    _zsl1_src._LIE_SETTLE_SEC = 0.0
    cmd_q = queue.Queue()
    result_q = queue.Queue()
    stop = threading.Event()
    worker = threading.Thread(
        target=_zsl1_src.sdk_worker,
        args=(cmd_q, '127.0.0.1', 1, '127.0.0.1', result_q, 0.0, stop),
        kwargs={'app_factory': App},
        daemon=True,
    )
    worker.start()
    assert result_q.get(timeout=2.0)[0] == 'ready'
    cmd_q.put(('waypoint_pose', (ACTION_TURN_LEFT, 0.0, 0.0, 0.0, 1, 0.4, 0.0)))
    cmd, rc = result_q.get(timeout=2.0)
    assert cmd == 'waypoint_pose'
    assert rc == 0
    assert calls == [
        ('move', (0.0, 0.0, 0.0)),
        ('move', (0.0, 0.0, 0.4)),
        ('move', (0.0, 0.0, 0.0)),
        ('standUp',),
        ('standUp',),
    ]
    stop.set()
    cmd_q.put(('exit', ()))
    worker.join(timeout=2.0)
    _zsl1_src._LIE_SETTLE_SEC = old_settle


def test_zsl1_action_101_reuses_lie_passive():
    adapter, cmd_q = _zsl1_adapter()
    result = adapter.execute_action(ACTION_DAMP)
    assert result.success is True
    assert cmd_q.get_nowait()[0] == 'passive'
    assert adapter.is_standing() is False


def _zsl1_posture_adapter(**overrides):
    node = FakeNode()
    cmd_q = queue.Queue()
    params = {
        'start_process': False,
        'cmd_queue': cmd_q,
        'pitch_rate': 0.25,
        'height_rate': 0.15,
        'turn_rate': 0.6,
        'posture_duration': 10.0,
        'posture_repeats': 2,
        'turn_duration': 10.0,
    }
    params.update(overrides)
    return Zsl1Adapter(node, params), cmd_q


def _drain(cmd_q):
    items = []
    while True:
        try:
            items.append(cmd_q.get_nowait())
        except queue.Empty:
            return items


def test_zsl1_pitch_and_height_use_attitude_cycle():
    adapter, cmd_q = _zsl1_posture_adapter()
    adapter.stand()
    _drain(cmd_q)
    assert adapter.execute_action(ACTION_PITCH).success is True
    cmd, args = cmd_q.get_nowait()
    assert cmd == 'attitude_cycle'
    assert args == (0.0, 0.25, 0.0, 0.0, 10.0, 2)
    adapter.apply_velocity(0.5, 0.0, 0.0)
    assert _drain(cmd_q) == []

    adapter, cmd_q = _zsl1_posture_adapter(posture_repeats=3)
    assert adapter.execute_action(ACTION_HEIGHT).success is True
    assert cmd_q.get_nowait() == ('attitude_cycle', (0.0, 0.0, 0.0, 0.15, 10.0, 3))


def test_zsl1_idle_zero_velocity_is_not_resent():
    adapter, cmd_q = _zsl1_posture_adapter()
    adapter.stand()
    _drain(cmd_q)
    adapter.apply_velocity(0.0, 0.0, 0.0)
    assert cmd_q.get_nowait() == ('move', (0.0, 0.0, 0.0))
    adapter.apply_velocity(0.0, 0.0, 0.0)
    assert _drain(cmd_q) == []
    adapter.apply_velocity(0.4, 0.0, 0.0)
    assert cmd_q.get_nowait()[0] == 'move'
    adapter.apply_velocity(0.0, 0.0, 0.0)
    assert cmd_q.get_nowait() == ('move', (0.0, 0.0, 0.0))


def test_zsl1_attitude_cycle_then_walk():
    adapter, cmd_q = _zsl1_posture_adapter(posture_duration=0.05, posture_repeats=1)
    adapter.stand()
    _drain(cmd_q)
    assert adapter.execute_action(ACTION_PITCH).success is True
    assert cmd_q.get_nowait()[0] == 'attitude_cycle'
    deadline = time.monotonic() + 3.0
    while adapter._action_busy and time.monotonic() < deadline:
        time.sleep(0.02)
    assert adapter._action_busy is False
    adapter.apply_velocity(0.0, 0.0, 0.0)
    cmd, args = cmd_q.get_nowait()
    assert cmd == 'move'
    assert args == (0.0, 0.0, 0.0)
    adapter.apply_velocity(0.4, 0.0, 0.0)
    cmd, args = cmd_q.get_nowait()
    assert cmd == 'move'
    assert args[0] == 0.4


def test_zsl1_turn_uses_stepping_move():
    adapter, cmd_q = _zsl1_posture_adapter()
    adapter.stand()
    _drain(cmd_q)
    assert adapter.execute_action(ACTION_TURN_LEFT).success is True
    cmd, args = cmd_q.get_nowait()
    assert cmd == 'turn_pulse'
    assert args[0] == 0.6
    assert args[1] == 10.0
    adapter.apply_velocity(0.4, 0.0, 0.0)
    assert _drain(cmd_q) == []

    adapter, cmd_q = _zsl1_posture_adapter(turn_duration=0.05)
    adapter.stand()
    _drain(cmd_q)
    assert adapter.execute_action(ACTION_TURN_RIGHT).success is True
    cmd, args = cmd_q.get_nowait()
    assert cmd == 'turn_pulse'
    assert args[0] == -0.6
    time.sleep(0.12)
    adapter.apply_velocity(0.0, 0.0, 0.0)
    cmd, args = cmd_q.get_nowait()
    assert cmd == 'move'
    assert args == (0.0, 0.0, 0.0)


def test_zsl1_stand_cancels_posture_hold():
    adapter, cmd_q = _zsl1_posture_adapter(posture_duration=0.05, posture_repeats=1)
    assert adapter.execute_action(ACTION_PITCH).success is True
    cmd_q.get_nowait()
    assert adapter.stand().success is True
    assert cmd_q.get_nowait()[0] == 'standUp'
    time.sleep(0.12)
    adapter.apply_velocity(0.0, 0.0, 0.0)
    cmd, args = cmd_q.get_nowait()
    assert cmd == 'move'
    assert args == (0.0, 0.0, 0.0)


def test_zsl1_shutdown_signals_worker_exit():
    finished = threading.Event()

    def worker(
        cmd_queue,
        local_ip,
        local_port,
        dog_ip,
        result_queue=None,
        stand_settle=0,
        stop_event=None,
        **_kwargs,
    ):
        while stop_event is None or not stop_event.is_set():
            time.sleep(0.01)
        finished.set()

    adapter = Zsl1Adapter(
        FakeNode(),
        {
            'start_process': True,
            'cmd_queue': queue.Queue(),
            'max_restart': 0,
        },
        worker=worker,
    )
    adapter.shutdown()
    assert finished.wait(1.0)
    assert adapter.proc is None


def test_ysc_lite3_does_not_toggle_when_already_standing():
    udp = FakeUdp(states=[6, 6])
    node = FakeNode(udp=udp)
    adapter = YscLite3Adapter(node, {'mode': 'auto'}, udp=udp)
    result = adapter.stand()
    assert result.success is True
    assert CMD_STAND not in udp.simple


def test_ysc_lite3_sends_stand_when_lying():
    udp = FakeUdp(states=[1, 6])
    node = FakeNode(udp=udp)
    adapter = YscLite3Adapter(
        node, {'mode': 'auto', 'stand_timeout': 0.2, 'state_wait': 0.05}, udp=udp,
    )
    result = adapter.stand()
    assert result.success is True
    assert CMD_STAND in udp.simple


def test_ysc_lite3_skips_stand_without_state_packet():
    udp = FakeUdp(states=[])
    node = FakeNode(udp=udp)
    adapter = YscLite3Adapter(node, {'mode': 'auto', 'state_wait': 0.05}, udp=udp)
    result = adapter.stand()
    assert result.success is False
    assert CMD_STAND not in udp.simple


def test_ysc_lite3_is_standing_uses_cache_and_does_not_block():
    udp = FakeUdp(states=[9])
    node = FakeNode(udp=udp)
    adapter = YscLite3Adapter(node, {'mode': 'auto', 'state_wait': 2.0}, udp=udp)
    assert adapter.is_standing() is False
    assert udp.states == [9]
    udp.available = [6]
    assert adapter.is_standing() is True
    assert adapter.stand().success is True
    assert CMD_STAND not in udp.simple


def test_ysc_lite3_prepare_from_lie_sends_toggle():
    udp = FakeUdp(available=[1], states=[6])
    node = FakeNode(udp=udp)
    adapter = YscLite3Adapter(
        node, {'mode': 'auto', 'stand_timeout': 0.2, 'state_wait': 0.05}, udp=udp,
    )
    result = adapter.prepare()
    assert result.success is True
    assert CMD_STAND in udp.simple


def test_ysc_lite3_skips_lie_without_state_packet():
    udp = FakeUdp(states=[])
    node = FakeNode(udp=udp)
    adapter = YscLite3Adapter(node, {'mode': 'auto', 'state_wait': 0.05}, udp=udp)
    result = adapter.lie()
    assert result.success is False
    assert CMD_STAND not in udp.simple


def test_ysc_lite3_action_101_lies_when_standing():
    udp = FakeUdp(states=[6, 1])
    node = FakeNode(udp=udp)
    adapter = YscLite3Adapter(
        node, {'mode': 'auto', 'stand_timeout': 0.2, 'state_wait': 0.05}, udp=udp,
    )
    result = adapter.execute_action(ACTION_DAMP)
    assert result.success is True
    assert CMD_STAND in udp.simple


def test_ysc_lite3_action_101_skips_when_already_lying():
    udp = FakeUdp(states=[1])
    node = FakeNode(udp=udp)
    adapter = YscLite3Adapter(node, {'mode': 'auto', 'state_wait': 0.05}, udp=udp)
    result = adapter.execute_action(ACTION_DAMP)
    assert result.success is True
    assert CMD_STAND not in udp.simple


def test_ysc_lite3_action_103_sends_crawl_gait():
    udp = FakeUdp()
    node = FakeNode(udp=udp)
    adapter = YscLite3Adapter(node, {'mode': 'auto'}, udp=udp)
    result = adapter.execute_action(ACTION_LIE)
    assert result.success is True
    assert CMD_MANUAL_MODE in udp.simple
    assert CMD_MOVE_MODE in udp.simple
    assert CMD_CRAWL in udp.simple
    assert CMD_STAND not in udp.simple
    assert not hasattr(adapter, '_status_sock')
    assert node.subscriptions


def test_unitree_posture_codes_and_b2_wave_unsupported():
    posted = []
    node = FakeNode()
    adapter = UnitreeAdapter(node, {'unitree_model': 'b2'}, publish=posted.append)
    assert adapter.owns_velocity is False
    assert adapter.stand().success is True
    assert posted[-1] == POSTURE_STAND
    assert adapter.lie().success is True
    assert posted[-1] == POSTURE_LIE
    wave = adapter.execute_action(ACTION_WAVE)
    assert wave.success is False
    go2 = UnitreeAdapter(node, {'unitree_model': 'go2w'}, publish=posted.append)
    assert go2.execute_action(ACTION_WAVE).success is True
    assert posted[-1] == POSTURE_HELLO
    assert go2.execute_action(ACTION_STAND).success is True
    assert go2.execute_action(ACTION_LIE).success is True
    go2_standalone = UnitreeAdapter(node, {'unitree_model': 'go2'}, publish=posted.append)
    assert go2_standalone.execute_action(ACTION_DAMP).success is True
    assert posted[-1] == POSTURE_LIE
    b2_damp = UnitreeAdapter(node, {'unitree_model': 'b2'}, publish=posted.append)
    assert b2_damp.execute_action(ACTION_DAMP).success is True
    assert posted[-1] == POSTURE_DAMP
