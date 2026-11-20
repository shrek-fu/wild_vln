from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional

from robot_controller.adapters.base import AdapterResult, RobotAdapter
from robot_controller.command_ids import (
    ACTION_DAMP,
    ACTION_HEIGHT,
    ACTION_LIE,
    ACTION_PITCH,
    ACTION_STAND,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    ACTION_WAVE,
)

# Python binding: attitudeControl(yaw_vel, roll_vel, pitch_vel, height_vel)
_ATTITUDE_RATE_MAX = 0.5
_HEIGHT_RATE_MAX = 0.5
_STAND_OK = 0
_QUEUE_WAIT_SEC = 0.05
_POSTURE_RECOVER_SEC = 2.0
# Gait must leave "running" before lieDown or standUp, or the SDK refuses
# ("robot is runing, can't transition to ...").
_LIE_SETTLE_SEC = 1.0
_LIE_HOLD_SEC = 3.0
_DOG_VERSIONS = ('old', 'new', 'new-v2')
# value_limit(min_vel, zero_limit) per axis: vx, vy, wz
# new-v2 uses SDK 1.1.5, with the same deadband as new.
_VEL_LIMITS = {
    'old': ((0.4, 0.2), (0.4, 0.2), (0.35, 0.1)),
    'new': ((0.05, 0.05), (0.1, 0.05), (0.02, 0.02)),
    'new-v2': ((0.05, 0.05), (0.1, 0.05), (0.02, 0.02)),
}


def value_limit(vel: float, max_vel: float, min_vel: float, zero_limit: float) -> float:
    vel = max(-max_vel, min(vel, max_vel))
    if abs(vel) < zero_limit:
        return 0.0
    if abs(vel) < min_vel:
        return min_vel * (1 if vel > 0 else -1)
    return vel


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, float(value)))


def _waypoint_pose(app, args, stop_event, stand_settle: float):
    """Stop, run one waypoint pose, then stand. None means the worker is stopping."""
    (
        control_id,
        pitch_rate,
        height_rate,
        posture_duration,
        posture_repeats,
        turn_wz,
        turn_duration,
    ) = args
    control_id = int(control_id)
    app.move(0.0, 0.0, 0.0)
    if _sleep_or_stop(stop_event, _LIE_SETTLE_SEC):
        return None

    action_rc = 0
    if control_id == ACTION_DAMP:
        app.passive()
        if _sleep_or_stop(stop_event, _LIE_HOLD_SEC):
            return None
    elif control_id == ACTION_STAND:
        rc = app.standUp()
        if _sleep_or_stop(stop_event, stand_settle):
            return None
        return 0 if _sdk_rc_ok(rc) else -1
    elif control_id == ACTION_LIE:
        rc = app.lieDown()
        _sdk_log(f'waypoint lieDown rc={rc}')
        if not _sdk_rc_ok(rc):
            if _sleep_or_stop(stop_event, _LIE_SETTLE_SEC):
                return None
            rc = app.lieDown()
            _sdk_log(f'waypoint lieDown retry rc={rc}')
        if _sdk_rc_ok(rc):
            if _sleep_or_stop(stop_event, _LIE_HOLD_SEC):
                return None
            action_rc = 0
        else:
            action_rc = int(rc)
    elif control_id == ACTION_WAVE:
        app.shakeHand()
        if _sleep_or_stop(stop_event, 3.0):
            return None
    elif control_id in (ACTION_PITCH, ACTION_HEIGHT):
        if control_id == ACTION_PITCH:
            yaw, roll, pitch, height = 0.0, float(pitch_rate), 0.0, 0.0
        else:
            yaw, roll, pitch, height = 0.0, 0.0, 0.0, float(height_rate)
        for _ in range(max(1, int(posture_repeats))):
            app.attitudeControl(yaw, roll, pitch, height)
            if _sleep_or_stop(stop_event, posture_duration):
                return None
            app.attitudeControl(-yaw, -roll, -pitch, -height)
            if _sleep_or_stop(stop_event, posture_duration):
                return None
        if _sleep_or_stop(stop_event, _POSTURE_RECOVER_SEC):
            return None
    elif control_id in (ACTION_TURN_LEFT, ACTION_TURN_RIGHT):
        app.move(0.0, 0.0, float(turn_wz))
        if _sleep_or_stop(stop_event, turn_duration):
            return None
        app.move(0.0, 0.0, 0.0)
        # move() keeps the running gait; standUp is rejected until it settles.
        if _sleep_or_stop(stop_event, _LIE_SETTLE_SEC):
            return None
    else:
        action_rc = -4

    rc = app.standUp()
    _sdk_log(f'waypoint standUp rc={rc}')
    if not _sdk_rc_ok(rc):
        if _sleep_or_stop(stop_event, _LIE_SETTLE_SEC):
            return None
        rc = app.standUp()
        _sdk_log(f'waypoint standUp retry rc={rc}')
    if _sleep_or_stop(stop_event, stand_settle):
        return None
    if not _sdk_rc_ok(rc):
        return -1
    return action_rc


def _sdk_rc_ok(rc) -> bool:
    if rc is None:
        return True
    try:
        return int(rc) == 0
    except (TypeError, ValueError):
        return False


def _reply(result_queue, command: str, rc: int = 0) -> None:
    if result_queue is None:
        return
    result_queue.put((command, int(rc)))


def _sdk_log(message: str) -> None:
    print(f'[zsl1-sdk] {message}', flush=True)
    logging.info(message)


def _is_connected(app) -> bool:
    checker = getattr(app, 'checkConnect', None)
    if not callable(checker):
        return True
    try:
        return bool(checker())
    except Exception:
        return False


def _sleep_or_stop(stop_event, seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, float(seconds))
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            return True
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    return False


class _DisconnectedApp:
    """Stand-in when the 1.1.5 binding cannot be created. standUp rc=-2 means offline."""

    def checkConnect(self) -> bool:
        return False

    def move(self, *_args):
        return 0

    def attitudeControl(self, *_args):
        return 0

    def standUp(self) -> int:
        return -2

    def lieDown(self) -> int:
        return -2

    def passive(self) -> int:
        return -2

    def shakeHand(self) -> int:
        return -2


def _open_sdk_app(
    dog_version: str,
    local_ip: str,
    local_port: int,
    dog_ip: str,
    send_port: int,
    recv_port: int,
    vel_max,
):
    if str(dog_version).strip().lower() == 'new-v2':
        try:
            from robot_controller.adapters.zsl1_v2 import open_zsibot

            return open_zsibot(dog_ip, send_port, recv_port, vel_max)
        except Exception as exc:
            _sdk_log(
                f'new-v2 SDK init failed: {exc}. '
                'Build and source package zsibot_sdk_py (SDK 1.1.5).'
            )
            return _DisconnectedApp()
    try:
        import mc_sdk_zsl_1_py as mc_sdk_zsl
    except ImportError:
        import mc_sdk_py as mc_sdk_zsl  # type: ignore

    app = mc_sdk_zsl.HighLevel()
    app.initRobot(local_ip, int(local_port), dog_ip)
    return app


def _release_sdk(app) -> None:
    if app is None:
        return
    try:
        app.passive()
    except Exception:
        pass
    for name in ('close', 'unInit', 'uninit', 'release'):
        closer = getattr(app, name, None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
            break


def sdk_worker(
    cmd_queue,
    local_ip: str,
    local_port: int,
    dog_ip: str,
    result_queue=None,
    stand_settle: float = 6.0,
    stop_event=None,
    app_factory=None,
    dog_version: str = 'old',
    send_port: int = 8081,
    recv_port: int = 8080,
    vel_max=(1.5, 1.0, 3.0),
    motion_blocked=None,
) -> None:
    if app_factory is None:
        app = _open_sdk_app(
            dog_version,
            local_ip,
            local_port,
            dog_ip,
            int(send_port),
            int(recv_port),
            vel_max,
        )
    else:
        app = app_factory()
    connected = _is_connected(app)
    _sdk_log(
        f'init done version={dog_version} connected={connected} '
        f'local={local_ip}:{local_port} dog={dog_ip} '
        f'sdk_port={send_port}/{recv_port}'
    )
    if not connected:
        _sdk_log(
            f'SDK not connected. If bind said Address already in use, '
            f'kill the leftover process on port {local_port}'
        )
    _reply(result_queue, 'ready', 1 if connected else 0)

    while True:
        if stop_event is not None and stop_event.is_set():
            _release_sdk(app)
            return
        try:
            cmd, args = cmd_queue.get(timeout=_QUEUE_WAIT_SEC)
        except queue.Empty:
            continue
        if cmd == 'move':
            if motion_blocked is not None and motion_blocked.is_set():
                continue
            app.move(*args)
        elif cmd == 'attitude':
            app.attitudeControl(*args)
        elif cmd == 'attitude_cycle':
            yaw, roll, pitch, height, duration, repeats = args
            interrupted = False
            for _ in range(max(1, int(repeats))):
                app.attitudeControl(yaw, roll, pitch, height)
                if _sleep_or_stop(stop_event, duration):
                    interrupted = True
                    break
                app.attitudeControl(-yaw, -roll, -pitch, -height)
                if _sleep_or_stop(stop_event, duration):
                    interrupted = True
                    break
            if not interrupted:
                app.standUp()
                _sleep_or_stop(stop_event, _POSTURE_RECOVER_SEC)
        elif cmd == 'turn_pulse':
            wz, duration = args
            app.move(0.0, 0.0, float(wz))
            _sleep_or_stop(stop_event, duration)
            app.move(0.0, 0.0, 0.0)
        elif cmd == 'standUp':
            rc = app.standUp()
            _sdk_log(f'standUp rc={rc}')
            if _sleep_or_stop(stop_event, stand_settle):
                _release_sdk(app)
                return
            _reply(result_queue, 'standUp', rc if rc is not None else -1)
        elif cmd == 'waypoint_pose':
            rc = _waypoint_pose(app, args, stop_event, stand_settle)
            if rc is None:
                _release_sdk(app)
                return
            _reply(result_queue, 'waypoint_pose', int(rc))
        elif cmd == 'lieDown':
            app.lieDown()
            _sleep_or_stop(stop_event, 3)
        elif cmd == 'passive':
            rc = 0
            try:
                result = app.passive()
                if result is not None:
                    rc = int(result)
            except Exception:
                rc = 1
            finally:
                # Stay connected while the body finishes. 1012 kills this process
                # as soon as the service returns.
                _sleep_or_stop(stop_event, _LIE_HOLD_SEC)
                _reply(result_queue, 'passive', rc)
        elif cmd == 'shakeHand':
            app.shakeHand()
            _sleep_or_stop(stop_event, 3)
        elif cmd == 'exit':
            _release_sdk(app)
            return


class Zsl1Adapter(RobotAdapter):
    """ZSL1 HighLevel SDK on one in-process thread, matching the official demo."""

    def __init__(self, node: Any, params: Dict[str, Any], worker: Optional[Callable] = None):
        super().__init__(node, params)
        self.local_ip = str(params.get('local_ip', '192.168.168.99'))
        self.local_port = int(params.get('local_port', 43988))
        self.dog_ip = str(params.get('dog_ip', '192.168.168.168'))
        self.max_restart = int(params.get('max_restart', 5))
        self.vx_max = float(params.get('max_linear_x', 1.5))
        self.vy_max = float(params.get('max_linear_y', 1.0))
        self.wz_max = float(params.get('max_angular_z', 3.0))
        version = str(params.get('dog_version', 'new')).strip().lower()
        if version not in _DOG_VERSIONS:
            version = 'new'
        self.dog_version = version
        self.vx_min, self.vx_zero = _VEL_LIMITS[version][0]
        self.vy_min, self.vy_zero = _VEL_LIMITS[version][1]
        self.wz_min, self.wz_zero = _VEL_LIMITS[version][2]
        self.send_port = int(params.get('sdk_send_port', 8081))
        self.recv_port = int(params.get('sdk_recv_port', 8080))
        self.pitch_rate = _clamp(params.get('pitch_rate', 0.25), _ATTITUDE_RATE_MAX)
        self.height_rate = _clamp(params.get('height_rate', 0.40), _HEIGHT_RATE_MAX)
        self.posture_duration = max(0.0, float(params.get('posture_duration', 1.5)))
        self.posture_repeats = max(1, int(params.get('posture_repeats', 2)))
        self.turn_rate = abs(float(params.get('turn_rate', 0.8)))
        self.turn_duration = max(0.0, float(params.get('turn_duration', 2.0)))
        self.stand_settle = max(0.0, float(params.get('stand_settle', 3.0)))
        self._worker = worker or sdk_worker
        self._standing = False
        self._lying = False
        self._stand_on_next_motion = False
        self._motion_blocked = threading.Event()
        self._lock = threading.Lock()
        self._action_busy = False
        self._last_move_zero = False
        self._action_timer: Optional[threading.Timer] = None
        self._pulse_gen = 0
        self.cmd_queue = params.get('cmd_queue') or queue.Queue()
        self.result_queue = params.get('result_queue')
        self.proc = None
        self.restart_count = 0
        self._stop_event = threading.Event()
        self._monitor = None
        if bool(params.get('start_process', True)):
            if self.result_queue is None:
                self.result_queue = queue.Queue()
            self._start_process()
            self._monitor = threading.Thread(target=self._monitor_proc, daemon=True)
            self._monitor.start()

    def _start_process(self) -> None:
        if self.proc is not None and self.proc.is_alive():
            return
        self.proc = threading.Thread(
            target=self._worker,
            args=(
                self.cmd_queue,
                self.local_ip,
                self.local_port,
                self.dog_ip,
                self.result_queue,
                self.stand_settle,
                self._stop_event,
            ),
            kwargs={
                'dog_version': self.dog_version,
                'send_port': self.send_port,
                'recv_port': self.recv_port,
                'vel_max': (self.vx_max, self.vy_max, self.wz_max),
                'motion_blocked': self._motion_blocked,
            },
            daemon=True,
            name='zsl1-sdk',
        )
        self.proc.start()
        self.restart_count += 1
        self.log().info(
            f'ZSL1 SDK thread started ({self.restart_count}) version={self.dog_version}'
        )

    def _monitor_proc(self) -> None:
        while not self._stop_event.is_set():
            if self.proc is not None and not self.proc.is_alive():
                if self.restart_count < self.max_restart:
                    self.log().warn('ZSL1 SDK thread died, restarting')
                    self._standing = False
                    self._clear_motion_overlay()
                    self._start_process()
                else:
                    self.log().error('ZSL1 SDK thread exceeded max restarts')
                    break
            self._stop_event.wait(2.0)

    def run_sdk_command(self, command: str, args=()) -> bool:
        if bool(self.params.get('start_process', True)):
            if self.proc is None or not self.proc.is_alive():
                self.log().error(f'ZSL1 SDK unavailable, skip {command}')
                return False
        self.cmd_queue.put((command, args))
        return True

    def _wait_result(self, command: str, timeout: float) -> Optional[int]:
        if self.result_queue is None:
            return _STAND_OK
        deadline = time.monotonic() + max(0.1, float(timeout))
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                cmd, rc = self.result_queue.get(timeout=min(0.2, remaining))
            except queue.Empty:
                continue
            if cmd == command:
                return int(rc)
        return None

    def _clear_motion_overlay(self) -> None:
        with self._lock:
            self._pulse_gen += 1
            timer = self._action_timer
            self._action_timer = None
            self._action_busy = False
            self._last_move_zero = False
        if timer is not None:
            timer.cancel()

    def _begin_pulse(self, duration: float) -> int:
        with self._lock:
            self._pulse_gen += 1
            token = self._pulse_gen
            timer = self._action_timer
            self._action_timer = None
            self._action_busy = True
            self._last_move_zero = False
        if timer is not None:
            timer.cancel()

        def finish() -> None:
            with self._lock:
                if token != self._pulse_gen:
                    return
                self._action_busy = False
                self._action_timer = None

        pulse = threading.Timer(max(0.0, float(duration)), finish)
        pulse.daemon = True
        with self._lock:
            if token != self._pulse_gen:
                return token
            self._action_timer = pulse
        pulse.start()
        return token

    def is_standing(self) -> bool:
        return self._standing

    def stand(self) -> AdapterResult:
        self._clear_motion_overlay()
        self._standing = False
        self._lying = False
        self._motion_blocked.clear()
        if not self.run_sdk_command('standUp'):
            return AdapterResult(False, 'zsl1: SDK unavailable')
        if not bool(self.params.get('start_process', True)):
            self._standing = True
            self._stand_on_next_motion = False
            return AdapterResult(True, 'zsl1: standUp queued')
        timeout = self.stand_settle + 5.0
        rc = self._wait_result('standUp', timeout)
        if rc is None:
            return AdapterResult(False, 'zsl1: standUp timed out')
        if rc == -2:
            return AdapterResult(
                False,
                f'zsl1: SDK not connected ({self.local_ip}:{self.local_port}). '
                'port busy or leftover SDK process',
            )
        if rc != _STAND_OK:
            return AdapterResult(False, f'zsl1: standUp failed rc=0x{rc:x}')
        self._standing = True
        self._stand_on_next_motion = False
        return AdapterResult(True, 'zsl1: standUp ok')

    def lie(self) -> AdapterResult:
        if self._lying:
            return AdapterResult(True, 'zsl1: already lying')
        # Drop queued sticks first. A zero move in front of passive enters move
        # mode again and can outlast the 1012 kill.
        self._standing = False
        self._lying = True
        self._motion_blocked.set()
        self._clear_motion_overlay()
        self._discard_motion_commands()
        if not self.run_sdk_command('passive'):
            self._lying = False
            self._motion_blocked.clear()
            return AdapterResult(False, 'zsl1: SDK unavailable')
        if not bool(self.params.get('start_process', True)):
            return AdapterResult(True, 'zsl1: passive queued')
        rc = self._wait_result('passive', _LIE_HOLD_SEC + 8.0)
        if rc is None:
            return AdapterResult(False, 'zsl1: passive timed out')
        if rc != 0:
            return AdapterResult(False, f'zsl1: passive failed rc={rc}')
        return AdapterResult(True, 'zsl1: passive ok')

    def _discard_motion_commands(self) -> None:
        kept = []
        while True:
            try:
                item = self.cmd_queue.get_nowait()
            except queue.Empty:
                break
            cmd = item[0] if item else ''
            if cmd in ('move', 'turn_pulse', 'attitude', 'attitude_cycle'):
                continue
            kept.append(item)
        for item in kept:
            self.cmd_queue.put(item)

    def apply_velocity(self, linear_x: float, linear_y: float, angular_z: float) -> None:
        vx = value_limit(linear_x, self.vx_max, self.vx_min, self.vx_zero)
        vy = value_limit(linear_y, self.vy_max, self.vy_min, self.vy_zero)
        wz = value_limit(angular_z, self.wz_max, self.wz_min, self.wz_zero)
        with self._lock:
            busy = self._action_busy
            resume = self._stand_on_next_motion and not self._standing
        if busy:
            return
        if resume:
            if vx == 0.0 and vy == 0.0 and wz == 0.0:
                return
            result = self.stand()
            if not result.success:
                self.log().error(f'zsl1: stand before motion failed: {result.message}')
                return
        if not self._standing:
            return

        with self._lock:
            if self._action_busy:
                return
            if vx == 0.0 and vy == 0.0 and wz == 0.0:
                if self._last_move_zero:
                    return
                self._last_move_zero = True
            else:
                self._last_move_zero = False
        self.run_sdk_command('move', (vx, vy, wz))

    def stop(self) -> AdapterResult:
        self._clear_motion_overlay()
        if not self._standing:
            return AdapterResult(True, 'zsl1: already stopped')
        ok = self.run_sdk_command('move', (0.0, 0.0, 0.0))
        return AdapterResult(ok, 'zsl1: stopped' if ok else 'zsl1: SDK unavailable')

    def _queue_attitude_cycle(
        self,
        yaw: float,
        roll: float,
        pitch: float,
        height: float,
        label: str,
    ) -> AdapterResult:
        duration = self.posture_duration
        repeats = self.posture_repeats
        total = duration * 2.0 * repeats + _POSTURE_RECOVER_SEC
        self._begin_pulse(total)
        ok = self.run_sdk_command(
            'attitude_cycle',
            (
                _clamp(yaw, _ATTITUDE_RATE_MAX),
                _clamp(roll, _ATTITUDE_RATE_MAX),
                _clamp(pitch, _ATTITUDE_RATE_MAX),
                _clamp(height, _HEIGHT_RATE_MAX),
                duration,
                repeats,
            ),
        )
        if not ok:
            self._clear_motion_overlay()
            return AdapterResult(False, 'zsl1: SDK unavailable')
        return AdapterResult(True, f'zsl1: {label} queued repeats={repeats}')

    def _queue_turn_pulse(self, yaw_sign: float, label: str) -> AdapterResult:
        wz = value_limit(yaw_sign * self.turn_rate, self.wz_max, 0.02, 0.02)
        duration = self.turn_duration
        self._begin_pulse(duration)
        ok = self.run_sdk_command('turn_pulse', (wz, duration))
        if not ok:
            self._clear_motion_overlay()
            return AdapterResult(False, 'zsl1: SDK unavailable')
        return AdapterResult(True, f'zsl1: {label} queued')

    def execute_action(self, control_id: int) -> AdapterResult:
        """1043 remote action. Leaves the dog in the requested pose."""
        if control_id == ACTION_DAMP:
            return self.lie()
        if control_id == ACTION_STAND:
            return self.stand()
        if control_id == ACTION_LIE:
            self.stop()
            if not self.run_sdk_command('lieDown'):
                return AdapterResult(False, 'zsl1: SDK unavailable')
            self._standing = False
            self._lying = False
            return AdapterResult(True, 'zsl1: lieDown queued')
        if control_id == ACTION_WAVE:
            self._clear_motion_overlay()
            ok = self.run_sdk_command('shakeHand')
            return AdapterResult(ok, 'zsl1: shakeHand' if ok else 'zsl1: SDK unavailable')
        if control_id == ACTION_PITCH:
            return self._queue_attitude_cycle(0.0, self.pitch_rate, 0.0, 0.0, 'pitch')
        if control_id == ACTION_HEIGHT:
            return self._queue_attitude_cycle(0.0, 0.0, 0.0, self.height_rate, 'height')
        if control_id == ACTION_TURN_LEFT:
            return self._queue_turn_pulse(1.0, 'turn left')
        if control_id == ACTION_TURN_RIGHT:
            return self._queue_turn_pulse(-1.0, 'turn right')
        return AdapterResult(False, f'zsl1: unsupported control_id {control_id}')

    def execute_waypoint_action(self, control_id: int) -> AdapterResult:
        """Stop, run the pose, then stand. 1043 does not use this."""
        control_id = int(control_id)
        if control_id not in range(101, 109):
            return AdapterResult(False, f'zsl1: unsupported control_id {control_id}')
        sign = 0.0
        if control_id == ACTION_TURN_LEFT:
            sign = 1.0
        elif control_id == ACTION_TURN_RIGHT:
            sign = -1.0
        turn_wz = value_limit(sign * self.turn_rate, self.wz_max, 0.02, 0.02) if sign else 0.0
        payload = (
            control_id,
            float(self.pitch_rate),
            float(self.height_rate),
            float(self.posture_duration),
            int(self.posture_repeats),
            float(turn_wz),
            float(self.turn_duration),
        )
        self._clear_motion_overlay()
        with self._lock:
            self._action_busy = True
        try:
            if not self.run_sdk_command('waypoint_pose', payload):
                return AdapterResult(False, 'zsl1: SDK unavailable')
            if not bool(self.params.get('start_process', True)):
                return AdapterResult(True, 'zsl1: waypoint_pose queued')
            pose = self.posture_duration * 2.0 * self.posture_repeats + _POSTURE_RECOVER_SEC
            timeout = (
                _LIE_SETTLE_SEC * 3.0
                + _LIE_HOLD_SEC
                + max(pose, self.turn_duration, 3.0)
                + self.stand_settle
                + 5.0
            )
            rc = self._wait_result('waypoint_pose', timeout)
            if rc is None:
                self._stand_on_next_motion = True
                return AdapterResult(False, 'zsl1: waypoint action timed out')
            if rc < 0:
                self._standing = False
                self._stand_on_next_motion = True
                return AdapterResult(False, f'zsl1: waypoint stand failed rc={rc}')
            self._standing = True
            self._lying = False
            self._stand_on_next_motion = False
            if rc == 0:
                return AdapterResult(True, f'zsl1: waypoint {control_id} ok')
            return AdapterResult(False, f'zsl1: waypoint {control_id} failed rc={rc}')
        finally:
            with self._lock:
                self._action_busy = False

    def shutdown(self) -> None:
        self._stop_event.set()
        self._clear_motion_overlay()
        if self._monitor is not None and self._monitor.is_alive():
            self._monitor.join(timeout=1.0)
        if self.proc is not None and self.proc.is_alive():
            try:
                self.cmd_queue.put(('exit', ()))
            except Exception:
                pass
            self.proc.join(timeout=2.0)
        self.proc = None
