"""Zsibot SDK 1.1.5 backend for dog_version new-v2.

vx, vy and yaw are written to the stick as the commanded m/s and rad/s.
The SDK clamps a stick whose absolute value is greater than 1 into [-1, 1].
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Callable, Optional, Sequence

# CmdCode values from zsibot_define.h
CMD_EMERGENCY_STOP = 0x5A
CMD_STAND_UP = 0x7A
CMD_SIT_DOWN = 0x6A
CMD_MOVE_MODE = 0x8A
CMD_SLOW_SPEED = 0xAE
CMD_BALANCE_STAND_MODE = 0x9A
CMD_GREET = 0x04
CMD_ENTER_LAB_MODE = 0xC1
CMD_SDK_CONTROL_RIGHT = 0xB3
CMD_DISABLE_AUTO_STAND_UP = 0xC8

# ControlMode values
CM_STAND_UP = 1
CM_SIT_DOWN = 2
CM_EMERGENCY_STOP = 4
CM_MOVE_MODE = 5
CM_BALANCE_STAND_MODE = 6

_STAND_MODES = frozenset((CM_STAND_UP, CM_MOVE_MODE, CM_BALANCE_STAND_MODE))
_ATTITUDE_SCALE = 0.5
# 105 pitch_rate 0.25 used to be half stick. A little more reach, still inside [-1, 1].
_PITCH_GAIN = 1.5
_POLL_SEC = 0.05
_POSE_TIMEOUT_SEC = 3.0
_SPEED_SETTLE_SEC = 0.2
_SLOW_WAIT_SEC = 1.0
_SLOW_RETRY_SEC = 0.5
# SpeedLevel::SL_SLOW. Speed gear is independent of move mode.
SL_SLOW = 1
_CONTROL_RIGHT_WAIT_SEC = 0.2
_LAB_WAIT_SEC = 0.2
# SDK 1.1.5 drops a stick command unless it is repeated. HighLevel latched it.
_HOLD_SEC = 0.05


def _unit(value: float) -> float:
    """Pass the speed through, and fold |value| > 1 into [-1, 1]."""
    return max(-1.0, min(1.0, float(value)))


def _stick(value: float, limit: float) -> float:
    if limit <= 0.0:
        return 0.0
    scaled = float(value) / float(limit)
    return max(-1.0, min(1.0, scaled))


class ZsibotApp:
    """Same method names the ZSL1 worker already calls on HighLevel."""

    def __init__(
        self,
        executor,
        vel_max: Sequence[float] = (1.5, 1.0, 3.0),
        sleep: Optional[Callable[[float], None]] = None,
        repeat: bool = True,
    ):
        self._ex = executor
        self._sleep = sleep or time.sleep
        self._balance = False
        self._move_mode = False
        self._slow = False
        self._slow_sent_at = 0.0
        self._passive = False
        self._hold_lock = threading.Lock()
        self._held: Optional[tuple] = None
        self._alive = True
        self._hold_thread = None
        if repeat:
            self._hold_thread = threading.Thread(
                target=self._hold_loop,
                daemon=True,
                name='zsl1-sdk-hold',
            )
            self._hold_thread.start()

    def acquire(self) -> bool:
        ready = bool(self._ex.wait_ready(5000))
        self._ex.set_cmd(CMD_SDK_CONTROL_RIGHT)
        self._sleep(_CONTROL_RIGHT_WAIT_SEC)
        self._ex.set_cmd(CMD_DISABLE_AUTO_STAND_UP)
        if self._ex is None:
            return False
        return ready and bool(self._ex.is_connected())

    def checkConnect(self) -> bool:
        if self._ex is None:
            return False
        try:
            return bool(self._ex.is_connected())
        except Exception:
            return False

    def initRobot(self, *_args) -> None:
        return None

    def move(self, vx: float, vy: float, wz: float) -> int:
        # A later zero-stick left in the queue must not enter move mode again.
        if self._passive:
            return 0
        if self._balance or not self._move_mode:
            self._begin_move()
        else:
            self._ensure_slow()
        # On this dog, stick [1] strafes and [2] yaws. Header text has those two swapped.
        joy = (_unit(vx), _unit(vy), _unit(wz), 0.0)
        self._hold_remote(joy, (0.0,) * 14)
        return 0

    def standUp(self) -> int:
        self._release_stick()
        self._ex.set_cmd(CMD_STAND_UP)
        ok = self._wait_mode(_STAND_MODES, _POSE_TIMEOUT_SEC)
        self._passive = False
        self._begin_move()
        return 0 if ok else 1

    def lieDown(self) -> int:
        """匍匐. On this dog CMD_SIT_DOWN holds the body up on the forearms."""
        self._release_stick()
        self._slow = False
        self._move_mode = False
        self._balance = False
        self._passive = False
        self._ex.set_cmd(CMD_SIT_DOWN)
        return 0 if self._wait_mode(frozenset((CM_SIT_DOWN,)), _POSE_TIMEOUT_SEC) else 1

    def passive(self) -> int:
        """趴下. CMD_EMERGENCY_STOP drops and disables the legs.

        1012 and emergency both use this. CMD_SIT_DOWN is the separate 匍匐 pose.
        """
        self._release_stick()
        self._slow = False
        self._move_mode = False
        self._balance = False
        if self._passive or self._ex is None:
            return 0
        # Latch before the commands so a move that arrives during the send cannot stand.
        self._passive = True
        for _ in range(3):
            self._release_stick()
            self._sleep(0.05)
        self._sleep(0.8)
        self._ex.set_cmd(CMD_DISABLE_AUTO_STAND_UP)
        self._sleep(0.05)
        self._ex.set_cmd(CMD_EMERGENCY_STOP)
        if not callable(getattr(self._ex, 'control_mode', None)):
            return 0
        try:
            int(self._ex.control_mode())
        except Exception:
            return 0
        return 0 if self._wait_mode(frozenset((CM_EMERGENCY_STOP,)), _POSE_TIMEOUT_SEC) else 1

    def shakeHand(self) -> int:
        self._ex.set_cmd(CMD_ENTER_LAB_MODE)
        self._sleep(_LAB_WAIT_SEC)
        self._ex.set_cmd(CMD_GREET)
        return 0

    def attitudeControl(self, _yaw: float, body_pitch: float, _pitch: float, height: float) -> int:
        """Match the HighLevel call this worker already makes.

        105 passes pitch_rate in the second slot; that slot pitches the body.
        106 passes height_rate in the fourth slot. SDK 1.1.5 uses the balance
        stand pitch stick and the height buttons for those two axes.
        """
        if not self._balance:
            self._release_stick()
            self._ex.set_cmd(CMD_BALANCE_STAND_MODE)
            self._balance = True
            self._move_mode = False
            self._slow = False
        joy = (
            0.0,
            0.0,
            0.0,
            max(-1.0, min(1.0, _stick(body_pitch, _ATTITUDE_SCALE) * _PITCH_GAIN)),
        )
        button = [0.0] * 14
        if height > 0.02:
            button[7] = 1.0
        elif height < -0.02:
            button[4] = 1.0
        self._hold_remote(joy, button)
        return 0

    def close(self) -> None:
        self._alive = False
        self._stop_hold()
        thread = self._hold_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.3)
        ex = self._ex
        self._ex = None
        if ex is None:
            return
        try:
            ex.set_remote((0.0, 0.0, 0.0, 0.0), (0.0,) * 14)
        except Exception:
            pass
        closer = getattr(ex, 'close', None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass

    def _begin_move(self) -> None:
        self._stop_hold()
        self._ex.set_cmd(CMD_MOVE_MODE)
        # Speed gear is a separate command. Wait until move mode is active before setting it.
        self._wait_mode(frozenset((CM_MOVE_MODE,)), _POSE_TIMEOUT_SEC)
        self._sleep(_SPEED_SETTLE_SEC)
        self._balance = False
        self._move_mode = True
        self._slow = False
        self._push_slow(wait=True)

    def _push_slow(self, wait: bool) -> None:
        if self._ex is None:
            return
        self._ex.set_cmd(CMD_SLOW_SPEED)
        self._slow_sent_at = time.monotonic()
        if self._read_speed_level() == SL_SLOW:
            self._mark_slow()
            return
        if not wait or self._read_speed_level() < 0:
            return
        deadline = time.monotonic() + _SLOW_WAIT_SEC
        while time.monotonic() < deadline:
            self._sleep(_POLL_SEC)
            if self._read_speed_level() == SL_SLOW:
                self._mark_slow()
                return
            self._ex.set_cmd(CMD_SLOW_SPEED)
            self._slow_sent_at = time.monotonic()
        print('[zsl1-sdk] slow speed not confirmed, retrying with motion', flush=True)

    def _ensure_slow(self) -> None:
        if self._ex is None:
            return
        level = self._read_speed_level()
        if level == SL_SLOW:
            self._mark_slow()
            return
        if level > 0:
            self._slow = False
        if self._slow:
            return
        now = time.monotonic()
        if now - self._slow_sent_at < _SLOW_RETRY_SEC:
            return
        self._ex.set_cmd(CMD_SLOW_SPEED)
        self._slow_sent_at = now

    def _mark_slow(self) -> None:
        if self._slow:
            return
        self._slow = True
        print('[zsl1-sdk] speed level slow', flush=True)

    def _read_speed_level(self) -> int:
        reader = getattr(self._ex, 'speed_level', None)
        if not callable(reader):
            return -1
        try:
            return int(reader())
        except Exception:
            return -1

    def _hold_remote(self, joy, button) -> None:
        packet = (tuple(float(value) for value in joy), tuple(float(value) for value in button))
        with self._hold_lock:
            self._held = packet
            ex = self._ex
        if ex is not None:
            ex.set_remote(packet[0], packet[1])

    def _stop_hold(self) -> None:
        with self._hold_lock:
            self._held = None

    def _release_stick(self) -> None:
        self._stop_hold()
        ex = self._ex
        if ex is None:
            return
        try:
            ex.set_remote((0.0, 0.0, 0.0, 0.0), (0.0,) * 14)
        except Exception:
            pass

    def _hold_loop(self) -> None:
        while self._alive:
            with self._hold_lock:
                packet = self._held
                ex = self._ex
            if packet is not None and ex is not None:
                try:
                    ex.set_remote(packet[0], packet[1])
                except Exception:
                    pass
            time.sleep(_HOLD_SEC)

    def _wait_mode(self, modes: frozenset, timeout: float) -> bool:
        if not callable(getattr(self._ex, 'control_mode', None)):
            return False
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            try:
                mode = int(self._ex.control_mode())
            except Exception:
                mode = 0
            if mode in modes:
                return True
            if time.monotonic() >= deadline:
                return False
            self._sleep(_POLL_SEC)


def _bridge_path() -> str:
    from ament_index_python.packages import get_package_prefix

    path = os.path.join(
        get_package_prefix('zsibot_sdk_py'),
        'lib',
        'zsibot_sdk_py',
        'zsibot_sdk_bridge',
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f'zsibot bridge not installed: {path}')
    return path


def _drain_stderr(stream) -> None:
    try:
        for line in stream:
            text = line.strip()
            if text:
                print(f'[zsl1-sdk] {text}', flush=True)
    except Exception:
        return


class BridgeExecutor:
    """Talks to zsibot_sdk_bridge. libzsibot.a is not PIC, so it stays a process."""

    def __init__(self, dog_ip: str, send_port: int, recv_port: int):
        self._io = threading.Lock()
        self._proc = subprocess.Popen(
            [_bridge_path(), str(dog_ip), str(int(send_port)), str(int(recv_port))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self._proc.stdout.readline() if self._proc.stdout is not None else ''
        if not ready.startswith('OK'):
            detail = ''
            if self._proc.stderr is not None:
                try:
                    detail = self._proc.stderr.read().strip()
                except Exception:
                    detail = ''
            self._proc.kill()
            raise RuntimeError(detail or 'zsibot bridge failed to start')
        if self._proc.stderr is not None:
            threading.Thread(
                target=_drain_stderr,
                args=(self._proc.stderr,),
                daemon=True,
                name='zsl1-sdk-stderr',
            ).start()

    def wait_ready(self, timeout_ms: int = 5000) -> bool:
        return self._request(f'WAIT {int(timeout_ms)}').split()[-1] == '1'

    def is_connected(self) -> bool:
        return self._request('CONNECTED').split()[-1] == '1'

    def set_cmd(self, code: int) -> None:
        self._request(f'CMD {int(code)}')

    def set_remote(self, joy, button) -> None:
        values = [float(value) for value in tuple(joy) + tuple(button)]
        payload = ' '.join(str(value) for value in values)
        self._request(f'REMOTE {payload}')

    def control_mode(self) -> int:
        return int(self._request('MODE').split()[-1])

    def speed_level(self) -> int:
        return int(self._request('SPEED').split()[-1])

    def close(self) -> None:
        with self._io:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._proc = None
                return
            try:
                if proc.stdin is not None:
                    proc.stdin.write('EXIT\n')
                    proc.stdin.flush()
            except Exception:
                proc.kill()
                self._proc = None
                return
        try:
            proc.wait(timeout=1.0)
        except Exception:
            proc.kill()
        self._proc = None

    def _request(self, line: str) -> str:
        with self._io:
            proc = self._proc
            if proc is None or proc.poll() is not None or proc.stdin is None or proc.stdout is None:
                raise RuntimeError('zsibot bridge is not running')
            proc.stdin.write(line + '\n')
            proc.stdin.flush()
            reply = proc.stdout.readline().strip()
            if not reply:
                raise RuntimeError('zsibot bridge closed')
            if reply.startswith('ERR'):
                raise RuntimeError(reply)
            return reply


def open_zsibot(
    dog_ip: str,
    send_port: int,
    recv_port: int,
    vel_max: Sequence[float],
) -> ZsibotApp:
    executor = BridgeExecutor(str(dog_ip), int(send_port), int(recv_port))
    app = ZsibotApp(executor, vel_max)
    try:
        app.acquire()
    except Exception:
        executor.close()
        raise
    return app
