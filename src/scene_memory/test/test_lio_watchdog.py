import math

from scene_memory.nodes.lio_watchdog_node import DivergenceDetector


def _walk(det, t0, x0, n, speed, dt=0.1):
    tripped = []
    for i in range(n):
        t = t0 + i * dt
        tripped.append(det.update(t, (x0 + speed * (t - t0), 0.0, 0.0)))
    return tripped


def test_normal_motion_never_trips():
    det = DivergenceDetector(max_speed=5.0, trips=3)
    assert not any(_walk(det, 0.0, 0.0, 100, 1.5))
    assert det.last_good_t is not None


def test_single_glitch_is_tolerated():
    det = DivergenceDetector(max_speed=5.0, trips=3)
    _walk(det, 0.0, 0.0, 10, 1.0)
    assert not det.update(1.0, (5.0, 0.0, 0.0))  # 41 m/s once
    assert not any(_walk(det, 1.1, 5.0, 10, 1.0))


def test_runaway_trips_after_n_poses():
    det = DivergenceDetector(max_speed=5.0, trips=3)
    _walk(det, 0.0, 0.0, 10, 1.0)
    good = det.last_good_t
    assert _walk(det, 1.0, 100.0, 3, 1000.0) == [False, False, True]
    assert det.last_good_t == good


def test_non_finite_trips():
    det = DivergenceDetector(max_speed=5.0, trips=2)
    det.update(0.0, (0.0, 0.0, 0.0))
    assert not det.update(0.1, (math.nan, 0.0, 0.0))
    assert det.update(0.2, (0.0, math.inf, 0.0))
