import numpy as np

from scene_memory.navigation import grid_nav
from scene_memory.nodes.goal_server_node import GOAL_ABORTED, GOAL_SUCCEEDED, nav_outcome, outcome_reply


def test_nav_outcome():
    goal = [5.0, 0.0, 0.0]
    assert nav_outcome(GOAL_SUCCEEDED, [4.8, 0.0, 0.0], goal) == "arrived"
    assert nav_outcome(GOAL_SUCCEEDED, [3.9, 0.0, 0.0], goal) == "arrived_near"    # stall watchdog accepted it
    assert nav_outcome(GOAL_ABORTED, [4.0, 0.0, 0.0], goal) == "arrived_near"
    assert nav_outcome(GOAL_ABORTED, [2.0, 0.0, 0.0], goal) == "blocked"
    assert nav_outcome(GOAL_ABORTED, None, goal) == "blocked"
    assert nav_outcome(5, [4.8, 0.0, 0.0], goal) == "other"
    assert "停在了附近" in outcome_reply("arrived_near", "椅子", True, 1.2)
    assert "another chair" in outcome_reply("alternative", "chair", False)


def test_approach_pose_avoids_the_blocked_spot():
    traj = np.array([[x, 0.0, 0.5] for x in np.arange(0.0, 6.01, 0.1)])
    box = np.array([[3.0 + dx, 1.5 + dy, z] for dx in np.linspace(-0.2, 0.2, 5) for dy in np.linspace(-0.2, 0.2, 5)
                    for z in (0.3, 0.6)])
    grid = grid_nav.NavGrid(box, traj, sensor_height=0.5)
    obj = {"center": [3.0, 1.5, 0.45], "size": [0.4, 0.4, 0.3]}
    first = grid_nav.approach_pose(grid, obj, [0.0, 0.0])
    assert first is not None and 0.4 <= first["standoff_m"] <= 1.2
    second = grid_nav.approach_pose(grid, obj, [0.0, 0.0], avoid=[first["xy"]])
    assert second is not None and np.linalg.norm(np.subtract(second["xy"], first["xy"])) >= 0.8
    assert 0.4 <= grid_nav.footprint_dist(second["xy"], obj) <= 1.2
    wp = grid_nav.approach_waypoint(grid, second)
    assert wp["position"][:2] == second["xy"] and abs(wp["orientation"][3] ** 2 + wp["orientation"][2] ** 2 - 1) < 1e-5
