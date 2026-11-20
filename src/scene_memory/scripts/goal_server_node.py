#!/usr/bin/env python3
"""Run ``scene_memory.nodes.goal_server_node`` from the source tree (no colcon install needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_memory.nodes.goal_server_node import main  # noqa: E402

if __name__ == "__main__":
    main()
