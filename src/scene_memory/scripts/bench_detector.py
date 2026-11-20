#!/usr/bin/env python3
"""Run ``scene_memory.perception.bench_detector`` from the source tree (no colcon install needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_memory.perception.bench_detector import main  # noqa: E402

if __name__ == "__main__":
    main()
