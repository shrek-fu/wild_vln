import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prompt_depth import ViewGeom, build_prompt_depth, knn_pooling, project_sparse_depth  # noqa: E402


def _view(w=70, h=70) -> ViewGeom:
    K = np.array([[35.0, 0, w / 2 - 0.5], [0, 35.0, h / 2 - 0.5], [0, 0, 1.0]])
    # optical = level for the test
    return ViewGeom("front", w, h, K, np.eye(4))


def test_project_and_knn_fills_holes():
    view = _view()
    # a wall of points 2 m in front
    xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, 20), np.linspace(-0.5, 0.5, 20))
    xyz = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 2.0)])
    sparse = project_sparse_depth(xyz, view, scale=1.0)
    assert (sparse > 0).sum() > 50
    # punch a hole
    sparse[30:40, 30:40] = 0
    filled = knn_pooling(sparse, pool_size=12, k=4)
    assert (filled[32:38, 32:38] > 1.5).mean() > 0.8


def test_build_prompt_upscales():
    view = _view(640, 640)
    xs = np.linspace(-1, 1, 40)
    ys = np.linspace(-1, 1, 40)
    xx, yy = np.meshgrid(xs, ys)
    xyz = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 3.0)])
    prompt, stats = build_prompt_depth(xyz, view, project_scale=1 / 7, knn_size=10, knn_k=2,
                                       out_size=(640, 640))
    assert prompt.shape == (640, 640)
    assert stats["hits"] > 10
    assert abs(float(np.median(prompt[prompt > 0])) - 3.0) < 0.3
