"""Per-name map folders: data/maps/<name>/<name>_* (map_prefix resolution, stvln.sh map_migrate)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

WS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WS / "src" / "planning" / "launch"))
import stack_common  # noqa: E402


@pytest.fixture
def maps(tmp_path, monkeypatch):
    monkeypatch.setattr(stack_common, "MAPS_DIR", tmp_path / "maps")
    monkeypatch.setattr(stack_common, "QINAV_MAPS_DIR", tmp_path / "qinav")
    return tmp_path / "maps"


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


def test_folder_map_first(maps):
    touch(maps / "demo" / "demo_map.pcd")
    touch(maps / "demo_map.pcd")
    assert stack_common.map_prefix("demo") == maps / "demo" / "demo"
    assert stack_common.saved_map_prefix("demo") == maps / "demo" / "demo"


def test_flat_and_qinav_fallback(maps):
    touch(maps / "old_map.pcd")
    touch(maps.parent / "qinav" / "q_map.pcd")
    assert stack_common.map_prefix("old") == maps / "old"
    assert stack_common.map_prefix("q") == maps.parent / "qinav" / "q"
    with pytest.raises(RuntimeError):
        stack_common.map_prefix("missing")


def test_map_migrate_keeps_other_maps(tmp_path):
    maps = tmp_path / "maps"
    for f in ("office_map.pcd", "office_traversable.pcd", "office_semantic.json", "office_scene/topo_graph.json",
              "office_loop8_map.pcd", "office_loop8_scene/topo_graph.json"):
        touch(maps / f)
    env = dict(os.environ, STVLN_MAPS=str(maps))
    out = subprocess.run(["bash", str(WS / "src/scripts/stvln.sh"), "map_migrate", "office"], env=env,
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert sorted(p.name for p in (maps / "office").iterdir()) == [
        "office_map.pcd", "office_scene", "office_semantic.json", "office_traversable.pcd"]
    assert (maps / "office_loop8_map.pcd").is_file() and (maps / "office_loop8_scene").is_dir()
    out = subprocess.run(["bash", str(WS / "src/scripts/stvln.sh"), "map_migrate", "--all"], env=env,
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert (maps / "office_loop8" / "office_loop8_scene" / "topo_graph.json").is_file()
    assert sorted(p.name for p in maps.iterdir()) == ["office", "office_loop8"]
