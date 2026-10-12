"""``benchmarks.calcity.use`` selects Cal City for the shared engine, in its
process and the children that process spawns."""

import json
import os
import subprocess
import sys

_SELECTED = """
import json, subprocess, sys
from benchmarks.calcity import paths as cc, use
use(); use()
from benchmarks.citynav import paths
from maps.perception.ortho.ground_level import GROUND_LEVEL
child = subprocess.run([sys.executable, "-c", "from benchmarks.citynav import paths; print(paths.TILES_DIR)"],
                       capture_output=True, text=True, check=True).stdout.strip()
print(json.dumps({
    "roots": [str(getattr(paths, a)) == str(getattr(cc, b)) for a, b in (
        ("TILES_DIR", "TILES_DIR"), ("SPLITS_DIR", "SPLITS_DIR"), ("OBJECTS_JSON", "OBJECTS_JSON"),
        ("CITYNAV_NATIVE_RGBD_DIR", "CALCITY_NATIVE_RGBD_DIR"))],
    "split": paths.split_path("calcity") == cc.split_path("calcity"),
    "criterion": [paths.SUCCESS_CRITERION, paths.DEFAULT_SPLIT],
    "ground": GROUND_LEVEL["calcity"], "child_tiles": child == str(cc.TILES_DIR)}))
"""
_DEFAULT = "from benchmarks.citynav import paths; print(paths.SUCCESS_CRITERION, paths.DEFAULT_SPLIT)"
_LATE = "from benchmarks.citynav import paths; from benchmarks.calcity import use; use()"


def _run(code, tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SEARCH_", "CITYNAV_", "CALCITY_"))}
    env["SEARCH_CALCITY_TILES"] = str(tmp_path / "tiles")
    env["SEARCH_CALCITY_DATA"] = str(tmp_path / "data")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


def test_use_selects_calcity_for_the_process_and_its_children(tmp_path):
    out = _run(_SELECTED, tmp_path)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == {"roots": [True] * 4, "split": True, "criterion": ["shape", "calcity"],
                                      "ground": 0.0, "child_tiles": True}
    assert _run(_DEFAULT, tmp_path).stdout.split() == ["ring", "val_seen"]
    late = _run(_LATE, tmp_path)
    assert late.returncode != 0 and "call benchmarks.calcity.use() first" in late.stderr
