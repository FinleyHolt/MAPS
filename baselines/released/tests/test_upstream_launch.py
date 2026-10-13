"""The launcher's run-time adaptations, against stand-in upstream modules.

``upstream_eval/launch.py`` runs under the upstream interpreter, so it is
loaded here by path, the way that interpreter sees it.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from collections import namedtuple
from pathlib import Path
from typing import NamedTuple

import pytest

LAUNCH = Path(__file__).resolve().parents[1] / "upstream_eval" / "launch.py"


@pytest.fixture(scope="module")
def launch():
    spec = importlib.util.spec_from_file_location("released_launch", LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launch_imports_only_the_standard_library():
    import ast

    tree = ast.parse(LAUNCH.read_text())
    top = {
        alias.name.split(".")[0]
        for node in tree.body if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module.split(".")[0] for node in tree.body
         if isinstance(node, ast.ImportFrom) and node.module}
    assert len(top) >= 5  # vacuity floor: the scan saw the imports
    assert top <= set(sys.stdlib_module_names) | {"__future__"}


def _mapdata():
    class Bounds(NamedTuple):
        x_min: float
        y_min: float
        x_max: float
        y_max: float

    m = types.ModuleType("mapdata")
    m.Bounds = Bounds
    m.GROUND_LEVEL = {"birmingham_block_0": 16.0}
    m.MAP_BOUNDS = {"birmingham_block_0": Bounds(0, 0, 1, 1)}
    return m


def test_register_blocks_extends_the_dicts_in_place(launch):
    m = _mapdata()
    bound_elsewhere = m.MAP_BOUNDS  # what ``from mapdata import MAP_BOUNDS`` holds
    launch.register_blocks(m, {"GROUND_LEVEL": {"calcity_block_4": 0.0},
                               "MAP_BOUNDS": {"calcity_block_4": [10.0, 20.0, 410.0, 420.0]}})
    assert bound_elsewhere["calcity_block_4"].y_max == 420.0
    assert isinstance(bound_elsewhere["calcity_block_4"], m.Bounds)
    assert m.GROUND_LEVEL == {"birmingham_block_0": 16.0, "calcity_block_4": 0.0}


def test_filter_splits(launch):
    loader = launch.filter_splits(lambda split, *a, **k: [split, a, k], {"test_unseen"})
    assert loader("test_unseen", "all", 50) == ["test_unseen", ("all", 50), {}]
    assert loader("val_seen", "all", 50) == []


def test_zero_cache_reads_empty_for_any_caption(launch):
    cache = launch.ZeroCache((1, 240, 240))
    arr = cache["calcity_block_3-the red roof"][0]
    assert arr.shape == (240, 240) and float(arr.max()) == 0.0


def test_bert_path_is_remapped_and_nothing_else(launch):
    calls = []

    class Model:
        @classmethod
        def from_pretrained(cls, name, *args, **kwargs):
            calls.append((cls, name, args, kwargs))
            return name

    launch.patch_from_pretrained([Model], "bert-base-uncased")
    assert Model.from_pretrained(launch.HETT_BERT_PATH) == "bert-base-uncased"
    assert Model.from_pretrained("other/model", revision="x") == "other/model"
    assert calls[-1] == (Model, "other/model", (), {"revision": "x"})


def test_alias_package(launch, tmp_path, monkeypatch):
    pkg = tmp_path / "upstreampkg"
    pkg.mkdir()
    (pkg / "space.py").write_text("VALUE = 3\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        launch.alias_package("direction", "upstreampkg", ["space"])
        from direction.space import VALUE  # noqa: F401  # the import HETT's dataset module makes

        assert VALUE == 3
    finally:
        for name in ("direction", "direction.space", "upstreampkg", "upstreampkg.space"):
            sys.modules.pop(name, None)


def test_capture_record(launch):
    Pose4D = namedtuple("Pose4D", "x y z yaw")
    record = launch.capture_record(
        "hett", "test_unseen",
        {("calcity_block_2", 9_999_999, 0): [Pose4D(1, 2, 50, 0), Pose4D(3, 4, 50, 0)]},
        errors=[("calcity_block_5", 9_999_999, 0)], metrics={"sr": 0.0},
    )
    assert record["episodes"] == [{"key": ["calcity_block_2", 9_999_999, 0],
                                   "trajectory": [[1.0, 2.0], [3.0, 4.0]]}]
    assert record["errors"] == [["calcity_block_5", 9_999_999, 0]]
