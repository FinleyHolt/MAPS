"""``calcity_bench`` describes Cal City for the shared engine, in its process
and, through the environment it carries, in the children that process spawns."""

import os
import subprocess
import sys

from benchmarks.calcity.paths import calcity_bench
from benchmarks.citynav.paths import citynav_bench, selected_bench
from maps.perception.ortho.ground_level import GROUND_LEVEL

_CHILD = (
    "import os; from benchmarks.citynav.paths import selected_bench; b = selected_bench(os.environ); "
    "print(b.name, b.success_criterion, b.default_split, b.tiles_dir, b.rgbd_dir, b.split_path('calcity'))"
)


def _environ(tmp_path) -> dict[str, str]:
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("SEARCH_", "CITYNAV_", "CALCITY_"))
    }
    env["SEARCH_CALCITY_TILES"] = str(tmp_path / "tiles")
    env["SEARCH_CALCITY_DATA"] = str(tmp_path / "data")
    return env


def test_the_bench_selects_calcity_for_the_process_and_its_children(tmp_path):
    bench = calcity_bench(_environ(tmp_path))
    assert (bench.name, bench.success_criterion, bench.default_split) == (
        "calcity",
        "shape",
        "calcity",
    )
    assert bench.tiles_dir == tmp_path / "tiles" and bench.root == tmp_path / "data"
    assert bench.objects_json == tmp_path / "data" / "data" / "cityrefer" / "objects.json"
    assert GROUND_LEVEL["calcity"] == 0.0
    # What a run records and a child inherits: the bench by name and its roots under the shared runner's names.
    assert {
        k: v
        for k, v in bench.environ.items()
        if k.startswith(("SEARCH_BENCHMARK", "SEARCH_CITYNAV_"))
    } == {
        "SEARCH_BENCHMARK_NAME": "calcity",
        "SEARCH_CITYNAV_DATA": str(bench.root),
        "SEARCH_CITYNAV_SPLITS": str(bench.splits_dir),
        "SEARCH_CITYNAV_TILES": str(bench.tiles_dir),
        "SEARCH_CITYNAV_RGBD_DIR": str(bench.rgbd_dir),
    }
    child = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        check=True,
        env=dict(bench.environ),
    ).stdout.split()
    assert child == [
        "calcity",
        "shape",
        "calcity",
        str(bench.tiles_dir),
        str(bench.rgbd_dir),
        str(bench.split_path("calcity")),
    ]


def test_an_environment_that_names_no_bench_selects_citynav(tmp_path):
    bench = selected_bench(_environ(tmp_path))
    assert (bench.name, bench.success_criterion, bench.default_split) == (
        "citynav",
        "ring",
        "val_seen",
    )


def test_a_citynav_bench_built_first_stands_beside_calcity(tmp_path):
    """Cal City is described after CityNav in one process, and neither moves the other."""
    environ = _environ(tmp_path)
    citynav = citynav_bench(environ)
    before = (
        citynav.tiles_dir,
        citynav.splits_dir,
        citynav.success_criterion,
        dict(citynav.environ),
    )
    calcity = calcity_bench(environ)
    assert calcity.tiles_dir == tmp_path / "tiles" != citynav.tiles_dir
    assert (
        citynav.tiles_dir,
        citynav.splits_dir,
        citynav.success_criterion,
        dict(citynav.environ),
    ) == before
    assert (
        "SEARCH_BENCHMARK_NAME" not in citynav.environ
        and "SEARCH_CITYNAV_TILES" not in citynav.environ
    )
