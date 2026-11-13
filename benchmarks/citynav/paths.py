"""The CityNav bench as a value, built from an environment mapping.

:func:`citynav_bench` reads, each overridable:

- ``SEARCH_CITYNAV_DATA`` (else ``CITYNAV_ROOT``, else ``~/datasets/citynav-arxiv``):
  the ``citynav-arxiv`` checkout (``data/citynav``, ``data/cityrefer``, ``data/rgbd_upstream``).
- ``SEARCH_CITYNAV_TILES`` (else ``<repo>/tiles/citynav_2m_osmv3``): the tile bake.
- ``SEARCH_CITYNAV_SPLITS`` (else ``$CITYNAV_ROOT/data/citynav``): the split JSONs
  alone, e.g. HETT's refined ``val_seen``/``val_unseen``/``test_unseen``
  (github crotonyl/HETT, arXiv 2512.14222), whose ``objects.json`` is
  byte-identical, so catalog and tiles stay put.
- ``SEARCH_CITYNAV_RGBD_DIR`` (else ``data/rgbd_upstream``): upstream's 0.1 m/px
  nadir rasters, from ``data_prep/rasterize_citynav_native.sh``.

:func:`selected_bench` is the shared runner's: ``SEARCH_BENCHMARK_NAME``
(default ``citynav``) picks the constructor, which is how an attempt child
learns the bench its driver flies.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from benchmarks import REPO_ROOT
from benchmarks.bench import Bench, env_path
from benchmarks.calcity.paths import calcity_bench

# The tile bake (data_prep/bake_tiles.sh): one tags-only OSM ``prior`` list per
# block, masks clipped to the block's imagery. An SR measured against another
# bake is not comparable.
DEFAULT_TILE_BAKE = "tiles/citynav_2m_osmv3"

#: The episode engine CityNav flies, and Cal City with it.
ENGINE_MODULE = "benchmarks.citynav.engine"

SPLIT_FILES: dict[str, str] = {
    # No test_seen split exists; easy/medium/hard are trajectory-length tiers.
    "train_seen": "citynav_train_seen.json",
    "val_seen": "citynav_val_seen.json",
    "val_seen_easy": "citynav_val_seen_easy.json",
    "val_seen_medium": "citynav_val_seen_medium.json",
    "val_seen_hard": "citynav_val_seen_hard.json",
    "val_unseen": "citynav_val_unseen.json",
    "val_unseen_easy": "citynav_val_unseen_easy.json",
    "val_unseen_medium": "citynav_val_unseen_medium.json",
    "val_unseen_hard": "citynav_val_unseen_hard.json",
    "test_unseen": "citynav_test_unseen.json",
    "test_unseen_easy": "citynav_test_unseen_easy.json",
    "test_unseen_medium": "citynav_test_unseen_medium.json",
    "test_unseen_hard": "citynav_test_unseen_hard.json",
}


def _assert_available(bench: Bench, split: str | None = None) -> None:
    """Raise with the fetch recipe when the splits, ``split``'s file or ``objects.json``
    are missing."""
    required = [bench.splits_dir, *([bench.split_path(split)] if split else []), bench.objects_json]
    missing = [p for p in required if not p.exists()]
    if missing:
        recipe = (
            f"CityNav data not found under {bench.root}.\n"
            "Fetch it, from the repo root:\n"
            f"  CITYNAV_ROOT={bench.root} benchmarks/citynav/data_prep/setup.sh --sensaturban <data_release.zip>\n"
            "Override the root via CITYNAV_ROOT=<path>, or the episode\n"
            "splits alone via SEARCH_CITYNAV_SPLITS=<dir>.\n\n"
            f"Missing: {', '.join(str(p) for p in missing)}"
        )
        raise FileNotFoundError(recipe)


def citynav_bench(environ: Mapping[str, str]) -> Bench:
    """The CityNav bench ``environ`` describes."""
    root = env_path(
        environ,
        "SEARCH_CITYNAV_DATA",
        "CITYNAV_ROOT",
        default=Path.home() / "datasets" / "citynav-arxiv",
    )
    data = root / "data"
    return Bench(
        name="citynav",
        root=root,
        splits_dir=env_path(environ, "SEARCH_CITYNAV_SPLITS", default=data / "citynav"),
        objects_json=data / "cityrefer" / "objects.json",
        rgbd_dir=env_path(environ, "SEARCH_CITYNAV_RGBD_DIR", default=data / "rgbd_upstream"),
        tiles_dir=env_path(environ, "SEARCH_CITYNAV_TILES", default=REPO_ROOT / DEFAULT_TILE_BAKE),
        split_files=SPLIT_FILES,
        success_criterion="ring",
        default_split="val_seen",
        engine_module=ENGINE_MODULE,
        availability=_assert_available,
        environ=dict(environ),
    )


def selected_bench(environ: Mapping[str, str]) -> Bench:
    """The bench ``SEARCH_BENCHMARK_NAME`` names in ``environ``, CityNav when unset."""
    name = environ.get("SEARCH_BENCHMARK_NAME", "").strip() or "citynav"
    return {"citynav": citynav_bench, "calcity": calcity_bench}[name](environ)
