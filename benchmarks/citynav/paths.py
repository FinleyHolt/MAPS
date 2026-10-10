"""Filesystem paths for the CityNav integration.

Resolves everything via env vars so the same code runs on a laptop and on
HPC without edits:

- ``SEARCH_CITYNAV_DATA`` (preferred) or ``CITYNAV_ROOT`` for the dataset
  root (the ``citynav-arxiv`` checkout: ``data/citynav/*.json``,
  ``data/cityrefer/*``, ``data/rgbd/*``, ``raw_ply/*``).
- ``SEARCH_CITYNAV_TILES`` for the prebaked feature-graph tile sidecars
  (defaults to ``<repo>/tiles/citynav_2m_osmv3/``, the OSM-only bake that
  is the tile source of truth).
- ``SEARCH_CITYNAV_SPLITS`` for the episode split JSONs alone (defaults
  to ``$CITYNAV_ROOT/data/citynav``). Set this to the HETT refined
  splits dir for an opt-in eval-only A/B; the CityRefer catalog and
  every baked tile stay pointed at the canonical tree (HETT's
  ``objects.json`` is byte-identical, so no re-bake). Refined ships only
  the three base eval splits (``val_seen``/``val_unseen``/``test_unseen``);
  the easy/medium/hard tiers and ``train_seen`` are original-only.

Both defaults assume a laptop layout (``~/datasets/citynav-arxiv``,
``<repo>/tiles/citynav_2m_osmv3``); override either env var on HPC.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def _env_value(*names: str, default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


CITYNAV_ROOT = (
    Path(
        _env_value(
            "SEARCH_CITYNAV_DATA",
            "CITYNAV_ROOT",
            default=str(Path.home() / "datasets" / "citynav-arxiv"),
        )
    )
    .expanduser()
    .resolve()
)

DATA_DIR = CITYNAV_ROOT / "data"
# Episode splits are independently overridable so the HETT refined set
# (github crotonyl/HETT, arXiv 2512.14222) can be A/B'd without touching
# the catalog or any baked tile. Default expands to exactly
# ``$CITYNAV_ROOT/data/citynav`` — zero change to the canonical path.
SPLITS_DIR = (
    Path(
        _env_value(
            "SEARCH_CITYNAV_SPLITS",
            default=str(DATA_DIR / "citynav"),
        )
    )
    .expanduser()
    .resolve()
)
OBJECTS_JSON = DATA_DIR / "cityrefer" / "objects.json"
RGBD_DIR = DATA_DIR / "rgbd"

# Upstream CityNav prebaked nadir rasters for the byte-faithful
# ``observation_backend: ortho`` path. Kept in a SEPARATE dir from
# ``RGBD_DIR`` because the upstream ``<map>.png`` (0.1 m/px RGB ortho) share
# filenames with this repo's own ``RGBD_DIR/<map>.png`` placeholders (1 m/px
# mean-RGB, used by the existing FlightSim/raster path). Generate with
# ``benchmarks/citynav/data_prep/rasterize_citynav_native.sh``. Override with
# ``SEARCH_CITYNAV_RGBD_DIR``.
CITYNAV_NATIVE_RGBD_DIR = (
    Path(
        _env_value(
            "SEARCH_CITYNAV_RGBD_DIR",
            default=str(DATA_DIR / "rgbd_upstream"),
        )
    )
    .expanduser()
    .resolve()
)

# Per-split prebaked feature-graph tiles. Baked by bake_graphs.py.
REPO_ROOT = Path(__file__).resolve().parents[2]
# The tags-only OSM 2 m bake is the tile source of truth: ONE ``prior`` list
# per block, rows = mask + raw OSM tags verbatim, masks coverage-clipped to
# the block's imagery (no prior over no-data). Earlier bakes
# (``tiles/citynav_2m_osmv2`` typed classes, ``tiles/citynav_2m``,
# ``tiles/citynav``) are retired; ``assert_current_tile_schema`` refuses
# them rather than let a run ground against the wrong priors.
# THE bake, by name (an SR measured against another bake is not comparable).
DEFAULT_TILE_BAKE = "tiles/citynav_2m_osmv3"
TILES_DIR = (
    Path(
        _env_value(
            "SEARCH_CITYNAV_TILES",
            default=str(REPO_ROOT / DEFAULT_TILE_BAKE),
        )
    )
    .expanduser()
    .resolve()
)

# WHAT COUNTS AS ARRIVING, for this benchmark. One value per bench, and the
# only thing that decides which mask ``SR`` is a verdict under:
#
#   "ring"   within ``success_radius_m`` of the target point (CityNav, 20 m)
#   "shape"  inside the target's own annotated footprint (Cal City)
#
# Rebound by ``benchmarks.calcity._citynav_bridge`` like every other name
# here, so a Cal City run reads "shape" without the CityNav package holding a
# Cal City fact. READ IT AT CALL TIME (``paths.SUCCESS_CRITERION``): a
# ``from ... import`` captures the CityNav default at import and keeps
# scoring the ring on a Cal City run, silently and with the same signature.
SUCCESS_CRITERION = "ring"

SPLIT_FILES: dict[str, str] = {
    # CityNav ships no test_seen split. Easy/medium/hard are difficulty tiers
    # within each test_unseen / val_* split, pre-partitioned by trajectory length.
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


def split_path(split: str) -> Path:
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}. Known: {sorted(SPLIT_FILES)}")
    return SPLITS_DIR / SPLIT_FILES[split]


def rgbd_png(map_name: str) -> Path:
    """Rasterized top-down semantic ortho for a SensatUrban block."""
    return RGBD_DIR / f"{map_name}.png"


def rgbd_tiff(map_name: str) -> Path:
    """Height-map (depth) ortho for a SensatUrban block."""
    return RGBD_DIR / f"{map_name}.tiff"


def tile_stem(map_name: str) -> str:
    """Stem path for the prebaked feature-graph sidecar (no suffix)."""
    return str(TILES_DIR / map_name)


# Healthy SensatUrban PLYs run ~30-90 MB; the corrupt stubs upstream
# rasterize.py blacklists are 3-4 MB. 10 MB cleanly separates the two
# (and admits cambridge_block_32 at ~37 MB, which the older 100 MB guard
# was silently rejecting).
_RAW_PLY_MIN_BYTES = 10_000_000

# Mirrors the hardcoded skip in upstream CityNav rasterize.py:97. Of these
# three, blocks 0 and 1 also fail the size guard (~3-4 MB), but block_34's
# PLY weighs in at ~38 MB so it would otherwise be admitted. Listing all
# three keeps the policy explicit and resilient to PLY-size drift.
_RAW_PLY_BLACKLIST = frozenset(
    {"cambridge_block_0", "cambridge_block_1", "cambridge_block_34"}
)


def raw_ply(map_name: str) -> Path | None:
    """Return the raw SensatUrban colored PLY for ``map_name`` if one exists.

    CityNav's own demo videos render from these PLYs via Potree. Layout::

        $CITYNAV_ROOT/raw_ply/<block>.ply
        $CITYNAV_ROOT/raw_ply/<split>/<block>.ply
        $CITYNAV_ROOT/raw_ply_fast/<block>.ply

    Returns ``None`` when no complete file is on disk — partial ``.part``
    stubs (mid-download) are ignored by the size guard.
    """
    if map_name in _RAW_PLY_BLACKLIST:
        return None
    candidates = [
        CITYNAV_ROOT / "raw_ply" / f"{map_name}.ply",
        CITYNAV_ROOT / "raw_ply" / "train" / f"{map_name}.ply",
        CITYNAV_ROOT / "raw_ply" / "test" / f"{map_name}.ply",
        CITYNAV_ROOT / "raw_ply_fast" / f"{map_name}.ply",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > _RAW_PLY_MIN_BYTES:
            return p
    return None


def assert_available() -> None:
    """Fail loudly if the core CityNav JSONs aren't fetched.

    The rasterized RGBD tiles are optional for planning — the adapter falls
    back to an open-terrain grid when absent — but the stepwise runner
    (`run_benchmark/`) needs them for observations. We only hard-require the
    splits + objects.json here; observation.py enforces RGBD presence
    lazily when loaded.
    """
    missing = [p for p in (SPLITS_DIR, OBJECTS_JSON) if not p.exists()]
    if missing:
        recipe = (
            f"CityNav data not found under {CITYNAV_ROOT}.\n"
            "Fetch it:\n"
            f"  git clone https://github.com/water-cookie/citynav-arxiv {CITYNAV_ROOT}\n"
            f"  cd {CITYNAV_ROOT} && sh scripts/download_data.sh\n"
            "  # Then fetch SensatUrban .ply files and run scripts/rasterize.sh.\n"
            "Override the root via CITYNAV_ROOT=<path>, or the episode\n"
            "splits alone via SEARCH_CITYNAV_SPLITS=<dir> (HETT refined A/B).\n\n"
            f"Missing: {', '.join(str(p) for p in missing)}"
        )
        raise FileNotFoundError(recipe)


# A run against a retired bake grounds on entirely different priors and still
# completes, so the only symptom is a wrong SR -- which is how a stale
# SEARCH_CITYNAV_TILES went unnoticed across a whole calibration arm (SR 0.351
# vs 0.169 on the same episodes). The current bake writes ONE ``prior`` list
# per block (tags-only, PRIOR_CONTRACT config B): rows are mask + ALL raw tags
# verbatim + ``source`` provenance, nothing minted. The retired shapes are the
# typed-class bake (``building``/``footpath``/... keys, rows minting
# ``prior_role``/``prior_source``) and the older opaque bucket whose rows
# carry no ``source`` at all.


def assert_current_tile_schema(features: dict, source: str) -> None:
    """Refuse a features payload from a retired tile bake.

    ``features`` is a loaded ``<block>_features.json``. Raises when it carries
    typed class keys, or ``prior`` rows minting the retired
    ``prior_role``/``prior_source`` fields or missing ``source`` provenance.
    """
    if not isinstance(features, dict) or not features:
        return
    _remedy = (
        "Grounding against it silently produces a wrong SR. Point "
        "SEARCH_CITYNAV_TILES at the current bake (tiles/citynav_2m_osmv3) "
        "or re-bake with `benchmarks/citynav/data_prep/bake_tiles.sh`."
    )
    extra = sorted(set(features) - {"prior"})
    if extra:
        raise RuntimeError(
            f"{source} is a RETIRED tile bake: its features carry typed class "
            f"keys {extra} instead of the one tags-verbatim 'prior' list. "
            f"{_remedy}"
        )
    rows = features.get("prior") or []
    if any(
        not isinstance(r, dict)
        or "prior_role" in r
        or "prior_source" in r
        or not r.get("source")
        for r in rows
    ):
        raise RuntimeError(
            f"{source} is a RETIRED tile bake: its 'prior' rows mint "
            f"prior_role/prior_source or carry no source provenance, instead "
            f"of the tags-verbatim contract. {_remedy}"
        )


def bake_sha256(tiles_dir: Path | None = None) -> str | None:
    """Content identity of a tile bake: one hash over every file in it.

    ``assert_current_tile_schema`` refuses retired SHAPES; this pins the
    retired CONTENT that shares a shape. Two bakes of the same block set
    live at the same path under the same name and pass every suffix check
    the manifest makes, and a run flown against the wrong one scores a
    different SR with nothing failing (0.351 vs 0.169 on identical
    episodes, measured). A run stamps this so a later reader can say which
    bake produced the number instead of trusting a directory name.

    Hashes names and bytes, so a renamed, added, or dropped file changes
    the result. Returns ``None`` when the directory does not exist -- an
    unhashable bake is an unknown one, never a passing one.
    """
    root = Path(tiles_dir) if tiles_dir is not None else TILES_DIR
    if not root.is_dir():
        return None
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()
