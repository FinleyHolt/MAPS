"""Filesystem paths of the benchmark this process runs, each overridable by environment.

``SEARCH_BENCHMARK_NAME`` (read at import, default ``citynav``) picks the
success criterion, default split and split files; ``benchmarks.calcity.use``
sets it and points the four roots below at Cal City's data.

- ``SEARCH_CITYNAV_DATA`` (else ``CITYNAV_ROOT``, else ``~/datasets/citynav-arxiv``):
  the ``citynav-arxiv`` checkout (``data/citynav``, ``data/cityrefer``, ``data/rgbd_upstream``).
- ``SEARCH_CITYNAV_TILES`` (else ``<repo>/tiles/citynav_2m_osmv3``): the tile bake.
- ``SEARCH_CITYNAV_SPLITS`` (else ``$CITYNAV_ROOT/data/citynav``): the split JSONs
  alone, e.g. HETT's refined ``val_seen``/``val_unseen``/``test_unseen``
  (github crotonyl/HETT, arXiv 2512.14222), whose ``objects.json`` is
  byte-identical, so catalog and tiles stay put.
- ``SEARCH_CITYNAV_RGBD_DIR`` (else ``data/rgbd_upstream``): upstream's 0.1 m/px
  nadir rasters, from ``data_prep/rasterize_citynav_native.sh``.
"""
from __future__ import annotations

import hashlib
import json
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

REPO_ROOT = Path(__file__).resolve().parents[2]
# The tile bake (data_prep/bake_tiles.sh): one tags-only OSM ``prior`` list per
# block, masks clipped to the block's imagery. An SR measured against another
# bake is not comparable.
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

BENCHMARK = os.environ.get("SEARCH_BENCHMARK_NAME", "").strip() or "citynav"

# What counts as arriving (``eval.score_arrival``), the runner's default split,
# and the split files, per benchmark.
SUCCESS_CRITERION, DEFAULT_SPLIT, SPLIT_FILES = {
    "citynav": ("ring", "val_seen", {
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
    }),
    "calcity": ("shape", "calcity", {"calcity": "calcity.json"}),
}[BENCHMARK]


def split_path(split: str) -> Path:
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}. Known: {sorted(SPLIT_FILES)}")
    return SPLITS_DIR / SPLIT_FILES[split]


def tile_stem(map_name: str) -> str:
    """Stem path for the prebaked feature-graph sidecar (no suffix)."""
    return str(TILES_DIR / map_name)


def assert_available() -> None:
    """Raise with the fetch recipe when the splits or ``objects.json`` are missing."""
    if BENCHMARK == "calcity":
        from benchmarks.calcity import paths as calcity_paths  # noqa: PLC0415

        calcity_paths.assert_available()
        return
    missing = [p for p in (SPLITS_DIR, OBJECTS_JSON) if not p.exists()]
    if missing:
        recipe = (
            f"CityNav data not found under {CITYNAV_ROOT}.\n"
            "Fetch it, from the repo root:\n"
            f"  CITYNAV_ROOT={CITYNAV_ROOT} benchmarks/citynav/data_prep/fetch_data.sh\n"
            f"  CITYNAV_ROOT={CITYNAV_ROOT} benchmarks/citynav/data_prep/rasterize_citynav_native.sh\n"
            "  PY=.venv/bin/python benchmarks/citynav/data_prep/bake_tiles.sh\n"
            "Override the root via CITYNAV_ROOT=<path>, or the episode\n"
            "splits alone via SEARCH_CITYNAV_SPLITS=<dir>.\n\n"
            f"Missing: {', '.join(str(p) for p in missing)}"
        )
        raise FileNotFoundError(recipe)


def assert_current_tile_schema(features: dict, source: str) -> None:
    """Refuse a loaded ``<block>_features.json`` of another schema.

    The current bake writes one ``prior`` list whose rows carry mask, raw tags
    and ``source``. A run against another bake still completes with a wrong
    SR, so typed class keys, ``prior_role``/``prior_source`` rows and rows
    without ``source`` raise.
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


#: The per-block files the bake writes, the content :func:`bake_sha256` hashes.
BAKE_OUTPUT_SUFFIXES = ("_meta.json", "_features.json", "_graph.npz", "_nodes.json")
#: When and from which commit a block was baked: provenance, not content.
_BAKE_WHEN_WHO = ("baked_at_utc", "bake_git_sha")


def _bake_content(f: Path) -> bytes:
    """``f``'s bytes, a block meta's bake time and commit left out."""
    data = f.read_bytes()
    if not f.name.endswith("_meta.json"):
        return data
    meta = json.loads(data)
    prov = meta.get("provenance")
    if isinstance(prov, dict):
        meta["provenance"] = {k: v for k, v in prov.items() if k not in _BAKE_WHEN_WHO}
    return json.dumps(meta, sort_keys=True).encode()


def bake_sha256(tiles_dir: Path | None = None) -> str | None:
    """Content identity of a tile bake: one hash over the name and content of
    every ``BAKE_OUTPUT_SUFFIXES`` file, a meta's bake time and commit left out.

    Two bakes of one block set can share a path and a schema yet score
    different SRs, so a run stamps this rather than trust a directory name. ``None`` when the directory does not exist.
    """
    root = Path(tiles_dir) if tiles_dir is not None else TILES_DIR
    if not root.is_dir():
        return None
    h = hashlib.sha256()
    for f in sorted(p for p in root.iterdir()
                    if p.is_file() and p.name.endswith(BAKE_OUTPUT_SUFFIXES)):
        h.update(f.name.encode())
        h.update(b"\0")
        h.update(hashlib.sha256(_bake_content(f)).digest())
    return h.hexdigest()
