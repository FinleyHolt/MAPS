"""A bench as a value: where its data lies, how it scores, and the episode engine it flies.

``benchmarks.citynav.paths.citynav_bench`` and
``benchmarks.calcity.paths.calcity_bench`` each build one from an environment
mapping; an entry point builds its bench once, from ``os.environ``, and passes
it to what needs it, so one process can hold both.

This module and the two constructors import the standard library only: the
data fetch script checks a bench with a bare interpreter.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

#: The per-block files a tile bake writes, the content :func:`bake_sha256` hashes.
BAKE_OUTPUT_SUFFIXES = ("_meta.json", "_features.json")


def _bake_content(f: Path) -> bytes:
    """``f``'s bytes; a block meta's as canonical JSON (sorted keys)."""
    data = f.read_bytes()
    return (
        json.dumps(json.loads(data), sort_keys=True).encode()
        if f.name.endswith("_meta.json")
        else data
    )


def bake_sha256(tiles_dir: Path) -> str | None:
    """Content identity of a tile bake: one hash over the name and content of
    every ``BAKE_OUTPUT_SUFFIXES`` file.

    Two bakes of one block set can share a path and a schema yet score
    different SRs, so a run stamps this rather than trust a directory name. ``None`` when the directory does not exist.
    """
    root = Path(tiles_dir)
    if not root.is_dir():
        return None
    h = hashlib.sha256()
    for f in sorted(
        p for p in root.iterdir() if p.is_file() and p.name.endswith(BAKE_OUTPUT_SUFFIXES)
    ):
        h.update(f.name.encode())
        h.update(b"\0")
        h.update(hashlib.sha256(_bake_content(f)).digest())
    return h.hexdigest()


def missing_bake(stem: str) -> str | None:
    """What the bake at ``stem`` lacks for an episode to load, or None."""
    recipe = (
        "python -m benchmarks.calcity.data_prep.bake_tiles"
        if Path(stem).name == "calcity"
        else "benchmarks/citynav/data_prep/bake_tiles.sh"
    )
    for path in (Path(f"{stem}{s}") for s in BAKE_OUTPUT_SUFFIXES):
        if not path.exists():
            return f"Missing {path}: bake this block first ({recipe})."
    return None


class EpisodeEngine(Protocol):
    """The episode build a bench flies: what turns a split row into a hydrated
    search stack and a mission conversation over it. The flight harness and the
    station reach the engine through their bench, never by import."""

    def load_split(self, split_file: Path) -> list[dict]: ...

    def find_row(self, rows: list[dict], episode_id: str) -> tuple[int, dict] | None: ...

    def parse_episode(self, row: dict, row_index: int | None = None) -> Any: ...

    def authored_episode(self, episode_id: str, start_xy: tuple[float, float]) -> Any:
        """An episode flown under the operator's own words: the map its id
        names, the start given, and no target."""

    def load_assets(self, bench: Bench, map_name: str) -> Any: ...

    def start_altitude_m(
        self, cfg: Any, map_name: str, start_z: float | None
    ) -> tuple[float, float]: ...

    def build_episode(
        self,
        cfg: Any,
        episode: Any,
        assets: Any,
        bench: Bench,
        *,
        settings: Any,
        workdir: Path | None = None,
        routing_key: str | None = None,
    ) -> Any:
        """The hydrated search stack for ``episode`` under ``settings`` (the
        engine's, ``maps.settings.Settings``), recording under ``workdir``."""

    def open_session(self, h: Any, *, recorder: Any = None, observe: Any = None) -> Any: ...

    def load_objects(self, objects_json: Path) -> dict:
        """The scorer's answer key: map name -> its catalog objects (``id``, ``contour``)."""

    def fly_attempt(self, **attempt: Any) -> dict:
        """One attempt flown, scored and written as the bench's runner
        dispatches it (its ``dispatch_one``, whose keywords these are), and
        its row. ``converse`` drives the mission conversation over the
        episode's session in place of the bench's one turn: the station's,
        which holds the conversation open for an operator."""

    def is_infra_fault(self, exc: BaseException) -> bool: ...


@dataclass(frozen=True)
class Bench:
    """One bench's description. Paths are resolved; nothing here reads the
    environment or the disk at construction."""

    #: ``citynav`` or ``calcity``: names an attempt directory and the default output root.
    name: str
    #: The data root (``data/cityrefer``, ``data/rgbd_upstream`` under it by default).
    root: Path
    #: The directory holding the split JSONs.
    splits_dir: Path
    #: The CityRefer-format object catalog, the scorer's answer key.
    objects_json: Path
    #: The imagery rasters the ortho crops read.
    rgbd_dir: Path
    #: The tile bake: one ``<block>_meta.json`` and ``<block>_features.json`` per block.
    tiles_dir: Path
    #: Split name -> file name under ``splits_dir``.
    split_files: Mapping[str, str]
    #: What counts as arriving (``eval.score_arrival``): ``ring`` or ``shape``.
    success_criterion: str
    #: The split a runner flies when none is named.
    default_split: str
    #: The module whose ``ENGINE`` is the episode build this bench flies
    #: (:attr:`engine`). Named, not held: describing a bench, checking its data
    #: and reading its bake need the standard library alone.
    engine_module: str
    #: Raises with the bench's preparation recipe when its runtime data is absent.
    availability: Callable[[Bench], None]
    #: The environment of the bench's runs: the mapping it was built from, plus
    #: the variables the bench itself exports and what an entry point settles
    #: (:meth:`with_environ`). A run's manifest and rows record it, the engine's
    #: settings are built from it, and an attempt child inherits it. Nothing
    #: downstream of an entry point reads ``os.environ``.
    environ: Mapping[str, str] = field(repr=False)  # it can hold the reasoner's key

    @property
    def engine(self) -> EpisodeEngine:
        """The episode build this bench flies, imported when first asked for."""
        return importlib.import_module(self.engine_module).ENGINE

    def split_path(self, split: str) -> Path:
        if split not in self.split_files:
            raise ValueError(f"Unknown split {split!r}. Known: {sorted(self.split_files)}")
        return self.splits_dir / self.split_files[split]

    def tile_stem(self, map_name: str) -> str:
        """Stem path of a block's bake files (no suffix)."""
        return str(self.tiles_dir / map_name)

    def assert_available(self) -> None:
        self.availability(self)

    def bake_sha256(self) -> str | None:
        """The content identity of this bench's tile bake (:func:`bake_sha256`)."""
        return bake_sha256(self.tiles_dir)

    def with_environ(self, changes: Mapping[str, str | None]) -> Bench:
        """This bench under an environment with ``changes`` applied: a value sets
        its variable, ``None`` unsets it."""
        environ = {**self.environ, **{k: v for k, v in changes.items() if v is not None}}
        for k, v in changes.items():
            if v is None:
                environ.pop(k, None)
        return replace(self, environ=environ)
