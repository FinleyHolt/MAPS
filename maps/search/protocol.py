"""Compiler contract for the parse-first compiler.

The compiler takes operator natural language plus a world-state
snapshot and produces an ordered tour over candidate observation
points. A deterministic broken filter drops reasoners whose
``tour_plan_observations`` reach no grounded question-tree leaves;
alive reasoners contribute observations to the tour synth.

CompilerInput fields:
  mission          natural-language query, under ``mission['instruction']``
  grid_shape       (H, W) of the tile
  sensor_config    sensor / platform config at step n
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Union

from maps.search.pose import RobotPose

if TYPE_CHECKING:
    from maps.sws import WorldState


@dataclass
class CompilerInput:
    """Everything an approach needs to produce a tour over observation
    points."""

    mission: dict  # Carries Q under mission["instruction"]
    grid_shape: tuple[int, int]  # (H, W) of the tile
    features: dict | None = None  # PRIOR TIER ONLY: per-feature metadata
    # keyed by open-vocab class name (e.g. "roads", "buildings", "landuse").
    # This is the OSM/CityNav prebake. The compiler does not expect a
    # pre-merged catalog here — Stage 1.5 in compile() builds a canonical
    # concept union (string-dedup only) from the reasoner's sam_concepts and
    # forwards it to the runtime; the per-frame image-mode ``/segment`` calls
    # in flight segment that union and ingest detections (one tracked upsert
    # per tour stop) into the one shared runtime ObjectGraph. The
    # persistent reasoner queries that single graph through its
    # ``_feature_index``, which points at the shared graph: prior tier plus
    # the shared SAM tier, one feature_id namespace.
    tile_name: str | None = None  # Stable key for cross-call graph caching
    graph: object | None = None  # The world-state ObjectGraph: the prebake
    # seed (OSM / CityRefer) plus every in-flight detection upserted by the
    # caller via ``upsert_detected``. One graph; nodes carry ``provenance``
    # so prior, runtime, and confirmed features are distinguishable without
    # a second instance. Skips O(n^2) build when supplied.
    robot_pose: RobotPose | None = None  # current drone pose; what the
    # body-frame projection of each prior-graph node is computed from
    # (the per-turn sidecar in ``analyze_scene.tools.ToolContext``)
    tile_origin_xy: tuple[float, float] | None = None  # (x_min, y_min) of the
    # tile in the same world frame as ``robot_pose``; needed to convert the
    # drone's world coords to grid (row, col). Defaults to (0.0, 0.0).
    cell_size_m: float | None = None  # grid cell size in meters; per-tile
    # value from the tile's ``_meta.json`` (also stored on
    # ``ObjectGraph.cell_size_m``).
    step_idx: int = 0
    # Flight-step index of this compile turn. Used by the compiler's
    # run-checkpoint store to partition on-disk artifacts under
    # ``<workdir>/compile/turn_<step_idx:04d>/`` so a re-run with
    # ``--resume-from`` can skip earlier phases per-turn rather than
    # only on the last turn of an episode.
    # When set, the compiler skips every phase strictly before the named
    # one and reads its inputs from the per-turn checkpoint directory
    # instead. Allowed: ``segment``, ``ingest``. ``None`` (or
    # ``"stage1"``) runs the full pipeline. Raises FileNotFoundError if
    # a required checkpoint is missing.
    fallback_marker_xy: tuple[float, float] | None = None
    # Operator marker mean-xy (the admissible CityNav ``marker_positions``
    # input, NOT the withheld GT ``target_positions``). Consumed ONLY by the
    # compiler's fallback-tour installer when iter-0 exhausts its retries with
    # no committable plan: it synthesizes one nadir viewpoint over the marker
    # so the drone flies + perceives real evidence instead of scoring a blind
    # pre-takeoff zero. ``None`` -> the installer falls back to ``robot_pose``
    # (the start pose).
    start_altitude_m: float | None = None
    # Altitude hint for the orthographic picker (ortho).
    # Caller passes the current runtime altitude_m so the picker emits
    # poses at the in-band altitude the drone is already flying at.
