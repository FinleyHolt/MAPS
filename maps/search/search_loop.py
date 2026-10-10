"""Per-episode belief state machine.

Wraps ``MapsCompiler`` with the surrounding orchestration: graph
upsert, target lock, per-frame lock refresh, paint-layer TTL,
target-evidence assembly, perception_update wiring, parse-compile call,
and the per-turn artifact build (target spec, resolved features,
reasoner thread, committed detected ids).

ONE reasoner on a ReAct loop drives all of it, with no subagents.

``update_perception(detections, ...)`` folds one flight into the belief:
graph upsert, lock refresh, paint TTL, target-evidence assembly. It
mutates internal state; ``snapshot()`` returns the current view at any
point, and the recorder writes its per-step snapshot from that.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Callable
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from maps.search.compile_info import CompileTurnResult
from maps.search.io.atomic import atomic_write_json
from maps.search.parse_compile.compiler import MapsCompiler
from maps.search.pose import RobotPose
from maps.search.protocol import CompilerInput
from maps.util.mask_codec import encode_mask_record


# ``p0``, ``P3``, ``p0@(393.5, 358.8)`` style peak-id citations the
# scout may put in its reasoning. The apply-time gate parses these
# and verifies that every cited peak's world_xy sits within the
# success radius of the candidate's centroid.
def _parse_bool(raw: str) -> bool:
    """Truthy parsing matching the legacy ``=="1"`` checks but a bit
    more forgiving so ``SEARCH_FOO=true`` works the way humans expect."""
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_override(envvar: str, default, parser):
    """Return ``parser(os.environ[envvar])`` if the env var is set,
    else ``default``.

    This is the single override primitive for ``SearchLoopConfig``:
    dataclass defaults are the source of truth, env vars are pure
    overrides. Use the right ``parser`` for the field type
    (``int`` / ``float`` / ``_parse_bool`` / ``str``).
    """
    raw = os.environ.get(envvar)
    if raw is None:
        return default
    return parser(raw)


def _lookup_class_roles(
    label: str, class_to_roles: dict[str, tuple[str, ...]]
) -> tuple[str, ...]:
    """Look up the role tag(s) for a SAM detection's class label.

    Tries (a) exact case-insensitive match, then (b) substring match
    in either direction so a composite runtime label matches a
    sub-part union class (or vice versa). Multiple union classes that
    match yield the union of their roles.
    """
    if not label or not class_to_roles:
        return ()
    norm = label.strip().lower()
    if not norm:
        return ()
    if norm in class_to_roles:
        return class_to_roles[norm]
    matched: set[str] = set()
    for cls, roles in class_to_roles.items():
        if cls in norm or norm in cls:
            for r in roles:
                matched.add(r)
    return tuple(sorted(matched))


def _class_to_roles_from_specs(
    union_specs: "list[dict] | None",
) -> dict[str, tuple[str, ...]]:
    """Build a lowercased ``class -> role tuple`` map from Stage-1
    ``sam_concepts`` union specs (each ``{"class": str, "roles": [...]}``).

    Roles stay a tuple so a class that is BOTH target and anchor (e.g.
    "car next to a car") carries ``("anchor","target")`` and the consumer
    can flag the ambiguity rather than force a binary label.
    """
    out: dict[str, set[str]] = {}
    for spec in union_specs or []:
        if not isinstance(spec, dict):
            continue
        cls = str(spec.get("class") or "").strip().lower()
        if not cls:
            continue
        roles = [str(r).strip().lower() for r in (spec.get("roles") or []) if r]
        out.setdefault(cls, set()).update(roles or ["target"])
    return {cls: tuple(sorted(roles)) for cls, roles in out.items()}


log = logging.getLogger(__name__)

# THE RUNAWAY BACKSTOP, and nothing else. A footprint runs to 418k cells
# here, so a pass needs SOME bound or a degenerate shape walks forever --
# but the bound is a guard against broken looping, never a budget the pass
# is supposed to husband. The caller passes its own liveness limit (the
# mission passes its flight budget); this default only covers a caller
# that states none.
#
# It was 12, an eighth of the mission's 96 flights, and that is what a cap
# costs: covering Neuralia Road takes ~111 stops at the observation's own
# spacing, so the pass truncated to 12, reported itself SAMPLED, and told
# the model to narrow and explore a part instead. The model then settled
# for a prior pool 1,790 m off the road -- having reasoned, in its own
# words, that it should "explore the area around Neuralia Road to detect
# more pools that might not be in the prior map". SR is the goal; a longer
# pass is the price of an ambiguous cue, not a fault to be capped.
_COVER_MAX_STOPS = 1000


@dataclass
class SearchLoopConfig:
    """The engine's tunable knobs; the field defaults are the contract."""

    # The ``look`` tool: a NON-TERMINAL vision fetch (nadir candidate crops
    # into the multimodal reasoner) at each per-stop round. It is the ONLY
    # path that puts an image in front of the model — with it off the reasoner
    # is entirely text — so it is on by default and ``SEARCH_LOOK_TOOL=0`` is
    # the off-switch. Only wires on the ortho backend (the renderer is the
    # upstream crop path).
    look_tool: bool = True

    @classmethod
    def from_env(cls) -> "SearchLoopConfig":
        """Build a config from ``SEARCH_*`` env overrides.

        The dataclass field defaults above are the single source of
        truth for tuning knobs. ``from_env`` applies the matching env
        var only when it is set; unset env vars leave the dataclass
        default in place. This prevents the "double-default" failure
        mode where a stale literal in this method silently overrides
        a bumped field default (e.g. raising a deadline at the field
        site but forgetting to update a duplicate string here).

        Every knob goes through ``_env_override``; unset env vars
        leave the dataclass default in place.
        """
        base = cls()
        return dataclasses.replace(
            base,
            look_tool=_env_override(
                "SEARCH_LOOK_TOOL",
                base.look_tool,
                _parse_bool,
            ),
        )


@dataclass(frozen=True)
class SearchSnapshot:
    """Read-only view consumed by the recorder, the planner, and tests.

    Fields that were once mutable lists / sets / dicts on ``EpisodeState``
    are returned as immutable tuples / frozensets to keep the snapshot
    safe to share across callers.
    """

    runtime_graph: object  # ObjectGraph
    thread_history: tuple[dict, ...]
    committed_detected_ids: frozenset[str]
    locked_detection_id: str | None
    confirmed_target_xy: tuple[float, float] | None
    lock_step: int | None
    termination_decisions: tuple[dict, ...]
    planned_rollout_xy: tuple[tuple[float, float], ...]
    # Monotonic counter incremented once per successful compile drain.
    # Lets event-driven consumers (e.g. the planner replan gate) tell
    # "no-op turn" from "fresh intel applied" without diffing engine
    # state.
    compile_seq: int = 0
    # Which path bound the commit (instrument). Copied from _State so the
    # full-flight commit trace records it (run_benchmark reads the snapshot,
    # not the live state).
    commit_bind_phase: str | None = None
    # Every evidence-column name the reasoner's programs attached over the
    # episode, copied from _State on the same route as commit_bind_phase.
    # The offline read of what the agent actually acquired.


@dataclass
class _State:
    """Mutable internal state. Mirrors the autonomy fields that used to
    live on ``EpisodeState``."""

    runtime_graph: object  # ObjectGraph
    # {fid, class, xy} for the detections the LAST update_perception
    # folded -- the flight report serves them so acting on a fresh
    # detection is one step, not an analyze away: most observed-and-
    # zero-row episodes never analyze after acquisition.
    last_flight_detections: list[dict] = field(default_factory=list)
    thread_history: list[dict] = field(default_factory=list)
    committed_detected_ids: set[str] = field(default_factory=set)
    planned_rollout_xy: list[tuple[float, float]] = field(default_factory=list)
    locked_detection_id: str | None = None
    confirmed_target_xy: tuple[float, float] | None = None
    # Flight step the commit bound at. ``None`` = nothing flew.
    lock_step: int | None = None
    termination_decisions: list[dict] = field(default_factory=list)
    # Monotonic turn counter. Collapsed turns do not bump it.
    compile_seq: int = 0
    # ----------------------------------------------------------------
    # Persistent reasoner state (per-stop conversations).
    # ----------------------------------------------------------------
    # The live ``Reasoner`` per reasoner id, so a resume continues the SAME
    # conversation (APC discount on the shared prefix).
    reasoners: dict[int, Any] = field(default_factory=dict)
    # Per-reasoner status. Set to "alive" at compile-commit time
    # and stays "alive" for the whole episode: reasoners are
    # immortal across stops. ``reasoner_alive_rids`` is the
    # canonical alive-set; this dict is retained as a tombstone
    # surface for the SPA's compile-tab "status" column.
    reasoner_status: dict[int, str] = field(default_factory=dict)
    # Per-reasoner stop history. One entry per stop the reasoner has
    # already seen. Used as a defensive idempotency check so the same
    # reasoner never sees the same stop_id twice in one episode.
    reasoner_stop_history: dict[int, list[str]] = field(default_factory=dict)
    # Preflight target/anchor separation, lowercased class -> role tuple
    # (e.g. {"parking lot": ("target",), "road": ("anchor",), "car":
    # ("anchor","target")}). Built from the Stage-1 sam_concepts union
    # specs (which carry per-class roles). Consumed in-flight to tag each
    # candidate by the role(s) its class plays and to prefer a target-role
    # candidate at commit — the per-task agent's own anchor/target split,
    # not a hardcoded rule.
    sam_class_to_roles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Pre-flight broken filter: alive reasoner ids carried forward
    # into the runtime. Read by the preflight arming path, the
    # degenerate ``_forced_preflight_choice`` fallback, and the
    # workdir dump. Broken reasoners are not entered here.
    reasoner_alive_rids: set[int] = field(default_factory=set)
    # C4 three-verb surface: the per-episode PerceptionDirector — the live
    # SAM watchlist (initial = the round-0 sam_concepts union), the
    # queued/active boosts, and the addressable pending-stop registry
    # (qstop_N). Rebuilt fresh at each compile commit (new intel = new
    # perception plan); None until the first compile pushes a union.
    perception_director: Any = None
    # Which path bound the final commit (instrument). The mission binds
    # ``final_flight``; a run that flew nothing leaves it None.
    commit_bind_phase: str | None = None
    # First pose seen this episode (probe-parity start_xy for the
    # clean-room selection message). Set once by update_perception.
    episode_start_xy: tuple[float, float] | None = None
    # Prior-color enrichment frames that raised this episode; the warning
    # prints on the first and every 50th.
    prior_enrich_fails: int = 0


class SearchLoop:
    """Per-episode belief pipeline orchestrator."""

    def __init__(
        self,
        cfg: SearchLoopConfig,
        world_state,  # WorldState
        *,
        tile_origin_xy: tuple[float, float],
        tile_name: str,
        base_features: dict,
        compiler: MapsCompiler,
        sam_concept_setter: Callable[[list[str]], None] | None = None,
        sam_watchlist_setter: Callable[[list[str]], None] | None = None,
        ortho_ground_z_m: float | None = None,
        rgbd_dir: str | None = None,
        marker_xy: tuple[float, float] | None = None,
    ):
        self._cfg = cfg
        self._world_state = world_state
        self._tile_origin_xy = (
            float(tile_origin_xy[0]),
            float(tile_origin_xy[1]),
        )
        self._tile_name = tile_name
        self._base_features = base_features
        self._compiler = compiler
        self._sam_concept_setter = sam_concept_setter
        # C4 three-verb surface: the FREEZE-EXEMPT agent-owned watchlist
        # channel (``ReasonerUnionConceptProvider.apply_watchlist``). The
        # PerceptionDirector pushes the live watchlist (+ active boost
        # concepts) through it so mid-flight ``view{watchlist}`` mutations
        # actually reach the per-stop /segment calls (``set_concepts`` is
        # frozen after preflight and would silently no-op them).
        self._sam_watchlist_setter = sam_watchlist_setter
        # Observation backend the runtime crops frames through. Drives
        # viewpoint-picker dispatch (perception.viewpoint).
        # Per-block upstream ground datum for the orthographic picker's
        # view-square sizing (ortho). ``None`` for backends that
        # don't need it (the perspective picker ignores it).
        self._ortho_ground_z_m = ortho_ground_z_m
        # Upstream rgbd raster dir (``CITYNAV_NATIVE_RGBD_DIR``) — the ONLY
        # consumer is the opt-in ``look`` tool's crop renderer
        # (``_render_candidate_crops``). ``None`` keeps look inert. Passed by the
        # benchmark so maps.search stays benchmark-agnostic (no import benchmark).
        self._rgbd_dir = rgbd_dir
        #: What THIS RUN has captured, handed in by a live runner
        #: (``benchmarks/flight/run_imagery.RunImagery``): the imagery
        #: ``look`` crops instead of the bake's raster. None is the bench,
        #: whose camera is the raster.
        self.look_imagery = None
        # Operator marker mean-xy (admissible CityNav ``marker_positions``
        # input, NOT the GT target). Threaded onto ``CompilerInput`` as
        # ``fallback_marker_xy`` and consumed ONLY by the compiler's
        # fallback-tour installer when round 0 exhausts its retries with no
        # committable plan. ``None`` -> the installer falls back to the start
        # pose (robot_pose).
        self._marker_xy = (
            (float(marker_xy[0]), float(marker_xy[1])) if marker_xy else None
        )
        self._resolve_look_gate(cfg)
        # Keep the reasoner's system-prompt look legend on the SAME gate as
        # tool exposure by stamping the resolved 3-way gate onto the run
        # context the reasoner reads — otherwise the legend would fire on the
        # env flag alone and could advertise look on a non-citynav backend
        # where the tool is never exposed.
        try:
            if self._compiler is not None:
                self._compiler.look_enabled = self._look_tool_enabled
        except Exception:  # noqa: BLE001 — never block ctor on an odd compiler
            pass
        # World-space tracker (UCMCTrack-core, constant-position). Assigns
        # a stable cross-frame ``track_id`` to per-frame detections so
        # ``_find_track``'s primary exact-id match collapses duplicates,
        # replacing the fragile per-frame world-IoU re-id fallback. One
        # instance per episode (SearchLoop is per-episode); always on.
        from maps.perception.world_tracker import WorldTracker

        self._world_tracker = WorldTracker()
        self._graph: object | None = None  # frozen prebake graph
        self._state: _State | None = None
        # Re-entrant lock so a worker thread and the main thread can both
        # reach ``update_perception`` / ``snapshot`` / the commit predicates
        # without racing.
        # Every public mutator + reader of ``_State`` acquires this lock so
        # the worker and main thread serialize on shared state without
        # data races. ``RLock`` because some methods call others that also
        # take the lock (e.g. a snapshot taken inside a commit path).
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(
        self,
        *,
        prebake_graph,
        runtime_graph,
    ) -> None:
        self._graph = prebake_graph
        self._state = _State(runtime_graph=runtime_graph)
        # One ByteTrack over BOTH ingest paths: stash the episode's world
        # tracker on the shared graph so the compile-time union ingest
        # (``parse_compile``/``llm.tools``) routes its detections through the
        # SAME tracker the runtime per-stop ingest uses. Same physical object
        # (union vs runtime, or runtime cross-frame) → one stable track_id →
        # one node; ByteTrack is the sole dedup.
        if runtime_graph is not None and hasattr(runtime_graph, "__dict__"):
            runtime_graph.__dict__["_world_tracker"] = self._world_tracker

    # ------------------------------------------------------------------ #
    # Grid frame conversions
    # ------------------------------------------------------------------ #

    @property
    def _cell_size_m(self) -> float:
        """Tile-grid cell size in meters, sourced from ``maps.sws.grid_domain``.

        This is the single source of truth for the
        ``centroid_grid → world_xy`` conversion used by lock fire,
        tour synthesis, and tool-context construction. The WorldState
        the engine is constructed with always carries a populated
        ``grid_domain`` (see ``benchmarks.citynav.run_benchmark``), so a
        missing ``grid_domain`` indicates a programming error rather
        than a runtime state to silently absorb. The legacy
        ``getattr(self._world_state, "cell_size_m", 5.0)`` form had
        the silent fallback that produced 2.5× off ``confirmed_target_xy``
        on the 2 m CityNav tiles.
        """
        gd = getattr(self._world_state, "grid_domain", None)
        if gd is None:
            raise AttributeError(
                "SearchLoop._world_state.grid_domain is None; cannot "
                "resolve cell_size_m. Construct WorldState with a "
                "populated GridDomain."
            )
        return float(gd.cell_size_m)

    def _grid_centroid_to_world_xy(
        self, centroid: object
    ) -> tuple[float, float] | None:
        """Convert a ``(row, col)`` grid centroid to world XY in meters.

        Returns ``None`` when ``centroid`` is not a 2-tuple of numbers.
        Reads ``cell_size_m`` from :attr:`_cell_size_m` (canonical) and
        the per-block frame origin from :attr:`_tile_origin_xy`.
        """
        if not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
            return None
        try:
            row = int(centroid[0])
            col = int(centroid[1])
        except (TypeError, ValueError):
            return None
        cell = self._cell_size_m
        tx, ty = self._tile_origin_xy
        return (
            float(tx) + (col + 0.5) * cell,
            float(ty) + (row + 0.5) * cell,
        )

    def _prior_surface_xy_locked(self) -> dict:
        """``{fid: (x, y)}`` for every PRIOR row, as the surface serves it.

        Caller MUST hold ``self._lock``. Built once and kept: the prebake
        graph is frozen for the mission, so it cannot go stale, and
        rebuilding costs ~10 s on the calcity bake.
        """
        cached = getattr(self, "_prior_xy_cache", None)
        if cached is not None:
            return cached
        out: dict = {}
        if self._graph is not None and hasattr(self._graph, "nodes"):
            from maps.search.analyze_scene.geom_program import (  # noqa: PLC0415
                build_surface_from_graph,
            )

            _c, anchors = build_surface_from_graph(
                self._graph.nodes.items(), self._tile_origin_xy,
                float(self._cell_size_m))
            out = {str(r["fid"]): (float(r["xy"][0]), float(r["xy"][1]))
                   for r in anchors if r.get("xy")}
        self._prior_xy_cache = out
        return out

    def _world_xy_for_node_locked(self, target_fid: str) -> tuple[float, float] | None:
        """Resolve a fid to WHERE ITS ROW SAYS IT IS.

        ONE POSITION PER ROW. A row's ``xy`` is the position the model
        reasons on, orders against and names in a commit, and it is what
        ``act.move`` flies to. They used to be different quantities computed
        by paths no test reconciled: the row served ``representative_point``
        over the feature's own geometry while the flight took
        ``_grid_centroid_to_world_xy(node.centroid)``, the int cell-mean of a
        merged feature's LEAD way. Measured over one calcity run, 17.3% of
        prior flights landed more than 20 m from the ``xy`` their own row had
        just served, 11.4% more than a kilometre, the worst 11,149 m; the 86
        episodes that flew one scored 0.291 against a 0.516 run.

        The precedence mirrors ``geom_program.build_surface._xy`` exactly,
        because it answers the same question:

        * a DETECTION's ``world_xy`` — the tracker's sub-cell measurement;
        * a PRIOR's served row point, off the frozen surface;
        * the node's cell centroid, for a fid on neither. That is a cluster
          MEMBER rather than its row's own fid — nothing serves one, so the
          model cannot name one, and the fallback exists for readers walking
          the graph directly.

        Caller MUST hold ``self._lock``. Tries the live runtime graph first,
        then the frozen prebake (prior) graph, so a pre-flight
        ``cityref_<id>`` candidate still grounds when it has not been folded
        into the runtime graph. Returns ``None`` only when the fid is in
        neither graph or has no position at all.
        """
        s = self.state
        node = None
        rg = getattr(s, "runtime_graph", None)
        if rg is not None and hasattr(rg, "nodes"):
            node = rg.nodes.get(target_fid)
        if node is None and self._graph is not None and hasattr(self._graph, "nodes"):
            node = self._graph.nodes.get(target_fid)
        if node is None:
            return None
        w = getattr(node, "world_xy", None)
        if w is not None and len(w) >= 2:
            return (float(w[0]), float(w[1]))
        served = self._prior_surface_xy_locked().get(str(target_fid))
        if served is not None:
            return served
        return self._grid_centroid_to_world_xy(getattr(node, "centroid", None))

    def world_xy_for_node(self, target_fid: str) -> tuple[float, float] | None:
        """Thread-safe public wrapper for :meth:`_world_xy_for_node_locked`."""
        with self._lock:
            return self._world_xy_for_node_locked(target_fid)

    def _node_for_fid_locked(self, fid: str):
        """The graph node for ``fid`` (runtime graph first, then the frozen
        prebake), or ``None``. Caller MUST hold ``self._lock``. Mirrors the
        node-resolution in :meth:`_world_xy_for_node_locked`."""
        rg = getattr(self.state, "runtime_graph", None)
        if rg is not None and hasattr(rg, "nodes"):
            n = rg.nodes.get(fid)
            if n is not None:
                return n
        if self._graph is not None and hasattr(self._graph, "nodes"):
            return self._graph.nodes.get(fid)
        return None

    def _cells_xy_locked(self, target_fid: str) -> list[tuple[float, float]] | None:
        """The centres of this fid's cells in world metres, its footprint as
        the bake rasterized it; None off both graphs or without cells. Under
        the lock."""
        node = self._node_for_fid_locked(target_fid)
        cells = getattr(node, "cells", None) if node is not None else None
        # `not cells` is a numpy TRUTHINESS TRAP: a node's cells arrive as
        # an (N, 2) array on the runtime graph, where bool(arr) raises
        # "truth value of an array with more than one element is
        # ambiguous" and takes the whole episode down as an
        # unclassified no-prediction row. Length is the honest test.
        if cells is None or len(cells) == 0:
            return None
        cell = self._cell_size_m
        tx, ty = self._tile_origin_xy
        return [(float(tx) + (int(c[1]) + 0.5) * cell, float(ty) + (int(c[0]) + 0.5) * cell) for c in cells]

    def node_cells_xy(self, target_fid: str) -> list[tuple[float, float]] | None:
        """The ground this fid stands on, as the centres of its cells in
        world metres: the area a flight to it has to hold before it has
        arrived (the station's arrival rule). None off both graphs or
        without cells."""
        with self._lock:
            return self._cells_xy_locked(target_fid)

    def node_coverage_stops(self, target_fid: str, spacing_m: float,
                            max_stops: int = _COVER_MAX_STOPS,
                            minor_spacing_m: float | None = None
                            ) -> list | None:
        """Points ON this fid's footprint, in world metres, walking the
        shape along its length — the stops an ``explore`` flies.

        ``None`` when the fid is on neither graph or carries no cells.
        Flying to the centroid alone lands within 20 m of the target on
        65.6% of the calcity set; a point anywhere in this footprint does
        on 97.6%. That 32-point gap used to be reached by the model naming
        a coordinate, which it cannot do (probed with crops and the world
        window it returned the crop's CENTER on 20/21 replies, midpoint
        arithmetic with no visual pointing in it). It is reached HERE
        instead: covering the shape at the sensor's own footprint puts a
        detection on the part that matters, and the model flies to that
        row.

        ``spacing_m`` is how much of the shape's length one stop is
        credited with covering. The caller passes it from the run's own
        observation extent rather than a constant here, because a bake or
        an altitude change moves it and a number written down twice goes
        stale on one side. ``minor_spacing_m`` is the footprint's SHORT
        side from the same source; passing it arms the serpentine tiling
        for shapes wider than it in both axes, and leaving it ``None``
        keeps the 1-D walk everywhere.

        A BOX IS NOT SERVED and never was worth serving. A shape does not
        fill its box: a road's is 50% road at the median over this bake and
        2.9% at p10, so the box describes the empty corner as confidently
        as the road. The stops come off the CELLS.
        """
        with self._lock:
            points = self._cells_xy_locked(target_fid)
            if points is None:
                return None
            cell = self._cell_size_m
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            half = cell / 2.0
            x0, y0 = min(xs) - half, min(ys) - half
            x1, y1 = max(xs) + half, max(ys) + half
            # ONE REPRESENTATION: THE CELLS. Every prior carries them —
            # 6188 of 6188 on this bake — because the bake rasterizes
            # everything into the same 2D form: a polygon fills, a
            # LineString inflates to one cell of width (roads measure 1.00
            # cells per unit of length at the median), a node is one cell.
            # So the shape is already uniform and there is nothing to
            # special-case.
            #
            # The vector world_contour is NOT that form and must not be
            # mixed in. It exists for 64% of rows, and it is the UNCLIPPED
            # OSM ring while cells are CLIPPED to imagery coverage — a
            # feature straddling the block window keeps its whole ring and
            # loses the cells beneath part of it, so ring vertices named
            # points that were not on the thing at all, on 19% of contour
            # priors, one of them 2.4 km off its own mask. The aid and the
            # offset both come off the CELLS, so they cannot disagree.
            #
            # The stops walk the shape ALONG ITS LENGTH, its two extremes on
            # that axis first, ordered and deduped. Ordered because an
            # explore is a PASS: stops in shape order fly a line, stops in
            # any other order fly a scribble over the same ground for the
            # same money.
            #
            # The 1-D walk covers any shape whose minor extent fits inside
            # one observation, which is every building and every road on
            # this bake. A thing wider than the footprint's SHORT side in
            # BOTH axes (a whole block, an open field) tiles SERPENTINE
            # instead when the caller passes that side as
            # ``minor_spacing_m``: rows along the dominant axis at that
            # pitch on the minor axis, row direction alternating so
            # consecutive rows join at the near end and the pass stays one
            # line. Without ``minor_spacing_m`` the 1-D walk stands and
            # the caller's report says how much of the shape was reached.
            span = max(1.0, float(spacing_m))
            if (minor_spacing_m is not None
                    and (x1 - x0) > float(minor_spacing_m)
                    and (y1 - y0) > float(minor_spacing_m)):
                pitch = max(1.0, float(minor_spacing_m))
                x_major = (x1 - x0) >= (y1 - y0)
                _maj = (lambda p: p[0]) if x_major else (lambda p: p[1])
                _mnr = (lambda p: p[1]) if x_major else (lambda p: p[0])
                lo_min = y0 if x_major else x0
                rows: dict[int, list] = {}
                for p in zip(xs, ys):
                    rows.setdefault(
                        int((_mnr(p) - lo_min) // pitch), []).append(p)
                picks = []
                for ri, ridx in enumerate(sorted(rows)):
                    row = sorted(rows[ridx], key=_maj,
                                 reverse=bool(ri % 2))
                    r_reach = abs(_maj(row[-1]) - _maj(row[0]))
                    rk = max(1, min(int(max_stops),
                                    int(r_reach // span) + 1, len(row)))
                    if rk == 1:
                        picks.append(row[(len(row) - 1) // 2])
                    else:
                        stride = (len(row) - 1) / (rk - 1)
                        picks.extend(row[int(round(i * stride))]
                                     for i in range(rk))
                if len(picks) > int(max_stops) > 1:
                    # over the stop bound the pass thins even-stride,
                    # ends kept: the caller's own reach arithmetic then
                    # says SAMPLED rather than pretending coverage.
                    stride = (len(picks) - 1) / (int(max_stops) - 1)
                    picks = [picks[int(round(i * stride))]
                             for i in range(int(max_stops))]
            else:
                along = sorted(zip(xs, ys),
                               key=(lambda p: p[0])
                               if (x1 - x0) >= (y1 - y0)
                               else (lambda p: p[1]))
                reach = max(x1 - x0, y1 - y0)
                # A shape shorter than one stop's own footprint is covered
                # by standing in the middle of it: k == 1, and an explore
                # of a small building is one flight, not a tour of a shed.
                k = max(1, min(int(max_stops), int(reach // span) + 1))
                k = min(k, len(along))
                if k == 1:
                    # THE MIDDLE, not an end. The one-stop case is the
                    # shape that fits inside one observation, and standing
                    # on its far end sees half of it — `along` is sorted
                    # on the dominant axis, so its median is a real cell
                    # at the middle of the length.
                    picks = [along[(len(along) - 1) // 2]]
                else:
                    stride = (len(along) - 1) / (k - 1)
                    picks = [along[int(round(i * stride))]
                             for i in range(k)]
            on_it: list = []
            for px, py in picks:
                q = [round(px), round(py)]
                if q not in on_it:
                    on_it.append(q)
            return on_it

    def _render_candidate_crops(self, requests: list[dict]) -> list[dict]:
        """The ``look`` renderer (ortho only), handed to the per-stop
        round as ``look_renderer`` when ``self._look_tool_enabled`` — the
        pure render and its window/upscale/badge constants live in
        ``look_render``; this binds the loop's tile/raster identity."""
        from maps.search.render.look_render import render_candidate_crops  # noqa: PLC0415

        return render_candidate_crops(
            requests,
            map_name=str(self._tile_name).split("::")[-1],
            rgbd_dir=self._rgbd_dir,
            imagery=self.look_imagery,
        )

    def _resolve_look_gate(self, cfg) -> None:
        """Set ``_look_tool_enabled``, or raise if look was asked for and
        cannot be wired.

        The gate is the flag (cfg / ``SEARCH_LOOK_TOOL``) AND the ortho backend
        AND a raster dir. ``view{crops}`` is the only path that puts an image
        in front of the reasoner, so a False gate means a text-only run.

        That silence is what this loudness exists for: the tool was off from
        ~13 to 25 July 2026 and every eval in the window flew a blind reasoner
        with nothing saying so. Asking for look and not getting it is a startup
        error now, not a quiet degrade — flying text-only has to be chosen.
        """
        want_look = bool(getattr(cfg, "look_tool", False))
        self._look_tool_enabled = bool(want_look and self._rgbd_dir)
        if want_look and not self._look_tool_enabled:
            raise ValueError(
                "look tool requested but its renderer cannot be wired: "
                f"rgbd_dir={self._rgbd_dir!r} (needs a path). "
                "Set SEARCH_LOOK_TOOL=0 to fly the reasoner text-only on "
                "purpose."
            )

    def _look_audit_dir(self) -> str | None:
        """``<workdir>/flight/look`` — the per-episode look-audit root, keyed by
        stop_id + call id. ``None`` when look is off or no episode
        workdir is known (``compiler.workdir`` / ``SEARCH_COMPILE_WORKDIR``)."""
        if not self._look_tool_enabled:
            return None
        wd = getattr(self._compiler, "workdir", None) or os.environ.get(
            "SEARCH_COMPILE_WORKDIR"
        )
        if not wd:
            return None
        return str(Path(wd) / "flight" / "look")


    def _fid_is_detected(self, fid: str) -> bool:
        """True if the candidate is a runtime SAM detection (provenance
        ``detected``) rather than a prebake catalog prior. At a stop the
        drone actually reached, the perceived instance sits on the physical
        object while a prior CityRefer node can be ~20-50m offset, so the
        detection is the more precise commit. Missing node / unknown
        provenance -> False (treated as non-detected)."""
        nodes = getattr(self.state.runtime_graph, "nodes", None)
        node = nodes.get(fid) if (nodes is not None and hasattr(nodes, "get")) else None
        return bool(node is not None and getattr(node, "provenance", "") == "detected")

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict) -> None:
        """Telemetry write: atomic, and never raises into the round.

        Delegates to the shared writer -- which mints its tempfile with
        ``mkstemp`` rather than a fixed ``.json.tmp`` suffix, so two of the
        concurrent stop rounds writing under one workdir cannot clobber each
        other's partial file. A telemetry write that fails is logged, not
        raised: it must not take the episode down.
        """
        try:
            atomic_write_json(path, payload)
        except OSError as e:
            log.warning("telemetry write failed at %s: %s", path, e)

    def restore_from_workdir(
        self,
        src_workdir: Path | str,
        *,
        instruction: str,
        turn: int | None = None,
    ) -> dict:
        """Rebuild engine state from a prior run's on-disk checkpoint.

        Reads the canonical on-disk artifacts of a prior run and overlays
        them onto the live state, so the next call to
        ``update_perception`` proceeds as if the engine had run up through
        ``turn`` already. ``start``
        must be called first so the runtime graph and the
        ``sam_concept_setter`` callback are wired.

        Sources read from ``src_workdir``:

        - ``compile/turn_<NNNN>/finalized/r*.json`` -> the concept union
          rebuild (the SAM concept provider via the ``sam_concept_setter``
          injected at construction, the per-class roles)
        - ``state.compile_seq = turn + 1``

        ``turn`` defaults to the highest ``compile/turn_<NNNN>/`` that
        carries a finalized reasoner with reconstructable sam_concepts.
        Returns a dict summarizing what was restored (turn index, sam
        concept count, source workdir).
        """
        if self._state is None:
            raise RuntimeError("SearchLoop.start must be called before restore.")
        src = Path(src_workdir).resolve()
        if not src.is_dir():
            raise FileNotFoundError(f"resume workdir not found: {src}")

        compile_root = src / "compile"
        turn_dir_re = re.compile(r"turn_(\d+)$")

        def _compile_turn_dir(idx: int) -> Path:
            return compile_root / f"turn_{idx:04d}"

        from maps.search.parse_compile.compiler import _extract_reasoner_concepts

        def _record_restorable(d: dict) -> bool:
            """A finalized reasoner record is restorable when it carries at
            least one reconstructable SAM concept. The check routes through
            the SAME ``_extract_reasoner_concepts`` the union rebuild below
            consumes (it drops null-token/blank/non-dict classes), so the
            restorability gate and the consumer agree — a record whose
            ``sam_concepts`` are all junk (e.g. ``[{"class": "none"}]``) is
            NOT restorable, and the turn selection skips it rather than
            selecting it and then hard-raising."""
            if not isinstance(d, dict):
                return False
            try:
                return bool(_extract_reasoner_concepts(d))
            except Exception:
                return False

        def _has_restorable_record(idx: int) -> bool:
            fdir = _compile_turn_dir(idx) / "finalized"
            if not fdir.is_dir():
                return False
            for hp in sorted(fdir.glob("r*.json")):
                try:
                    with hp.open("r") as fh:
                        d = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    continue
                if _record_restorable(d):
                    return True
            return False

        def _compile_turn_indices() -> list[int]:
            if not compile_root.is_dir():
                return []
            out = []
            for p in compile_root.iterdir():
                m = turn_dir_re.match(p.name)
                if m:
                    out.append(int(m.group(1)))
            return sorted(out)

        if turn is None:
            candidates = [
                t for t in _compile_turn_indices() if _has_restorable_record(t)
            ]
            if not candidates:
                raise FileNotFoundError(
                    f"no turn under {compile_root} has a restorable finalized "
                    f"reasoner (reconstructable sam_concepts)"
                )
            turn = candidates[-1]

        ctd = _compile_turn_dir(turn)
        finalized_records: list[dict] = []
        fdir = ctd / "finalized"
        if fdir.is_dir():
            for hp in sorted(fdir.glob("r*.json")):
                try:
                    with hp.open("r") as fh:
                        d = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(d, dict):
                    continue
                finalized_records.append(d)

        sam_concepts: list[str] = []
        union_specs: list[dict] = []
        # PRIMARY source of truth for the resumed target/anchor split: the
        # round-0 roles each reasoner committed (the record's sam_concepts),
        # rebuilt with the SAME canonical union builder the fresh compile
        # uses — a role-less rebuild would tag every class "target" and hand
        # a building target a car-dominated ballot on resume.
        try:
            from maps.search.parse_compile.concept_union import ConceptUnionBuilder

            proposals: dict = {}
            for d in finalized_records:
                rid = d.get("reasoner_id")
                if rid is not None:
                    specs = _extract_reasoner_concepts(d)
                    if specs:
                        proposals[rid] = specs
            if proposals:
                ur = ConceptUnionBuilder().build(proposals)
                sam_concepts = list(ur.union)
                union_specs = [
                    {"class": str(s.get("class") or ""),
                     "roles": [str(r) for r in (s.get("roles") or [])]}
                    for s in (ur.union_specs or [])
                    if isinstance(s, dict) and s.get("class")
                ]
        except Exception as exc:  # noqa: BLE001
            log.warning("restore: role union rebuild failed (%s)", exc)

        # The finalized record always carries sam_concepts, so the
        # rebuild above is the sole concept source. A restorable turn
        # (see _record_restorable) guarantees at least one reconstructable
        # concept; empty here means a genuinely malformed source -> fail loud
        # rather than silently flying with no SAM concepts.
        if not sam_concepts:
            raise FileNotFoundError(
                f"restore: no reconstructable sam_concepts under {fdir} "
                f"(sam_concepts missing on every reasoner)"
            )

        s = self._state
        s.compile_seq = max(s.compile_seq, turn + 1)
        s.sam_class_to_roles = _class_to_roles_from_specs(union_specs)
        restored_finalized: dict[int, dict] = {}
        for d in finalized_records:
            if not isinstance(d.get("reasoner_id"), int):
                continue
            if d.get("canceled") or d.get("error"):
                continue
            if not _record_restorable(d):
                continue
            restored_finalized[int(d["reasoner_id"])] = dict(d)
        s.reasoner_alive_rids = set(restored_finalized)

        # Reasoner re-arm: ``s.reasoners`` holds in-memory LLM sessions that
        # are NEVER serialized (a restore strips ``_persistent_reasoner``),
        # so a restore that only rehydrates the finalized records leaves
        # ``s.reasoners`` empty and the runtime reasoning path is never
        # exercised on resume. Rebuild a live ``Reasoner`` per restored rid.
        #
        # The per-stop packet is THIN and does NOT re-push the operator
        # instruction (it lives on the persistent round-0 transcript, which is
        # NOT on disk for a resumed source), and the system prompt carries no
        # cue. A bare reasoner would be CUE-BLIND — unable to rank candidates
        # toward the cue, nulling any runtime lever on BOTH arms of an A/B.
        # Seed each rebuilt reasoner with the operator instruction + its
        # committed round-0 intel. No new persistence: the instruction is
        # an argument; the sam_concepts are on disk.
        from maps.search.parse_compile.reasoner_agent import Reasoner

        s.reasoners = {}
        s.reasoner_status = {}
        s.reasoner_stop_history = {}
        sam_cs = [str(c) for c in sam_concepts if str(c).strip()]
        for rid, d in restored_finalized.items():
            seed = d.get("seed")
            reasoner = Reasoner(
                reasoner_url=self._compiler.reasoner_url,
                endpoint_spec=self._compiler.endpoint_spec,
                # Resume rebuilds the in-memory client only; reuse the
                # compiler's already-resolved credential without serializing
                # it into the checkpoint. Omitting this silently falls back to
                # the local-vLLM ``EMPTY`` sentinel on hosted endpoints.
                api_key=self._compiler.api_key,
                seed=int(seed) if isinstance(seed, int) else None,
                reasoner_index=int(rid),
                sensor_description=getattr(self._compiler, "sensor_description", ""),
            )
            reasoner.append_user(
                "RESUMED MISSION CONTEXT -- you already completed the "
                "pre-takeoff planning round of this search; that transcript is "
                "summarized below. Continue from here: the drone is now flying "
                "the tour you planned, and each following packet is one "
                "stop.\n\n"
                "--- Operator instruction ---\n"
                f"{instruction.strip()}\n\n"
                "--- Your committed mission intel ---\n"
                f"sam_concepts: {', '.join(sam_cs)}"
            )
            s.reasoners[rid] = reasoner
            s.reasoner_status[rid] = "alive"
            s.reasoner_stop_history[rid] = []

        if self._sam_concept_setter is not None and sam_concepts:
            self._sam_concept_setter(list(sam_concepts))
        # C4: arm the PerceptionDirector on resume too (watchlist = the
        # restored union), so view{watchlist}/boost work on resumed runs.
        try:
            self._arm_perception_director(list(sam_concepts))
        except Exception:  # noqa: BLE001 — never fail the restore
            log.exception("perception director arm failed on restore")

        # Resume hydrates the pre-flight state ("when we started flying"),
        # NOT the in-flight lock state. Carrying forward locked_detection_id
        # / confirmed_target_xy from a prior run's mid-flight or landing
        # phase means each resume starts post-lock, which makes iterating
        # on the runtime / commit loop impossible.
        summary = {
            "src_workdir": str(src),
            "turn": int(turn),
            "sam_concepts": sam_concepts,
            "alive_rids": sorted(s.reasoner_alive_rids),
            "reasoners_armed": len(s.reasoners),
            "compile_seq": int(s.compile_seq),
        }
        log.info(
            "SearchLoop.restore_from_workdir: src=%s turn=%d "
            "sam_concepts=%d reasoners_armed=%d",
            src,
            turn,
            len(sam_concepts),
            len(s.reasoners),
        )
        return summary

    @property
    def state(self) -> _State:
        if self._state is None:
            raise RuntimeError("SearchLoop.start must be called before update.")
        return self._state

    @property
    def runtime_graph(self):
        return self.state.runtime_graph

    # ------------------------------------------------------------------ #
    # C4 three-verb surface: PerceptionDirector plumbing
    # ------------------------------------------------------------------ #

    def _arm_perception_director(self, concepts: list) -> None:
        """(Re)build the per-episode PerceptionDirector with ``concepts`` as
        the initial live watchlist (the round-0 sam_concepts union).
        Called under ``self._lock`` at compile commit + resume-restore; a
        fresh compile turn (new intel) rebuilds it outright."""
        from maps.search.parse_compile.perception_director import (
            PerceptionDirector,
        )

        if self._sam_watchlist_setter is None:
            log.info(
                "PerceptionDirector armed WITHOUT a live watchlist channel "
                "(no sam_watchlist_setter): view{watchlist}/boost mutate "
                "engine state only"
            )
        director = PerceptionDirector(push=self._sam_watchlist_setter)
        director.retro_perceiver = getattr(self, "_retro_perceiver_fn", None)
        director.init_watchlist([str(c) for c in (concepts or [])])
        self.state.perception_director = director

    def _watchlist_apply_cb(self, add, remove) -> dict:
        """``view{watchlist}`` channel handed to the per-stop round."""
        with self._lock:
            d = self.state.perception_director
            if d is None:
                # Honest degradation: no armed director (no compile landed a
                # union yet) — say so rather than acking a phantom mutation.
                return {"error": "watchlist is not armed this episode "
                                 "(no compile committed a concept union)"}
            return d.watchlist_apply(add, remove)

    def _boost_queue_cb(self, concepts, where) -> dict:
        """``view{boost}`` channel handed to the per-stop round."""
        with self._lock:
            d = self.state.perception_director
            if d is None:
                return {"error": "boost queueing is not armed this episode"}
            return d.boost_queue(concepts, where)


    def set_retro_perceiver(self, fn) -> None:
        """Install the RETROSPECTIVE-perception executor: the
        runtime closure that re-renders a flown stop's recorded pose,
        re-segments it with a boost vocabulary, ingests through the normal
        upsert path, and returns the delta. Threaded onto the live director
        (and any director armed later)."""
        self._retro_perceiver_fn = fn
        with self._lock:
            if self._state is not None:
                d = self._state.perception_director
                if d is not None:
                    d.retro_perceiver = fn

    def activate_pending_boosts(self) -> list[str]:
        """Runtime hook (run_benchmark, top of each perception round): move
        pending boosts to ACTIVE and push watchlist+boost concepts live.
        Returns the activated concepts ([] when none pending)."""
        with self._lock:
            if self._state is None:
                return []
            d = self._state.perception_director
            if d is None:
                return []
            activated = d.activate_pending()
            if activated:
                log.info("boost activation: +%s for this round", activated)
            return activated

    def retire_boosts_after_round(self, flown: list[dict]) -> None:
        """Runtime hook (run_benchmark, after a round's Phase-A perception):
        consume matched/where-less active boosts, return unmatched targeted
        boosts to pending, mark flown pending stops, and re-push the live
        concept set."""
        with self._lock:
            if self._state is None:
                return
            d = self._state.perception_director
            if d is not None:
                d.retire_after_round(flown)

    def declare_mission_concepts(self, cleaned: list[dict]) -> dict:
        """Runtime hook (the mission engine's ``act.sam_concepts`` channel,
        latest-wins): re-arm the SAME surfaces the compile fold arms — the
        segmenter setter, the live director watchlist (armed here on the
        first declaration; DIFFED afterwards, never rebuilt, so pending
        qstops/boosts survive), and the per-class roles.

        ``cleaned`` is the validated ``[{class, role}]`` list the mission
        loop recorded. Returns the declaration receipt the act result
        carries."""
        from maps.search.parse_compile.compiler import (
            _extract_reasoner_concepts,
        )
        from maps.search.parse_compile.concept_union import (
            ConceptUnionBuilder,
        )

        specs = _extract_reasoner_concepts({"sam_concepts": list(cleaned or [])})
        ur = ConceptUnionBuilder().build({0: specs})
        union = list(ur.union)
        union_specs = [
            {
                "class": str(u.get("class") or ""),
                "roles": [str(x) for x in (u.get("roles") or [])],
            }
            for u in (ur.union_specs or [])
            if isinstance(u, dict) and u.get("class")
        ]
        setter_err: str | None = None
        with self._lock:
            if self._sam_concept_setter is not None:
                try:
                    self._sam_concept_setter(union)
                except Exception as e:  # noqa: BLE001
                    # NEVER swallowed into a clean receipt: a declaration
                    # whose ack reads armed while the segmenter kept its
                    # old class set is a lie in the transcript -- it LOOKS
                    # live and no flight ever segments the class. The
                    # receipt carries the failure; the say channel serves
                    # it loud.
                    log.warning("sam concept setter failed: %s", e)
                    setter_err = str(e)
            d = self.state.perception_director
            if d is None:
                self._arm_perception_director(union)
            else:
                have = list(getattr(d, "watchlist", []) or [])
                adds = [c for c in union if c not in have]
                rems = [c for c in have if c not in union]
                if adds or rems:
                    d.watchlist_apply(adds, rems)
            self._arm_roles(union_specs)
        rcpt = {
            "declared": [
                f"{c.get('class')} ({c.get('role')})" for c in (cleaned or [])
            ]
        }
        if setter_err:
            rcpt["error"] = (
                "the segmenter did not take this declaration "
                f"({setter_err}); say it again"
            )
        return rcpt


    # ------------------------------------------------------------------ #
    # Turn boundary
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Per-frame perception (graph upsert, target evidence accumulation)
    # ------------------------------------------------------------------ #

    def update_perception(
        self,
        *,
        step_idx: int,
        pose_xy: tuple[float, float],
        detections: list,
        projected_masks: dict[int, object] | None = None,
        bounds_xy_min: tuple[float, float] | None = None,
        cell_size_m: float | None = None,
    ) -> list[dict]:
        """Fold one flight's detections into the runtime graph.

        Returns the per-step detections record (the serialisable shape
        the recorder expects).
        """
        with self._lock:
            return self._update_perception_locked(
                step_idx=step_idx,
                pose_xy=pose_xy,
                detections=detections,
                projected_masks=projected_masks,
                bounds_xy_min=bounds_xy_min,
                cell_size_m=cell_size_m,
            )

    def _update_perception_locked(
        self,
        *,
        step_idx: int,
        pose_xy: tuple[float, float],
        detections: list,
        projected_masks: dict[int, object] | None = None,
        bounds_xy_min: tuple[float, float] | None = None,
        cell_size_m: float | None = None,
    ) -> list[dict]:
        s = self.state
        if s.episode_start_xy is None:
            s.episode_start_xy = (float(pose_xy[0]), float(pose_xy[1]))
        bounds_min = bounds_xy_min or self._tile_origin_xy
        cell = float(cell_size_m or self._world_state.grid_domain.cell_size_m)

        now = time.time()

        upsert_ids, step_detections = self._upsert_detections(
            detections=detections,
            projected_masks=projected_masks or {},
            step=step_idx,
            now=now,
            bounds_min=bounds_min,
            cell_size_m=cell,
        )

        # THIS flight's folded detections, as the handles the analyze
        # surface serves (same graph, same fids; the loop refreshes its
        # surface after every fly, so they are addressable the turn the
        # report lands). xy comes off the detection itself -- the durable
        # coordinate if a later dedup re-homes the fid. Replaced whole
        # every step: a step folding nothing leaves an empty list, never
        # the previous flight's.
        _seen: dict[str, dict] = {}
        for _i, _fid in sorted(upsert_ids.items()):
            try:
                _det = detections[_i]
            except (IndexError, TypeError):
                continue
            _wxy = getattr(_det, "world_xy", None)
            if not isinstance(_fid, str) or _wxy is None:
                continue
            try:
                _conf = float(getattr(_det, "confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                _conf = 0.0
            _seen.setdefault(_fid, {
                "fid": _fid,
                "class": str(getattr(_det, "label", "") or ""),
                "xy": [round(float(_wxy[0]), 1), round(float(_wxy[1]), 1)],
                # The detector's own score, so the report's head can show
                # the best of each class instead of the first folded.
                "conf": round(_conf, 3),
            })
        s.last_flight_detections = list(_seen.values())
        return step_detections

    def last_flight_detections(self) -> list[dict]:
        """``[{fid, class, xy}]`` folded by the LAST perception update.

        The flight report's source: what THIS flight just put into the
        world, by the same fids the analyze surface serves.
        """
        with self._lock:
            return [dict(d) for d in self.state.last_flight_detections]

    # ------------------------------------------------------------------ #
    # Phase 8: parse-compile + per-turn artifacts
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    # Termination
    # ------------------------------------------------------------------ #


    def upsert_prior_percept(self, fid: str, attrs: dict, *, px: int = 0) -> bool:
        """Merge a PERCEIVED attribute set onto a prebake node in the runtime
        graph from the rendered pixels inside the prior's projected footprint
        (the SAME deterministic postproc SAM detections get).

        ``attrs`` is e.g. ``{"color": "gray", "color_rgb": "128,130,131"}``
        from :func:`perception.sam.detect_client._set_color_attr`. The
        write bumps the prior's ``observation_count`` so it surfaces as an
        observed, color-bearing candidate (``collect_observed_target_candidates``)
        and the color reads off ``node.semantic_attributes`` in the candidate
        listing. Best-effort: never raises, so the per-frame enrichment sweep
        in the runner can call it inline."""
        try:
            from maps.sws.graph.runtime import (
                upsert_prior_percept as _upsert_prior_percept,
            )

            with self._lock:
                graph = self.state.runtime_graph
                if graph is None:
                    return False
                return _upsert_prior_percept(
                    graph, str(fid), dict(attrs or {}), px=px
                )
        except Exception:  # noqa: BLE001
            log.debug("upsert_prior_percept failed for %s", fid, exc_info=True)
            return False


    def _tool_ctx_for_examiner(self):
        """Build a :class:`ToolContext` the runtime reasoner rounds run against.

        Mirrors the per-reasoner context the Stage-2 reasoner used: the
        prebake graph + runtime graph (shared mutable reference) and the
        tile metadata for spatial filters. ``grid_shape`` is the REAL tile
        grid: the perceive{map} painter clamps every cell to it, and the
        old (0, 0) "filler" silently dropped every cell on every live map
        render — 182/182 maps in framestage_1000 delivered dark with an
        empty class legend while the offline painter (fed a real shape)
        painted fine.
        """
        from maps.search.analyze_scene.tools import ToolContext

        s = self.state
        cell_size_m = self._cell_size_m
        gd = getattr(self._world_state, "grid_domain", None)
        grid_shape = (
            (int(gd.height), int(gd.width)) if gd is not None else (0, 0)
        )
        return ToolContext(
            grid_shape=grid_shape,
            features=self._base_features or {},
            graph=s.runtime_graph,
            tile_name=self._tile_name,
            tile_origin_xy=self._tile_origin_xy,
            cell_size_m=cell_size_m,
        )

    def snapshot(self) -> SearchSnapshot:
        with self._lock:
            s = self.state
            return SearchSnapshot(
                runtime_graph=s.runtime_graph,
                thread_history=tuple(s.thread_history),
                committed_detected_ids=frozenset(s.committed_detected_ids),
                locked_detection_id=s.locked_detection_id,
                confirmed_target_xy=s.confirmed_target_xy,
                lock_step=s.lock_step,
                termination_decisions=tuple(s.termination_decisions),
                planned_rollout_xy=tuple(s.planned_rollout_xy),
                compile_seq=s.compile_seq,
                commit_bind_phase=s.commit_bind_phase,
            )


    # ------------------------------------------------------------------ #
    # Commit predicate (PR2): planner-side descend_and_land trigger
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Internal phase helpers
    # ------------------------------------------------------------------ #

    def _upsert_detections(
        self,
        *,
        detections: list,
        projected_masks: dict[int, object],
        step: int,
        now: float,
        bounds_min: tuple[float, float],
        cell_size_m: float,
    ) -> tuple[dict[int, str], list[dict]]:
        """Project + upsert detections and resolve unresolved anchors."""
        from maps.sws.graph import (
            upsert_detected_centroid_only,
            upsert_tracked_detected_batch,
        )

        s = self.state
        # World-space tracker: assign a STABLE cross-frame ``track_id``
        # ('wt_<n>') in place on each detection BEFORE the upsert, so
        # ``_find_track``'s primary exact-id match re-identifies the
        # object directly instead of relying on the per-frame world-IoU
        # fallback (which churns and folds thin diagonals together). Runs
        # on whatever detections arrive (image-mode SAM, video, or
        # synthetic) — it is situation-agnostic.
        self._world_tracker.update(detections)
        # Pass the world-tracker ``track_id`` straight through to the
        # world-state upsert. That ByteTrack id is the sole re-id signal:
        # ``_find_track`` resolves it by exact match, so cross-frame re-ID
        # is preserved without a parallel canonical-UUID layer.
        tracked_by_idx: dict[int, str] = {
            i: self._detection_id(det)
            for i, det in enumerate(detections)
            if (det.metadata or {}).get("track_id")
        }

        upsert_ids: dict[int, str] = {}
        if projected_masks:
            batch_replaces: list[dict] = []
            batch_index_map: list[int] = []
            for i, tp in projected_masks.items():
                track_id = tracked_by_idx.get(i)
                if track_id is None:
                    continue
                det = detections[i]
                base = _detection_to_mapobject(det, bounds_min, cell_size_m, now)
                batch_replaces.append(
                    {
                        "track_id": track_id,
                        "base_node": base,
                        # Sparse occupied cells, never a dense tile raster:
                        # a full-city tile is ~10 MB per detection dense
                        # against a few kB sparse.
                        "rows": tp.rows,
                        "cols": tp.cols,
                        "confidence": float(det.confidence),
                        "now": now,
                    }
                )
                batch_index_map.append(i)
            fids_out = upsert_tracked_detected_batch(
                s.runtime_graph,
                batch_replaces,
            )
            upsert_ids.update(zip(batch_index_map, fids_out))
            s.runtime_graph.step_index = step
            s.runtime_graph.tau = now

        step_detections: list[dict] = []
        for i, det in enumerate(detections):
            if i not in upsert_ids:
                node = _detection_to_mapobject(det, bounds_min, cell_size_m, now)
                upsert_detected_centroid_only(
                    s.runtime_graph,
                    node,
                    now=now,
                    step_index=step,
                )
            det_record = {
                "world_xy": list(det.world_xy),
                "label": det.label,
                "confidence": float(det.confidence),
                "class_hint": det.class_hint,
                "source_tier": det.source_tier,
                "attributes": dict(det.attributes or {}),
                "metadata": {k: _json_safe(v) for k, v in (det.metadata or {}).items()},
            }
            if det.mask is not None:
                mask_record = encode_mask_record(
                    det.mask.crop, bbox_rcrc=det.mask.bbox
                )
                if mask_record is not None:
                    det_record.update(mask_record)
            step_detections.append(det_record)

        return upsert_ids, step_detections

    @staticmethod
    def _detection_id(det) -> str:
        """Stable per-detection id.

        Precedence: the world tracker's stable id ('wt_<n>') wins when
        present — it is the canonical cross-frame identity authority for
        ``_find_track``'s exact-id match. A FRAME-LOCAL INDEX IS NEVER A
        RE-ID KEY: the retired ``sam3_1_<object_id>`` fallback was the
        per-frame post-NMS index, so stop N's detection 3 exact-matched
        stop N+1's unrelated detection 3 and ``_find_track`` re-homed the
        node 300 m — a pass DESTROYED its own rows as it flew (140#138's
        GT lot, four runs). Every SAM detection with a world position
        carries ``wt_``; a non-churning adapter track id (gt_perfect's
        stable object ids) still rides, and everything else keys on
        label@xy — a per-frame identity that collides only when two
        sightings are genuinely co-located, which is the fold the graph
        wants.
        """
        meta = getattr(det, "metadata", None) or {}
        if isinstance(meta, dict):
            track = meta.get("track_id")
            if isinstance(track, str) and track.startswith("wt_"):
                return track
            if track and not str(track).startswith("sam3_1_img_"):
                return str(track)
        label = str(getattr(det, "label", "?") or "?")
        xy = getattr(det, "world_xy", None) or (0.0, 0.0)
        return f"{label}@{float(xy[0]):.0f},{float(xy[1]):.0f}"

    def _arm_roles(self, union_specs: list[dict]) -> None:
        """Arm the per-class roles from union specs (the mid-flight
        ``act.sam_concepts`` declaration)."""
        s = self.state
        # The in-flight commit + stop packet read the target/anchor split
        # (an empty map reads every role "unknown").
        s.sam_class_to_roles = _class_to_roles_from_specs(union_specs)
        log.info(
            "target/anchor split: %d classes (%s)",
            len(s.sam_class_to_roles),
            {k: list(v) for k, v in list(s.sam_class_to_roles.items())[:6]},
        )


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _detection_to_mapobject(det, bounds_min, cell_size_m, now):
    """Convert one DetectedObject into a graph MapObject node."""
    from maps.sws.graph.model import MapObject

    meta = det.metadata or {}
    # Prefer the perception adapter's rasterized tile cells when present —
    # they are the ground truth for mask-IoU re-identification. Falls
    # back to a 1x1 cell at the projected centroid when the adapter did
    # not stamp cells (no-tile harnesses).
    raw_cells = meta.get("tile_cells")
    cells: np.ndarray | None = None
    if raw_cells is not None:
        try:
            cells_arr = np.asarray(raw_cells, dtype=np.int32).reshape(-1, 2)
            if cells_arr.size > 0:
                cells = cells_arr
        except (TypeError, ValueError):
            cells = None

    # WORLD_CONTOUR substrate: fine world-meter footprint polygon stamped by
    # the perceiving side (gt_perfect exact CityRefer contour, tightness-gated;
    # or the real-SAM ortho path via geolocate.world_contour_from_pixel_mask).
    # Carried onto the base_node so BOTH ingest paths (in-process
    # upsert_tracked_detected_batch and the pickled-payload ingest server)
    # deliver it to the graph node; upsert re-stamps it on repeat sightings.
    raw_contour = meta.get("world_contour")
    world_contour: list[list[float]] | None = None
    if raw_contour is not None:
        try:
            wc = [[float(p[0]), float(p[1])] for p in raw_contour]
            if len(wc) >= 3:
                world_contour = wc
        except (TypeError, ValueError, IndexError):
            world_contour = None

    col = int((det.world_xy[0] - bounds_min[0]) / cell_size_m)
    row = int((det.world_xy[1] - bounds_min[1]) / cell_size_m)
    if cells is not None:
        centroid = (
            int(round(float(cells[:, 0].mean()))),
            int(round(float(cells[:, 1].mean()))),
        )
        bbox = (
            int(cells[:, 0].min()),
            int(cells[:, 1].min()),
            int(cells[:, 0].max()),
            int(cells[:, 1].max()),
        )
        cell_count = int(cells.shape[0])
    else:
        centroid = (row, col)
        bbox = (row, col, row, col)
        cell_count = 1
    fid_placeholder = "detected_pending"
    semantic_attributes: dict[str, dict[str, int]] = {
        key: {value: 1} for key, value in (det.attributes or {}).items()
    }
    # Best-view color: the in-view mask pixel count rides in on det.attributes
    # ['color_px'] as a TRANSIENT carrier (M6 sites stamp it) — promote it to the
    # node's first-class best_view_px and strip it so it never becomes a surface
    # attribute. The runtime upsert compares it to REPLACE color only on a larger view.
    best_view_px = 0
    _bv_bucket = semantic_attributes.pop("color_px", None)
    if _bv_bucket:
        try:
            best_view_px = int(next(iter(_bv_bucket)))
        except (TypeError, ValueError):
            best_view_px = 0
    label = str(det.label or "").strip()
    if label:
        label_bucket = semantic_attributes.setdefault("sam_label", {})
        label_bucket[label] = label_bucket.get(label, 0) + 1
    return MapObject(
        id=fid_placeholder,
        # Carry the FINE SAM concept label as the node class so the
        # per-task target/anchor role lookup (_lookup_class_roles) can match
        # the reasoner's sam_concepts; fall back to the coarse class_hint
        # bucket only when the detection has no label. Attribution is
        # geometry-only (maps.sws.graph.runtime), so a fine class here
        # cannot misattach a detection to a prebake footprint.
        class_name=(str(det.label or "").strip() or det.class_hint),
        name=det.label,
        tags={
            "label": det.label,
            "source_tier": det.source_tier,
        },
        centroid=centroid,
        bbox=bbox,
        cell_count=cell_count,
        orientation=None,
        orientation_confidence=0.0,
        provenance="detected",
        first_seen_t=now,
        last_seen_t=now,
        observation_count=1,
        best_view_px=best_view_px,
        confidence=float(det.confidence),
        # Detected nodes do not STORE the dense per-node mask machinery:
        # `centroid`/`bbox`/`cell_count` are derived from the local `cells`
        # var above, and the live SR/commit path
        # reads only those scalars + class/color/provenance — never the stored
        # `cells` array or the re-id `embedding`. The viewpoint path grounds
        # target_xy from `centroid` (see tour_synth._runtime_graph_to_viewpoint_features),
        # and embedding-sim dedup is inert on the canonical track_id-only path.
        embedding=None,
        semantic_attributes=semantic_attributes,
        cells=None,
        world_contour=world_contour,
        # The tracker's continuous world position — the sub-cell truth the
        # int cell centroid quantizes away (build_surface prefers it).
        world_xy=(float(det.world_xy[0]), float(det.world_xy[1])),
    )


def _json_safe(v):
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    return str(v)


__all__ = [
    "SearchLoop",
    "SearchLoopConfig",
    "SearchSnapshot",
]
