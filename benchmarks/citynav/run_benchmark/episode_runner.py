"""run_episode: per-episode setup -> mission conversation (fly-on-request)
-> trajectory synthesis -> scored result record.
"""

from __future__ import annotations

from benchmarks.citynav.adapter import OBJTYPE_TO_LABEL, parse_episode
from benchmarks.citynav.episode import EpisodeState
from benchmarks.citynav.live_recorder import LiveRecorder
from benchmarks.citynav.metrics import classify_referent
from benchmarks.citynav.paths import CITYNAV_NATIVE_RGBD_DIR, OBJECTS_JSON, tile_stem
from benchmarks.citynav.run_benchmark.config import (
    _REALSTACK_PERCEPTION_TIER,
    RunConfig,
    _assert_realstack_gt_isolation,
)
from collections.abc import Callable
from pathlib import Path
from typing import Any
from maps.llm import endpoint_spec as endpoint_spec_module
from maps.llm import flight_summary
from maps.perception.geometry.geolocate import TileFrame
from maps.perception.geometry.sensor_spec import resolve_sensor
from maps.perception.observation import crop_observation
from maps.perception.ortho.detection_stamping import (
    project_and_stamp_ortho_detections,
)
from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider
from maps.perception.sam.detect_client import SamDetectClient
from maps.search import (
    MapsCompiler,
    SearchLoop,
    SearchLoopConfig,
)
from maps.sws import GridDomain, WorldState
from maps.search.withhold import withheld
from maps.sws.graph import attach_mask_cache, build_object_graph
import json
import math
import numpy as np
import os
import threading
import time
from benchmarks.citynav.run_benchmark.assets import (
    load_episode_assets,
    load_episode_graph,
)
from benchmarks.citynav.run_benchmark.mission_engine import run_mission_episode
from benchmarks.citynav.run_benchmark.osm_prior import (
    _OsmPriorCtx,
    _build_osm_prior_ctx_from_graph,
    _gt_nearest_prior_fid,
    _load_osm_prior_ctx,
    _resolve_osm_prior_graph_fids,
    _write_osm_percepts_sidecar,
)
from benchmarks.citynav.run_benchmark.perception import make_perception
from benchmarks.citynav.run_benchmark.record import _build_provisional_rec
from benchmarks.citynav.run_benchmark.resume import hydrate_resume_artifacts
from benchmarks.citynav.run_benchmark.trajectory import synthesize_compliant_trajectory
from benchmarks.citynav.run_benchmark.reasoner_warmup import (
    ReasonerUnreachableError,
    _reasoner_warmup_ping,
    _write_pre_flight_health,
)
import logging

log = logging.getLogger("run_benchmark")


def start_altitude_m(cfg, map_name: str, episode_id: str) -> tuple[float, float]:
    """``(start altitude AMSL, block ground datum)`` for one episode.

    The ortho profile samples the upstream 100-150 m band (paper SS3.2)
    per episode, deterministic (seeded by ``episode_id``) so a run is
    reproducible and parallel-safe; ``start_altitude_agl_m`` replaces the
    AMSL band with a uniform above-ground cruise. The result is clamped
    into ``[ground + 1, altitude_cap_m]``.

    This pair IS the crop window's observation ceiling — the mission's
    ``obs_span_m = 2 * (alt - ground)`` — so it is a named function a test
    can call. ``cfg.altitude_m`` is deliberately NOT read: it is the inert
    fallback for the non-ortho paths, and reading it as the start pose is
    what put a 25.8 m median span in the tree that no episode has flown
    (the real median is ~192 m).
    """
    import hashlib  # noqa: PLC0415

    from maps.perception.ortho.ground_level import (  # noqa: PLC0415
        GROUND_LEVEL,
    )

    ground = float(GROUND_LEVEL.get(map_name, 0.0))
    if cfg.start_altitude_agl_m is not None:
        alt = ground + float(cfg.start_altitude_agl_m)
    else:
        lo_m, hi_m = (
            float(cfg.start_altitude_band_m[0]),
            float(cfg.start_altitude_band_m[1]),
        )
        seed = int(
            hashlib.sha256(str(episode_id).encode()).hexdigest()[:16], 16
        )
        alt = lo_m + (seed / float(0xFFFFFFFFFFFFFFFF)) * (hi_m - lo_m)
    return (
        float(min(max(alt, ground + 1.0), float(cfg.altitude_cap_m))),
        ground,
    )


def _episode_graph(stem: str, assets) -> tuple:
    """The episode's object graph and the prior features it starts from.
    The ``prior`` withhold starts both empty."""
    if withheld("prior"):
        return build_object_graph({}, cell_size_m=float(assets.cell_size_m)), {}
    try:
        return load_episode_graph(stem), assets.base_features
    except FileNotFoundError:
        log.warning("  %s: no baked graph, building on the fly", stem)
        return build_object_graph(
            assets.base_features, cell_size_m=float(assets.cell_size_m),
        ), assets.base_features


def _success_mask(cfg, episode, objects_by_map: dict) -> tuple[str, dict]:
    """``(criterion, mask kwargs)`` for ``eval.score_arrival``.

    The criterion is the BENCH's, read off ``paths.SUCCESS_CRITERION`` at
    call time so the Cal City bridge's rebind reaches it -- a
    ``from ... import`` here would capture CityNav's ring and score every
    Cal City run on it, silently.

    The shape is the linked object's contour when the episode names one,
    else its hand-drawn/derived mask. The fallback radius comes from the ROW
    when the split states one per episode, else from the run config; on a
    corpus where every row resolves a shape it never fires.
    """
    from benchmarks.citynav import paths as bench_paths

    contour = None
    if episode.target_object_ids:
        _tids = set(episode.target_object_ids)
        for _o in objects_by_map.get(episode.map_name, []) or []:
            if getattr(_o, "id", None) in _tids:
                contour = getattr(_o, "contour", None)
                break
    row_radius = getattr(episode, "success_radius_m", None)
    return bench_paths.SUCCESS_CRITERION, {
        "gt_polygon": contour if episode.target_object_ids else getattr(
            episode, "target_mask", None
        ),
        "radius_m": float(
            row_radius if row_radius is not None else cfg.success_radius_m
        ),
        "tolerance_m": float(getattr(cfg, "precise_tolerance_m", 0.0)),
        "floor_radius_m": float(getattr(cfg, "precise_floor_radius_m", 5.0)),
    }


def run_episode(
    row: dict,
    compiler: MapsCompiler | None,
    objects_by_map: dict,
    cfg: RunConfig,
    recorder: LiveRecorder | None = None,
    row_index: int | None = None,
    provisional_sink: Callable[[dict], None] | None = None,
) -> dict | None:
    episode = parse_episode(row, row_index=row_index)
    objs_for_map = objects_by_map.get(episode.map_name, [])
    if not objs_for_map:
        log.warning("  skip %s: no CityRefer objects", episode.episode_id)
        return None

    assets = load_episode_assets(episode.map_name, objs_for_map, cfg.cell_size_m)
    graph_stem = tile_stem(episode.map_name)
    log.info("tile_stem=%s map_name=%s", graph_stem, episode.map_name)
    graph, prior_features = _episode_graph(graph_stem, assets)

    # Attach a mask cache so runtime upserts keep the fid ledger consistent
    # with graph.nodes. Seeded from the per-tile features.json so prebake
    # masks use their precise cell lists; detected nodes extend it row by row.
    #
    # World-state cleanout: the mask cache is a trivial scalar ledger
    # (fid -> centroid/cell_count) rebuilt from graph.nodes in ~ms — there is
    # no dense disk twin to resume and no shared-tile_stem concurrent clobber.
    attach_mask_cache(graph, assets.grid_shape, features=prior_features)

    if cfg.spawn == "uniform":
        # Seeded uniform spawn over the tile (run_benchmark.spawn). The
        # episode object itself is rebound so every downstream reader —
        # flight record, metrics, crash floors — sees the FLOWN start,
        # not the dataset's.
        import dataclasses as _dataclasses

        from benchmarks.citynav.run_benchmark.spawn import uniform_spawn_xy

        episode = _dataclasses.replace(
            episode,
            start_xy=uniform_spawn_xy(
                (
                    float(assets.bounds.x_min),
                    float(assets.bounds.y_min),
                    float(assets.bounds.x_max),
                    float(assets.bounds.y_max),
                ),
                cfg.spawn_seed,
                episode.episode_id,
            ),
        )
        log.info(
            "  uniform spawn: (%.1f, %.1f) (spawn_seed=%d, episode=%s)",
            episode.start_xy[0],
            episode.start_xy[1],
            int(cfg.spawn_seed),
            episode.episode_id,
        )
    elif cfg.spawn != "episode":
        raise ValueError(
            f"unknown spawn mode {cfg.spawn!r} — expected 'episode' or 'uniform'"
        )
    start_xy = episode.start_xy
    if os.environ.get("CITYNAV_START_CENTER") == "1":
        start_xy = (
            0.5 * (float(assets.bounds.x_min) + float(assets.bounds.x_max)),
            0.5 * (float(assets.bounds.y_min) + float(assets.bounds.y_max)),
        )
    # The start pose's altitude and the block's ground datum. The datum is
    # threaded to the orthographic viewpoint picker so it sizes the view
    # square from the same scalar the crop uses (``side = 2 * (alt -
    # ground)``) instead of any per-object height prior.
    start_alt_m, ortho_ground_z_m = start_altitude_m(
        cfg, episode.map_name, episode.episode_id
    )
    log.info(
        "  ortho start altitude: %.2f m (%s, ground=%.2f, "
        "cap=%.0f, view_square=%.0fm, episode=%s)",
        start_alt_m,
        (
            f"AGL {cfg.start_altitude_agl_m:.0f} m"
            if cfg.start_altitude_agl_m is not None
            else f"band [{cfg.start_altitude_band_m[0]:.0f}, "
                 f"{cfg.start_altitude_band_m[1]:.0f}]"
        ),
        ortho_ground_z_m,
        float(cfg.altitude_cap_m),
        2.0 * (start_alt_m - ortho_ground_z_m),
        episode.episode_id,
    )
    state = EpisodeState(
        pose_xy=start_xy,
        altitude_m=start_alt_m,
        graph=graph,
    )
    # Runtime observations accumulate in the SAME graph as prebaked nodes;
    # ``provenance`` is the only tag separating them. The belief engine
    # mutates ``graph`` in place via maps.sws.graph.upsert_*; downstream
    # consumers that filter by provenance get the right detected subset.
    state.record_visit()

    grid_h, grid_w = assets.grid_shape
    world_state = WorldState(
        grid_domain=GridDomain(
            height=grid_h, width=grid_w, cell_size_m=float(assets.cell_size_m)
        ),
        object_graph=graph,
    )

    perception = make_perception(cfg, objs_for_map, episode.map_name)
    # C1 GT-isolation, post-construction half (blocker V1-R2): now that the
    # perception object exists, re-assert the gate WITH it so a sam
    # tier that somehow resolved to a synthetic/GT-perfect object fails LOUD
    # before any flight spend (folds in the SEARCH_SKIP_VLM leak vector).
    _assert_realstack_gt_isolation(cfg, perception=perception)
    # The world-space tracker owns cross-frame identity: it assigns a
    # stable object_id per tracked masklet which the adapter forwards into
    # MapObject.track_id. Reset per episode so a fresh session starts here
    # and object ids do not bleed across episodes.
    if perception is not None and hasattr(perception, "reset"):
        perception.reset()

    # Build the belief engine. The compiler carries the reasoner endpoint
    # facts the mission engine reads (None disables the LLM path entirely;
    # the engine still runs perception_update / paint in that mode). The
    # SAM concept setter pushes each act.sam_concepts union back into a
    # SAM-style perception adapter; non-SAM adapters get a no-op. The
    # mission declares concepts latest-wins, so the catalog is never
    # frozen mid-episode (``sam_concept_freezer`` fires only on the
    # resume-restore path, which re-arms a finished catalog).
    sam_concept_setter = None
    sam_concept_freezer = None
    sam_watchlist_setter = None
    if isinstance(perception, SamDetectClient):
        cp = perception.concept_provider
        if isinstance(cp, ReasonerUnionConceptProvider):
            sam_concept_setter = cp.set_concepts
            sam_concept_freezer = cp.freeze
            # C4: the freeze-exempt agent-owned watchlist channel — how a
            # view{watchlist} mutation / boost activation reaches the
            # per-flight /segment calls.
            sam_watchlist_setter = cp.apply_watchlist

    search_loop: SearchLoop | None = None
    # Tier-(c) oracle-set diagnostic: set only
    # under SEARCH_ORACLE_INJECT_GT=1, stamped onto the metrics row below.
    _oracle_fid: str | None = None
    _oracle_dist_m: float | None = None
    if compiler is not None:
        engine_cfg = SearchLoopConfig.from_env()
        # Operator marker mean-xy (admissible CityNav input, ~25 m from the
        # target; NOT the withheld GT). The mission engine resolves the
        # ``marked_region`` viewpoint token against it.
        _marker_positions = episode.marker_positions or []
        _marker_xy = (
            (
                float(sum(p[0] for p in _marker_positions) / len(_marker_positions)),
                float(sum(p[1] for p in _marker_positions) / len(_marker_positions)),
            )
            if _marker_positions
            else None
        )
        if os.environ.get("SEARCH_ORACLE_INJECT_GT", "").strip() == "1":
            # GT-nearest prior from the prebake graph at episode start
            # (the flight/osm_percepts.json sidecar the brief names is a
            # post-flight artifact and does not exist at compile time).
            _oracle_fid, _oracle_dist_m = _gt_nearest_prior_fid(
                graph, episode.target_xy
            )
            if _oracle_fid is None:
                raise RuntimeError(
                    "SEARCH_ORACLE_INJECT_GT=1 but the prebake graph for "
                    f"{episode.map_name} has no contour-bearing prior; "
                    "refusing to fly the diagnostic arm as a silent "
                    "baseline"
                )
            log.warning(
                "SEARCH_ORACLE_INJECT_GT=1: injecting GT-nearest prior "
                "%s (%.1f m from target_xy=(%.1f, %.1f)) into the "
                "compile viewpoint union",
                _oracle_fid,
                _oracle_dist_m,
                episode.target_xy[0],
                episode.target_xy[1],
            )
        search_loop = SearchLoop(
            engine_cfg,
            world_state,
            tile_origin_xy=(assets.bounds.x_min, assets.bounds.y_min),
            tile_name=f"citynav::{assets.map_name}",
            base_features=prior_features,
            compiler=compiler,
            sam_concept_setter=sam_concept_setter,
            sam_watchlist_setter=sam_watchlist_setter,
            satellite_view=assets.satellite_view,
            ortho_ground_z_m=ortho_ground_z_m,
            # Upstream raster dir for the opt-in ``look`` tool's crop
            # renderer. Passed as a str so maps.search stays
            # benchmark-agnostic (no ``import benchmark``); only consumed
            # when SEARCH_LOOK_TOOL + ortho are both active.
            rgbd_dir=str(CITYNAV_NATIVE_RGBD_DIR),
            marker_xy=_marker_xy,
            oracle_inject_fid=_oracle_fid,
        )
        search_loop.start(prebake_graph=graph, runtime_graph=graph)
        # Expose the engine on the EpisodeState so log/record surfaces can
        # read autonomy state without it leaking through the harness.
        state._engine_ref = search_loop

    tile_frame = TileFrame(
        x_min=float(assets.bounds.x_min),
        y_min=float(assets.bounds.y_min),
        cell_size_m=float(assets.cell_size_m),
        grid_shape=tuple(assets.grid_shape),
    )
    # Per-frame OSM↔SAM registration state (the ortho prior-enrichment ctx).
    osm_ctx: _OsmPriorCtx | None = None
    # True once ``osm_ctx`` is the CityRefer-free graph-sourced context (identity
    # fid map, SPEC §5c LEAK L1 fix) — then the GT-position registration step
    # (_resolve_osm_prior_graph_fids) is skipped (it is neither needed nor
    # allowed for the identity-keyed ctx). False for the legacy objects.json ctx.
    _osm_ctx_is_identity = False
    # The ortho backend gets an OSM prior-contour context so the per-frame
    # prior-color enrichment hook can PROJECT each in-view OSM prior footprint
    # onto the frame (build_prior_polys_ortho_from_contours, via
    # _maybe_build_prior_polys). Without it osm_ctx stays None and enrichment
    # is silently inert.
    #
    # SPEC §5c LEAK L1 FIX — the ortho backend sources the context
    # CityRefer-FREE from the baked OSM ``world_contour`` graph priors (identity
    # fid map), so objects.json GT geometry NEVER steers which prior is enriched
    # or the color-sampling footprint. z is unused in the ortho top-down
    # projection, so the 2-D OSM contours (z padded 0) are exact. The all-osm
    # capture goes to the side store + flight/osm_percepts.json sidecar; ONLY the
    # named subset graph_fid_by_fid maps is upserted onto the graph (headline
    # contract held).
    if osm_ctx is None:
        # Ortho pivot: re-source from the baked OSM world_contour priors.
        osm_ctx = _build_osm_prior_ctx_from_graph(graph)
        if osm_ctx is not None:
            _osm_ctx_is_identity = True
        if osm_ctx is None:
            if cfg.perception == _REALSTACK_PERCEPTION_TIER:
                # Real-SAM tier but no OSM world_contour source available →
                # honest no-GT: DISABLE enrichment. NEVER read objects.json here
                # (SPEC §5c LEAK L1). enrich_stats then show 0 enriched.
                log.warning(
                    "prior-enrichment: no OSM world_contour on graph priors for "
                    "%s (%s) under the %s tier — enrichment DISABLED (no "
                    "objects.json read; SPEC §5c LEAK L1)",
                    episode.map_name,
                    _REALSTACK_PERCEPTION_TIER,
                )
                osm_ctx = None
            else:
                # LEGACY DIAGNOSTIC path only (perspective backends, or a
                # non-osmv2 CityRefer diagnostic run off the real-SAM tier):
                # objects.json fallback. Unreachable under the pivot config —
                # the loader refuses sam.
                from benchmarks.citynav.paths import OBJECTS_JSON

                try:
                    osm_ctx = _load_osm_prior_ctx(
                        OBJECTS_JSON, episode.map_name, perception=cfg.perception
                    )
                    log.info(
                        "prior-enrichment: loaded %d CityRefer prior contours "
                        "for %s (LEGACY objects.json path)",
                        len(osm_ctx.contours_by_fid),
                        episode.map_name,
                    )
                except KeyError:
                    # map not in the CityRefer catalog (non-SensatUrban split)
                    osm_ctx = None
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "prior-enrichment: OSM prior ctx load failed for %s: %s "
                        "(enrichment disabled this episode)",
                        episode.map_name,
                        exc,
                    )
                    osm_ctx = None
    # Resolve each LEGACY (objects.json) OSM prior contour fid -> its runtime-
    # graph node id so the perspective prior-color enrichment upserts land on
    # real nodes (the headline bake keys priors ``<class>_<idx>``, not the OSM
    # cityref namespace). SKIPPED for the graph-sourced ctx, which is already
    # keyed by graph fid (identity) — re-resolving it via GT positions is
    # neither needed nor allowed (SPEC §5c LEAK L1). A no-op when no osm_ctx was
    # loaded (disabled / non-catalog maps).
    if osm_ctx is not None and not _osm_ctx_is_identity:
        _resolve_osm_prior_graph_fids(
            osm_ctx,
            graph,
            x_min=float(assets.bounds.x_min),
            y_min=float(assets.bounds.y_min),
            cell_size_m=float(assets.cell_size_m),
        )
    # TARGET-TYPE WALL (config B, PRIOR_CONTRACT.md): unnamed baked CityRefer
    # priors of the EPISODE'S TARGET annotation type pre-load the annotation's
    # target geometry, so build_surface_from_graph drops them (the vehicle
    # anti-cheat rule generalized — the target must always require live
    # perception). The wall value is the RAW CityRefer ``object_type`` (the
    # tag the bake stamps on cityrefer_unnamed rows); no label mapping sits on
    # the serving path.
    # Set-or-UNSET every episode: worker processes recycle across episodes and
    # a stale walled type from the previous episode must be impossible. The
    # env key is process-local per-episode state, read at the ONE graph→surface
    # seam; it is not a behavior A/B flag.
    #
    # SPEC §5c LEAK L1: this reads the target type only from a CITYREF-keyed
    # ``osm_ctx`` (the legacy objects.json ctx). Under the pivot the ctx is the
    # graph-sourced identity map (``cityref_id`` absent), so the lookup returns
    # nothing and the wall UNSETS — which is correct: the osmv2 bake never
    # rasterizes objects.json target geometry into a prior (base_features are
    # OSM), so there is no annotation-geometry prior to wall, and dropping the
    # GT-class read tightens C1. The wall stays live for the legacy path.
    os.environ.pop("SEARCH_WALLED_UNNAMED_TYPE", None)
    try:
        _tids = set(episode.target_object_ids or [])
        _wall_type = None
        if _tids and osm_ctx is not None:
            for _tid in _tids:
                _m = osm_ctx.meta_by_fid.get(f"cityref_{_tid}")
                if _m and _m.get("class_name"):
                    _wall_type = str(_m["class_name"])
                    break
        if _wall_type:
            os.environ["SEARCH_WALLED_UNNAMED_TYPE"] = _wall_type
            log.info(
                "target-type wall: unnamed '%s' priors dropped from the "
                "surface this episode", _wall_type,
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("target-type wall: resolution failed (%s); wall unset", exc)
    # Resolve the per-episode SensorSpec. Inline render.* keys still
    # override intrinsics for quick experiments; pitch overrides are
    # rejected because the runtime ingest expects the canonical nadir
    # SensorSpec (mount_pitch_deg=90) at the rotator-camera tier.
    episode_ply_path = None
    sensor = resolve_sensor(cfg.sensor) if cfg.sensor else None
    if sensor is not None:
        has_intr_override = any(
            k in cfg.render
            for k in ("hfov_deg", "vfov_deg", "image_width", "image_height")
        )
        if "pitch_deg" in cfg.render:
            raise ValueError(
                "cfg.render.pitch_deg overrides are no longer accepted; the "
                "runtime ingest requires a nadir SensorSpec (mount_pitch_deg=90)."
            )
        if has_intr_override:
            from maps.perception.ortho.camera_model import CameraIntrinsics as _CI
            from maps.perception.geometry.sensor_spec import SensorSpec as _Spec

            intr = sensor.intrinsics
            new_intr = _CI(
                hfov_deg=float(cfg.render.get("hfov_deg", intr.hfov_deg)),
                vfov_deg=float(cfg.render.get("vfov_deg", intr.vfov_deg)),
                image_width=int(cfg.render.get("image_width", intr.image_width)),
                image_height=int(cfg.render.get("image_height", intr.image_height)),
            )
            sensor = _Spec(
                name=sensor.name + "_override",
                intrinsics=new_intr,
                mount_pitch_deg=sensor.mount_pitch_deg,
                max_range_m=sensor.max_range_m,
                min_range_m=sensor.min_range_m,
                world_yaw_deg=sensor.world_yaw_deg,
                depth_subsample=sensor.depth_subsample,
                body_offset_x=sensor.body_offset_x,
                body_offset_y=sensor.body_offset_y,
                body_offset_z=sensor.body_offset_z,
            )
    state.sensor_spec = sensor
    if search_loop is not None:
        search_loop.state.sensor_spec = sensor
    # ---- RETROSPECTIVE PERCEPTION executor (spec C4) --------------------
    # The ortho ortho render is DETERMINISTIC over the static
    # raster, so an already-visited pose's observation is permanently
    # re-analyzable: a view{boost} targeting a FLOWN stop re-renders the
    # crop at the recorded pose (the same crop_observation path the flight
    # used), re-segments it with the boost vocabulary through the standard
    # /segment client, ingests through the NORMAL update_perception upsert
    # (world_contour + color-mixture stamping mirrored from the flight's
    # paint pass — keep the two in lockstep), and returns the delta
    # synchronously in the view tool result. Zero flight. crop_observation
    # is a pure raster read for the ortho backend, /segment is HTTP, and
    # update_perception takes the engine RLock (re-entrant under the boost
    # callback's lock).
    if search_loop is not None:
        def _retro_perceive(flown_entry: dict, concepts: list) -> dict:
            if perception is None or not hasattr(perception, "_segment"):
                return {"error": "no SAM segment client on this run"}
            pose_xy = flown_entry.get("pose_xy")
            if not (isinstance(pose_xy, (list, tuple)) and len(pose_xy) == 2):
                return {"error": "the flown observation's pose is unknown"}
            _px, _py = float(pose_xy[0]), float(pose_xy[1])
            _alt = flown_entry.get("alt_m")
            _alt = float(_alt) if _alt is not None else float(cfg.altitude_m)
            try:
                _obs = crop_observation(
                    assets.raster,
                    (_px, _py),
                    cfg.fov_m,
                    altitude_m=_alt,
                    yaw_deg=0.0,
                    sensor=sensor,
                    ply_path=episode_ply_path,
                    rgbd_dir=CITYNAV_NATIVE_RGBD_DIR,
                    map_name=episode.map_name,
                    rgb_upscale_px=cfg.rgb_upscale_px,
                )
            except Exception as exc:  # noqa: BLE001 — surfaced as a tool fact
                log.exception("retro re-render failed")
                return {"error": (
                    f"re-render failed: {type(exc).__name__}: {exc}")}
            if (_obs is None or _obs.rgb is None
                    or _obs.ortho_view_corners_xy is None):
                return {"error": "re-render produced no ortho frame"}
            try:
                _dets = perception._segment(_obs, list(concepts))
            except Exception as exc:  # noqa: BLE001 — surfaced as a tool fact
                log.exception("retro segment failed")
                return {"error": (
                    f"segment failed: {type(exc).__name__}: {exc}")}
            _obs.tile = tile_frame
            # Ortho project + stamp: the SAME shared paint pass a flight
            # runs (maps.perception.ortho.detection_stamping), so the
            # two ingest paths stay in lockstep by construction.
            _tracked: dict[int, object] = dict(
                project_and_stamp_ortho_detections(
                    _dets,
                    view_corners_xy=_obs.ortho_view_corners_xy,
                    rgb=_obs.rgb,
                    tile=tile_frame,
                )
            )

            def _detected_ids() -> set[str]:
                return {
                    nid for nid, node in
                    search_loop.state.runtime_graph.nodes.items()
                    if str(getattr(node, "provenance", "")) == "detected"
                }

            _before = _detected_ids()
            try:
                search_loop.update_perception(
                    step_idx=int(getattr(state, "step_idx", 0) or 0),
                    pose_xy=(_px, _py),
                    altitude_m=_alt,
                    heading_deg=0.0,
                    instruction=episode.description,
                    detections=_dets,
                    projected_masks=_tracked,
                    bounds_xy_min=(
                        float(assets.bounds.x_min),
                        float(assets.bounds.y_min),
                    ),
                    cell_size_m=float(assets.cell_size_m),
                    grid_shape=tuple(assets.grid_shape),
                    sensor_frames_rgb=[np.asarray(_obs.rgb)],
                )
            except Exception as exc:  # noqa: BLE001 — surfaced as a tool fact
                log.exception("retro ingest failed")
                return {"error": (
                    f"ingest failed: {type(exc).__name__}: {exc}")}
            _new = sorted(_detected_ids() - _before)
            _sid = flown_entry.get("stop_id")
            return {
                "n_dets": len(_dets),
                "n_new": len(_new),
                "new_fids": _new,
                "note": (
                    "re-segmented the recorded view at "
                    f"({_px:.0f}, {_py:.0f})"
                    + (f" ({_sid})" if _sid else "")
                    + f" for {list(concepts)}: "
                    + (f"{len(_new)} new detection(s) "
                       f"{_new}" if _new else "no new detections")
                ),
            }

        search_loop.set_retro_perceiver(_retro_perceive)

    t_start = time.monotonic()

    # The block frame and the rig, as every row of this attempt carries them.
    bounds_xyxy = [
        float(assets.bounds.x_min),
        float(assets.bounds.y_min),
        float(assets.bounds.x_max),
        float(assets.bounds.y_max),
    ]
    sensor_row = sensor.to_dict() if sensor is not None else None

    if cfg.resume_from_workdir and search_loop is not None:
        # One canonical resume path. ``restore_from_workdir`` loads
        # belief, target_spec, last_intel_text, SAM video concepts,
        # and compile_seq from the source run's on-disk artifacts.
        # ``hydrate_resume_artifacts`` mirrors the dashboard tree so
        # the SPA's compile-tab and reasoner panels populate.
        try:
            summary = search_loop.restore_from_workdir(
                cfg.resume_from_workdir,
                instruction=episode.description,
                turn=cfg.resume_turn,
            )
            if sam_concept_freezer is not None:
                sam_concept_freezer()
            state.belief_snapshot = search_loop.snapshot()
            hydrate_resume_artifacts(
                src_workdir=summary["src_workdir"],
                dst_workdir=os.environ.get("SEARCH_COMPILE_WORKDIR"),
                turn=int(summary["turn"]),
            )
        except Exception as exc:
            log.error("resume_from_workdir failed: %s", exc, exc_info=True)
            raise
    steps_taken = 0

    # The episode takeoff pose, snapshotted before the mission moves the
    # drone: the replay below flies from exactly here, so the submittable
    # trajectory starts where the episode did.
    _synth_start_pose_xyh = (
        float(state.pose_xy[0]),
        float(state.pose_xy[1]),
        float(state.heading_deg),
    )
    _synth_start_alt = float(state.altitude_m)

    # Pre-mission infrastructure warmup. The mission conversation is the
    # first (and only) LLM consumer, so prove the reasoner endpoint is
    # reachable before any flight spend.
    if search_loop is not None and compiler is not None:
        # Cold-start warmup: first prove the reasoner API is reachable, then
        # pre-pay representative thinking/schema/image requests. Without
        # this, the mission's first turn pays the cold
        # prefill/grammar/cudagraph path on the clock.
        _warmup_workdir_str = os.environ.get("SEARCH_COMPILE_WORKDIR", "").strip()
        _warmup_workdir = Path(_warmup_workdir_str) if _warmup_workdir_str else None
        try:
            _endpoint_spec = endpoint_spec_module.spec()
            _reasoner_warmup_ping(
                base_url=_endpoint_spec.base_url,
                model=_endpoint_spec.model_id,
                api_key=os.environ.get("SEARCH_REASONER_API_KEY", "EMPTY"),
                # Bound the warmup: a reachable vLLM answers on attempt 1
                # (cold FP8 compile ~2-3 min, well under this cap), so the
                # cap never false-aborts a live server but stops an
                # UNreachable endpoint from spinning "standing by for
                # tunnel" until the multi-hour episode timeout. On
                # exhaustion ReasonerUnreachableError is raised + caught below,
                # ending the episode so the bench's vLLM-health monitor can
                # re-queue it instead of hanging the slot for hours.
                total_wait_s=float(os.environ.get("SEARCH_REASONER_WARMUP_TOTAL_S", "900")),
                workdir=_warmup_workdir,
                # Only a dedicated engine we launched has a cold grammar /
                # CUDA-graph path worth pre-paying. A hosted endpoint is
                # already warm, and the warmup's tool-schema + image calls
                # would just be billed.
                pre_pay_cold_start=_endpoint_spec.server_declared_window,
            )
        except ReasonerUnreachableError as exc:
            log.error("Pre-flight aborted: vLLM unreachable (%s)", exc)
            return None
        # ``--stop-after ingest``: the graph seed (prebake load + mask cache
        # + SearchLoop.start) is the whole pre-flight, so this exits here,
        # after warmup and before any flight spend.
        if os.environ.get("SEARCH_STOP_AFTER", "").strip().lower() == "ingest":
            log.info("SEARCH_STOP_AFTER=ingest: exiting after the graph seed.")
            return None

    # Never-None floor: write a scored provisional row before the mission
    # flies. If the runner is SIGKILLed mid-flight (Slurm cgroup OOM), this
    # row is what the attempt's episode.json holds instead of nothing; the
    # landed row overwrites it at episode end. Best-effort — a failure here
    # must never abort the flight.
    if provisional_sink is not None:
        try:
            _prov_criterion, _prov_mask = _success_mask(cfg, episode, objects_by_map)
            provisional_sink(
                _build_provisional_rec(
                    episode=episode,
                    cfg=cfg,
                    snap=state.belief_snapshot,
                    feature_tour=None,
                    start_xy=state.pose_xy,
                    criterion=_prov_criterion,
                    mask=_prov_mask,
                    bounds=bounds_xyxy,
                    sensor=sensor,
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("provisional prediction write failed (non-fatal): %s", exc)

    # Mission branch: the episode is ONE reasoner conversation over the
    # seeded prior graph — requested viewpoints fly the moment they land,
    # the plain reply ends the mission, and the commit is the LAST flight's
    # target. ``synthesize_compliant_trajectory`` then rebuilds
    # ``visited_xy``/``pred_xy`` from a deterministic discrete-action
    # replay of the flown stops + committed final leg.
    mission_out: dict | None = None
    if search_loop is not None and compiler is not None:
        # True episode start for the packet/telemetry surfaces (the
        # SearchLoop otherwise only ever sees stop poses). Best-effort.
        try:
            search_loop.state.episode_start_xy = (
                float(episode.start_xy[0]),
                float(episode.start_xy[1]),
            )
        except Exception:  # noqa: BLE001
            pass
        # Mission telemetry floor: the director's watchlist/boost bumps and
        # the replay's truncation flags both land here. Aliased onto the
        # replay's EpisodeState so metrics read one dict.
        try:
            search_loop.state.react_stats = {
                "descent_starved": False,
                "descent_truncated": False,
            }
            state.react_stats = search_loop.state.react_stats
        except Exception:  # noqa: BLE001 — telemetry only, never fatal
            pass
        mission_out = run_mission_episode(
            cfg=cfg,
            state=state,
            assets=assets,
            episode=episode,
            sensor=sensor,
            tile_frame=tile_frame,
            episode_ply_path=episode_ply_path,
            perception=perception,
            search_loop=search_loop,
            recorder=recorder,
            osm_ctx=osm_ctx,
            compiler=compiler,
            t_start=t_start,
        )
        if not mission_out["flights"]:
            # ZERO flights -> honest zero: the drone never took off and no
            # target was committed. Scored-failure row, cause "no_flights".
            from benchmarks.citynav.eval import score_arrival

            _workdir = os.environ.get("SEARCH_COMPILE_WORKDIR")
            _write_pre_flight_health(
                Path(_workdir) if _workdir else None,
                {
                    "kind": "mission",
                    "status": "no_flights",
                    "cause": "no_flights",
                    "ended_by": mission_out["reply"].ended_by,
                    "turns_used": mission_out["reply"].turns_used,
                },
            )
            _pred_xy = (float(episode.start_xy[0]), float(episode.start_xy[1]))
            _pos_traj = [_pred_xy, _pred_xy]
            # ``flew=False`` is the honest zero: no target was committed and
            # the drone never took off, so EVERY success flag goes to 0 while
            # the distances stay true. The scorer owns that list. Naming the
            # flags here is what let the mirrored SR/OSR survive while the
            # four *_to_goal flags were zeroed -- and on a shape-scored bench
            # the corpus has an episode whose start pose sits INSIDE its own
            # target footprint, so a never-flown row scored SR=1.
            _criterion, _mask = _success_mask(cfg, episode, objects_by_map)
            _scores = score_arrival(
                criterion=_criterion,
                final_pos_xy=_pred_xy,
                pred_xy=_pred_xy,
                target_xy=episode.target_xy,
                pos_trajectory=_pos_traj,
                pred_trajectory=[_pred_xy],
                success_dist_m=cfg.success_radius_m,
                flew=False,
                **_mask,
            )
            _scores["referent_bucket"] = classify_referent(episode.description)
            _scores["steps_taken"] = 0
            _scores["compile_s"] = 0.0
            _scores["committed_fid"] = None
            _scores["committed_provenance"] = None
            _scores["committed_xy"] = None
            _scores["commit_to_target_m"] = None
            _scores["commit_would_score"] = None
            _scores["commit_step"] = None
            _scores["mission"] = {
                "flights_flown": 0,
                "ended_by": mission_out["reply"].ended_by,
                "turns_used": mission_out["reply"].turns_used,
            }
            return {
                "episode_id": episode.episode_id,
                "map_name": episode.map_name,
                "description": episode.description,
                "start_xy": list(episode.start_xy),
                "target_xy": list(episode.target_xy),
                "pred_xy": list(_pred_xy),
                "final_pos_xy": list(_pred_xy),
                "visited_xy": [list(p) for p in _pos_traj],
                "compliant_parallel": True,
                "bounds": bounds_xyxy,
                "sensor": sensor_row,
                "metrics": _scores,
            }
        steps_taken = synthesize_compliant_trajectory(
            cfg=cfg,
            state=state,
            episode=episode,
            feature_tour=mission_out["feature_tour"],
            start_pose_xyh=_synth_start_pose_xyh,
            start_alt=_synth_start_alt,
        )
    else:
        # No LLM (smoke configs): nothing can request a flight; hold at
        # start and score honestly.
        log.error(
            "no mission engine for %s (search_loop=%s, compiler=%s); "
            "holding at start",
            episode.episode_id,
            "ready" if search_loop is not None else "None",
            "ready" if compiler is not None else "None",
        )
        steps_taken = 0

    compile_s = time.monotonic() - t_start

    snap = state.belief_snapshot
    target_xy_world = (float(episode.target_xy[0]), float(episode.target_xy[1]))
    visited_dists = [
        math.hypot(float(p[0]) - target_xy_world[0], float(p[1]) - target_xy_world[1])
        for p in state.visited_xy
    ]
    min_dist_to_target = min(visited_dists) if visited_dists else float("inf")
    frames_within_R20 = sum(1 for d in visited_dists if d <= 20.0)
    frames_within_R200 = sum(1 for d in visited_dists if d <= 200.0)
    commit_step_for_log = snap.lock_step if snap is not None else None
    log.info(
        "EPISODE-END SA: episode=%s commit_step=%s commit_pose=%s "
        "min_dist_to_target=%.1fm frames_within_R20=%d frames_within_R200=%d "
        "frames_total=%d",
        episode.episode_id,
        str(commit_step_for_log),
        str((round(state.pose_xy[0], 1), round(state.pose_xy[1], 1)))
        if commit_step_for_log is not None
        else "None",
        min_dist_to_target,
        frames_within_R20,
        frames_within_R200,
        len(visited_dists),
    )

    # The committed target is flown as a normal trajectory leg. Prediction
    # output stays the replayed final drone pose so the row cannot imply
    # an unflown target was reached.
    final_pos_xy = (float(state.pose_xy[0]), float(state.pose_xy[1]))
    record_pred_xy = final_pos_xy

    # Commit-arrival integrity telemetry: the replay ending short of the
    # recorded commit must be LOUD, never silent.
    _committed_xy_end = (
        getattr(snap, "confirmed_target_xy", None) if snap is not None else None
    )
    if _committed_xy_end is not None:
        _short_of_commit_m = math.hypot(
            final_pos_xy[0] - float(_committed_xy_end[0]),
            final_pos_xy[1] - float(_committed_xy_end[1]),
        )
        if _short_of_commit_m > 5.0:
            log.warning(
                "EPISODE-END: flight ended %.1fm SHORT of the committed target "
                "%s (action budget exhausted mid-transit?) -> pred_xy is the "
                "final pose, NOT the commit (react_stats.descent_truncated)",
                _short_of_commit_m,
                (round(float(_committed_xy_end[0]), 1),
                 round(float(_committed_xy_end[1]), 1)),
            )
            _rs_end = getattr(state, "react_stats", None)
            if isinstance(_rs_end, dict):
                _rs_end["descent_truncated"] = True
                _rs_end["descent_short_m"] = round(_short_of_commit_m, 1)

    agent_traj = [(float(p[0]), float(p[1])) for p in state.visited_xy]
    # Upstream gsamllavanav/evaluate.py parity: 2D NE/SR/OSR with
    # both pos_to_goal (final agent pose) and pred_to_goal
    # (committed predicted target) families -- under THIS bench's success
    # mask, which decides what ``SR`` is a verdict on.
    from benchmarks.citynav.eval import score_arrival

    _criterion, _mask = _success_mask(cfg, episode, objects_by_map)
    scores = score_arrival(
        criterion=_criterion,
        final_pos_xy=final_pos_xy,
        pred_xy=record_pred_xy,
        target_xy=episode.target_xy,
        pos_trajectory=agent_traj,
        pred_trajectory=[record_pred_xy],
        success_dist_m=cfg.success_radius_m,
        **_mask,
    )
    _ne_pred = scores.get("ne_pred_to_goal")
    if _ne_pred is not None:
        _inv = math.hypot(
            record_pred_xy[0] - episode.target_xy[0],
            record_pred_xy[1] - episode.target_xy[1],
        )
        if abs(_inv - _ne_pred) > 0.5:
            log.error(
                "pred_xy invariant violated for %s: "
                "dist(pred_xy,target)=%.2f != ne_pred_to_goal=%.2f",
                episode.episode_id,
                _inv,
                _ne_pred,
            )
    # NOTE: ``referent_bucket`` is a NOISY TEXT PROXY (regex over the
    # description; matches "car" anywhere, so ~23% of building targets that
    # merely mention a nearby car are mislabeled "car"). The AUTHORITATIVE
    # target class is ``target_object_type`` (CityRefer GT, emitted by the
    # perception/discernment block below). Kept under both names so the proxy
    # is never mistaken for ground truth; downstream should prefer
    # ``target_object_type`` and treat ``referent_bucket*`` as a text heuristic.
    scores["referent_bucket"] = classify_referent(episode.description)
    scores["steps_taken"] = steps_taken
    scores["compile_s"] = round(compile_s, 2)
    # Present ONLY on SEARCH_ORACLE_INJECT_GT episodes — an injected row
    # must identify itself; a normal row must not carry an empty marker.
    if _oracle_fid is not None:
        scores["oracle_injected_fid"] = _oracle_fid
        scores["oracle_inject_dist_m"] = round(float(_oracle_dist_m), 1)
    # What the episode spent on the LLM, reduced to row size. The flight dir
    # already holds it twice over (transcript + net_meter), but neither
    # survives a rows-only telemetry pull and both cost a file walk per
    # episode to read, so the population view could not have tokens or turns
    # across a fleet. Read here, after the flight: the transcript writer
    # flushes per record, so the file is complete. Never raises.
    _flight_dir = os.environ.get("SEARCH_COMPILE_WORKDIR")
    scores["llm"] = (
        flight_summary.summarize(Path(_flight_dir) / "flight")
        if _flight_dir
        else flight_summary.blank()
    )
    # Commit identity comes from the mission engine's flight ledger: the
    # committed fid may be an "[x, y]" point token (provenance "point"),
    # which no prefix heuristic can classify.
    _mission_flights = mission_out["flights"] if mission_out is not None else []
    _committed_fid = (
        _mission_flights[-1]["target"] if _mission_flights else None
    )
    scores["committed_fid"] = _committed_fid
    scores["committed_provenance"] = (
        _mission_flights[-1]["provenance"] if _mission_flights else None
    )
    scores["mission"] = (
        {
            "flights_flown": len(_mission_flights),
            "ended_by": mission_out["reply"].ended_by,
            "turns_used": mission_out["reply"].turns_used,
            "no_call_turns": mission_out["reply"].no_call_turns,
        }
        if mission_out is not None
        else None
    )
    # The acquisition record (clause_coverage's read), off the mission
    # reasoner itself. Written ONLY when the mission ran: an absent key is
    # "instrument not present", which the reader must never conflate with
    # a measured empty.
    if mission_out is not None and "evidence_columns" in mission_out:
        scores["evidence_columns"] = list(mission_out["evidence_columns"])
    # Which branch bound the commit. The mission has exactly one --
    # "final_flight" -- so this is a tripwire: any other value (or None on a
    # flown episode) means something other than the last flight bound it.
    scores["bind_phase"] = (
        getattr(snap, "commit_bind_phase", None) if snap is not None else None
    )
    # PERCEPTION vs DISCERNMENT diagnostic. Per the anti-cheat protocol the
    # dynamic target classes (Car/Bike) are NOT in the prebake graph, so a
    # car target's GT is committable only if SAM detected that car in flight.
    # For each MISS we want to know: was a target-CLASS detection present near
    # the GT in the runtime graph? perception-limited (no target-class
    # detection near GT -> could not be chosen) vs discernment-limited (one
    # WAS there -> reasoner committed elsewhere). We emit the raw fields here
    # and classify offline from episodes.jsonl; the split is sharpest for the
    # Car/Bike buckets (structural targets always have a committable prior).
    #
    # The class MUST come from the target object's CityRefer ``object_type``
    # (scoring-side GT, admissible like target_xy), NOT classify_referent on
    # the description: the latter regex-matches "car" anywhere and mislabels
    # ~23% of building targets ("the house with a red car in front") as car.
    #   target_object_type: true CityRefer class of the GT target object
    #   target_class_for_diag: the open-vocab label the counters counted
    #   nearest_target_class_det_m: dist to nearest same-class detection
    #   n_target_class_dets: count of same-class detections anywhere
    #
    # The counters are written ONLY when the diagnostic ARMED. Hardcoding
    # them to 0/None first made every episode the diagnostic skipped read as
    # a measured perception failure: round 8's own first-draft attribution
    # scored 8 of 24 "coverage" misses off that artifact. An absent key is
    # "instrument not present" to every reader (discernment.band -> the
    # ``unrecorded`` bucket, episode_signal -> the ``class`` family absent
    # from ``recorded``), which is the fact.
    _target_object_type = None
    try:
        _tids = set(episode.target_object_ids or [])
        if _tids:
            for _o in objects_by_map.get(episode.map_name, []) or []:
                if getattr(_o, "id", None) in _tids:
                    _target_object_type = getattr(_o, "object_type", None)
                    break
    except Exception:  # noqa: BLE001
        _target_object_type = None
    scores["target_object_type"] = _target_object_type

    # EVERY referent bucket, off the one canonical CityRefer-type -> label
    # map. The hand-rolled car/bike/building subset that used to sit here
    # left Ground and Parking targets — 25 of a 300-run, 8.3% — outside the
    # diagnostic's domain forever, so they failed any
    # ``n_target_class_dets == 0`` test unconditionally.
    _target_class_for_diag = OBJTYPE_TO_LABEL.get(
        str(_target_object_type or "").strip()
    )
    scores["target_class_for_diag"] = _target_class_for_diag
    try:
        if _target_class_for_diag is not None and search_loop is not None:
            from maps.search.parse_compile.candidate_expansion import labels_match

            _rg_diag = getattr(search_loop.state, "runtime_graph", None)
            _n_dets = 0
            _nearest: float | None = None
            if _rg_diag is not None and hasattr(_rg_diag, "nodes"):
                for _fid, _node in _rg_diag.nodes.items():
                    if getattr(_node, "provenance", None) != "detected":
                        continue
                    if not labels_match(
                        getattr(_node, "class_name", ""), _target_class_for_diag
                    ):
                        continue
                    _n_dets += 1
                    _xy = search_loop.world_xy_for_node(_fid)
                    if _xy is None:
                        continue
                    _d = math.hypot(
                        float(_xy[0]) - target_xy_world[0],
                        float(_xy[1]) - target_xy_world[1],
                    )
                    if _nearest is None or _d < _nearest:
                        _nearest = _d
            scores["n_target_class_dets"] = _n_dets
            scores["nearest_target_class_det_m"] = (
                round(_nearest, 1) if _nearest is not None else None
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("perception/discernment diagnostic failed (non-fatal): %s", exc)

    # PRIOR-ENRICHMENT telemetry (makes the lever FALSIFIABLE). The per-flight
    # enrichment hook accumulates {frames, osm_seen, enriched, noop, fail} on the
    # SearchLoop state as it colors in-view OSM priors from the pixels; surface
    # it so the A/B + pd tooling can read whether the lever ACTUALLY FIRED (an
    # all-zero block = the enrichment did nothing this episode). ``backend`` is
    # the observation backend it ran under. Two derived fields make BOTH halves
    # falsifiable: ``osm_percepts`` (side-store size — the ALL-osm capture; must
    # be >> graph_enriched) and ``graph_enriched`` (NAMED-node upserts — the
    # committable, unchanged path; must stay > 0). The ``flight/osm_percepts.json``
    # sidecar (all-osm perceived color, named + unnamed) is also written here.
    # Emitted once per episode, where the prediction payload is written.
    _enrich = (
        getattr(search_loop.state, "prior_enrich_stats", None)
        if search_loop is not None
        else None
    )
    if isinstance(_enrich, dict):
        _enrich_block = dict(_enrich)
        _enrich_block["backend"] = "ortho"
        _enrich_block["graph_enriched"] = int(_enrich.get("enriched", 0))
        _osm_workdir = os.environ.get("SEARCH_COMPILE_WORKDIR")
        _percept_store = osm_ctx.percept_by_fid if osm_ctx is not None else {}
        _enrich_block["osm_percepts"] = len(_percept_store)
        _enrich_block["osm_sidecar_written"] = _write_osm_percepts_sidecar(
            _osm_workdir, _percept_store
        )
        scores["prior_enrich"] = _enrich_block
        log.info(
            "prior-enrichment: osm_seen=%d osm_percepts=%d graph_enriched=%d "
            "noop=%d frames=%d fail=%d backend=%s",
            _enrich.get("osm_seen", 0),
            _enrich_block["osm_percepts"],
            _enrich_block["graph_enriched"],
            _enrich.get("noop", 0),
            _enrich.get("frames", 0),
            _enrich.get("fail", 0),
            "ortho",
        )
    else:
        scores["prior_enrich"] = None

    # Loop telemetry, readable off episodes.jsonl instead of grepping runner
    # logs: the replay's truncation flags (descent_starved / descent_truncated
    # / descent_short_m) plus whatever the SearchLoop bumped in flight
    # (sam_concepts_redeclared, canceled_stops, boost counters).
    scores["react"] = (
        getattr(search_loop.state, "react_stats", None)
        if search_loop is not None
        else None
    )
    # SELECTION CEILING (decouples "picked the right place" from "the drone
    # physically reached it within the action budget"). ``committed_xy`` is
    # the last flight's target (snap.confirmed_target_xy); ``commit_to_target_m``
    # is its error vs ground truth; ``commit_would_score`` is True iff that
    # committed location itself SCORES under this bench's mask -- inside the
    # target's footprint where that is the criterion, within the success
    # radius where the ring is -- i.e. the episode
    # WOULD have scored had the replay flown there. Ring-basis on a
    # shape-scored row would put the funnel's commit stage on a different
    # question from its own last stage, and the two would stop nesting. The
    # honest Sankey split: a
    # failed episode with commit_would_score=True is an EXECUTION/budget miss
    # (the choice was right); commit_would_score=False is a SELECTION/grounding
    # miss (wrong region or wrong object). The mission flies its own commit, so
    # the execution arm should stay near-empty -- a growing one is a bug.
    _committed_xy = (
        snap.confirmed_target_xy if snap is not None else None
    )
    if _committed_xy is not None and episode.target_xy is not None:
        _cmt = math.hypot(
            float(_committed_xy[0]) - episode.target_xy[0],
            float(_committed_xy[1]) - episode.target_xy[1],
        )
        scores["committed_xy"] = [float(_committed_xy[0]), float(_committed_xy[1])]
        scores["commit_to_target_m"] = round(_cmt, 1)
        if _criterion == "shape":
            from benchmarks.citynav.eval import score_precise

            scores["commit_would_score"] = bool(
                score_precise(
                    final_xy=(float(_committed_xy[0]), float(_committed_xy[1])),
                    target_xy=episode.target_xy,
                    **_mask,
                )["SR"]
            )
        else:
            scores["commit_would_score"] = bool(_cmt <= cfg.success_radius_m)
    else:
        scores["committed_xy"] = None
        scores["commit_to_target_m"] = None
        scores["commit_would_score"] = None
    if os.environ.get("SEARCH_DUMP_REASONING_TRACE") and search_loop is not None:
        try:
            _wd = (os.environ.get("SEARCH_COMPILE_WORKDIR")
                   or getattr(cfg, "workdir", None) or getattr(state, "workdir", None))
            if _wd:
                _tp = Path(_wd) / "flight" / "reasoning_trace.jsonl"
                _tp.parent.mkdir(parents=True, exist_ok=True)
                # OBSERVED same-class set, INDEPENDENT of the color-qualified
                # candidate pool (POOL_HEADNOUN only refills on an EMPTY color pool,
                # so a wrong-color pool silently drops the GT car). This is the TRUE
                # observed set the offline within20 trichotomy keys off -- iterate the
                # runtime graph's detected nodes whose class matches the target type.
                _observed_same_class: list[dict] = []
                try:
                    from maps.search.parse_compile.candidate_expansion import labels_match
                    _rg_obs = getattr(search_loop.state, "runtime_graph", None)
                    _tot = scores.get("target_object_type") or ""
                    if _rg_obs is not None and hasattr(_rg_obs, "nodes") and _tot:
                        for _ofid, _onode in _rg_obs.nodes.items():
                            if getattr(_onode, "provenance", None) != "detected":
                                continue
                            _ocls = getattr(_onode, "class_name", "") or ""
                            if not labels_match(_ocls, _tot):
                                continue
                            _oxy = search_loop.world_xy_for_node(_ofid)
                            _observed_same_class.append({
                                "fid": _ofid,
                                "xy": ([float(_oxy[0]), float(_oxy[1])] if _oxy is not None else None),
                                "class": _ocls})
                except Exception:  # noqa: BLE001
                    _observed_same_class = []
                _crec = {"kind": "commit", "episode_id": episode.episode_id,
                    "target_object_type": scores.get("target_object_type"),
                    "target_xy": [target_xy_world[0], target_xy_world[1]],  # GT — OFFLINE dump only, never to the model
                    # The criterion says which of these two decided
                    # ``commit_would_score``; the radius is a fallback that
                    # fires on no row of a fully-shaped corpus.
                    "success_criterion": _criterion,
                    "success_radius_m": float(cfg.success_radius_m),
                    "committed_fid": getattr(snap, "locked_detection_id", None) if snap is not None else None,
                    "committed_xy": scores.get("committed_xy"),
                    "commit_to_target_m": scores.get("commit_to_target_m"),
                    "commit_would_score": scores.get("commit_would_score"),
                    "bind_phase": scores.get("bind_phase"),
                    "observed_same_class": _observed_same_class}
                with _tp.open("a") as _fh:
                    _fh.write(json.dumps(_crec, ensure_ascii=False, default=str) + "\n")
        except Exception as _exc:  # noqa: BLE001
            log.warning("reasoning_trace commit-record failed (non-fatal): %s", _exc)
    # (map-gap vs true-recall split is derived offline from the prebaked
    # FOOTPRINTS -- a node's centroid can sit >R from a GT on its own
    # footprint, so a centroid-based in-runner emit would mislabel large
    # buildings as map-gap. Footprint distance is computed offline against
    # the static per-tile features.json.)
    # commit_step: scout-fired step (preserves longitudinal score
    # comparability with pre-refactor traces).
    scores["commit_step"] = snap.lock_step if snap is not None else None

    # Content-derived, split-order-independent key (the ``episode_id`` ``#N``
    # suffix is a split-row index that shifts when the split is re-enumerated
    # or subset, so it can't be reconstructed offline -- this can). Lets
    # cross-run joins + custom subsets work without split-name masquerading.
    _ann = (episode.raw.get("ann_ids") or [None])[0]
    _obj = (episode.target_object_ids or [None])[0]
    episode_key = f"{episode.map_name}::ann{_ann}:obj{_obj}"
    rec: dict[str, Any] = {
        "episode_id": episode.episode_id,
        "episode_key": episode_key,
        "map_name": episode.map_name,
        "description": episode.description,
        "start_xy": list(episode.start_xy),
        "target_xy": list(episode.target_xy),
        "pred_xy": list(record_pred_xy),
        "final_pos_xy": list(final_pos_xy),
        "visited_xy": [list(p) for p in state.visited_xy],
        "compliant_parallel": True,
        "protocol": {
            "observation_backend": "ortho",
            "scoring_basis": scores.get("scoring_basis", "final_pos_xy"),
            "pred_xy_role": "final_drone_pose",
            "compliant_parallel": True,
            # The DISCOVERED served model (None on a --no-llm run).
            "llm_model": (
                endpoint_spec_module.spec().model_id
                if endpoint_spec_module.spec_installed()
                else None
            ),
            "perception": cfg.perception,
        },
        "bounds": bounds_xyxy,
        "sensor": sensor_row,
        "metrics": scores,
    }

    # World-state cleanout: no mask-cache disk twin to save (the ledger
    # rebuilds from graph.nodes in ~ms).
    return rec
