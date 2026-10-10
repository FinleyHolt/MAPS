"""run_episode: per-episode setup -> mission conversation (fly-on-request)
-> trajectory synthesis -> scored result record.
"""

from __future__ import annotations

from benchmarks.citynav.adapter import parse_episode
from benchmarks.citynav.episode import EpisodeState
from benchmarks.citynav.live_recorder import LiveRecorder
from benchmarks.citynav.metrics import classify_referent
from benchmarks.citynav.paths import CITYNAV_NATIVE_RGBD_DIR, tile_stem
from benchmarks.citynav.run_benchmark.config import RunConfig
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
)
from maps.sws import GridDomain
from maps.search.withhold import withheld
from maps.sws.graph import build_object_graph, load_graph
import math
import os
import time
from benchmarks.citynav.run_benchmark.assets import load_episode_assets
from benchmarks.citynav.run_benchmark.mission_engine import run_mission_episode
from benchmarks.citynav.run_benchmark.perception import make_perception
from benchmarks.citynav.run_benchmark.trajectory import synthesize_compliant_trajectory
from benchmarks.citynav.run_benchmark.reasoner_warmup import (
    ReasonerUnreachableError,
    _reasoner_warmup_ping,
)
import logging

log = logging.getLogger("run_benchmark")


def start_altitude_m(cfg, map_name: str, start_z: float | None) -> tuple[float, float]:
    """``(start altitude AMSL, block ground datum)`` for one episode.

    The altitude is upstream's start z, the split row's ``trajectory[0]``
    z; ``start_altitude_agl_m`` replaces it with a uniform above-ground
    cruise. The result is held to ``[ground + 1, altitude_cap_m]``.

    This pair IS the crop window's observation ceiling: the mission's
    ``obs_span_m = 2 * (alt - ground)``. ``cfg.altitude_m`` is not the
    start pose.
    """
    from maps.perception.ortho.ground_level import (  # noqa: PLC0415
        GROUND_LEVEL,
    )

    ground = float(GROUND_LEVEL.get(map_name, 0.0))
    if cfg.start_altitude_agl_m is not None:
        alt = ground + float(cfg.start_altitude_agl_m)
    elif start_z is None:
        raise ValueError(
            f"{map_name}: the episode has no start z and the config sets no "
            "start_altitude_agl_m"
        )
    else:
        alt = float(start_z)
    return (
        float(min(max(alt, ground + 1.0), float(cfg.altitude_cap_m))),
        ground,
    )


def _episode_graph(stem: str, assets):
    """The episode's object graph. The ``prior`` withhold starts it empty."""
    if withheld("prior"):
        return build_object_graph({}, cell_size_m=float(assets.cell_size_m))
    try:
        return load_graph(stem)
    except FileNotFoundError:
        log.warning("  %s: no baked graph, building on the fly", stem)
        return build_object_graph(
            assets.base_features, cell_size_m=float(assets.cell_size_m),
        )


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
        "tolerance_m": float(cfg.precise_tolerance_m),
        "floor_radius_m": float(cfg.precise_floor_radius_m),
    }


def run_episode(
    row: dict,
    compiler: MapsCompiler | None,
    objects_by_map: dict,
    cfg: RunConfig,
    recorder: LiveRecorder | None = None,
    row_index: int | None = None,
) -> dict | None:
    episode = parse_episode(row, row_index=row_index)
    objs_for_map = objects_by_map.get(episode.map_name, [])
    if not objs_for_map:
        log.warning("  skip %s: no CityRefer objects", episode.episode_id)
        return None

    assets = load_episode_assets(episode.map_name, cfg.cell_size_m)
    graph_stem = tile_stem(episode.map_name)
    log.info("tile_stem=%s map_name=%s", graph_stem, episode.map_name)
    graph = _episode_graph(graph_stem, assets)

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
    # The start pose's altitude and the block's ground datum. The datum is
    # threaded to the orthographic viewpoint picker so it sizes the view
    # square from the same scalar the crop uses (``side = 2 * (alt -
    # ground)``) instead of any per-object height prior.
    start_alt_m, ortho_ground_z_m = start_altitude_m(
        cfg, episode.map_name, episode.start_z
    )
    log.info(
        "  ortho start altitude: %.2f m (%s, ground=%.2f, "
        "cap=%.0f, view_square=%.0fm, episode=%s)",
        start_alt_m,
        (
            f"AGL {cfg.start_altitude_agl_m:.0f} m"
            if cfg.start_altitude_agl_m is not None
            else f"row z={episode.start_z:.2f}"
        ),
        ortho_ground_z_m,
        float(cfg.altitude_cap_m),
        2.0 * (start_alt_m - ortho_ground_z_m),
        episode.episode_id,
    )
    state = EpisodeState(
        pose_xy=episode.start_xy,
        altitude_m=start_alt_m,
        graph=graph,
    )
    # Runtime observations accumulate in the SAME graph as prebaked nodes;
    # ``provenance`` is the only tag separating them. The belief engine
    # mutates ``graph`` in place via maps.sws.graph.upsert_*; downstream
    # consumers that filter by provenance get the right detected subset.
    state.record_visit()

    grid_h, grid_w = assets.grid_shape
    grid = GridDomain(
        height=grid_h, width=grid_w, cell_size_m=float(assets.cell_size_m)
    )

    perception = make_perception(cfg)

    # Build the belief engine. The compiler carries the reasoner endpoint
    # facts the mission engine reads (None disables the LLM path entirely;
    # the engine still runs perception_update / paint in that mode). The
    # SAM concept setter pushes each act.sam_concepts union back into a
    # SAM-style perception adapter; non-SAM adapters get a no-op. The
    # mission declares concepts latest-wins.
    sam_concept_setter = None
    sam_watchlist_setter = None
    if isinstance(perception, SamDetectClient):
        cp = perception.concept_provider
        if isinstance(cp, ReasonerUnionConceptProvider):
            sam_concept_setter = cp.set_concepts
            # The agent-owned watchlist channel: how a view{watchlist}
            # mutation / boost activation reaches the per-flight /segment
            # calls.
            sam_watchlist_setter = cp.apply_watchlist

    search_loop: SearchLoop | None = None
    if compiler is not None:
        search_loop = SearchLoop(
            grid,
            tile_origin_xy=(assets.bounds.x_min, assets.bounds.y_min),
            tile_name=f"citynav::{assets.map_name}",
            compiler=compiler,
            sam_concept_setter=sam_concept_setter,
            sam_watchlist_setter=sam_watchlist_setter,
            ortho_ground_z_m=ortho_ground_z_m,
            # Upstream raster dir for the ``look`` tool's crop renderer,
            # passed as a str so maps.search stays benchmark-agnostic.
            rgbd_dir=str(CITYNAV_NATIVE_RGBD_DIR),
        )
        search_loop.start(graph)

    tile_frame = TileFrame(
        x_min=float(assets.bounds.x_min),
        y_min=float(assets.bounds.y_min),
        cell_size_m=float(assets.cell_size_m),
        grid_shape=tuple(assets.grid_shape),
    )
    # Resolve the per-episode SensorSpec (a registered name or an inline
    # sensor override; resolve_sensor).
    sensor = resolve_sensor(cfg.sensor) if cfg.sensor else None
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
                    (_px, _py),
                    altitude_m=_alt,
                    yaw_deg=0.0,
                    sensor=sensor,
                    rgbd_dir=CITYNAV_NATIVE_RGBD_DIR,
                    map_name=episode.map_name,
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
                    detections=_dets,
                    projected_masks=_tracked,
                    bounds_xy_min=(
                        float(assets.bounds.x_min),
                        float(assets.bounds.y_min),
                    ),
                    cell_size_m=float(assets.cell_size_m),
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

    steps_taken = 0

    # The episode takeoff pose, snapshotted before the mission moves the
    # drone: the replay below flies from exactly here, facing upstream's
    # start heading, so the submittable trajectory starts where the episode
    # did and the turn to the first stop costs actions. The mission works
    # north-up, as every frame the model reads is.
    _synth_start_pose_xyh = (
        float(state.pose_xy[0]),
        float(state.pose_xy[1]),
        float(episode.start_heading_deg),
    )
    _synth_start_alt = float(state.altitude_m)

    # The mission conversation is the first (and only) LLM consumer, so
    # prove the reasoner endpoint is reachable before any flight spend.
    if search_loop is not None and compiler is not None:
        try:
            _endpoint_spec = endpoint_spec_module.spec()
            _reasoner_warmup_ping(
                base_url=_endpoint_spec.base_url,
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

    # Mission branch: the episode is ONE reasoner conversation over the
    # seeded prior graph — requested viewpoints fly the moment they land,
    # the plain reply ends the mission, and the commit is the LAST flight's
    # target. ``synthesize_compliant_trajectory`` then rebuilds
    # ``visited_xy``/``pred_xy`` from a deterministic discrete-action
    # replay of the flown stops + committed final leg.
    mission_out: dict | None = None
    if search_loop is not None and compiler is not None:
        mission_out = run_mission_episode(
            cfg=cfg,
            state=state,
            assets=assets,
            episode=episode,
            sensor=sensor,
            tile_frame=tile_frame,
            perception=perception,
            search_loop=search_loop,
            recorder=recorder,
            compiler=compiler,
            t_start=t_start,
        )
        if not mission_out["flights"]:
            # ZERO flights -> honest zero: the drone never took off and no
            # target was committed. Scored-failure row, cause "no_flights".
            from benchmarks.citynav.eval import score_arrival

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
            _scores["commit_would_score"] = None
            _scores["sam_tiles_failed"] = int(getattr(perception, "sahi_tiles_failed", 0))
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
                "final pose, NOT the commit",
                _short_of_commit_m,
                (round(float(_committed_xy_end[0]), 1),
                 round(float(_committed_xy_end[1]), 1)),
            )

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
    # ``referent_bucket`` is a NOISY TEXT PROXY (regex over the description;
    # matches "car" anywhere, so ~23% of building targets that merely mention
    # a nearby car are mislabeled "car").
    scores["referent_bucket"] = classify_referent(episode.description)
    scores["steps_taken"] = steps_taken
    scores["compile_s"] = round(compile_s, 2)
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
    # SELECTION CEILING (decouples "picked the right place" from "the drone
    # physically reached it within the action budget"). ``committed_xy`` is
    # the last flight's target (snap.confirmed_target_xy); ``commit_would_score``
    # is True iff that
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
        scores["commit_would_score"] = None
    # SAHI tiles the segmenter could not answer: their objects are missing
    # from those frames, which the episode survives.
    scores["sam_tiles_failed"] = int(getattr(perception, "sahi_tiles_failed", 0))

    rec: dict[str, Any] = {
        "episode_id": episode.episode_id,
        "map_name": episode.map_name,
        "description": episode.description,
        "start_xy": list(episode.start_xy),
        "target_xy": list(episode.target_xy),
        "pred_xy": list(record_pred_xy),
        "final_pos_xy": list(final_pos_xy),
        "visited_xy": [list(p) for p in state.visited_xy],
        "metrics": scores,
    }

    # World-state cleanout: no mask-cache disk twin to save (the ledger
    # rebuilds from graph.nodes in ~ms).
    return rec
