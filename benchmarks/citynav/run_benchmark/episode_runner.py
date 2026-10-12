"""One episode's engine and, on the bench, its run: build the search stack
(shared with the flight station), fly the mission conversation, replay the
flown stops in the six-action alphabet, and score the final pose.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.citynav.adapter import CityNavEpisode, parse_episode
from benchmarks.citynav.eval import score_arrival
from benchmarks.citynav.live_recorder import LiveRecorder
from benchmarks.citynav.paths import CITYNAV_NATIVE_RGBD_DIR, tile_stem
from benchmarks.citynav.run_benchmark.assets import BlockAssets, load_episode_assets
from benchmarks.citynav.run_benchmark.config import RunConfig
from benchmarks.citynav.run_benchmark.mission_engine import (
    MissionSession,
    perceive,
    run_mission_episode,
)
from benchmarks.citynav.run_benchmark.trajectory import synthesize_compliant_trajectory
from maps.llm import flight_summary
from maps.perception.geometry.geolocate import TileFrame
from maps.perception.geometry.sensor_spec import resolve_sensor
from maps.perception.observation import crop_observation
from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider
from maps.perception.sam.detect_client import SamDetectClient
from maps.search.search_loop import SearchLoop
from maps.search.withhold import withheld
from maps.sws.graph.builder import build_object_graph
from maps.sws.graph.io import load_graph
from maps.sws.graph.model import GridDomain, ObjectGraph

log = logging.getLogger("run_benchmark")


@dataclass
class EpisodeState:
    """The harness's pose, visit history, prior graph and bound commit; the
    runtime graph lives on the ``SearchLoop``."""

    pose_xy: tuple[float, float]
    altitude_m: float
    graph: ObjectGraph  # frozen prior graph (prebaked OSM/CityRefer)
    visited_xy: list[tuple[float, float]] = field(default_factory=list)
    step_idx: int = 0
    heading_deg: float = 0.0  # 0 = facing north (+y); derived from motion vector.
    # The last flight's resolved point, bound when the mission ends; None
    # when nothing flew. The trajectory replay flies to it.
    confirmed_target_xy: tuple[float, float] | None = None

    def record_visit(self) -> None:
        self.visited_xy.append(self.pose_xy)


def make_perception() -> SamDetectClient:
    """The SAM client. Its class set is the concept union of every active
    reasoner thread, pushed via ``set_concepts``; before the first push it
    detects nothing."""
    return SamDetectClient(concept_provider=ReasonerUnionConceptProvider())


def uniform_spawn_xy(
    bounds_xyxy: tuple[float, float, float, float],
    spawn_seed: int,
    episode_id: str,
) -> tuple[float, float]:
    """The ``spawn: uniform`` start: a point uniform over the tile bounds,
    deterministic per ``(spawn_seed, episode_id)`` from a hash fraction, so
    workers share no RNG state."""
    x_min, y_min, x_max, y_max = (float(v) for v in bounds_xyxy)
    digest = hashlib.sha256(
        f"spawn:{int(spawn_seed)}:{episode_id}".encode()
    ).hexdigest()
    fx = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    fy = int(digest[16:32], 16) / float(0xFFFFFFFFFFFFFFFF)
    return (x_min + fx * (x_max - x_min), y_min + fy * (y_max - y_min))


_CAR_PAT = re.compile(
    r"\b(car|cars|vehicle|truck|van|bus|taxi|suv|sedan|lorry)\b",
    re.IGNORECASE,
)
_BUILDING_PAT = re.compile(
    r"\b(building|buildings|tower|church|cathedral|hall|museum|hotel|shop|store|library)\b",
    re.IGNORECASE,
)


def classify_referent(description: str) -> str:
    """A row's ``referent_bucket``: car, building or other, by regex over the description."""
    if _CAR_PAT.search(description):
        return "car"
    if _BUILDING_PAT.search(description):
        return "building"
    return "other"


def start_altitude_m(cfg, map_name: str, start_z: float | None) -> tuple[float, float]:
    """``(start altitude AMSL, block ground datum)`` for one episode: the
    split row's start z, or ground + ``start_altitude_agl_m`` when set, held
    to ``[ground + 1, altitude_cap_m]``. The pair sets the observation span,
    ``2 * (alt - ground)``.
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

    The criterion is the bench's ``paths.SUCCESS_CRITERION``. The shape is the linked
    object's contour when the episode names one, else its mask; the fallback
    radius is the row's when it states one, else the config's.
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
    }


def score_episode(cfg, episode, objects_by_map: dict, final_xy, trajectory, *, flew: bool = True) -> dict:
    """The episode's metrics for an agent that ended at ``final_xy`` over
    ``trajectory``, under this bench's success mask (``eval.score_arrival``).
    The one scoring call of the bench row, its zero-flight row, the
    commit's would-score check and the flight station's mission row."""
    criterion, mask = _success_mask(cfg, episode, objects_by_map)
    return score_arrival(
        criterion=criterion,
        final_pos_xy=final_xy,
        target_xy=episode.target_xy,
        pos_trajectory=trajectory,
        success_dist_m=cfg.success_radius_m,
        flew=flew,
        **mask,
    )


@dataclass
class Hydrated:
    """The search stack for one episode, built once by :func:`build_episode`
    for the bench and the flight station alike."""

    cfg: RunConfig
    state: EpisodeState
    assets: BlockAssets
    episode: CityNavEpisode
    sensor: Any
    tile_frame: TileFrame
    perception: Any
    search_loop: SearchLoop


def build_episode(cfg: RunConfig, episode: CityNavEpisode, assets: BlockAssets) -> Hydrated:
    """The engine for one episode: its graph, the start pose, the segmenter
    with both of the agent's concept channels wired (the declared union and
    the director's watchlist, which a view{watchlist} change or a placed
    boost pushes), and the started SearchLoop."""
    graph = _episode_graph(tile_stem(episode.map_name), assets)
    start_alt_m, ground = start_altitude_m(cfg, episode.map_name, episode.start_z)
    # Runtime observations accumulate in this same graph; ``provenance``
    # separates them from the prebaked nodes.
    state = EpisodeState(pose_xy=episode.start_xy, altitude_m=start_alt_m, graph=graph)
    state.record_visit()
    # A module global looked up at call time: the flight's forced replay swaps it.
    perception = make_perception()
    grid_h, grid_w = assets.grid_shape
    search_loop = SearchLoop(
        GridDomain(height=grid_h, width=grid_w, cell_size_m=float(assets.cell_size_m)),
        tile_origin_xy=(assets.bounds.x_min, assets.bounds.y_min),
        map_name=assets.map_name,
        sam_setter=perception.concept_provider.set_concepts,
        # The mission's observation span is ``2 * (alt - ground)`` off it.
        ortho_ground_z_m=ground,
        # The look tool's crop raster (a live run crops its own imagery).
        rgbd_dir=str(CITYNAV_NATIVE_RGBD_DIR),
    )
    search_loop.start(graph)
    return Hydrated(
        cfg=cfg, state=state, assets=assets, episode=episode,
        sensor=resolve_sensor(cfg.sensor) if cfg.sensor else None,
        tile_frame=TileFrame(x_min=float(assets.bounds.x_min), y_min=float(assets.bounds.y_min),
                             cell_size_m=float(assets.cell_size_m), grid_shape=tuple(assets.grid_shape)),
        perception=perception, search_loop=search_loop,
    )


def _detected_fids(search_loop) -> set[str]:
    return {nid for nid, node in search_loop.state.runtime_graph.nodes.items()
            if str(getattr(node, "provenance", "")) == "detected"}


def retro_perceive(h: Hydrated, flown_entry: dict, concepts: list) -> dict:
    """The bench's retrospective perception: a view{boost} at a flown stop
    re-renders the raster crop at the recorded pose (north-up), re-segments
    it with the boost vocabulary and folds it as a flight does
    (:func:`mission_engine.perceive`); the delta is the tool result.
    Bench-only: the flight's camera is not the raster."""
    pose_xy = flown_entry.get("pose_xy")
    if not (isinstance(pose_xy, (list, tuple)) and len(pose_xy) == 2):
        return {"error": "the flown observation's pose is unknown"}
    px, py = float(pose_xy[0]), float(pose_xy[1])
    alt = float(flown_entry["alt_m"])
    try:
        obs = crop_observation(
            (px, py),
            altitude_m=alt,
            yaw_deg=0.0,
            sensor=h.sensor,
            rgbd_dir=CITYNAV_NATIVE_RGBD_DIR,
            map_name=h.episode.map_name,
        )
    except Exception as exc:  # noqa: BLE001 — surfaced as a tool fact
        log.exception("retro re-render failed")
        return {"error": f"re-render failed: {type(exc).__name__}: {exc}"}
    if obs is None or obs.rgb is None or obs.ortho_view_corners_xy is None:
        return {"error": "re-render produced no ortho frame"}
    try:
        dets = h.perception._segment(obs, list(concepts))
    except Exception as exc:  # noqa: BLE001 — surfaced as a tool fact
        log.exception("retro segment failed")
        return {"error": f"segment failed: {type(exc).__name__}: {exc}"}
    obs.tile = h.tile_frame
    before = _detected_fids(h.search_loop)
    try:
        perceive(h, obs, dets)
    except Exception as exc:  # noqa: BLE001 — surfaced as a tool fact
        log.exception("retro ingest failed")
        return {"error": f"ingest failed: {type(exc).__name__}: {exc}"}
    new = sorted(_detected_fids(h.search_loop) - before)
    sid = flown_entry.get("stop_id")
    return {
        "n_dets": len(dets),
        "n_new": len(new),
        "new_fids": new,
        "note": (
            f"re-segmented the recorded view at ({px:.0f}, {py:.0f})"
            + (f" ({sid})" if sid else "")
            + f" for {list(concepts)}: "
            + (f"{len(new)} new detection(s) {new}" if new else "no new detections")
        ),
    }


def run_episode(
    row: dict,
    objects_by_map: dict,
    cfg: RunConfig,
    recorder: LiveRecorder | None = None,
    row_index: int | None = None,
) -> dict | None:
    """Fly one split row and return its scored record (None: no CityRefer
    objects on its map, nothing to score)."""
    episode = parse_episode(row, row_index=row_index)
    if not objects_by_map.get(episode.map_name, []):
        log.warning("  skip %s: no CityRefer objects", episode.episode_id)
        return None

    assets = load_episode_assets(episode.map_name)
    log.info("tile_stem=%s map_name=%s", tile_stem(episode.map_name), episode.map_name)
    if cfg.spawn == "uniform":
        # The FLOWN start replaces the dataset's on the episode itself, so
        # every downstream reader sees it.
        b = assets.bounds
        episode = dataclasses.replace(episode, start_xy=uniform_spawn_xy(
            (float(b.x_min), float(b.y_min), float(b.x_max), float(b.y_max)),
            cfg.spawn_seed, episode.episode_id))
        log.info(
            "  uniform spawn: (%.1f, %.1f) (spawn_seed=%d, episode=%s)",
            episode.start_xy[0],
            episode.start_xy[1],
            int(cfg.spawn_seed),
            episode.episode_id,
        )
    h = build_episode(cfg, episode, assets)
    state = h.state
    ground = h.search_loop._ortho_ground_z_m
    log.info(
        "  ortho start altitude: %.2f m (%s, ground=%.2f, "
        "cap=%.0f, view_square=%.0fm, episode=%s)",
        state.altitude_m,
        (
            f"AGL {cfg.start_altitude_agl_m:.0f} m"
            if cfg.start_altitude_agl_m is not None
            else f"row z={episode.start_z:.2f}"
        ),
        ground,
        float(cfg.altitude_cap_m),
        2.0 * (state.altitude_m - ground),
        episode.episode_id,
    )
    h.search_loop.set_retro_perceiver(functools.partial(retro_perceive, h))

    t_start = time.monotonic()
    # The takeoff pose, before the mission moves the drone: the replay flies
    # from exactly here, facing upstream's start heading.
    start_pose_xyh = (float(state.pose_xy[0]), float(state.pose_xy[1]),
                      float(episode.start_heading_deg))
    start_alt = float(state.altitude_m)
    session = MissionSession(h, recorder=recorder)
    reply = run_mission_episode(session)
    flights = session.flights
    if flights:
        steps_taken = synthesize_compliant_trajectory(
            cfg=cfg, state=state, episode=episode, feature_tour=session.feature_tour,
            start_pose_xyh=start_pose_xyh, start_alt=start_alt,
        )
        compile_s = round(time.monotonic() - t_start, 2)
        final_xy = (float(state.pose_xy[0]), float(state.pose_xy[1]))
        visited = state.visited_xy
    else:
        # ZERO flights: the drone never took off and nothing was committed.
        steps_taken, compile_s = 0, 0.0
        final_xy = (float(episode.start_xy[0]), float(episode.start_xy[1]))
        visited = [final_xy, final_xy]

    scores = score_episode(cfg, episode, objects_by_map, final_xy,
                           [(float(p[0]), float(p[1])) for p in visited], flew=bool(flights))
    # ``referent_bucket`` is a noisy text proxy (regex over the description).
    scores["referent_bucket"] = classify_referent(episode.description)
    scores["steps_taken"] = steps_taken
    scores["compile_s"] = compile_s
    # What the episode spent on the LLM, at row size, read after the mission
    # from its transcript; a mission that never flew spent turns too.
    flight_dir = os.environ.get("SEARCH_COMPILE_WORKDIR")
    scores["llm"] = (flight_summary.summarize(Path(flight_dir) / "flight")
                     if flight_dir else flight_summary.blank())
    # Commit identity is the last flight's: it may be an "[x, y]" point
    # token (provenance "point"), which no fid prefix classifies.
    last = flights[-1] if flights else {}
    scores["committed_fid"] = last.get("target")
    scores["committed_provenance"] = last.get("provenance")
    scores["mission"] = {"ended_by": reply.ended_by}
    # Would the committed point itself score? Separates execution misses
    # from selection misses.
    if flights:
        cxy = (float(last["resolved_xy"][0]), float(last["resolved_xy"][1]))
        scores["committed_xy"] = list(cxy)
        scores["commit_would_score"] = bool(score_episode(cfg, episode, objects_by_map, cxy, [cxy])["SR"])
    else:
        scores["committed_xy"] = scores["commit_would_score"] = None
    # SAHI tiles the segmenter could not answer: their objects are missing
    # from those frames, which the episode survives.
    scores["sam_tiles_failed"] = int(h.perception.sahi_tiles_failed)

    return {
        "episode_id": episode.episode_id,
        "map_name": episode.map_name,
        "description": episode.description,
        "start_xy": list(episode.start_xy),
        "target_xy": list(episode.target_xy),
        "final_pos_xy": list(final_xy),
        "visited_xy": [list(p) for p in visited],
        "metrics": scores,
    }
