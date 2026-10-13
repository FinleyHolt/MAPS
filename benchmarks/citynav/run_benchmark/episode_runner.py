"""One episode's engine and, on the bench, its run: build the search stack
(shared with the flight mission), fly the mission conversation, replay the
flown stops in the six-action alphabet, and score the final pose.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import logging
import os
import re
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
from maps.sws.graph.model import GridDomain

log = logging.getLogger("run_benchmark")


@dataclass
class EpisodeState:
    """The harness's pose and visit history; the graph lives on the
    ``SearchLoop``."""

    pose_xy: tuple[float, float]
    altitude_m: float
    visited_xy: list[tuple[float, float]] = field(default_factory=list)
    step_idx: int = 0

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


def _success_shape(episode, objects_by_map: dict):
    """The shape ``eval.score_arrival`` tests: the linked object's contour
    when the episode names one, else its mask."""
    if not episode.target_object_ids:
        return episode.target_mask
    tids = set(episode.target_object_ids)
    return next((o.contour for o in objects_by_map.get(episode.map_name, []) or [] if o.id in tids), None)


def score_episode(cfg, episode, objects_by_map: dict, final_xy, trajectory, *,
                  commit_xy=None, flew: bool = True) -> dict:
    """The episode's metrics for an agent that ended at ``final_xy`` over
    ``trajectory``, having committed at ``commit_xy``, under this bench's
    success criterion (``paths.SUCCESS_CRITERION``). The one scoring call of
    the bench row and the flight mission's row."""
    from benchmarks.citynav import paths as bench_paths

    return score_arrival(
        criterion=bench_paths.SUCCESS_CRITERION,
        final_pos_xy=final_xy,
        target_xy=episode.target_xy,
        pos_trajectory=trajectory,
        success_dist_m=cfg.success_radius_m,
        gt_polygon=_success_shape(episode, objects_by_map),
        commit_xy=commit_xy,
        flew=flew,
    )


@dataclass
class Hydrated:
    """The search stack for one episode, built once by :func:`build_episode`
    for the bench and the flight mission alike."""

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
    state = EpisodeState(pose_xy=episode.start_xy, altitude_m=start_alt_m)
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
    # Runtime observations accumulate in this same graph; ``provenance``
    # separates them from the prebaked nodes.
    search_loop.start(graph)
    return Hydrated(
        cfg=cfg, state=state, assets=assets, episode=episode,
        sensor=resolve_sensor(cfg.sensor),
        tile_frame=TileFrame(x_min=float(assets.bounds.x_min), y_min=float(assets.bounds.y_min),
                             cell_size_m=float(assets.cell_size_m), grid_shape=tuple(assets.grid_shape)),
        perception=perception, search_loop=search_loop,
    )


def _detected_fids(search_loop) -> set[str]:
    return {nid for nid, node in search_loop.state.runtime_graph.nodes.items()
            if node.provenance == "detected"}


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
            cfg=cfg, state=state, flights=flights,
            start_pose_xyh=start_pose_xyh, start_alt=start_alt,
        )
        final_xy = (float(state.pose_xy[0]), float(state.pose_xy[1]))
        visited = state.visited_xy
    else:
        # ZERO flights: the drone never took off and nothing was committed.
        steps_taken = 0
        final_xy = (float(episode.start_xy[0]), float(episode.start_xy[1]))
        visited = [final_xy, final_xy]

    # Commit identity is the last flight's: it may be an "[x, y]" point
    # token (provenance "point"), which no fid prefix classifies.
    last = flights[-1] if flights else {}
    commit_xy = (float(last["resolved_xy"][0]), float(last["resolved_xy"][1])) if flights else None
    scores = score_episode(cfg, episode, objects_by_map, final_xy,
                           [(float(p[0]), float(p[1])) for p in visited],
                           commit_xy=commit_xy, flew=bool(flights))
    # ``referent_bucket`` is a noisy text proxy (regex over the description).
    scores["referent_bucket"] = classify_referent(episode.description)
    scores["steps_taken"] = steps_taken
    # What the episode spent on the LLM, at row size, read after the mission
    # from its transcript; a mission that never flew spent turns too.
    flight_dir = os.environ.get("SEARCH_COMPILE_WORKDIR")
    scores["llm"] = (flight_summary.summarize(Path(flight_dir) / "flight")
                     if flight_dir else flight_summary.blank())
    scores["committed_provenance"] = last.get("provenance")
    scores["mission"] = {"ended_by": reply.ended_by}
    # SAHI tiles the segmenter could not answer: their objects are missing
    # from those frames, which the episode survives.
    scores["sam_tiles_failed"] = int(h.perception.sahi_tiles_failed)

    return {
        "episode_id": episode.episode_id,
        "map_name": episode.map_name,
        "description": episode.description,
        "start_xy": list(episode.start_xy),
        "visited_xy": [list(p) for p in visited],
        "metrics": scores,
    }
