"""One episode's engine and, on the bench, its run: build the search stack
(shared with the flight mission), fly the mission conversation, replay the
flown stops in the six-action alphabet, and score the final pose.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

from benchmarks.bench import Bench
from benchmarks.citynav.adapter import CityNavEpisode
from benchmarks.citynav.eval import score_arrival
from benchmarks.run_record.episode_workdir import LiveRecorder
from benchmarks.citynav.run_benchmark.assets import BlockAssets, load_episode_assets
from benchmarks.run_record.config import RunConfig
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
from maps.perception.ortho.ground_level import GROUND_LEVEL
from maps.perception.sam.detect_client import SamDetectClient
from maps.mission.episode_world import EpisodeWorld
from maps.settings import Settings
from maps.world_state.prior_graph import build_object_graph
from maps.world_state.graph import GridDomain

log = logging.getLogger("run_benchmark")


@dataclass
class EpisodeState:
    """The aircraft's pose during the mission and the stop it is flying; the
    graph lives on the ``EpisodeWorld``."""

    pose_xy: tuple[float, float]
    altitude_m: float
    step_idx: int = 0


def make_perception(settings: Settings) -> SamDetectClient:
    """The SAM client over the replicas ``settings`` names; it detects the
    classes last pushed through ``set_concepts`` (none before the first push)."""
    return SamDetectClient(endpoint=settings.endpoint)


def uniform_spawn_xy(
    bounds_xyxy: tuple[float, float, float, float],
    spawn_seed: int,
    episode_id: str,
) -> tuple[float, float]:
    """The ``spawn: uniform`` start: a point uniform over the tile bounds,
    deterministic per ``(spawn_seed, episode_id)`` from a hash fraction, so
    workers share no RNG state."""
    x_min, y_min, x_max, y_max = (float(v) for v in bounds_xyxy)
    digest = hashlib.sha256(f"spawn:{int(spawn_seed)}:{episode_id}".encode()).hexdigest()
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
    """``(start altitude AMSL, block ground datum)`` for one episode: ground
    + ``start_altitude_agl_m`` when set, else the split row's start z, held to
    ``[ground + 1, altitude_cap_m]``."""
    ground = float(GROUND_LEVEL[map_name])
    if cfg.start_altitude_agl_m is not None:
        alt = ground + float(cfg.start_altitude_agl_m)
    elif start_z is None:
        raise ValueError(
            f"{map_name}: the episode has no start z and the config sets no start_altitude_agl_m"
        )
    else:
        alt = float(start_z)
    return (
        float(min(max(alt, ground + 1.0), float(cfg.altitude_cap_m))),
        ground,
    )


def _episode_graph(assets):
    """The episode's object graph, built from the bake's features."""
    return build_object_graph(assets.base_features, cell_size_m=float(assets.cell_size_m))


def success_shape(episode, objects_by_map: dict):
    """The shape ``eval.score_arrival`` tests: the linked object's contour
    when the episode names one, else its mask."""
    if not episode.target_object_ids:
        return episode.target_mask
    tids = set(episode.target_object_ids)
    return next(
        (o.contour for o in objects_by_map.get(episode.map_name) or [] if o.id in tids), None
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
    # The block's ground datum: every height above ground is read off it.
    ground_z_m: float
    perception: Any
    world: EpisodeWorld
    #: The bench the episode is flown on: its rasters, its bake, its criterion.
    bench: Bench
    #: The run's engine settings (``maps.settings``): endpoints, sampling.
    settings: Settings
    #: The attempt's directory, where the mission records (its wire log, its
    #: frames, its tour cursor); None records nothing.
    workdir: Path | None = None
    #: The mission's router-affinity key (the episode's id on a run).
    routing_key: str | None = None

    def crop(self, xy: tuple[float, float], altitude_m: float):
        """The raster's north-up frame through the run's sensor at this pose."""
        return crop_observation(
            (float(xy[0]), float(xy[1])),
            altitude_m=float(altitude_m),
            sensor=self.sensor,
            rgbd_dir=self.bench.rgbd_dir,
            map_name=self.episode.map_name,
        )


def build_episode(
    cfg: RunConfig,
    episode: CityNavEpisode,
    assets: BlockAssets,
    bench: Bench,
    *,
    settings: Settings,
    workdir: Path | None = None,
    routing_key: str | None = None,
) -> Hydrated:
    """The engine for one episode on ``bench`` under ``settings``: its graph,
    the start pose, the segmenter wired to the director's watchlist (the
    declared classes plus the boosts riding a flight), and the started
    EpisodeWorld. ``workdir`` and ``routing_key`` are the mission's own
    (:class:`Hydrated`)."""
    graph = _episode_graph(assets)
    start_alt_m, ground = start_altitude_m(cfg, episode.map_name, episode.start_z)
    state = EpisodeState(pose_xy=episode.start_xy, altitude_m=start_alt_m)
    # ``make_perception`` is a module global looked up at call time: a test swaps it.
    perception = make_perception(settings)
    grid_h, grid_w = assets.grid_shape
    world = EpisodeWorld(
        GridDomain(height=grid_h, width=grid_w, cell_size_m=float(assets.cell_size_m)),
        tile_origin_xy=(assets.bounds.x_min, assets.bounds.y_min),
        map_name=assets.map_name,
        sam_setter=perception.set_concepts,
        # The look tool's crop raster (a live run crops its own imagery).
        rgbd_dir=str(bench.rgbd_dir),
    )
    # Runtime observations accumulate in this same graph; ``provenance``
    # separates them from the prebaked nodes.
    world.start(graph)
    return Hydrated(
        cfg=cfg,
        state=state,
        assets=assets,
        episode=episode,
        sensor=resolve_sensor(cfg.sensor),
        tile_frame=TileFrame(
            x_min=float(assets.bounds.x_min),
            y_min=float(assets.bounds.y_min),
            cell_size_m=float(assets.cell_size_m),
            grid_shape=tuple(assets.grid_shape),
        ),
        ground_z_m=ground,
        perception=perception,
        world=world,
        bench=bench,
        settings=settings,
        workdir=workdir,
        routing_key=routing_key,
    )


def _detected_fids(world) -> set[str]:
    return {
        nid
        for nid, node in world.state.runtime_graph.nodes.items()
        if node.provenance == "detected"
    }


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
        obs = h.crop((px, py), alt)
    except Exception as exc:  # noqa: BLE001 - surfaced as a tool fact
        log.exception("retro re-render failed")
        return {"error": f"re-render failed: {type(exc).__name__}: {exc}"}
    try:
        dets = h.perception.segment(obs, list(concepts))
    except Exception as exc:  # noqa: BLE001 - surfaced as a tool fact
        log.exception("retro segment failed")
        return {"error": f"segment failed: {type(exc).__name__}: {exc}"}
    before = _detected_fids(h.world)
    try:
        perceive(h, obs, dets)
    except Exception as exc:  # noqa: BLE001 - surfaced as a tool fact
        log.exception("retro ingest failed")
        return {"error": f"ingest failed: {type(exc).__name__}: {exc}"}
    new = sorted(_detected_fids(h.world) - before)
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
    episode: CityNavEpisode,
    objects_by_map: dict,
    cfg: RunConfig,
    bench: Bench,
    recorder: LiveRecorder | None = None,
    *,
    settings: Settings,
    workdir: Path | None = None,
    routing_key: str | None = None,
    converse: Callable[[MissionSession], Any] | None = None,
) -> dict:
    """Fly one episode of ``bench`` under ``settings`` and return its scored
    record; ``workdir`` is the attempt's directory and ``routing_key`` its
    router-affinity key.
    ``converse`` drives the mission conversation over the episode's session in
    place of :func:`run_mission_episode` and returns its last reply (the
    station's, which holds the conversation open for an operator); the build,
    the replay and the scoring around it are this function's either way."""
    assets = load_episode_assets(bench, episode.map_name)
    log.info("tile_stem=%s map_name=%s", bench.tile_stem(episode.map_name), episode.map_name)
    if cfg.spawn == "uniform":
        # The FLOWN start replaces the dataset's on the episode itself, so
        # every downstream reader sees it.
        b = assets.bounds
        episode = dataclasses.replace(
            episode,
            start_xy=uniform_spawn_xy(
                (float(b.x_min), float(b.y_min), float(b.x_max), float(b.y_max)),
                cfg.spawn_seed,
                episode.episode_id,
            ),
        )
        log.info(
            "  uniform spawn: (%.1f, %.1f) (spawn_seed=%d, episode=%s)",
            *episode.start_xy,
            int(cfg.spawn_seed),
            episode.episode_id,
        )
    h = build_episode(
        cfg, episode, assets, bench, settings=settings, workdir=workdir, routing_key=routing_key
    )
    state = h.state
    ground = h.ground_z_m
    agl = cfg.start_altitude_agl_m
    log.info(
        "  ortho start altitude: %.2f m (%s, ground=%.2f, cap=%.0f, episode=%s)",
        state.altitude_m,
        f"AGL {agl:.0f} m" if agl is not None else f"row z={episode.start_z:.2f}",
        ground,
        float(cfg.altitude_cap_m),
        episode.episode_id,
    )
    h.world.set_retro_perceiver(functools.partial(retro_perceive, h))

    # The takeoff pose, before the mission moves the drone: the replay flies
    # from exactly here, facing upstream's start heading.
    start_pose_xyh = (
        float(state.pose_xy[0]),
        float(state.pose_xy[1]),
        float(episode.start_heading_deg),
    )
    start_alt = float(state.altitude_m)
    session = MissionSession(h, recorder=recorder)
    reply = (converse or run_mission_episode)(session)
    flights = session.flights
    if flights:
        visited = synthesize_compliant_trajectory(
            cfg=cfg,
            flights=flights,
            start_pose_xyh=start_pose_xyh,
            start_alt=start_alt,
        )
        steps_taken = len(visited)
        final_xy = visited[-1] if visited else start_pose_xyh[:2]
    else:
        # ZERO flights: the drone never took off and nothing was committed.
        steps_taken = 0
        final_xy = (float(episode.start_xy[0]), float(episode.start_xy[1]))
        visited = [final_xy, final_xy]

    # Commit identity is the last flight's: it may be an "[x, y]" point
    # token (provenance "point"), which no fid prefix classifies.
    last = flights[-1] if flights else {}
    commit_xy = (float(last["resolved_xy"][0]), float(last["resolved_xy"][1])) if flights else None
    scores = score_arrival(
        criterion=bench.success_criterion,
        final_pos_xy=final_xy,
        target_xy=episode.target_xy,
        pos_trajectory=[(float(p[0]), float(p[1])) for p in visited],
        success_dist_m=cfg.success_radius_m,
        gt_polygon=success_shape(episode, objects_by_map),
        commit_xy=commit_xy,
        flew=bool(flights),
    )
    # ``referent_bucket`` is a noisy text proxy (regex over the description).
    scores["referent_bucket"] = classify_referent(episode.description)
    scores["intel_quality"] = episode.intel_quality
    scores["steps_taken"] = steps_taken
    # What the episode spent on the LLM, at row size, read after the mission
    # from its transcript; a mission that never flew spent turns too.
    scores["llm"] = (
        flight_summary.summarize(Path(workdir) / "flight", settings.endpoint.tokenizer_name)
        if workdir
        else flight_summary.blank()
    )
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
        "target_xy": list(episode.target_xy) if episode.target_xy is not None else None,
        # The point the episode is judged at on a shape bench; None when it
        # never flew. The ring scores the replay's final pose instead.
        "commit_xy": list(commit_xy) if commit_xy is not None else None,
        "visited_xy": [list(p) for p in visited],
        "metrics": scores,
    }
