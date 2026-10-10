"""Scoring/telemetry row assembly: the provisional pre-flight floor,
recorder step kwargs, and per-turn/graph histograms.
"""

from __future__ import annotations

from benchmarks.citynav.adapter import CityNavEpisode
from benchmarks.citynav.episode import EpisodeState
from benchmarks.citynav.episodes import STATUS_FLYING
from benchmarks.citynav.eval import WorldBounds
from maps.search.parse_compile.tour_state import FeatureTour
from benchmarks.citynav.metrics import classify_referent
from benchmarks.citynav.run_benchmark.config import RunConfig
from typing import Any
from maps.search import SearchSnapshot
import math
import numpy as np
import time


def _build_provisional_rec(
    *,
    episode: CityNavEpisode,
    cfg: RunConfig,
    snap,
    feature_tour: "FeatureTour | None",
    start_xy: tuple[float, float],
    criterion: str,
    mask: dict,
    bounds: tuple[float, float, float, float],
    sensor: Any = None,
) -> dict:
    """A scored, never-None pre-flight row, status ``flying``.

    Written to the attempt's ``episode.json`` right after pre-flight
    compile so a mid-flight crash (Slurm cgroup SIGKILL) leaves the
    compiler's best pre-flight binding on disk instead of nothing. The
    landed row overwrites it; the driver never scores it
    (``Driver._landed_row`` returns None on a ``flying`` row). ``pred_xy``
    is the best available pre-flight target: the engine's bound centroid
    if one exists, else the first tour stop, else the start pose. The row
    carries ``provisional: true`` so it is auditable and never confused
    with a flown result.
    """
    binding_xy: tuple[float, float] = (float(start_xy[0]), float(start_xy[1]))
    confirmed = getattr(snap, "confirmed_target_xy", None) if snap is not None else None
    if confirmed is not None:
        binding_xy = (float(confirmed[0]), float(confirmed[1]))
    elif feature_tour is not None and feature_tour.viewpoints:
        vp0 = feature_tour.viewpoints[0]
        binding_xy = (float(vp0.xy[0]), float(vp0.xy[1]))

    from benchmarks.citynav.eval import score_arrival

    # Under the same criterion the committed row will carry: this floor is
    # only ever read when a crash left nothing else, and a floor scored on a
    # different mask from the run around it is worse than no floor.
    scores = score_arrival(
        criterion=criterion,
        final_pos_xy=binding_xy,
        pred_xy=binding_xy,
        target_xy=episode.target_xy,
        pos_trajectory=[(float(start_xy[0]), float(start_xy[1])), binding_xy],
        pred_trajectory=[binding_xy],
        success_dist_m=cfg.success_radius_m,
        **mask,
    )
    scores["referent_bucket"] = classify_referent(episode.description)
    scores["steps_taken"] = 0
    scores["compile_s"] = 0.0
    scores["committed_fid"] = None
    scores["committed_provenance"] = None
    scores["committed_xy"] = None
    scores["commit_to_target_m"] = None
    scores["commit_would_score"] = None
    scores["commit_step"] = None
    return {
        "episode_id": episode.episode_id,
        "map_name": episode.map_name,
        "description": episode.description,
        "start_xy": list(episode.start_xy),
        "target_xy": list(episode.target_xy),
        "pred_xy": list(binding_xy),
        "final_pos_xy": list(binding_xy),
        "visited_xy": [],
        "compliant_parallel": True,
        "provisional": True,
        "status": STATUS_FLYING,
        "bounds": list(bounds),
        "sensor": sensor.to_dict() if sensor is not None else None,
        "metrics": scores,
    }


def _json_safe(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    return str(v)


def _build_write_step_kwargs(
    *,
    state: "EpisodeState",
    cfg: "RunConfig",
    step: int,
    t_start: float,
    step_detections: list[dict],
    frame_px: int | None,
    # The sensor the frame was rendered for, by registry name — present only
    # on sensor-footprint runs, where it (not frame_px) fixes the frame's
    # width × height for the re-render.
    sensor_name: str | None = None,
    # The observation's own crop corners. The frame IS the ground footprint
    # on this build, so the map draws what was cropped rather than a frustum
    # projected from a mount angle the ortho path never applies.
    fov_corners_xy: list[tuple[float, float]] | None,
    obs_class_hist: list[dict],
    compile_info: dict | None,
    planned_route: list[tuple[float, float]],
    replan_triggered: bool,
    graph_payload: dict | None,
    discrete_action: str | None = None,
    discrete_steps_emitted: int = 0,
) -> dict[str, Any]:
    """Assemble the kwargs both ``recorder.write_step`` call sites share.

    The early-step write (perception-only) and the post-compile write
    differ on ``compile_info / planned_route / replan_triggered /
    graph_payload``; everything else is per-step state + sensor
    cadence. Centralised so the schema only changes in one place.
    """
    return dict(
        step=step,
        pose_xy=state.pose_xy,
        altitude_m=state.altitude_m,
        heading_deg=state.heading_deg,
        visited_xy=list(state.visited_xy),
        fov_corners_xy=fov_corners_xy,
        detections=step_detections,
        compile_info=compile_info,
        wall_t_monotonic=time.monotonic() - t_start,
        frame_px=frame_px,
        sensor=sensor_name,
        obs_class_histogram=obs_class_hist,
        detected_node_class_histogram=_detected_node_class_histogram(
            state.belief_snapshot.runtime_graph
            if state.belief_snapshot is not None
            else state.graph
        ),
        planned_route=planned_route,
        replan_triggered=replan_triggered,
        graph_payload=graph_payload,
        sample_rate_hz=(
            cfg.flight_model.camera_hz if cfg.flight_model is not None else None
        ),
        discrete_action=discrete_action,
        discrete_steps_emitted=discrete_steps_emitted,
    )




def _detected_node_class_histogram(graph) -> list[dict]:
    """Per-class counts of ``provenance="detected"`` nodes in the runtime
    graph, sorted by descending count.

    Telemetry for the two-layer perception contract: the ``car`` count
    layer and the overlapping ``gray car`` color layer must appear as
    separate classes (a collapse would show one class only). Read directly
    off the ``steps.jsonl`` row to verify a fresh preflight formed both
    layers without reconstructing the graph.
    """
    import collections

    counts: collections.Counter = collections.Counter()
    nodes = getattr(graph, "nodes", None) or {}
    for n in nodes.values():
        if getattr(n, "provenance", None) == "detected":
            counts[n.class_name] += 1
    return [
        {"class_name": cls, "count": int(cnt)}
        for cls, cnt in counts.most_common()
    ]


def _obs_class_histogram(
    semantic: np.ndarray,
    alpha: np.ndarray | None = None,
    height: np.ndarray | None = None,
) -> list[dict]:
    """Return a short summary describing the FoV crop.

    For CityNav's photoreal rasters the R channel carries no semantic
    meaning, so we report coverage (fraction of pixels with point cloud
    data) and the height range over covered pixels. Returns a list of
    one-key dicts so the dashboard's `FoV classes:` caption can render
    `coverage 95%  •  ground 4.2-12.1 m`.
    """
    items: list[dict] = []
    if alpha is not None:
        mask = alpha > 0
        coverage = 100.0 * float(mask.mean())
        items.append(
            {
                "class_id": -1,
                "name": "coverage",
                "count": int(mask.sum()),
                "percent": round(coverage, 1),
            }
        )
        if height is not None and mask.any():
            h = height[mask]
            items.append(
                {
                    "class_id": -2,
                    "name": f"height {float(h.min()):.1f}-{float(h.max()):.1f} m",
                    "count": int(mask.sum()),
                    "percent": round(100.0 * float(mask.mean()), 1),
                }
            )
    elif height is not None and height.size:
        items.append(
            {
                "class_id": -2,
                "name": f"height {float(height.min()):.1f}-{float(height.max()):.1f} m",
                "count": int(height.size),
                "percent": 100.0,
            }
        )
    return items
