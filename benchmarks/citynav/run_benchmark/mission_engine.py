"""Mission engine: the CityNav bench episode as ONE reasoner conversation.

The operator instruction is the single user message; every ``act.viewpoints``
item FLIES the moment it lands (:func:`fly_one` renders, segments, and folds
that stop's perception into the live world-state); a plain no-tool-call reply
ends the mission. The commit is the LAST flight's target — no ground-walk, no
fallback tiers. Zero flights is an honest zero the episode runner scores.
"""

from __future__ import annotations

from benchmarks.citynav.adapter import CityNavEpisode
from benchmarks.citynav.episode import EpisodeState
from benchmarks.citynav.live_recorder import LiveRecorder
from benchmarks.citynav.paths import CITYNAV_NATIVE_RGBD_DIR
from benchmarks.citynav.run_benchmark.assets import BlockAssets
from benchmarks.citynav.run_benchmark.config import RunConfig
from benchmarks.citynav.run_benchmark.osm_prior import _build_frame_enrich
from benchmarks.citynav.run_benchmark.tour_cursor import _write_tour_cursor
from maps.search.withhold import withheld
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from maps.perception.observation import crop_observation
from maps.perception.ortho.detection_stamping import (
    project_and_stamp_ortho_detections,
)
from maps.perception.ortho.ground_level import GROUND_LEVEL
from maps.search import SearchLoop, MapsCompiler
from maps.search.parse_compile.conversation import (
    CONVERSATION_JSONL,
    ConversationLog,
)
from maps.search.parse_compile.reasoner_loop import MissionSpec, run_mission
from maps.search.parse_compile.tour_state import (
    VIEW_POSITION_CURRENT,
    VIEW_POSITION_MARKER,
    FeatureTour,
    TourViewpoint,
)
from maps.search.pose import RobotPose
from maps.util.xy_token import parse_xy_token
import os
import time
import logging

log = logging.getLogger("run_benchmark")

# The mission's flight budget: a NEVER-SERVED liveness backstop (the model
# hears about no budget; flights are bounded by turns anyway). Sits far
# above every observed episode (max 10 flights over 1,100 flown missions).
MISSION_FLIGHT_BUDGET = 1000

# The wall-clock half of the same backstop. The harness exports an ABSOLUTE
# epoch (see ``bench_fullset``) because it owns the one true episode start and
# the cap; re-deriving an elapsed here would drift past whatever pre-flight
# pose resolution and ingest already spent. Unset = no clock, which is what a
# non-harness caller (the field station, a bare smoke) gets.
MISSION_DEADLINE_ENV = "CITYNAV_EPISODE_DEADLINE_EPOCH"


def _mission_deadline_epoch() -> float | None:
    """The absolute epoch a pass must stop flying at, or ``None``.

    Read at CALL time rather than import time: the runner is a warm process
    that flies many episodes, and each one is handed its own deadline.
    """
    raw = os.environ.get(MISSION_DEADLINE_ENV, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        # A malformed deadline is a wiring bug, and falling back to "no
        # clock" would restore the exact failure this exists to end -- but
        # so would refusing to fly. Say so and fly unbounded.
        log.warning("%s=%r is not a number; flying with no wall-clock "
                    "backstop", MISSION_DEADLINE_ENV, raw)
        return None


class _StopPhases:
    """Where one stop's wall time went, as a log line per stop.

    A Cal City stop measured 37-67 s across every arm on the bake, and the
    breakdown was nowhere: `wall_t_monotonic` in the step record says how long
    the stop took and nothing about what it spent. Two episodes then burned a
    whole 4 h cap inside one `explore`, and answering "on what" meant guessing.

    MARKS, not nested context managers: a phase boundary is a single line
    inserted between two statements, so adding one never reindents the code it
    measures. `perf_counter` is ~50 ns, so this is free against phases that
    run in tenths of seconds.
    """

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._last = self._t0
        self._marks: list[tuple[str, float]] = []

    def at(self, name: str) -> None:
        now = time.perf_counter()
        self._marks.append((name, now - self._last))
        self._last = now

    def note(self, sub: dict | None) -> None:
        """Fold a callee's own breakdown into this stop's line.

        A phase that is one call from here can still be two passes inside it,
        and the mark cannot see that. ``_build_frame_enrich`` returns its own
        split rather than taking a timer argument, so the seam stays a return
        value and not a parameter threaded through the signature.
        """
        for k, v in (sub or {}).items():
            self._marks.append((k, float(v)))

    def line(self) -> str:
        parts = " ".join(f"{n}={d:.2f}" for n, d in self._marks)
        return f"{parts} total={self._last - self._t0:.2f}"


def resolve_flight_target(
    token: str,
    *,
    node_xy,
    pose_xy: tuple[float, float],
    marker_xy: tuple[float, float] | None,
) -> dict:
    """Resolve one viewpoint target token to a world point + provenance.

    ``token`` is the item's ``target_fid``: a literal fid (``node_xy(fid) ->
    ((x, y), "detected"|"prior") | None`` over the LIVE graph), a
    view-position sentinel, or the canonical ``"[x, y]"`` world-point token.
    Returns ``{"xy": (x, y), "provenance": ...}`` or ``{"error": ...}`` (the
    fid is on neither graph).

    A NAMED THING FLIES ITS MIDDLE, and there is no aim to move it. The
    model naming a point on the thing was the other way to reach a PART of
    a region, and it cannot: probed directly with crops and the world
    window it returned the crop's CENTER on 20/21 parsed replies --
    midpoint arithmetic with no visual pointing in it -- and every aim of
    the first live code-surface episodes landed 23-38 m off the footprint
    it named. A part is reached by COVERING the thing (``explore``) and
    flying to the row the segmenter put there: SAM's target-class
    detection sits a median 1.5 m from GT on 83% of rows, which is
    measurement where the aim was estimate.
    """
    xy = parse_xy_token(token)
    if xy is not None:
        return {"xy": (float(xy[0]), float(xy[1])), "provenance": "point"}
    if token == VIEW_POSITION_CURRENT:
        return {
            "xy": (float(pose_xy[0]), float(pose_xy[1])),
            "provenance": "point",
        }
    if token == VIEW_POSITION_MARKER:
        if marker_xy is None:
            return {"error": "no marked region on this episode"}
        return {
            "xy": (float(marker_xy[0]), float(marker_xy[1])),
            "provenance": "point",
        }
    resolved = node_xy(token)
    if resolved is None:
        return {"error": f"unknown fid {token!r}; it is not on the surface"}
    nxy, provenance = resolved
    return {"xy": (float(nxy[0]), float(nxy[1])),
            "provenance": str(provenance)}


# The flight report serves what the flight just found, head-capped like a
# rendered block (geom_program's 12-row head). Never a silent cap: the
# report's ``new_detections`` count is the total, and ``detected_more``
# names what the head left out.
_DETECTED_HEAD = 12


def detected_digest(dets: list[dict], cap: int = _DETECTED_HEAD) -> dict:
    """Report fields for THIS flight's folded detections.

    Acting on a fresh detection should be one step, not an analyze away:
    105 of the 135 observed-and-zero-row episodes on
    f40088a:calcity_full_replyrepair never analyzed again after the
    detection existed.

    EVERY ARMED CLASS IS REPRESENTED. The head was the first ``cap`` in
    FOLD order, which is the order the segmenter happened to return, and
    one class routinely swamps it: a calcity pass folded 63 houses and 1
    pool, so 12 slots of fold order can spend all 12 on houses. In
    calcity::200#198 the ground-truth car was folded on five consecutive
    flights at conf 0.86-0.88 and 1.5 m from truth, and never once
    appeared in a report -- it sat among the 52 the head left out, and
    the model committed a car 46 m away that it already had.

    So the head takes the best of EACH class in turn: classes ordered by
    their strongest detection, then round-robin, each class's own rows by
    detector confidence. A class with one detection cannot be crowded out
    by a class with sixty. This is an ORDERING of what the head shows --
    the count is still the total and ``detected_more`` still names the
    remainder, so nothing is withheld that was not withheld before.
    """
    if not dets:
        return {}
    by_class: dict = {}
    for _i, d in enumerate(dets):
        by_class.setdefault(str(d.get("class") or ""), []).append((_i, d))
    for rows in by_class.values():
        # conf descending; fold order breaks a tie, so a run with no
        # confidences is exactly the old head.
        rows.sort(key=lambda t: (-float(t[1].get("conf") or 0.0), t[0]))
    lanes = sorted(by_class.values(),
                   key=lambda rows: (-float(rows[0][1].get("conf") or 0.0),
                                     rows[0][0]))
    head: list = []
    for _rank in range(max((len(r) for r in lanes), default=0)):
        for rows in lanes:
            if _rank < len(rows) and len(head) < cap:
                head.append(rows[_rank][1])
        if len(head) >= cap:
            break
    out: dict = {"detected": [
        {"fid": d["fid"], "class": d["class"], "xy": list(d["xy"])}
        for d in head]}
    if len(dets) > len(head):
        out["detected_more"] = len(dets) - len(head)
    return out


def echo_pose(ctx, report: dict, *, fallback_alt: float, heading_deg: float) -> None:
    """After every successful flight the ToolContext's pose echo follows the
    drone, so the loop's surface refresh reads the folded world from where
    the drone is. The graph needs no following: the context holds the live
    ``state.runtime_graph`` itself, which every fold mutates in place."""
    if not isinstance(report, dict) or "error" in report:
        return
    pxy = report.get("pose_xy") or ()
    if len(pxy) == 2:
        ctx.robot_pose = RobotPose(
            x=float(pxy[0]), y=float(pxy[1]),
            z=float(report.get("alt_m") or fallback_alt),
            yaw_deg=float(heading_deg),
            timestamp=time.time(),
        )


def fly_one(
    item: dict,
    *,
    flights: list[dict],
    feature_tour: FeatureTour,
    node_xy,
    marker_xy: tuple[float, float] | None,
    cfg: RunConfig,
    state: EpisodeState,
    assets: BlockAssets,
    episode: CityNavEpisode,
    sensor,
    tile_frame,
    perception,
    search_loop: SearchLoop,
    recorder: "LiveRecorder | None",
    t_start: float,
    observe=None,
) -> dict:
    """One flight: resolve -> teleport -> render -> segment -> fold.

    ``observe`` is the live station's: a callable returning what the
    camera sees NOW, posed by the receiver, or ``None`` when it has no
    frame or no fix. When it is given the observation is the aircraft's
    real frame, perceived from the pose it was taken at, and the stop is
    the target the block named, exactly as on the bench: the pass geometry
    is the crew's. The bench never passes it and this path is unchanged.

    The per-frame unit of the mission: render, detect, project, paint,
    enrich — run for ONE requested viewpoint the moment the act lands. Returns the model-facing flight report (compact;
    detections are pulled via analyze), or ``{"error": ...}`` when the
    target does not resolve. Appends the executed flight to ``flights``
    (the commit reads the last entry) and to ``feature_tour`` (the discrete
    replay flies the same stops).
    """
    token = str(item.get("target_fid") or "")
    resolved = resolve_flight_target(
        token,
        node_xy=node_xy,
        pose_xy=state.pose_xy,
        marker_xy=marker_xy,
    )
    if "error" in resolved:
        return {"error": resolved["error"]}
    target_xy = resolved["xy"]
    new_xy = target_xy

    # Hold the cruise altitude, clamped ground+1 .. cap (the ortho crop is
    # valid at any altitude; the clamp keeps the view square finite).
    _g = float(GROUND_LEVEL.get(episode.map_name, 0.0))
    new_alt = float(
        min(max(float(state.altitude_m), _g + 1.0), float(cfg.altitude_cap_m))
    )
    new_heading = float(state.heading_deg)

    # Pending view{boost} vocabulary activates for THIS flight's /segment.
    try:
        search_loop.activate_pending_boosts()
    except Exception:  # noqa: BLE001 — perception direction never kills a flight
        log.exception("boost activation failed; flight proceeds un-boosted")

    step_idx = len(flights)
    stop_id = f"stop_{step_idx:03d}"
    state.step_idx = step_idx
    if observe is not None:
        obs = observe()
        if obs is None:
            return {"error": "no live observation: the camera or the receiver has nothing to pose"}
        new_xy = (float(obs.pose_xy[0]), float(obs.pose_xy[1]))
        new_alt = float(obs.altitude_m)
        new_heading = float(obs.yaw_deg)
    state.pose_xy = new_xy
    state.altitude_m = new_alt
    state.heading_deg = new_heading

    # NORTH-UP, not the episode heading. Over a nadir ortho the heading
    # carries zero information and only rotates the segmenter's input, and a
    # target that reads north-up can vanish from detection at other yaws.
    # The served captions promise north-up pictures, and the retro-boost
    # re-render (episode_runner) draws at yaw 0 -- this makes the flight
    # observation say the same thing. The pose keeps its heading: the
    # replay's motion model still turns.
    _ph = _StopPhases()
    if observe is None:
        obs = crop_observation(
            new_xy,
            altitude_m=new_alt,
            yaw_deg=0.0,
            sensor=sensor,
            rgbd_dir=CITYNAV_NATIVE_RGBD_DIR,
            map_name=episode.map_name,
        )
    _ph.at("render")
    if obs is not None:
        obs.tile = tile_frame

    detections = (
        perception.detect(obs, new_xy, instruction_hint=episode.description)
        if perception is not None
        else []
    )
    _ph.at("segment")

    # Project detection masks to tile cells through the observation's
    # view square (an orthographic rendering of the sensor's footprint).
    tracked_projections: dict[int, object] = {}
    if obs is not None:
        tracked_projections = dict(
            project_and_stamp_ortho_detections(
                detections,
                view_corners_xy=obs.ortho_view_corners_xy,
                rgb=obs.rgb,
                tile=tile_frame,
            )
        )

    _ph.at("project")
    step_detections: list[dict] = []
    step_detections = search_loop.update_perception(
        step_idx=step_idx,
        detections=detections,
        projected_masks=tracked_projections,
        bounds_xy_min=(float(assets.bounds.x_min), float(assets.bounds.y_min)),
        cell_size_m=float(assets.cell_size_m),
    )
    _ph.at("fold")
    state.belief_snapshot = search_loop.snapshot()
    _ph.at("snapshot")

    if recorder is not None and obs is not None:
        try:
            recorder.write_step(
                step=step_idx,
                pose_xy=state.pose_xy,
                altitude_m=state.altitude_m,
                heading_deg=state.heading_deg,
                visited_xy=list(state.visited_xy),
                # The observation's own crop corners: the frame IS the ground
                # footprint, so the map draws what was cropped.
                fov_corners_xy=(
                    [tuple(c) for c in obs.ortho_view_corners_xy]
                    if obs.ortho_view_corners_xy is not None
                    else None
                ),
                detections=step_detections,
                wall_t_monotonic=time.monotonic() - t_start,
                frame_px=(int(obs.rgb.shape[0]) if obs.rgb is not None else None),
                sensor=(obs.sensor.name if obs.sensor is not None else None),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("mission step write failed step=%d: %s", step_idx, exc)

    _ph.at("record")
    # Prior color enrichment: one frame, built inline (no pool), upserted
    # onto the named priors of the graph.
    view_corners = (
        tuple(obs.ortho_view_corners_xy)
        if obs is not None and obs.ortho_view_corners_xy is not None
        else None
    )
    _built: dict | None = None
    if obs is not None and obs.rgb is not None:
        _built = _build_frame_enrich(
            obs=obs,
            step_idx=step_idx,
            view_corners=view_corners,
            assets=assets,
                build_prior=not withheld("prior"),
        )
        for pfid, pattrs, ppx in _built["prior"]:
            search_loop.upsert_prior_percept(pfid, pattrs, px=ppx)
        if _built["prior_exc"] is not None:
            search_loop.state.prior_enrich_fails += 1
            _nfail = search_loop.state.prior_enrich_fails
            if _nfail == 1 or _nfail % 50 == 0:
                log.warning(
                    "prior percept enrichment failed (n=%d) step=%d: %s",
                    _nfail,
                    step_idx,
                    _built["prior_exc"],
                    exc_info=_built["prior_exc"],
                )

    _ph.at("enrich")
    _ph.note(_built.get("phase_s") if _built else None)
    if observe is not None:
        # THE STOP IS THE THING ASKED FOR. The frame's own pose fed the fold
        # above and the recorder's step; from here the flight reads as it
        # does on the bench, at the target's own point. The pass geometry is
        # the crew's: it lives in frames.jsonl, steps.jsonl and the station
        # transcript, and never reaches the model. Served, it read as a miss
        # to fix: one sortie re-requested a correct target six times over a
        # 98-387 m offset before the aircraft left the area
        # (20260911T201524Z run03).
        new_xy = (float(target_xy[0]), float(target_xy[1]))
        state.pose_xy = new_xy
    # The flight is perceived: retire matched boosts, log the flown stop into
    # the retrospective address space, and re-push the plain watchlist.
    try:
        search_loop.retire_boosts_after_round(
            [
                {
                    "pose_xy": new_xy,
                    "alt_m": new_alt,
                    "stop_id": stop_id,
                    "target_fids": [token] if token else [],
                }
            ]
        )
    except Exception:  # noqa: BLE001 — perception direction never kills a flight
        log.exception("boost retirement failed (non-fatal)")

    flights.append(
        {
            "stop_id": stop_id,
            "target": token,
            "pose_xy": [float(new_xy[0]), float(new_xy[1])],
            "resolved_xy": [float(new_xy[0]), float(new_xy[1])],
            "provenance": str(resolved.get("provenance", "unresolved")),
            "step_idx": step_idx,
        }
    )
    feature_tour.viewpoints.append(
        TourViewpoint(
            fid=token,
            xy=(float(new_xy[0]), float(new_xy[1])),
            altitude_m=new_alt,
        )
    )
    _write_tour_cursor(feature_tour)

    report: dict = {
        "stop_id": stop_id,
        "target": token,
        "pose_xy": [round(float(new_xy[0]), 1), round(float(new_xy[1]), 1)],
        "alt_m": round(new_alt, 1),
        "new_detections": len(step_detections),
        **detected_digest(search_loop.last_flight_detections()),
        # The commit is the LAST flight's target, verbatim: state the
        # standing answer on every flight, so the fact the reply consumes
        # is present at decision time (measured: 9/100 episodes committed
        # a scouting point while the reply named an object). No budget
        # fact rides the report -- the model hears about no cap.
        "answer_if_you_reply_now": token,
        # RE2 cue echo: the operator instruction rides every flight report.
        "cue": episode.description,
    }
    cp = getattr(perception, "concept_provider", None)
    if cp is not None and hasattr(cp, "concepts") and not withheld("look_for"):
        report["watchlist"] = list(cp.concepts)
    _ph.at("report")
    # One line per stop, to the runner log. A pass flies its stops back to
    # back with no reasoner between them, so this is the only place the cost
    # of a stop is stated at all.
    log.info("stop %s phases: %s", stop_id, _ph.line())
    return report


@dataclass
class MissionSession:
    """Everything a mission conversation flies over, built once.

    The engine's half of the mission: the ToolContext the model reasons
    against, the MissionSpec its verbs call back into, and the two lists
    the flights accumulate in. Held open across operator turns -- the
    field console re-tasks a live mission by driving another turn over
    THIS session rather than rebuilding one, which is what keeps the
    world-state the first turn folded.
    """

    ctx: object
    mission_spec: MissionSpec
    flights: list[dict]
    feature_tour: FeatureTour
    boost: "object"
    obs_span_m: float | None
    obs_minor_m: float | None


def build_mission_session(
    *,
    cfg: RunConfig,
    state: EpisodeState,
    assets: BlockAssets,
    episode: CityNavEpisode,
    sensor,
    tile_frame,
    perception,
    search_loop: SearchLoop,
    recorder: "LiveRecorder | None",
    t_start: float,
    observe=None,
) -> MissionSession:
    """Wire the mission surface over the live SearchLoop.

    The seeded prior graph IS the analyze surface.
    """
    _markers = episode.marker_positions or []
    marker_xy = (
        (
            float(sum(p[0] for p in _markers) / len(_markers)),
            float(sum(p[1] for p in _markers) / len(_markers)),
        )
        if _markers
        else None
    )

    def _node_xy(fid: str):
        xy = search_loop.world_xy_for_node(fid)
        if xy is None:
            return None
        provenance = "detected" if search_loop._fid_is_detected(fid) else "prior"
        return ((float(xy[0]), float(xy[1])), provenance)

    def _declare_concepts(cleaned: list[dict]) -> dict:
        return search_loop.declare_mission_concepts(cleaned)

    # Arm the perception-direction surface (empty watchlist) so
    # view{watchlist}/view{boost} are live from the first turn; the model's
    # act.sam_concepts declarations mutate it latest-wins.
    search_loop.declare_mission_concepts([])

    # ToolContext over the LIVE runtime graph — the same construction the
    # examiner surface is built from, plus the start pose for the echo.
    with search_loop._lock:
        ctx = search_loop._tool_ctx_for_examiner()
    ctx.robot_pose = RobotPose(
        x=float(state.pose_xy[0]),
        y=float(state.pose_xy[1]),
        z=float(state.altitude_m),
        yaw_deg=float(state.heading_deg),
        timestamp=time.time(),
    )

    flights: list[dict] = []
    feature_tour = FeatureTour.empty()
    _fly_inner = partial(
        fly_one,
        observe=observe,
        flights=flights,
        feature_tour=feature_tour,
        node_xy=_node_xy,
        marker_xy=marker_xy,
        cfg=cfg,
        state=state,
        assets=assets,
        episode=episode,
        sensor=sensor,
        tile_frame=tile_frame,
        perception=perception,
        search_loop=search_loop,
        recorder=recorder,
        t_start=t_start,
    )

    def _fly(item: dict) -> dict:
        report = _fly_inner(item)
        echo_pose(ctx, report, fallback_alt=float(state.altitude_m),
                  heading_deg=float(state.heading_deg))
        return report

    from maps.perception.observation import (
        observation_extent_m,
        observation_minor_extent_m,
    )

    _ground = search_loop._ortho_ground_z_m
    obs_span_m = (
        observation_extent_m(float(state.altitude_m), float(_ground), sensor)
        if _ground is not None
        else None
    )
    # The footprint's SHORT side, same source: the row pitch the serpentine
    # coverage tiling can rely on whichever way the footprint is turned. It
    # binds into the stops callable HERE so the MissionSpec contract stays
    # ``(fid, spacing_m) -> stops``.
    _obs_minor_m = (
        observation_minor_extent_m(
            float(state.altitude_m), float(_ground), sensor)
        if _ground is not None
        else None
    )

    def _coverage_stops(fid: str, spacing_m: float):
        # THE MISSION'S OWN LIVENESS LIMIT is the pass's bound -- a stop is
        # a flight, so the flight budget already says how far a pass can
        # go, and a second smaller number would be a budget the pass
        # husbands. The model is never told either.
        return search_loop.node_coverage_stops(
            fid, spacing_m, max_stops=MISSION_FLIGHT_BUDGET,
            minor_spacing_m=_obs_minor_m)

    mission_spec = MissionSpec(
        fly=_fly,
        declare_concepts=_declare_concepts,
        # WHERE AN EXPLORE'S STOPS COME FROM. The loop owns the pass (it
        # flies each stop and the return leg through ``fly``); the shape
        # is the world-state's, so the geometry stays on the SearchLoop
        # and reaches the loop the same way every other engine fact does.
        coverage_stops=_coverage_stops,
        flight_budget=MISSION_FLIGHT_BUDGET,
        turn_cap=100,
        deadline_epoch=_mission_deadline_epoch(),
    )
    return MissionSession(
        ctx=ctx,
        mission_spec=mission_spec,
        flights=flights,
        feature_tour=feature_tour,
        boost=search_loop._boost_queue_cb,
        obs_span_m=obs_span_m,
        obs_minor_m=_obs_minor_m,
    )


def run_mission_episode(
    *,
    cfg: RunConfig,
    state: EpisodeState,
    assets: BlockAssets,
    episode: CityNavEpisode,
    sensor,
    tile_frame,
    perception,
    search_loop: SearchLoop,
    recorder: "LiveRecorder | None",
    compiler: MapsCompiler,
    t_start: float,
) -> dict:
    """Drive one mission conversation and commit the last flight's target.

    Builds the session (:func:`build_mission_session`), runs
    :func:`run_mission` over it, then binds the commit onto the engine state
    the trajectory replay and metrics read (``confirmed_target_xy`` /
    ``lock_step``). Returns ``{"flights", "reply",
    "feature_tour", ...}``; zero flights leaves the engine state
    unbound so the runner scores the honest zero.
    """
    session = build_mission_session(
        cfg=cfg,
        state=state,
        assets=assets,
        episode=episode,
        sensor=sensor,
        tile_frame=tile_frame,
        perception=perception,
        search_loop=search_loop,
        recorder=recorder,
        t_start=t_start,
    )
    ctx = session.ctx
    flights = session.flights
    feature_tour = session.feature_tour
    mission_spec = session.mission_spec
    obs_span_m = session.obs_span_m
    _obs_minor_m = session.obs_minor_m
    _boost = session.boost
    seed = compiler._resolve_seed_base() & 0x7FFFFFFF
    workdir = os.environ.get("SEARCH_COMPILE_WORKDIR", "").strip() or None
    transcript = ConversationLog(
        Path(workdir) / CONVERSATION_JSONL if workdir else None
    )
    diag: dict = {}
    reply, _ = run_mission(
        instruction=episode.description,
        ctx=ctx,
        mission_spec=mission_spec,
        reasoner_url=compiler.reasoner_url,
        endpoint_spec=compiler.endpoint_spec,
        api_key=compiler.api_key,
        seed=seed,
        sensor_description=compiler.sensor_description,
        cancel_event=None,
        look_renderer=search_loop._render_candidate_crops,
        look_audit_dir=search_loop._look_audit_dir(),
        obs_span_m=obs_span_m,
        # The footprint's SHORT side arms the look coverage gate: pixels
        # are earned by flying, and a crop on ground no flight has covered
        # refuses toward explore.
        obs_minor_m=_obs_minor_m,
        boost_queue=_boost,
        transcript=transcript,
        diag=diag,
    )

    if flights:
        last = flights[-1]
        with search_loop._lock:
            s2 = search_loop.state
            s2.confirmed_target_xy = (
                float(last["resolved_xy"][0]),
                float(last["resolved_xy"][1]),
            )
            s2.lock_step = int(last["step_idx"])
        state.belief_snapshot = search_loop.snapshot()
        log.info(
            "mission commit: fid=%s xy=%s flights=%d ended_by=%s",
            last["target"],
            last["resolved_xy"],
            len(flights),
            reply.ended_by,
        )
    else:
        log.warning(
            "mission ended with ZERO flights (ended_by=%s); honest zero",
            reply.ended_by,
        )

    transcript.append(
        {
            "stage": "mission",
            "kind": "summary",
            "ended_by": reply.ended_by,
            "turns_used": reply.turns_used,
            "flights_flown": len(flights),
            "no_call_turns": reply.no_call_turns,
            "diag": diag,
        }
    )

    return {
        "flights": flights,
        "reply": reply,
        "feature_tour": feature_tour,
    }
