"""Mission engine: one episode as ONE reasoner conversation.

The operator instruction is the single user message; every move FLIES the
moment it lands (:meth:`MissionSession.fly` renders, segments, and folds that
stop's perception into the live world-state); a plain no-tool-call reply ends
the bench mission. The commit is the LAST flight's target. Zero flights is an
honest zero the episode runner scores.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from benchmarks.citynav.paths import CITYNAV_NATIVE_RGBD_DIR
from benchmarks.citynav.run_benchmark.osm_prior import _build_frame_enrich
from benchmarks.citynav.run_benchmark.tour_cursor import _write_tour_cursor
from maps.llm.endpoint_spec import spec as endpoint_spec
from maps.perception.geometry.sensor_spec import served_description
from maps.perception.observation import (
    crop_observation,
    observation_extent_m,
    observation_minor_extent_m,
)
from maps.perception.ortho.detection_stamping import project_and_stamp_ortho_detections
from maps.perception.ortho.ground_level import GROUND_LEVEL
from maps.search.parse_compile.conversation import CONVERSATION_JSONL, ConversationLog
from maps.search.parse_compile.reasoner_loop import MissionSpec, run_mission
from maps.search.parse_compile.tour_state import (
    VIEW_POSITION_CURRENT,
    VIEW_POSITION_MARKER,
    FeatureTour,
    TourViewpoint,
)
from maps.search.withhold import withheld
from maps.util.xy_token import parse_xy_token

log = logging.getLogger("run_benchmark")

# The mission's flight budget: a NEVER-SERVED liveness backstop (the model
# hears about no budget; flights are bounded by turns anyway). Sits far
# above every observed episode (max 10 flights over 1,100 flown missions).
MISSION_FLIGHT_BUDGET = 1000

# The wall-clock half of the same backstop: an ABSOLUTE epoch the bench driver
# exports, because it owns the episode start and the cap. Unset = no clock,
# which is what the field station and a bare smoke get.
MISSION_DEADLINE_ENV = "CITYNAV_EPISODE_DEADLINE_EPOCH"


def _seed_base() -> int:
    """``SEARCH_SEED_BASE`` if set, else a wall-clock nanosecond stamp, as a
    32-bit seed."""
    env_seed_base = os.environ.get("SEARCH_SEED_BASE")
    if env_seed_base is not None:
        return int(env_seed_base) & 0xFFFFFFFF
    return time.time_ns() & 0xFFFFFFFF


def _mission_deadline_epoch() -> float | None:
    """The absolute epoch a pass must stop flying at, or ``None``."""
    raw = os.environ.get(MISSION_DEADLINE_ENV, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        # A malformed deadline is a wiring bug; refusing to fly would cost
        # the episode, so say so and fly unbounded.
        log.warning("%s=%r is not a number; flying with no wall-clock "
                    "backstop", MISSION_DEADLINE_ENV, raw)
        return None


class _StopPhases:
    """Where one stop's wall time went, as one log line per stop: a mark
    between two statements per phase, so adding one never reindents the
    code it measures."""

    def __init__(self) -> None:
        self._t0 = self._last = time.perf_counter()
        self._marks: list[tuple[str, float]] = []

    def at(self, name: str) -> None:
        now = time.perf_counter()
        self._marks.append((name, now - self._last))
        self._last = now

    def note(self, sub: dict | None) -> None:
        """Fold a callee's own breakdown (``_build_frame_enrich``'s split)
        into this stop's line."""
        self._marks.extend((k, float(v)) for k, v in (sub or {}).items())

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

    A NAMED THING FLIES ITS MIDDLE, with no aim to move it: probed with crops,
    the model returned the crop's centre on 20/21 replies, and its aims landed
    23-38 m off the footprint it named. A part of a region is reached by
    covering it (``explore``) and flying to the row the segmenter put there
    (a median 1.5 m from GT on 83% of rows).
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
    """Report fields for THIS flight's folded detections, so acting on a
    fresh detection is one step, not an analyze away.

    EVERY ARMED CLASS IS REPRESENTED: classes ordered by their strongest
    detection, then round-robin, each class's rows by detector confidence,
    so one class with sixty rows cannot crowd out a class with one (a
    ground-truth car folded on five flights once never reached a report
    behind 52 houses). The count stays the total and ``detected_more``
    names the remainder.
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


def echo_pose(ctx, report: dict) -> None:
    """After every successful flight the ToolContext's pose echo follows the
    drone, so the loop's surface refresh reads the folded world from where
    the drone is. The graph needs no following: the context holds the live
    ``state.runtime_graph`` itself, which every fold mutates in place."""
    if not isinstance(report, dict) or "error" in report:
        return
    pxy = report.get("pose_xy") or ()
    if len(pxy) == 2:
        ctx.pose_xy = (float(pxy[0]), float(pxy[1]))


def perceive(h, obs, detections, step_idx: int, mark=lambda _phase: None) -> list[dict]:
    """Project and stamp one observation's detections through its view square
    and fold them into the live world-state; returns the folded rows. The
    one ingest path a flight and the bench's retrospective boost share."""
    projected = dict(project_and_stamp_ortho_detections(
        detections, view_corners_xy=obs.ortho_view_corners_xy, rgb=obs.rgb, tile=h.tile_frame,
    )) if obs is not None else {}
    mark("project")
    return h.search_loop.update_perception(
        step_idx=step_idx,
        detections=detections,
        projected_masks=projected,
        bounds_xy_min=(float(h.assets.bounds.x_min), float(h.assets.bounds.y_min)),
        cell_size_m=float(h.assets.cell_size_m),
    )


class MissionSession:
    """The engine's half of one mission conversation, built once over a
    hydrated episode (``episode_runner.Hydrated``): the ToolContext the
    model reasons against, the MissionSpec its verbs call back into, and
    the flights. The field station re-tasks a live mission by driving
    another turn over THIS session, which keeps the world-state the first
    turn folded.

    ``observe`` is the live station's: a callable returning what the camera
    sees NOW, posed by the receiver, or ``None`` when it has no frame or no
    fix. With it a flight perceives the aircraft's frame from the pose it
    was taken at, and the stop served is still the target the move named.
    The bench passes none and renders the raster crop at the target.
    """

    def __init__(self, h, *, recorder=None, t_start: float, observe=None) -> None:
        self.h = h
        self.recorder = recorder
        self.t_start = t_start
        self.observe = observe
        self.flights: list[dict] = []
        self.feature_tour = FeatureTour.empty()
        markers = h.episode.marker_positions or []
        self.marker_xy = (
            (float(sum(p[0] for p in markers) / len(markers)),
             float(sum(p[1] for p in markers) / len(markers)))
            if markers else None
        )
        loop = h.search_loop
        # Arm the perception-direction surface (empty watchlist) so
        # view{watchlist}/view{boost} are live from the first turn.
        loop.declare_mission_concepts([])
        with loop._lock:
            self.ctx = loop._tool_ctx_for_examiner()
        state = h.state
        self.ctx.pose_xy = (float(state.pose_xy[0]), float(state.pose_xy[1]))
        ground = loop._ortho_ground_z_m
        alt = float(state.altitude_m)
        obs_span_m = observation_extent_m(alt, float(ground), h.sensor) if ground is not None else None
        # The footprint's SHORT side: the row pitch an explore's serpentine
        # can rely on whichever way the footprint is turned, and the look
        # coverage gate (a crop on ground no flight covered refuses).
        self.obs_minor_m = (observation_minor_extent_m(alt, float(ground), h.sensor)
                            if ground is not None else None)
        self.mission_spec = MissionSpec(
            fly=self.fly,
            declare_concepts=loop.declare_mission_concepts,
            coverage_stops=self._coverage_stops,
            flight_budget=MISSION_FLIGHT_BUDGET,
            turn_cap=100,
            deadline_epoch=_mission_deadline_epoch(),
        )
        #: What every turn of the conversation takes, the first and each
        #: operator turn after it alike.
        self.run_kwargs = dict(
            ctx=self.ctx,
            mission_spec=self.mission_spec,
            look_renderer=loop._render_candidate_crops,
            look_audit_dir=loop._look_audit_dir(),
            obs_span_m=obs_span_m,
            obs_minor_m=self.obs_minor_m,
            boost_queue=loop._boost_queue_cb,
        )

    def run(self, instruction: str, **extra):
        """The conversation's first turn: ``run_mission`` over this session,
        the installed endpoint and the rig's served description. Returns
        ``(reply, reasoner)``."""
        h = self.h
        spec = endpoint_spec()
        return run_mission(
            instruction=instruction,
            reasoner_url=spec.base_url,
            endpoint_spec=spec,
            # "EMPTY" is the no-auth sentinel a local vLLM accepts.
            api_key=os.environ.get("SEARCH_REASONER_API_KEY", "EMPTY"),
            seed=_seed_base() & 0x7FFFFFFF,
            sensor_description=(served_description(h.sensor, h.cfg.start_altitude_agl_m)
                                if h.sensor is not None else ""),
            **self.run_kwargs,
            **extra,
        )

    def _coverage_stops(self, fid: str, spacing_m: float):
        # The flight budget bounds a pass: a stop IS a flight.
        return self.h.search_loop.node_coverage_stops(
            fid, spacing_m, max_stops=MISSION_FLIGHT_BUDGET, minor_spacing_m=self.obs_minor_m)

    def _node_xy(self, fid: str):
        loop = self.h.search_loop
        xy = loop.world_xy_for_node(fid)
        if xy is None:
            return None
        return ((float(xy[0]), float(xy[1])), "detected" if loop._fid_is_detected(fid) else "prior")

    def resolve(self, token: str) -> dict:
        """``token`` as :func:`resolve_flight_target` resolves it over the live graph."""
        return resolve_flight_target(token, node_xy=self._node_xy, pose_xy=self.h.state.pose_xy,
                                     marker_xy=self.marker_xy)

    def fly(self, item: dict) -> dict:
        """One flight (``MissionSpec.fly``), then the pose echo follows it."""
        report = self._fly(item)
        echo_pose(self.ctx, report)
        return report

    def _fly(self, item: dict) -> dict:
        """Resolve -> teleport -> render -> segment -> fold -> enrich for ONE
        move, the moment it lands. Returns the model-facing flight report, or
        ``{"error": ...}`` when the target does not resolve. Appends the flight
        to ``flights`` (the commit reads the last) and to ``feature_tour`` (the
        discrete replay flies the same stops)."""
        h = self.h
        state, episode, loop = h.state, h.episode, h.search_loop
        token = str(item.get("target_fid") or "")
        resolved = self.resolve(token)
        if "error" in resolved:
            return {"error": resolved["error"]}
        target_xy = resolved["xy"]
        new_xy = target_xy

        # Hold the cruise altitude, clamped ground+1 .. cap.
        _g = float(GROUND_LEVEL.get(episode.map_name, 0.0))
        new_alt = float(min(max(float(state.altitude_m), _g + 1.0), float(h.cfg.altitude_cap_m)))
        new_heading = float(state.heading_deg)

        # Pending view{boost} vocabulary activates for THIS flight's /segment.
        try:
            loop.activate_pending_boosts()
        except Exception:  # noqa: BLE001 — perception direction never kills a flight
            log.exception("boost activation failed; flight proceeds un-boosted")

        step_idx = len(self.flights)
        stop_id = f"stop_{step_idx:03d}"
        state.step_idx = step_idx
        if self.observe is not None:
            obs = self.observe()
            if obs is None:
                return {"error": "no live observation: the camera or the receiver has nothing to pose"}
            new_xy = (float(obs.pose_xy[0]), float(obs.pose_xy[1]))
            new_alt = float(obs.altitude_m)
            new_heading = float(obs.yaw_deg)
        state.pose_xy = new_xy
        state.altitude_m = new_alt
        state.heading_deg = new_heading

        # NORTH-UP, not the episode heading: over a nadir ortho the heading
        # only rotates the segmenter's input, the served captions promise
        # north-up pictures, and the retrospective re-render draws at yaw 0.
        # The pose keeps its heading: the replay's motion model still turns.
        _ph = _StopPhases()
        if self.observe is None:
            obs = crop_observation(
                new_xy,
                altitude_m=new_alt,
                yaw_deg=0.0,
                sensor=h.sensor,
                rgbd_dir=CITYNAV_NATIVE_RGBD_DIR,
                map_name=episode.map_name,
            )
        _ph.at("render")
        if obs is not None:
            obs.tile = h.tile_frame

        detections = (
            h.perception.detect(obs)
            if h.perception is not None
            else []
        )
        _ph.at("segment")
        step_detections = perceive(h, obs, detections, step_idx, mark=_ph.at)
        _ph.at("fold")

        if self.recorder is not None and obs is not None:
            try:
                self.recorder.write_step(step=step_idx, detections=step_detections)
            except Exception as exc:  # noqa: BLE001
                log.warning("mission step write failed step=%d: %s", step_idx, exc)

        _ph.at("record")
        # Prior colour enrichment: one frame, upserted onto the named priors.
        _built: dict | None = None
        if obs is not None and obs.rgb is not None:
            _built = _build_frame_enrich(
                obs=obs,
                step_idx=step_idx,
                view_corners=(tuple(obs.ortho_view_corners_xy)
                              if obs.ortho_view_corners_xy is not None else None),
                assets=h.assets,
                build_prior=not withheld("prior"),
            )
            for pfid, pattrs, ppx in _built["prior"]:
                loop.upsert_prior_percept(pfid, pattrs, px=ppx)
            if _built["prior_exc"] is not None:
                loop.state.prior_enrich_fails += 1
                _nfail = loop.state.prior_enrich_fails
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
        if self.observe is not None:
            # THE STOP IS THE THING ASKED FOR. The frame's own pose fed the
            # fold and the recorder above; from here the flight reads as on
            # the bench, at the target's own point. Served, the pass geometry
            # read as a miss to fix: one sortie re-requested a correct target
            # six times over a 98-387 m offset (20260911T201524Z run03).
            new_xy = (float(target_xy[0]), float(target_xy[1]))
            state.pose_xy = new_xy
        # The flight is perceived: retire matched boosts, log the flown stop
        # into the retrospective address space, re-push the plain watchlist.
        try:
            loop.retire_boosts_after_round(
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

        self.flights.append(
            {
                "stop_id": stop_id,
                "target": token,
                "pose_xy": [float(new_xy[0]), float(new_xy[1])],
                "resolved_xy": [float(new_xy[0]), float(new_xy[1])],
                "provenance": str(resolved.get("provenance", "unresolved")),
                "step_idx": step_idx,
            }
        )
        self.feature_tour.viewpoints.append(
            TourViewpoint(fid=token, xy=(float(new_xy[0]), float(new_xy[1])), altitude_m=new_alt)
        )
        _write_tour_cursor(self.feature_tour)

        report: dict = {
            "stop_id": stop_id,
            "target": token,
            "pose_xy": [round(float(new_xy[0]), 1), round(float(new_xy[1]), 1)],
            "alt_m": round(new_alt, 1),
            "new_detections": len(step_detections),
            **detected_digest(loop.last_flight_detections()),
            # The commit is the LAST flight's target, verbatim: state the
            # standing answer on every flight, so the fact the reply consumes
            # is present at decision time (measured: 9/100 episodes committed
            # a scouting point while the reply named an object).
            "answer_if_you_reply_now": token,
            # RE2 cue echo: the operator instruction rides every flight report.
            "cue": episode.description,
        }
        cp = getattr(h.perception, "concept_provider", None)
        if cp is not None and hasattr(cp, "concepts") and not withheld("look_for"):
            report["watchlist"] = list(cp.concepts)
        _ph.at("report")
        # One line per stop: a pass flies its stops back to back with no
        # reasoner between them, so this is the only place a stop's cost is stated.
        log.info("stop %s phases: %s", stop_id, _ph.line())
        return report


def run_mission_episode(session: MissionSession):
    """Drive the bench's one conversation and bind the last flight's target
    as the commit (``state.confirmed_target_xy``) the trajectory replay
    reads. Returns the reply; zero flights leaves it unbound, the honest
    zero."""
    h = session.h
    workdir = os.environ.get("SEARCH_COMPILE_WORKDIR", "").strip() or None
    transcript = ConversationLog(Path(workdir) / CONVERSATION_JSONL if workdir else None)
    diag: dict = {}
    reply, _ = session.run(h.episode.description, transcript=transcript, diag=diag)
    flights = session.flights
    if flights:
        last = flights[-1]
        h.state.confirmed_target_xy = (float(last["resolved_xy"][0]), float(last["resolved_xy"][1]))
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
    return reply
